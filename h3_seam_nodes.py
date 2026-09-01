"""h3_seam_nodes.py — the H3 timeline seam layer as PACK nodes.

Ports the three local core commits out of comfy_extras/nodes_minimax_h3.py,
comfy/ldm/minimax/model.py and comfy/model_base.py into guarded overrides so
the whole estate runs STOCK ComfyUI (replicatability requirement,
2026-09-01). Ported:

  c37888e  first_frame_latent input, SaveLatent/LoadLatent .h3latent pack
           nodes, AddGuide backport
  001bf9a  audio-only keyframe guides — grid rows + cond collection
  b518a90  seam context — multi-frame head pin keyframes + backwards
           audio window (ends at the join)
  (+ WIP)  len(z.shape) instead of Tensor.dim() for shape-only stub latents

Layers
------
NODE LAYER (no core changes needed): MiniMaxH3AddGuide, MiniMaxH3SaveLatent,
MiniMaxH3LoadLatent — verbatim node classes.

NODE OVERRIDE: MiniMaxH3ImageToVideo extended with first_frame_latent /
context_steps / audio_context_steps, registered under the SAME node id
(custom nodes load after comfy_extras, so the override wins in
NODE_CLASS_MAPPINGS; existing graphs keep working and gain the inputs).

MODEL LAYER: PackedLayout.__init__ (attention grid rows for multi-frame
latent tails + backwards audio keyframes) and MiniMaxH3.extra_conds (cond
payload: skip audio-only keyframes in cond_video_latents, collect
cond_audio_latents) are replaced with the seam-aware versions. The
replacement functions are exec'd with the TARGET MODULE'S globals so they
behave exactly as if defined there.

GUARDS: if the running core already has seam support (the old patched
checkout), the module detects it and no-ops — one pack, both estates,
never double-applied. Detection: PackedLayout.__init__ mentions
audio_backwards / comfy_extras has MiniMaxH3AddGuide.
"""
import hashlib
import math
import os

import torch
import torchaudio
import safetensors.torch

import folder_paths
import nodes
import comfy.model_management
import comfy.nested_tensor
import node_helpers
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_api.latest import io

# stock-core helpers the ported code reuses (present in stock v0.33+)
import comfy_extras.nodes_minimax_h3 as _core_h3

_resize = _core_h3._resize
_empty_av_latent = _core_h3._empty_av_latent


# ── helpers (ported verbatim) ────────────────────────────────────────

def _encode_ref_audio(audio_vae, audio):
    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]


def _seam_latent(prev_samples):
    """Extract the raw last-video-frame latent from a prior window's AV pack."""
    s = prev_samples["samples"] if isinstance(prev_samples, dict) else prev_samples
    video = s.unbind()[0] if getattr(s, "is_nested", False) else s
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError("first_frame_latent expects a MiniMax H3 AV latent (video [B,24,T,h,w])")
    return video[:, :, -1:, :, :]


def _seam_tail(prev_samples, video_steps, audio_steps):
    """Slice a motion-context tail (video) + a backwards audio window from a
    prior window's AV latent pack. video_steps L: latent frames from the end
    (1 = the classic single-frame seam). audio_steps A: audio latent steps
    from the end (0 = none). Returns (video_tail, audio_tail) raw latents —
    no decode, no re-encode; the numbers the model produced are the numbers
    the next window is conditioned on."""
    s = prev_samples["samples"] if isinstance(prev_samples, dict) else prev_samples
    if getattr(s, "is_nested", False):
        parts = s.unbind()
        video, audio = parts[0], parts[1]
    else:
        raise ValueError("seam context expects the joint AV pack (video+audio)")
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError("first_frame_latent expects a MiniMax H3 AV latent (video [B,24,T,h,w])")
    L = max(1, int(video_steps))
    if L > video.shape[2]:
        raise ValueError("context_steps {} exceeds the prior window's latent depth {}".format(L, video.shape[2]))
    video_tail = video[:, :, -L:, :, :]
    audio_tail = None
    A = int(audio_steps)
    if A > 0:
        if audio is None or audio.dim() != 4:
            raise ValueError("audio_context_steps > 0 needs the prior pack's audio stream [B,32,2,T]")
        if A > audio.shape[-1]:
            raise ValueError("audio_context_steps {} exceeds the prior window's audio depth {}".format(A, audio.shape[-1]))
        audio_tail = audio[:, :, :, -A:]
    return video_tail, audio_tail


# ── node layer: MiniMaxH3ImageToVideo override ──────────────────────

class MiniMaxH3ImageToVideoSeam(io.ComfyNode):
    """Stock MiniMaxH3ImageToVideo + the seam inputs. Registered under the
    stock node id so graphs transparently gain the seam capability."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ImageToVideo",
            display_name="MiniMax H3 Image to Video",
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested)"),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Latent.Input("first_frame_latent", optional=True,
                                tooltip="Prior window's H3 AV latent: its raw last-frame video latent is injected as the frame-0 keyframe with NO decode/encode roundtrip (decoded once only for the vision tokens). Canvas must match."),
                io.Int.Input("context_steps", default=1, min=1, max=32,
                             tooltip="Latent frames of the prior window's tail to pin as this clip's motion context (1 = single-frame seam; 6-7 ≈ 22 pixel frames ≈ 0.9s is the field-tested sweet spot). The pinned head regenerates the carried motion; the gateway trims it from delivery."),
                io.Int.Input("audio_context_steps", default=0, min=0, max=160,
                             tooltip="Audio latent steps of tail sound to pin ENDING at the join (40 = 1s at 40Hz). Backwards-reaching window: the model continues the same recording instead of writing a sound-alike. 0 = off."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, length,
                first_frame=None, last_frame=None, first_frame_latent=None,
                context_steps=1, audio_context_steps=0) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        images = []
        keyframes = []
        if first_frame is not None:
            # geometry anchor: plain stretch to canvas
            img = _resize(first_frame[:1], width, height, "disabled")
            images.append(img)
            keyframes.append({"resolved_frame_index": 0, "image": img})
        if last_frame is not None:
            # follower: aspect-preserving cover-crop
            img = _resize(last_frame[:1], width, height, "center")
            images.append(img)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

        if first_frame_latent is not None:
            if first_frame is not None:
                raise ValueError("give either first_frame (image) or first_frame_latent, not both")
            seam, seam_audio = _seam_tail(first_frame_latent, context_steps, audio_context_steps)
            if tuple(seam.shape[3:5]) != (height // 16, width // 16):
                raise ValueError("first_frame_latent is {}x{} but the canvas is {}x{}; match the window canvas".format(
                    seam.shape[4] * 16, seam.shape[3] * 16, width, height))
            img = vae.decode(seam[:, :, -1:, :, :])
            if img.ndim == 5:
                # comfy VAE wrapper: [B, T, H, W, C] with T=1 for a single-frame latent
                img = img[:, 0]
            img = _resize(img, width, height, "disabled")
            images.append(img)
            # SEAM CONTEXT: the multi-frame tail pins at this clip's head;
            # the tail-audio window rides as a backwards-reaching keyframe
            # (ends at the join — continuation, not a reference take).
            kf = {"resolved_frame_index": 0, "latent": seam}  # raw latent: no encode
            if seam_audio is not None:
                kf["audio_latent"] = seam_audio
                kf["audio_rt"] = int(seam_audio.shape[-1])
                kf["audio_backwards"] = True
            keyframes.append(kf)

        tokens = clip.tokenize(prompt, images=images)
        cond = clip.encode_from_tokens_scheduled(tokens)

        if keyframes:
            for kf in keyframes:
                if "latent" not in kf:
                    kf["latent"] = vae.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_keyframes": keyframes,
                "minimax_frame_count": frame_count,
            })
        return io.NodeOutput(cond, latent)


# ── node layer: AddGuide / SaveLatent / LoadLatent (verbatim) ───────

class MiniMaxH3AddGuide(io.ComfyNode):
    """Anchor image and/or audio guides at an arbitrary pixel frame of the target video."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AddGuide",
            display_name="Add Guide for MiniMax H3",
            category="model/conditioning/minimax",
            description="Anchor an image, a short clip, audio, or a clip with its soundtrack at any frame of a MiniMax H3 video. Chain several nodes to anchor several frames.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Vae.Input("vae", optional=True, tooltip="Video VAE, needed when an image is connected."),
                io.Vae.Input("audio_vae", optional=True, tooltip="Audio VAE, needed when an audio is connected."),
                io.Latent.Input("latent"),
                io.Image.Input("image", optional=True, tooltip="Image or video frames to anchor. Multi-frame batches are anchored as a clip and cropped down to the model's valid clip lengths: 5, 22, 39... (17k + 5) frames. Batches shorter than 5 frames use only the first image."),
                io.Audio.Input("audio", optional=True,
                               tooltip="Soundtrack to anchor starting at the same frame index, cropped to the video's remaining duration."),
                io.Int.Input("frame_idx", default=0, min=-9999, max=9999,
                             tooltip="Frame index to anchor the image or the clip's first frame at. Negative values are counted from the end of the video."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive")],
        )

    @classmethod
    def execute(cls, positive, latent, frame_idx, vae=None, audio_vae=None, image=None, audio=None) -> io.NodeOutput:
        samples = latent["samples"]
        if not samples.is_nested or len(samples.tensors) != 2 or samples.tensors[0].ndim != 5 or samples.tensors[0].shape[1] != 24:
            raise ValueError("MiniMaxH3AddGuide expects a MiniMax H3 AV latent")
        if image is None and audio is None:
            raise ValueError("MiniMaxH3AddGuide needs an image or an audio to anchor")
        video = samples.tensors[0]
        height = video.shape[3] * 16
        width = video.shape[4] * 16
        frame_count = sum(FRAME_PER_TOKEN[k % 5] for k in range(video.shape[2]))

        guide_frames = 1
        if image is not None:
            if vae is None:
                raise ValueError("anchoring guide frames needs the vae input")
            guide_frames = image.shape[0]
            if guide_frames < 5:
                guide_frames = 1
            else:
                while guide_frames % 17 != 5:
                    guide_frames -= 1

        resolved_frame_index = frame_idx if frame_idx >= 0 else frame_count + frame_idx
        if resolved_frame_index < 0 or resolved_frame_index + guide_frames > frame_count:
            if guide_frames == 1:
                raise ValueError("frame_idx {} is outside the video's {} frames".format(frame_idx, frame_count))
            raise ValueError("a {} frame guide clip at frame_idx {} does not fit in the video's {} frames".format(
                guide_frames, frame_idx, frame_count))

        keyframe = {"resolved_frame_index": resolved_frame_index}
        if image is not None:
            frames = _resize(image[:guide_frames], width, height, "center")
            keyframe["latent"] = vae.encode(frames)

        if audio is not None:
            if audio_vae is None:
                raise ValueError("anchoring guide audio needs the audio_vae input")
            audio_latent, audio_rt = _encode_ref_audio(audio_vae, audio)
            max_rt = math.floor(samples.tensors[1].shape[-1] - FRAME_RESCALE * resolved_frame_index)
            if max_rt < 1:
                raise ValueError("frame_idx {} is past the end of the video's audio track".format(frame_idx))
            if audio_rt > max_rt:
                audio_latent = audio_latent[..., :max_rt].clone()
                audio_rt = max_rt
            keyframe["audio_latent"] = audio_latent
            keyframe["audio_rt"] = audio_rt  # token count, read by the DiT grid builder

        keyframes = list(positive[0][1].get("minimax_keyframes", []))
        keyframes.append(keyframe)
        positive = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})
        return io.NodeOutput(positive)


class MiniMaxH3SaveLatent(io.ComfyNode):
    """Persist a MiniMax H3 AV latent pack (video+audio NestedTensor) to disk."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SaveLatent",
            display_name="Save MiniMax H3 Latent",
            category="model/latent/minimax",
            description="Writes the joint video+audio H3 latent pack as a .h3latent safetensors file (video + audio tensors) for lossless cross-run seams.",
            inputs=[
                io.Latent.Input("samples"),
                io.String.Input("filename_prefix", default="h3latents/seam"),
            ],
            outputs=[io.Latent.Output()],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, samples, filename_prefix) -> io.NodeOutput:
        s = samples["samples"]
        if not getattr(s, "is_nested", False) or len(s.tensors) != 2:
            raise ValueError("MiniMaxH3SaveLatent expects a MiniMax H3 AV latent pack")
        out_dir = folder_paths.get_output_directory()
        full_dir, filename, counter, subfolder, prefix = folder_paths.get_save_image_path(filename_prefix, out_dir)
        file = "{}_{:05}_.h3latent".format(filename, counter)
        path = os.path.join(full_dir, file)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        safetensors.torch.save_file({
            "video": s.tensors[0].detach().to("cpu", torch.float32, non_blocking=False).contiguous(),
            "audio": s.tensors[1].detach().to("cpu", torch.float32, non_blocking=False).contiguous(),
            "format": torch.tensor([]),
        }, path)
        return io.NodeOutput(samples, ui={"h3latents": [{"filename": file, "subfolder": subfolder, "type": "output"}]})


class MiniMaxH3LoadLatent(io.ComfyNode):
    """Load a MiniMax H3 AV latent pack saved by MiniMaxH3SaveLatent."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LoadLatent",
            display_name="Load MiniMax H3 Latent",
            category="model/latent/minimax",
            description="Reads a .h3latent pack back as an H3 AV latent (NestedTensor video+audio).",
            inputs=[
                io.String.Input("latent", default="", multiline=False,
                                tooltip=".h3latent filename inside ComfyUI's input directory"),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        path = folder_paths.get_annotated_filepath(latent)
        data = safetensors.torch.load_file(path, device="cpu")
        if "video" not in data or "audio" not in data:
            raise ValueError("{} is not a .h3latent pack".format(latent))
        dev = comfy.model_management.intermediate_device()
        pack = comfy.nested_tensor.NestedTensor((data["video"].to(dev), data["audio"].to(dev)))
        return io.NodeOutput({"samples": pack})

    @classmethod
    def IS_CHANGED(cls, latent):
        path = folder_paths.get_annotated_filepath(latent)
        m = hashlib.sha256()
        with open(path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()


# ── model layer: PackedLayout.__init__ (seam-aware grid rows) ────────
# exec'd with comfy.ldm.minimax.model's own globals — the function is
# indistinguishable from one defined inside model.py.

_PACKED_LAYOUT_INIT_SRC = '''
def _seam_packed_layout_init(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None, frame_count=None):
    frame, w_grid = _frame_grid(latent_h, latent_w)
    frame_rows = frame.shape[0]

    segments = [("text", text_len)]  # (kind, n_rows)
    g = torch.zeros(text_len, 3, dtype=torch.float64)
    g[:, 0] = torch.arange(text_len, dtype=torch.float64)
    pos = [g]  # per segment: [n, 3] float64 (t, h, w)

    img_pos, img_update = [], []
    audio_pos, audio_update = [], []
    cursor = text_len
    row = text_len
    target_audio_w = (float(w_grid[0]), float(w_grid[-1]))

    if keyframes:
        # fl2va: keyframe cond rows right after text, sharing the target spatial grid.
        # Audio-only keyframes (soundtrack anchors without a visual guide)
        # contribute audio rows at the anchor time and no visual rows.
        #
        # SEAM CONTEXT (2026-08-28): a keyframe "latent" may carry a
        # MULTI-FRAME tail pack [1,24,L,h,w] — L latent frames from the
        # previous window's end. Those L frames pin at this clip's HEAD
        # times (_video_t_grid(L, cond_t), identical to the target grid's
        # first L frames) so the model regenerates the carried motion;
        # the gateway trims those frames from the delivery before concat.
        # A keyframe with audio_backwards=True places its audio window
        # ENDING at the join (cursor cond_t-rt) instead of starting at
        # it: rows tile flush against this clip's target audio
        # [cond_t, ...) with no overlap, so the model reads them as
        # "this clip, earlier" — a continuation — rather than "a
        # separate clip that sounds like this" — a sound-alike take.
        for kf in keyframes:
            pixel_index = kf["resolved_frame_index"]
            if pixel_index == 0:
                cond_t = float(text_len)
            elif frame_count is not None and pixel_index == frame_count - 1:
                cond_t = float(text_len) + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
            else:
                raise ValueError("only first/last keyframe anchors are supported")
            if "latent" in kf:
                z = kf["latent"]
                # len(shape) instead of Tensor.dim(): the h3_motion_context
                # layout self-test drives this path with a shape-only
                # stub latent, and torch tensors give the same answer.
                n_ctx = int(z.shape[2]) if len(z.shape) == 5 else 1
                g = torch.empty(n_ctx * frame_rows, 3, dtype=torch.float64)
                if n_ctx > 1:
                    g[:, 0] = _video_t_grid(n_ctx, cond_t).repeat_interleave(frame_rows)
                else:
                    g[:, 0] = cond_t
                g[:, 1:] = frame.repeat(n_ctx, 1)
                segments.append(("cond", n_ctx * frame_rows))
                pos.append(g)
                img_pos.append(torch.arange(row, row + n_ctx * frame_rows))
                img_update.append(torch.zeros(n_ctx * frame_rows, dtype=torch.bool))
                row += n_ctx * frame_rows
            if kf.get("audio_latent") is not None:
                rt = int(kf.get("audio_rt", 0))
                if rt > 0:
                    # same row kind as ref audio blocks: tag 2, packed audio rows
                    segments.append(("ref_audio", rt * 2))
                    acursor = cond_t - rt if kf.get("audio_backwards") else cond_t
                    pos.append(_audio_grid(acursor, rt, *target_audio_w))
                    audio_pos.append(torch.arange(row, row + rt * 2))
                    audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                    row += rt * 2

    if refs:
        cursor = float(text_len)
        for blk in refs:
            kind = blk["kind"]
            if kind == "image":
                r_frame, _ = _frame_grid(blk["latent_h"], blk["latent_w"])
                n = r_frame.shape[0]
                g = torch.empty(n, 3, dtype=torch.float64)
                g[:, 0] = cursor
                g[:, 1:] = r_frame
                segments.append(("ref_img", n))
                pos.append(g)
                img_pos.append(torch.arange(row, row + n))
                img_update.append(torch.zeros(n, dtype=torch.bool))
                row += n
                cursor += 1.0
            elif kind == "audio":
                rt = blk["ref_audio_t"]
                if rt > 0:
                    segments.append(("ref_audio", rt * 2))
                    pos.append(_audio_grid(cursor, rt, *target_audio_w))
                    audio_pos.append(torch.arange(row, row + rt * 2))
                    audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                    row += rt * 2
                cursor += float(rt)
            elif kind in ("video", "video_audio"):
                # the block's audio rows pack immediately before its video
                # rows, both sharing the cursor origin
                rt = blk["ref_audio_t"]
                vt = blk["latent_t"]
                r_frame, r_w_grid = _frame_grid(blk["latent_h"], blk["latent_w"])
                if rt > 0:
                    segments.append(("ref_audio", rt * 2))
                    pos.append(_audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))
                    audio_pos.append(torch.arange(row, row + rt * 2))
                    audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                    row += rt * 2
                n = vt * r_frame.shape[0]
                segments.append(("ref_img", n))
                pos.append(_video_grid(vt, r_frame, cursor))
                img_pos.append(torch.arange(row, row + n))
                img_update.append(torch.zeros(n, dtype=torch.bool))
                row += n
                cursor += max(float(rt), sum(_video_t_spans(vt)))

    # target audio then target video, always the last two segments
    segments.append(("audio", audio_t * 2))
    pos.append(_audio_grid(cursor, audio_t, *target_audio_w))
    audio_pos.append(torch.arange(row, row + audio_t * 2))
    audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
    row += audio_t * 2

    n_video = latent_t * frame_rows
    segments.append(("video", n_video))
    pos.append(_video_grid(latent_t, frame, cursor))
    img_pos.append(torch.arange(row, row + n_video))
    img_update.append(torch.ones(n_video, dtype=torch.bool))
    row += n_video

    self.seq_len = row
    self.position_ids = torch.cat(pos)  # [S, 3] float64
    self.img_pos = torch.cat(img_pos)
    self.audio_pos = torch.cat(audio_pos)
    self.img_update = torch.cat(img_update)
    self.audio_update = torch.cat(audio_update)
    self.segments = segments
    self.latent_t = latent_t
    # newer cores (upstream master) fast-path on a signature tuple in
    # _forward; the v0.33-era init this port came from predates it
    self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
'''

# ── model layer: MiniMaxH3.extra_conds (audio-only kf + cond_audio) ──

_EXTRA_CONDS_SRC = '''
def _seam_extra_conds(self, **kwargs):
    import comfy as _comfy
    out = super(type(self), self).extra_conds(**kwargs)
    cross_attn = kwargs.get("cross_attn", None)
    if cross_attn is not None:
        # run condition_proj + token refiner once per sampling instead of per step
        cross_attn = self.diffusion_model.preprocess_text_embeds(
            cross_attn.to(device=kwargs["device"], dtype=self.get_dtype_inference()))
        out['c_crossattn'] = _comfy.conds.CONDRegular(cross_attn)

    latent_shapes = kwargs.get("latent_shapes", None)
    if latent_shapes is not None:
        out['latent_shapes'] = _comfy.conds.CONDConstant(latent_shapes)

    # Everything H3-specific rides in one dict so _apply_model's dtype cast
    # (which would flatten fp32 cond latents and long tags to bf16) skips it.
    payload = {}
    tags = kwargs.get("minimax_token_tags", None)
    if tags is not None:
        payload["text_token_tags"] = tags
    keyframes = kwargs.get("minimax_keyframes", None)
    if keyframes is not None:
        payload["keyframes"] = keyframes
        payload["frame_count"] = kwargs.get("minimax_frame_count", None)
        # audio-only keyframes carry no "latent": they anchor a soundtrack
        # without a visual guide, so the visual cond list must skip them.
        payload["cond_video_latents"] = [kf["latent"] for kf in keyframes if "latent" in kf]
        kf_audio = [kf["audio_latent"] for kf in keyframes if kf.get("audio_latent") is not None]
        if kf_audio:
            payload["cond_audio_latents"] = kf_audio + payload.get("cond_audio_latents", [])
    refs = kwargs.get("minimax_refs", None)
    if refs is not None:
        payload["refs"] = refs
        payload["cond_video_latents"] = [r["latent"] for r in refs if "latent" in r]
        payload["cond_audio_latents"] = [r["audio_latent"] for r in refs if r.get("audio_latent") is not None]
    if kwargs.get("minimax_visual_cond_noise_aug", None) is not None:
        payload["visual_cond_noise_aug"] = kwargs["minimax_visual_cond_noise_aug"]
    if kwargs.get("minimax_audio_cond_noise_aug", None) is not None:
        payload["audio_cond_noise_aug"] = kwargs["minimax_audio_cond_noise_aug"]
    payload["seed"] = kwargs.get("seed", 0)
    # same value process_latent_in/out used, so the model never undoes a scale that was not applied
    payload["audio_scale"] = self.audio_scale()
    if cross_attn is not None and latent_shapes is not None and len(latent_shapes) > 1:
        # packed layout built once per sampling run, h/w rounded up to the DiT's 2x2 patch
        vs = latent_shapes[0]
        payload["layout"] = _comfy.ldm.minimax.model.PackedLayout(
            cross_attn.shape[1], vs[2], (vs[3] + 1) // 2 * 2, (vs[4] + 1) // 2 * 2,
            latent_shapes[1][-1], keyframes=payload.get("keyframes"),
            refs=payload.get("refs"), frame_count=payload.get("frame_count"))
    out['minimax_payload'] = _comfy.conds.CONDConstant(payload)
    return out
'''


def _core_has_seam_support() -> bool:
    """True only when the running core already provides the seam layout
    (legacy patched checkout, or this pack's earlier bind in the same
    process). Keys on the stamp first, then the source marker."""
    if getattr(_mm.PackedLayout, "_seam_patched_by_pack", None):
        return True
    import inspect
    try:
        return "audio_backwards" in inspect.getsource(_mm.PackedLayout.__init__)
    except Exception:
        return False


# The ONE line of true seam delta the modern core lacks: a keyframe audio
# window placed BACKWARDS from the join (continuation, not a sound-alike).
_DYNAMIC_MARKER = "pos.append(_audio_grid(cond_t, rt, *target_audio_w))"


def _apply_dynamic_delta(body: str):
    """Insert the audio_backwards placement line, indent derived from the
    marker's own indentation (immune to upstream re-nesting)."""
    lines = body.split("\n")
    for i, ln in enumerate(lines):
        if _DYNAMIC_MARKER in ln:
            indent = ln[:len(ln) - len(ln.lstrip())]
            lines[i] = (indent + "acursor = cond_t - rt if kf.get(\"audio_backwards\") else cond_t\n"
                        + indent + "pos.append(_audio_grid(acursor, rt, *target_audio_w))")
            return "\n".join(lines)
    return None


def apply_model_layer_patches():
    """Bind seam-aware model-layer code. Modern cores (kf latents + cond_audio
    natively): patch the RUNNING CORE'S OWN __init__ source with the single
    audio_backwards delta — zero drift, whatever the core version is. Legacy
    cores (v0.33-era, no kf latent support): the full static port below.
    Returns the mode: 'dynamic' | 'legacy' | None."""
    import inspect
    import textwrap

    if _core_has_seam_support():
        return None

    mode = None
    try:
        stock_src = inspect.getsource(_mm.PackedLayout.__init__)
    except Exception:
        stock_src = ""

    if stock_src and _DYNAMIC_MARKER in stock_src:
        body = textwrap.dedent(stock_src)
        body = body.replace("def __init__(", "def _seam_packed_layout_init(", 1)
        body = _apply_dynamic_delta(body)
        ns = dict(vars(_mm))
        exec(compile(body, __file__ + " (PackedLayout.__init__, dynamic seam delta)", "exec"), ns)
        _mm.PackedLayout.__init__ = ns["_seam_packed_layout_init"]
        setattr(_mm.PackedLayout, "_seam_patched_by_pack", "dynamic")
        mode = "dynamic"
    elif stock_src and "resolved_frame_index" in stock_src:
        # modern core but the marker moved (upstream refactor): loud skip —
        # forward audio placement still works, backwards windows do not
        print("[h3_seam_nodes] WARNING: PackedLayout kf-audio marker not found; "
              "audio_backwards windows disabled on this core build")
        setattr(_mm.PackedLayout, "_seam_patched_by_pack", "marker-miss")
        mode = "marker-miss"
    else:
        # legacy core (v0.33-era): full static port
        ns_mm = dict(vars(_mm))
        exec(compile(_PACKED_LAYOUT_INIT_SRC, __file__ + " (PackedLayout.__init__, legacy port)", "exec"), ns_mm)
        _mm.PackedLayout.__init__ = ns_mm["_seam_packed_layout_init"]
        setattr(_mm.PackedLayout, "_seam_patched_by_pack", "legacy")
        mode = "legacy"

    # extra_conds: bind the static port ONLY when the running core's own
    # lacks keyframe audio handling (legacy). Modern cores already ship a
    # superset — overriding them is pure drift risk.
    try:
        mb_src = inspect.getsource(_mb.MiniMaxH3.extra_conds)
    except Exception:
        mb_src = ""
    if "cond_audio_latents" not in mb_src:
        ns_mb = dict(vars(_mb))
        exec(compile(_EXTRA_CONDS_SRC, __file__ + " (MiniMaxH3.extra_conds)", "exec"), ns_mb)
        _mb.MiniMaxH3.extra_conds = ns_mb["_seam_extra_conds"]
    return mode


import comfy.ldm.minimax.model as _mm  # noqa: E402
import comfy.model_base as _mb  # noqa: E402

# ── activation ───────────────────────────────────────────────────────
# On a seam-patched core everything no-ops (no duplicate node ids, no
# double monkeypatch); on stock core the seam layer activates.

_CORE_SEAM = _core_has_seam_support()

if _CORE_SEAM:
    SEAM_NODE_CLASS_MAPPINGS = {}
    SEAM_NODE_DISPLAY_NAME_MAPPINGS = {}
    MODEL_LAYER_PATCHED = False
else:
    apply_model_layer_patches()
    MODEL_LAYER_PATCHED = True
    SEAM_NODE_CLASS_MAPPINGS = {
        # superset of the stock node (identical when seam inputs unused);
        # always registered on stock so graphs gain the seam inputs
        "MiniMaxH3ImageToVideo": MiniMaxH3ImageToVideoSeam,
    }
    # node classes the running core doesn't already ship (upstream master
    # includes MiniMaxH3AddGuide natively; register only what's missing)
    if getattr(_core_h3, "MiniMaxH3AddGuide", None) is None:
        SEAM_NODE_CLASS_MAPPINGS["MiniMaxH3AddGuide"] = MiniMaxH3AddGuide
    if getattr(_core_h3, "MiniMaxH3SaveLatent", None) is None:
        SEAM_NODE_CLASS_MAPPINGS["MiniMaxH3SaveLatent"] = MiniMaxH3SaveLatent
    if getattr(_core_h3, "MiniMaxH3LoadLatent", None) is None:
        SEAM_NODE_CLASS_MAPPINGS["MiniMaxH3LoadLatent"] = MiniMaxH3LoadLatent
    SEAM_NODE_DISPLAY_NAME_MAPPINGS = {
        k: {"MiniMaxH3ImageToVideo": "MiniMax H3 Image to Video",
            "MiniMaxH3AddGuide": "Add Guide for MiniMax H3",
            "MiniMaxH3SaveLatent": "Save MiniMax H3 Latent",
            "MiniMaxH3LoadLatent": "Load MiniMax H3 Latent"}[k]
        for k in SEAM_NODE_CLASS_MAPPINGS
    }
    # Same-id override, bound directly. ComfyUI's custom-node loader drops
    # NODE_CLASS_MAPPINGS entries whose key collides with a core registration
    # (nodes.py init_external_custom_nodes passes base_node_names as `ignore`),
    # so the superset ImageToVideo would be silently skipped in the dict
    # route. The direct assignment happens at import time, inside this pack —
    # same mechanism as the model-layer patches, zero core files touched.
    # The class is a strict superset: stock graphs render identically, the
    # three seam inputs are optional additions.
    import nodes as _nodes  # noqa: E402
    _prev = _nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3ImageToVideo")
    _nodes.NODE_CLASS_MAPPINGS["MiniMaxH3ImageToVideo"] = MiniMaxH3ImageToVideoSeam
    MiniMaxH3ImageToVideoSeam.RELATIVE_PYTHON_MODULE = \
        getattr(_prev, "RELATIVE_PYTHON_MODULE", "comfy_extras.nodes_minimax_h3")
    _nodes.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3ImageToVideo"] = "MiniMax H3 Image to Video"
