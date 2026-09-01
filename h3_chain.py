"""RayH3SaveLatentTail + RayH3ChainLatentAV — the latent weave primitives.

See the package docstring for the mechanism. Both nodes are plain
old-style ComfyNode classes (INPUT_TYPES/FUNCTION/RETURN_TYPES), the
same registration shape as the other melite-* in-tree packs.

(2026-08-30 container law) ComfyUI runs CONTAINERIZED with an overlay
view of the repo: a node writing to an absolute repo path writes into
the container's OWN layer — the bytes never reach the host, the gateway
never sees the tail (proven live: window1..5_tail.pt existed inside
/proc/<comfy>/root/... and nowhere else, while every run "succeeded"
with zero tails). Tails now cross the boundary through ComfyUI's own
artifact doors, like every other artifact:

  SAVE   writes <output_dir>/<path> (+ .meta.json) and RETURNS both as
         UI outputs -> /history lists them -> the gateway downloads via
         /view into the run dir (canonical window<k>_tail.pt names).
  CHAIN  reads <input_dir>/<tail_path> — the gateway uploads the fetched
         tail (+meta) via /upload/image before submitting the seam
         window. Relative paths ONLY: output_dir for save, input_dir
         for chain. Absolute paths are rejected loudly (the old
         absolute-path contract was the container bug).
"""
from __future__ import annotations

import json
import os

import torch


# ── The 17k+5 grid (mirrors comfy_extras/nodes_minimax_h3.py) ──────────────

def _align_frame_count(n: int) -> int:
    while n % 17 != 5:
        n += 1
    return n


def _video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _carried_pixel_frames(carry_slots: int, total_slots: int) -> int:
    """Pixel frames covered by the LAST ``carry_slots`` of a window whose
    video latent has ``total_slots`` slots.

    Grid: slots 0..1 cover frames 0..4; each further slot covers 17/5
    frames. Inverse of ((F-5)//17)*5+2 for the slot counts the model
    actually emits ((total_slots-2) % 5 == 0).
    """
    if total_slots <= 2 or carry_slots >= total_slots:
        # The whole window is carried (degenerate, but keep it total).
        frames = 5 if total_slots == 2 else 5 + (total_slots - 2) * 17 // 5
        return frames
    head = min(carry_slots, 2)
    tail = carry_slots - head
    return head * 5 // 2 + tail * 17 // 5


def _clamp_carry(carry: int, total_slots: int, tail_slots: int) -> int:
    """The one true clamp — the node, the C++ twin and the plan all use
    this shape: never 0, never the whole window, never past the tail."""
    return max(1, min(int(carry), total_slots - 1, tail_slots))


# ── Path law (container edition) ──────────────────────────────────────────

def _resolve(kind: str, path: str, node: str) -> str:
    """Resolve a RELATIVE artifact path against ComfyUI's output/input
    directory. Absolute paths are the container bug — reject loudly."""
    import folder_paths
    if os.path.isabs(path):
        raise ValueError(
            f"{node}: path must be RELATIVE to the ComfyUI {kind} "
            f"directory — got {path!r}. Absolute paths write into the "
            f"container's overlay layer and never reach the gateway "
            f"(the 2026-08-30 tail-loss lesson)."
        )
    base = (folder_paths.get_output_directory() if kind == "output"
            else folder_paths.get_input_directory())
    return os.path.join(base, path)


class RayH3SaveLatentTail:
    """Passthrough: persist the sampled AV latent for the next window.

    Wired between KSampler and VAEDecode in the SOURCE window. Saves
    BOTH streams (video + audio) — the chain node slices at load time,
    so the source keeps full flexibility. Atomic write (tmp + rename):
    the chain node either sees a complete file or none of it.

    OUTPUT_NODE: the .pt and its .meta.json land in ComfyUI's output
    directory under ``path`` and are returned as UI outputs so /history
    lists them (the gateway fetches via /view). The meta is authored
    WHERE THE TAIL IS BORN, from the same grid math the chain node and
    the gateway plan use — one source of truth, cross-checked at both
    ends (the chain node verifies its own clamp against this meta; the
    gateway verifies the plan's trims against the fetched copy).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "path": ("STRING", {
                    "tooltip": "Run-scoped RELATIVE .pt path inside the "
                               "ComfyUI output directory "
                               "(<run_id>/window<k>_tail.pt)",
                }),
                "carry_latent_frames": ("INT", {
                    "default": 6, "min": 1, "max": 64,
                    "tooltip": "Video latent TAIL SLOTS the next window "
                               "will carry — authors the .meta.json "
                               "trim numbers (must match the plan)",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "melite/video"
    OUTPUT_NODE = True

    def execute(self, latent, path, carry_latent_frames):
        nt = latent["samples"]
        video, audio = nt.tensors[0], nt.tensors[1]
        total_slots = int(video.shape[2])
        c = _clamp_carry(carry_latent_frames, total_slots, total_slots)
        carried_px = _carried_pixel_frames(c, total_slots)
        ca = max(0, min(round(c * 17 / 3), int(audio.shape[3]) - 1))

        target = _resolve("output", path, "RayH3SaveLatentTail")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".tmp"
        torch.save(
            {"video": video.detach().to("cpu", copy=True),
             "audio": audio.detach().to("cpu", copy=True)},
            tmp,
        )
        os.replace(tmp, target)

        subfolder, filename = os.path.split(path)
        meta = {
            "carried_video_slots": c,
            "carried_audio_frames": ca,
            "carried_pixel_frames": carried_px,
            "total_video_slots": total_slots,
            "source_tail_path": path,
        }
        meta_name = filename + ".meta.json"
        mtmp = os.path.join(os.path.dirname(target), meta_name + ".tmp")
        with open(mtmp, "w") as f:
            json.dump(meta, f)
        os.replace(mtmp, os.path.join(os.path.dirname(target), meta_name))

        return {"ui": {"outputs": [
            {"filename": filename, "subfolder": subfolder,
             "type": "output"},
            {"filename": meta_name, "subfolder": subfolder,
             "type": "output"},
        ]},
            "result": (latent,)}


class RayH3ChainLatentAV:
    """Splice the previous window's saved latent tail into this window.

    video: [B,24,T,H,W]   — the tail's last ``carry_latent_frames`` slots
                            replace this window's HEAD slots (dim 2).
    audio: [B,32,2,Ta]    — the tail's last ``carry_audio`` frames replace
                            this window's head frames (dim 3), phase-locked
                            (17/3 audio frames per video latent slot).

    Output: {"samples": NestedTensor(spliced), "noise_mask": _CarryMask}.
    The mask rides the RAW noise_mask passthrough into KSamplerX0Inpaint,
    which holds the carried slots bit-exact while the free slots denoise —
    the carried region decodes to the source window's tail EXACTLY.

    tail_path is RELATIVE to the ComfyUI INPUT directory: the gateway
    uploads the fetched window<k>_tail.pt (+meta) there via /upload
    before submitting this window. If the uploaded .meta.json is present
    its carried_video_slots MUST match this node's clamp — drift between
    the save-side authoring and the chain-side splice is a hard error,
    never a silent seam.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "The target window latent (zeros from "
                               "EmptyMiniMaxH3LatentAV/ImageToVideo are "
                               "fine — the free slots get fresh noise)",
                }),
                "tail_path": ("STRING", {
                    "tooltip": "Run-scoped RELATIVE path of the uploaded "
                               "tail inside the ComfyUI input directory",
                }),
                "carry_latent_frames": ("INT", {
                    "default": 6, "min": 1, "max": 64,
                    "tooltip": "Video latent TAIL SLOTS to carry "
                               "(5 slots ≈ 17 pixel frames ≈ 0.71s)",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "melite/video"

    def execute(self, latent, tail_path, carry_latent_frames):
        import comfy.nested_tensor

        src = _resolve("input", tail_path, "RayH3ChainLatentAV")
        tail = torch.load(src, map_location="cpu", weights_only=True)
        tv, ta = tail["video"], tail["audio"]
        tgt = latent["samples"]
        wv, wa = tgt.tensors[0].to(tv.device), tgt.tensors[1].to(ta.device)

        if tuple(tv.shape[-2:]) != tuple(wv.shape[-2:]):
            raise ValueError(
                f"RayH3ChainLatentAV: resolution mismatch — the saved tail "
                f"is {tuple(tv.shape[-2:])} but this window is "
                f"{tuple(wv.shape[-2:])}. Latent chaining requires the "
                f"same canvas across the chain (re-render the source or "
                f"match width/height)."
            )

        total_slots = int(wv.shape[2])
        c = _clamp_carry(carry_latent_frames, total_slots, int(tv.shape[2]))
        # Audio carry: 17/5 pixel frames per video slot, 40 audio frames
        # per second of 24fps video -> 17/3 audio frames per slot.
        ca = max(0, min(round(c * 17 / 3), int(ta.shape[3]) - 1,
                        int(wa.shape[3]) - 1))

        meta_src = src + ".meta.json"
        if os.path.exists(meta_src):
            with open(meta_src) as f:
                saved = json.load(f)
            if int(saved.get("carried_video_slots", c)) != c:
                raise ValueError(
                    f"RayH3ChainLatentAV: seam drift — the tail's meta "
                    f"was authored for carry={saved['carried_video_slots']} "
                    f"but this window clamps to {c} "
                    f"(tail {tv.shape[2]} slots, window {total_slots}). "
                    f"The plan, the save node and the chain node "
                    f"disagree; refusing to splice."
                )

        video = torch.cat([tv[:, :, -c:], wv[:, :, : total_slots - c]], dim=2)
        audio = torch.cat(
            [ta[:, :, :, -ca:], wa[:, :, :, : int(wa.shape[3]) - ca]], dim=3,
        )

        vmask = torch.ones(1, 1, video.shape[2], 1, 1)
        vmask[:, :, :c] = 0.0
        amask = torch.ones(1, 1, 1, audio.shape[3])
        amask[:, :, :, :ca] = 0.0

        out = {"samples": comfy.nested_tensor.NestedTensor((video, audio)),
               "noise_mask": comfy.nested_tensor.NestedTensor(
                   (vmask, amask))}
        return (out,)
