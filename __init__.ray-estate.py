"""Melite H3 Motion Nodes — latent-space window chaining for MiniMax H3.

The video-timeline weave's ``extend`` transition mode (TRUE latent
chaining, the remake of the deleted "H3 Motion Context" extension):
the previous window's SAMPLED AV latent tail is carried into the next
window — no VAE round-trip, no conditioning softness, audio
phase-locked. The ``cut`` mode (pixel/reference conditioning via
MiniMaxH3ImageToVideo keyframes + MiniMaxH3AddGuide) stays native.

Nodes
-----
RayH3SaveLatentTail
    LATENT passthrough that torch.save()s the full AV latent (video +
    audio streams) to an absolute path. Wired between KSampler and
    VAEDecode in the SOURCE window's graph.
RayH3ChainLatentAV
    Splices the saved tail into the TARGET window's latent and attaches
    a temporal carry mask via latent["noise_mask"] (raw passthrough —
    nodes.py common_ksampler hands it to KSamplerX0Inpaint untouched).
    The carried slots are held at their exact values through sampling;
    the DiT attends to them in-sequence (motion/texture carry).

Mechanism (ComfyUI pin e01fb4c):
  - Latents are NestedTensor pairs: video [B,24,T,H/16,W/16],
    audio [B,32,2,T*40/24].
  - The carry mask rides latent["noise_mask"] as a NESTED
    per-stream mask (NestedTensor((vmask, amask))) — the sampler's
    NATIVE path (comfy/samplers.py: ``denoise_mask.is_nested →
    unbind() → prepare per stream → repack flat``). Every KSAMPLER
    wraps KSamplerX0Inpaint, which pastes per step:
    out = out * mask + latent_image * (1 - mask) — the carried slots
    come through BIT-EXACT from the spliced latent (the x re-anchor is
    the schedule-consistent re-noising of the carried x0).
  - The 17k+5 grid: frame_count % 17 == 5; latent T = ((F-5)//17)*5 + 2
    (2 slots cover the first 5 frames, then 5 slots per 17). Carry
    slot counts snap to 2+5k (the compressor's group boundaries) so
    the pixel-frame trim is EXACT (5, 22, 39… frames).

In-tree pack: deployed from custom_nodes/melite-h3-motion-nodes (declared in
plugins/melite-video/nodes/extensions.yaml — §16 E-EXTENSIONS; see also the
docker-compose.yml inference-comfyui volumes, bind-mounted read-only into
/root/ComfyUI/custom_nodes/).
"""
from .h3_chain import RayH3ChainLatentAV, RayH3SaveLatentTail
from . import h3_seam_nodes

NODE_CLASS_MAPPINGS = {
    "RayH3SaveLatentTail": RayH3SaveLatentTail,
    "RayH3ChainLatentAV": RayH3ChainLatentAV,
}
NODE_CLASS_MAPPINGS.update(h3_seam_nodes.SEAM_NODE_CLASS_MAPPINGS)

# ComfyUI's node-menu display names (the melite-h3 prefix keeps the two
# entries grouped under this pack in the add-node search).
NODE_DISPLAY_NAME_MAPPINGS = {
    "RayH3SaveLatentTail": "Melite H3 Save Latent Tail (AV)",
    "RayH3ChainLatentAV": "Melite H3 Chain Latent AV (window carry)",
}
NODE_DISPLAY_NAME_MAPPINGS.update(h3_seam_nodes.SEAM_NODE_DISPLAY_NAME_MAPPINGS)

# True when the seam model-layer patches were applied by this import
# (False on the legacy patched core — guards make it a no-op there).
SEAM_PORT_ACTIVE = h3_seam_nodes.MODEL_LAYER_PATCHED

__all__ = [
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "SEAM_PORT_ACTIVE",
    "RayH3ChainLatentAV", "RayH3SaveLatentTail",
]
