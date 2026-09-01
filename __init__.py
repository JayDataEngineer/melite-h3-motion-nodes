"""Melite H3 Motion Nodes — the seam port for MiniMax H3 timelines.

The in-tree seam layer, ported out of the patched core (2026-09-01):
``h3_seam_nodes.py`` registers the seam-aware ``MiniMaxH3ImageToVideo``
superset (+ first_frame_latent / context_steps / audio_context_steps),
``MiniMaxH3SaveLatent`` / ``MiniMaxH3LoadLatent`` (.h3latent packs),
``MiniMaxH3AddGuide`` where the running core lacks it, and binds
seam-aware ``PackedLayout.__init__`` + ``MiniMaxH3.extra_conds`` at
import. Stock-core ComfyUI only; on the legacy patched core the port
no-ops.

History (2026-09-01, node-migration move B): this pack previously also
shipped ``RayH3SaveLatentTail`` + ``RayH3ChainLatentAV``
(``h3_chain.py``) — a parallel latent-weave lineage that only the
dormant ``window*_v7`` / ``window1_load`` / ``window1_t4_audio``
templates referenced. The composer only ever emits the ``_t4``/``_t4r``
family (PlagueKind chain spine + base-pack latent save/load), so the
Ray lineage was deleted; ONE seam architecture remains.

In-tree pack: deployed from plugins/comfyui/custom_nodes/melite-h3-motion-nodes
(declared in plugins/comfyui/models/video/h3-timeline/nodes.yaml — the
h3-timeline family owns it). Published at
https://github.com/JayDataEngineer/melite-h3-motion-nodes and tracked in the
inference estate as a git submodule.
"""
from . import h3_seam_nodes

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(h3_seam_nodes.SEAM_NODE_CLASS_MAPPINGS)

# ComfyUI's node-menu display names.
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(h3_seam_nodes.SEAM_NODE_DISPLAY_NAME_MAPPINGS)

# True when the seam model-layer patches were applied by this import
# (False on the legacy patched core — guards make it a no-op there).
SEAM_PORT_ACTIVE = h3_seam_nodes.MODEL_LAYER_PATCHED

__all__ = [
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "SEAM_PORT_ACTIVE",
]
