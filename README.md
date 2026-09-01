# melite-h3-motion-nodes

THE SEAM LAYER for MiniMax H3 timelines — ported out of the patched core to stock-core ComfyUI.

Melite H3 Motion Nodes — the seam port for MiniMax H3 timelines.
> This pack monkey-patches stock ComfyUI (GPL-3.0) internals at import (seam-aware `PackedLayout.__init__` / `MiniMaxH3.extra_conds` via dynamic source delta) — it is therefore distributed under GPL-3.0.

## Nodes

- `MiniMaxH3ImageToVideo`

## Install

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/JayDataEngineer/melite-h3-motion-nodes.git
```

Restart ComfyUI.

## Provenance

Published from the inference estate (`inference.cpp` repo, `plugins/comfyui/custom_nodes/melite-h3-motion-nodes`) on 2026-09-01.

## License

GPL-3.0 — see LICENSE.
