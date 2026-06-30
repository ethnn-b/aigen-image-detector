"""Grad-CAM heatmaps over the backbone.

Shows which pixels drove a real-or-fake decision, so a flagged image can be checked
for a shortcut (a watermark or a border) versus a real generator fingerprint. Works
for both backbone families: a ViT's last block emits a token sequence, which is
reshaped back to a spatial grid, while an EfficientNet block already emits a feature
map, which is used as is.

No extra dependency: the hooks, the gradient pooling, and the colormap are written
here rather than pulled from pytorch-grad-cam.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _select_target_layer(model):
    """The last block is the usual Grad-CAM target: late enough to be semantic,
    still spatial. For a ViT, hook its norm1 (the block input) for a clean map."""
    blocks = getattr(model, "blocks", None)
    if blocks is not None and len(blocks) > 0:
        last = blocks[-1]
        return getattr(last, "norm1", last)
    raise ValueError("could not auto-select a Grad-CAM target layer; pass target_layer")


def _reshape_activation(model, act: torch.Tensor) -> torch.Tensor:
    """Coerce a captured activation to (B, C, H, W)."""
    if act.dim() == 4:
        return act
    if act.dim() == 3:
        # ViT tokens (B, N, C): drop the prefix (cls) tokens, fold the rest to a grid.
        n_prefix = int(getattr(model, "num_prefix_tokens", 1))
        tokens = act[:, n_prefix:, :]
        b, n, c = tokens.shape
        side = int(round(n**0.5))
        if side * side != n:
            raise ValueError(f"cannot fold {n} tokens into a square grid")
        return tokens.reshape(b, side, side, c).permute(0, 3, 1, 2)
    raise ValueError(f"unexpected activation rank {act.dim()}")


def gradcam(model, image_tensor: torch.Tensor, target_layer=None, target_class: int = 1) -> np.ndarray:
    """A heatmap (H, W) in [0, 1] for one image, for the fake class by default.

    Registers a forward and backward hook on the target layer, runs the image,
    backprops the chosen class logit, and weights the activations by the pooled
    gradients. The input is given requires_grad so the graph reaches the layer even
    when the backbone is frozen (the probe case).
    """
    model.eval()
    layer = target_layer if target_layer is not None else _select_target_layer(model)

    x = image_tensor
    if x.dim() == 3:
        x = x.unsqueeze(0)
    device = next(model.parameters()).device
    x = x.to(device).clone().requires_grad_(True)

    store: dict[str, torch.Tensor] = {}
    h1 = layer.register_forward_hook(lambda _m, _i, out: store.__setitem__("act", out))
    h2 = layer.register_full_backward_hook(lambda _m, _gi, go: store.__setitem__("grad", go[0]))
    try:
        logits = model(x)
        model.zero_grad(set_to_none=True)
        logits[:, target_class].sum().backward()
        act = _reshape_activation(model, store["act"])
        grad = _reshape_activation(model, store["grad"])
    finally:
        h1.remove()
        h2.remove()

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1))  # (B, h, w)
    cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0]
    cam = cam - cam.min()
    denom = cam.max()
    if denom > 0:
        cam = cam / denom
    return cam.detach().cpu().numpy()


def _colormap(cam: np.ndarray) -> np.ndarray:
    """A blue->green->red ramp, returned as uint8 RGB. No matplotlib needed."""
    stops = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype=np.float32)
    pos = np.clip(cam, 0, 1) * (len(stops) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = (pos - lo)[..., None]
    rgb = stops[lo] * (1 - frac) + stops[hi] * frac
    return rgb.astype(np.uint8)


def overlay_heatmap(pil_image, cam: np.ndarray, alpha: float = 0.5):
    """Blend a Grad-CAM heatmap over the original image and return a PIL image."""
    from PIL import Image

    base = pil_image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    heat = Image.fromarray(_colormap(cam))
    return Image.blend(base, heat, alpha)
