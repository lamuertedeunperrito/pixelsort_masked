import numpy as np
import torch

# -------------------------
# Helpers: tensors -> numpy
# -------------------------

def _images_to_np(images: torch.Tensor) -> np.ndarray:
    # IMAGE: [B,H,W,C] float 0..1
    return images.detach().cpu().numpy().astype(np.float32)

def _mask_to_np(mask: torch.Tensor) -> np.ndarray:
    # MASK: [B,H,W] o [H,W] float 0..1
    m = mask.detach().cpu().numpy().astype(np.float32)
    if m.ndim == 2:
        m = m[None, ...]
    return np.clip(m, 0.0, 1.0)

def _mask_from_images_np(mask_images_np: np.ndarray, mode: str = "luma") -> np.ndarray:
    # mask_images_np: [B,H,W,C] 0..1
    if mode == "luma":
        r, g, b = mask_images_np[..., 0], mask_images_np[..., 1], mask_images_np[..., 2]
        m = 0.2126 * r + 0.7152 * g + 0.0722 * b
    elif mode == "r":
        m = mask_images_np[..., 0]
    elif mode == "g":
        m = mask_images_np[..., 1]
    elif mode == "b":
        m = mask_images_np[..., 2]
    else:
        r, g, b = mask_images_np[..., 0], mask_images_np[..., 1], mask_images_np[..., 2]
        m = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.clip(m, 0.0, 1.0)  # [B,H,W]

def _broadcast_batch(arr: np.ndarray, B: int, name: str) -> np.ndarray:
    # arr: [b, ...]
    b = arr.shape[0]
    if b == B:
        return arr
    if b == 1 and B > 1:
        return np.repeat(arr, B, axis=0)
    raise ValueError(f"{name} batch ({b}) no coincide con images batch ({B}) y no es 1.")

# -------------------------
# Core: pixel sort variants
# -------------------------

def _rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    # img: [H,W,3] RGB 0..1
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = np.max(img, axis=-1)
    mn = np.min(img, axis=-1)
    diff = mx - mn + 1e-8

    # Hue
    h = np.zeros_like(mx)
    mask = diff > 1e-8
    # where mx==r
    idx = mask & (mx == r)
    h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6.0
    idx = mask & (mx == g)
    h[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2.0
    idx = mask & (mx == b)
    h[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4.0
    h = (h / 6.0) % 1.0

    # Saturation
    s = np.zeros_like(mx)
    s[mx > 1e-8] = diff[mx > 1e-8] / (mx[mx > 1e-8] + 1e-8)

    # Value
    v = mx

    hsv = np.stack([h, s, v], axis=-1)
    return hsv

def _sobel_edges(gray: np.ndarray) -> np.ndarray:
    # gray: [H,W] 0..1
    # Sobel simple (sin cv2)
    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)

    # pad reflect
    g = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")
    gx = (
        kx[0,0]*g[:-2,:-2] + kx[0,1]*g[:-2,1:-1] + kx[0,2]*g[:-2,2:] +
        kx[1,0]*g[1:-1,:-2] + kx[1,1]*g[1:-1,1:-1] + kx[1,2]*g[1:-1,2:] +
        kx[2,0]*g[2:,:-2] + kx[2,1]*g[2:,1:-1] + kx[2,2]*g[2:,2:]
    )
    gy = (
        ky[0,0]*g[:-2,:-2] + ky[0,1]*g[:-2,1:-1] + ky[0,2]*g[:-2,2:] +
        ky[1,0]*g[1:-1,:-2] + ky[1,1]*g[1:-1,1:-1] + ky[1,2]*g[1:-1,2:] +
        ky[2,0]*g[2:,:-2] + ky[2,1]*g[2:,1:-1] + ky[2,2]*g[2:,2:]
    )
    mag = np.sqrt(gx*gx + gy*gy)
    # normaliza suave
    mag = mag / (np.max(mag) + 1e-8)
    return np.clip(mag, 0.0, 1.0)

def pixel_sort_variant(
    img_rgb: np.ndarray,       # [H,W,3] RGB 0..1
    region01: np.ndarray,      # [H,W] 0..1
    axis: str,
    mode: str,                 # 5 modos
    threshold: float,
    min_run: int,
    descending: bool,
    strength: float,
):
    """
    5 modos:
    - "Luma Threshold": clásico: activa donde luma > threshold
    - "Luma Band": activa donde abs(luma - threshold) < band (usa threshold como centro)
    - "Edges": activa donde edges > threshold
    - "Saturation": activa donde sat > threshold
    - "Hue": activa donde hue > threshold (útil para “barridos” raros)
    """
    img = np.clip(img_rgb, 0.0, 1.0)
    h, w, _ = img.shape
    region = (region01 > 0.5)

    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b

    if mode == "Luma Threshold":
        active = region & (luma > threshold)
        key = luma
    elif mode == "Luma Band":
        band = 0.12  # ancho fijo (puedes exponerlo si quieres)
        active = region & (np.abs(luma - threshold) < band)
        key = luma
    elif mode == "Edges":
        edges = _sobel_edges(luma)
        active = region & (edges > threshold)
        key = edges
    elif mode == "Saturation":
        hsv = _rgb_to_hsv(img)
        sat = hsv[..., 1]
        active = region & (sat > threshold)
        key = sat
    elif mode == "Hue":
        hsv = _rgb_to_hsv(img)
        hue = hsv[..., 0]
        active = region & (hue > threshold)
        key = hue
    else:
        active = region & (luma > threshold)
        key = luma

    out = img.copy()

    if axis == "x":
        for y in range(h):
            row = active[y, :]
            if not np.any(row):
                continue
            idx = np.where(row)[0]
            splits = np.where(np.diff(idx) != 1)[0] + 1
            runs = np.split(idx, splits)
            for run in runs:
                if run.size < min_run:
                    continue
                seg = out[y, run, :]     # (len,3)
                seg_key = key[y, run]    # (len,)
                order = np.argsort(seg_key)
                if descending:
                    order = order[::-1]
                out[y, run, :] = seg[order, :]
    else:
        for x in range(w):
            col = active[:, x]
            if not np.any(col):
                continue
            idx = np.where(col)[0]
            splits = np.where(np.diff(idx) != 1)[0] + 1
            runs = np.split(idx, splits)
            for run in runs:
                if run.size < min_run:
                    continue
                seg = out[run, x, :]
                seg_key = key[run, x]
                order = np.argsort(seg_key)
                if descending:
                    order = order[::-1]
                out[run, x, :] = seg[order, :]

    if strength < 1.0:
        out = img * (1.0 - strength) + out * strength
        out = np.clip(out, 0.0, 1.0)

    debug = active.astype(np.float32)  # dónde actuó
    return out, debug


class PixelSortFramesPro:
    """
    images (IMAGE batch) ->
      - mask opcional (MASK)  [1,H,W] o [B,H,W]
      - mask_images opcional (IMAGE batch) (video máscara como frames)
    outputs:
      - images_out (IMAGE batch)
      - mask_debug (MASK batch)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "axis": (["x", "y"], {"default": "x"}),
                "mode": (["Luma Threshold", "Luma Band", "Edges", "Saturation", "Hue"], {"default": "Luma Threshold"}),
                "threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_run": ("INT", {"default": 12, "min": 2, "max": 4096, "step": 1}),
                "descending": ("BOOLEAN", {"default": False}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "debug_mask": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask": ("MASK",),
                "mask_images": ("IMAGE",),
                "mask_from_image_mode": (["luma", "r", "g", "b"], {"default": "luma"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "mask_debug")
    FUNCTION = "process"
    CATEGORY = "video/post"

    def process(
        self,
        images,
        axis,
        mode,
        threshold,
        min_run,
        descending,
        invert_mask,
        strength,
        debug_mask,
        mask=None,
        mask_images=None,
        mask_from_image_mode="luma",
    ):
        imgs = _images_to_np(images)  # [B,H,W,C]
        B, H, W, C = imgs.shape

        # region01: [B,H,W] float 0..1
        if mask is None and mask_images is None:
            region01 = np.ones((B, H, W), dtype=np.float32)  # sin máscara: aplica global
        elif mask is not None:
            region01 = _mask_to_np(mask)           # [b,H,W]
            region01 = _broadcast_batch(region01, B, "mask")
        else:
            mimgs = _images_to_np(mask_images)     # [b,H,W,C]
            mimgs = _broadcast_batch(mimgs, B, "mask_images")
            region01 = _mask_from_images_np(mimgs, mode=mask_from_image_mode)  # [B,H,W]

        if invert_mask:
            region01 = 1.0 - region01

        out_imgs = np.zeros_like(imgs, dtype=np.float32)
        out_dbg = np.zeros((B, H, W), dtype=np.float32)

        for i in range(B):
            out, dbg = pixel_sort_variant(
                img_rgb=imgs[i],
                region01=region01[i],
                axis=axis,
                mode=mode,
                threshold=threshold,
                min_run=min_run,
                descending=descending,
                strength=strength,
            )
            out_imgs[i] = out
            out_dbg[i] = dbg if debug_mask else (region01[i] > 0.5).astype(np.float32)

        out_imgs_t = torch.from_numpy(out_imgs).to(images.device)
        out_dbg_t = torch.from_numpy(out_dbg).to(images.device)

        return (out_imgs_t, out_dbg_t)


NODE_CLASS_MAPPINGS = {
    "PixelSortFramesPro": PixelSortFramesPro
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelSortFramesPro": "Pixel Sort PRO (Frames + Optional Mask)"
}
