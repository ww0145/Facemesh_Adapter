"""
Head region extraction from THuman meshes.

THuman2.1 meshes are Y-up, head is at high Y values.
Extracts head region by Y-axis threshold cropping.
Also provides head masking for base condition decoupling.
"""
import os
import torch
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from typing import Optional, Tuple


def extract_head_bbox(vertices: np.ndarray, head_ratio: float = 0.15) -> Tuple[float, int]:
    """
    Compute Y threshold for head region.

    Args:
        vertices: [N, 3] mesh vertices (Y-up)
        head_ratio: fraction of Y range to consider as head

    Returns:
        y_threshold: vertices with Y > threshold are head
        head_axis: always 1 (Y-up)
    """
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    y_threshold = y_max - head_ratio * (y_max - y_min)
    return float(y_threshold), 1


def extract_head_vertices(mesh: trimesh.Trimesh, head_ratio: float = 0.15) -> np.ndarray:
    """
    Get boolean mask of head vertices.

    Args:
        mesh: trimesh object (Y-up)
        head_ratio: fraction of Y range for head

    Returns:
        mask: [N] bool array
    """
    y_threshold, _ = extract_head_bbox(mesh.vertices, head_ratio)
    return mesh.vertices[:, 1] >= y_threshold


def crop_head_from_image(image: Image.Image, mesh: trimesh.Trimesh,
                         head_ratio: float = 0.15,
                         padding: float = 0.15,
                         output_size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """
    Crop head region from a rendered full-body image.

    Estimates head bounding box from mesh proportions and crops
    the corresponding region from the rendered image.

    Args:
        image: rendered full-body image
        mesh: trimesh object for proportion estimation
        head_ratio: fraction of body height for head
        padding: relative padding around head crop
        output_size: output image size

    Returns:
        head_image: cropped and resized PIL image
    """
    w, h = image.size
    v = mesh.vertices

    y_min, y_max = v[:, 1].min(), v[:, 1].max()
    body_height = y_max - y_min
    head_start_y = y_max - head_ratio * body_height

    # Map mesh Y to image Y (image Y is inverted: top=0)
    # head is at top of image
    head_frac = (y_max - head_start_y) / body_height
    pad = int(padding * head_frac * h)

    crop_top = max(0, int((1 - (y_max - y_min + 0.01) / body_height) * h / 2) - pad)
    crop_bottom = int(crop_top + head_frac * h * 1.2) + pad
    crop_bottom = min(h, crop_bottom)

    # Center crop horizontally
    x_range = v[:, 0].max() - v[:, 0].min()
    head_width_frac = x_range * head_ratio / (v[:, 0].max() - v[:, 0].min() + 1e-6)
    cx = w // 2
    half_w = int(w * head_width_frac * 2) + pad
    crop_left = max(0, cx - half_w)
    crop_right = min(w, cx + half_w)

    head_img = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    head_img = head_img.resize(output_size, Image.LANCZOS)
    return head_img


# ─────────────────────────────────────────────────────────
#  Head masking: mask head region in full-body image
#  Used to create decoupled base condition (body only)
# ─────────────────────────────────────────────────────────

def _detect_foreground_bbox(image: Image.Image, bg_thresh: int = 245):
    """
    Detect foreground (non-background) bounding box in a rendered image.

    Works for white-background renders (pyrender default bg=255).
    Returns (top, bottom, left, right) of the foreground region,
    or None if no foreground detected or image is not white-bg.
    """
    arr = np.array(image)
    # Check if image has a white-ish background (>30% pixels above thresh)
    white_pixels = (arr > bg_thresh).all(axis=2).sum()
    total_pixels = arr.shape[0] * arr.shape[1]
    if white_pixels / total_pixels < 0.15:
        # Not a white-bg image, skip foreground detection
        return None

    fg_mask = (arr < bg_thresh).any(axis=2)
    if not fg_mask.any():
        return None
    rows = np.where(fg_mask.any(axis=1))[0]
    cols = np.where(fg_mask.any(axis=0))[0]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _detect_face_bbox(image: Image.Image):
    """
    Detect face bounding box using OpenCV DNN or Haar Cascade.

    Returns (top, bottom, left, right) of the face region,
    or None if detection fails.
    """
    import cv2
    arr = np.array(image)
    h, w = arr.shape[:2]

    # Strategy 1: OpenCV DNN face detector (more robust)
    try:
        net_path = cv2.data.haarcascades.replace(
            'haarcascades', 'dnn'
        ).rstrip('/') if hasattr(cv2.data, 'haarcascades') else None

        # Use Haar as primary — always available, no extra files needed
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        # Try multiple cascades for robustness
        for cascade_name in [
            'haarcascade_frontalface_alt2.xml',
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt.xml',
        ]:
            cascade_path = cv2.data.haarcascades + cascade_name
            if not os.path.exists(cascade_path):
                continue
            cascade = cv2.CascadeClassifier(cascade_path)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
            )
            if len(faces) > 0:
                areas = [fw * fh for (_, _, fw, fh) in faces]
                x, y, fw, fh = faces[np.argmax(areas)]
                return int(y), int(y + fh), int(x), int(x + fw)
    except Exception:
        pass

    return None


def mask_head_in_image(image: Image.Image,
                       mesh: Optional[trimesh.Trimesh] = None,
                       head_ratio: float = 0.20,
                       expand: float = 1.3,
                       fill: str = "black") -> Image.Image:
    """
    Mask the head region in a full-body image.

    Layered strategy (tries in order):
      1. Face detection — precise, works on renders and photos
      2. Foreground detection — for white-bg renders if face det fails
      3. Top-of-image heuristic — last resort fallback

    Args:
        image: PIL Image, full-body rendering or photo
        mesh: optional trimesh (unused, kept for API compat)
        head_ratio: fraction of BODY height considered as head
        expand: expansion factor on head bbox (covers hair/forehead)
        fill: "black" | "noise" | "mean"

    Returns:
        PIL Image with head region masked
    """
    w, h = image.size

    # ── Strategy 1: Face detection (most accurate, works on any image) ──
    face_bbox = _detect_face_bbox(image)
    if face_bbox is not None:
        face_top, face_bottom, face_left, face_right = face_bbox
        face_h = face_bottom - face_top
        face_w = face_right - face_left
        face_cx = (face_left + face_right) // 2

        # Expand face bbox to cover full head (hair, forehead, ears)
        crop_top = max(0, int(face_top - face_h * 0.5))
        crop_bottom = min(h, int(face_bottom + face_h * 0.1))
        half_w = int(face_w * 0.5)
        crop_left = max(0, face_cx - half_w)
        crop_right = min(w, face_cx + half_w)

    else:
        # ── Strategy 2: Foreground detection (white-bg renders, if face det fails) ──
        fg_bbox = _detect_foreground_bbox(image)
        if fg_bbox is not None:
            fg_top, fg_bottom, fg_left, fg_right = fg_bbox
            body_height = fg_bottom - fg_top

            head_px = int(body_height * head_ratio * expand)
            crop_top = max(0, fg_top - 5)
            crop_bottom = min(h, fg_top + head_px)

            fg_cx = (fg_left + fg_right) // 2
            body_width = fg_right - fg_left
            head_half_w = int(body_width * 0.2 * expand)
            crop_left = max(0, fg_cx - head_half_w)
            crop_right = min(w, fg_cx + head_half_w)
        else:
            # ── Strategy 3: Heuristic fallback ──
            crop_top = 0
            crop_bottom = int(h * head_ratio * expand)
            crop_bottom = min(h, crop_bottom)
            margin = int(w * 0.15)
            crop_left = margin
            crop_right = w - margin

    # Apply mask
    masked = image.copy()
    if fill == "black":
        draw = ImageDraw.Draw(masked)
        draw.rectangle([crop_left, crop_top, crop_right, crop_bottom], fill=(0, 0, 0))
    elif fill == "noise":
        region_h = crop_bottom - crop_top
        region_w = crop_right - crop_left
        noise = np.random.randint(0, 255, (region_h, region_w, 3), dtype=np.uint8)
        masked.paste(Image.fromarray(noise), (crop_left, crop_top))
    elif fill == "mean":
        arr = np.array(image)
        mean_color = tuple(arr.mean(axis=(0, 1)).astype(np.uint8))
        draw = ImageDraw.Draw(masked)
        draw.rectangle([crop_left, crop_top, crop_right, crop_bottom], fill=mean_color)

    return masked


def render_mesh_to_image(mesh: trimesh.Trimesh,
                         resolution: Tuple[int, int] = (512, 512),
                         bg_color: Tuple[int, ...] = (255, 255, 255, 255),
                         angle: float = 0.0) -> Optional[Image.Image]:
    """
    Render trimesh to image. Tries pyrender (EGL), falls back to trimesh scene.

    Args:
        mesh: trimesh object
        resolution: output resolution
        bg_color: background RGBA
        angle: Y-axis rotation in radians

    Returns:
        PIL Image or None if rendering fails
    """
    import os

    # Apply rotation if needed
    if abs(angle) > 1e-6:
        rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        mesh = mesh.copy()
        mesh.apply_transform(rot)

    # Try pyrender with EGL
    try:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        import pyrender

        scene = pyrender.Scene(bg_color=bg_color, ambient_light=[0.5, 0.5, 0.5])
        py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        scene.add(py_mesh)

        # Camera: look at center
        center = mesh.centroid
        cam = pyrender.PerspectiveCamera(yfov=np.radians(30))
        cam_pose = np.eye(4)
        cam_pose[2, 3] = 2.5  # distance
        cam_pose[1, 3] = center[1]  # align to mesh center Y
        scene.add(cam, pose=cam_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=cam_pose)

        r = pyrender.OffscreenRenderer(*resolution)
        color, _ = r.render(scene)
        r.delete()
        return Image.fromarray(color)

    except (ImportError, Exception):
        pass

    # Fallback: trimesh scene export
    try:
        scene = mesh.scene()
        png_data = scene.save_image(resolution=resolution)
        if png_data is not None:
            import io
            return Image.open(io.BytesIO(png_data))
    except Exception:
        pass

    return None