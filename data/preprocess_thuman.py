#!/usr/bin/env python
"""
THuman2.1 offline preprocessing for FaceMesh-Adapter v2 training.

For each sample, generates:
  - Shape GT latent (SC-VAE encoded from O-Voxel)
  - Full-body DINOv3 cond tokens (head MASKED — body-only condition)
  - Head DINOv3 id_emb (for adapter)

Key change vs v1: base cond is encoded from head-masked full-body image,
so that the base model cannot leak facial identity information.
The adapter is the sole source of facial identity.

Prerequisites:
  1. THuman2.1 meshes at --thuman_dir (e.g. ~/TRELLIS/model/)
  2. O-Voxel .vxz files from convert_thuman_to_vxz.py
  3. TRELLIS.2 pipeline loaded

Usage:
    cd ~/TRELLIS.2
    CUDA_VISIBLE_DEVICES=2 python facemesh_adapter_v2/data/preprocess_thuman.py \
        --thuman_dir ~/TRELLIS/model \
        --vxz_dir data/thuman_vxz/dual_grid_512 \
        --out_dir data/adapter_train \
        --resolution 512 \
        --num_samples 10
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import argparse
import cv2
import torch
import numpy as np
import trimesh
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

import o_voxel
import trellis2.models as models
import trellis2.modules.sparse as sp

from facemesh_adapter_v2.libs.utils.head_extract import mask_head_in_image


def load_vxz(vxz_path):
    """Load .vxz and return encoder-ready SparseTensors."""
    coords, attr = o_voxel.io.read_vxz(vxz_path, num_threads=4)
    vertices = sp.SparseTensor(
        (attr['vertices'] / 255.0).float(),
        torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
    )
    intersected = vertices.replace(torch.cat([
        attr['intersected'] % 2,
        attr['intersected'] // 2 % 2,
        attr['intersected'] // 4 % 2,
    ], dim=-1).bool())
    return vertices, intersected


def render_mesh_front(mesh_path, resolution=512):
    """
    Render front view of a textured mesh.
    Tries pyrender (EGL), falls back to trimesh scene.

    Returns:
        (PIL Image, trimesh.Trimesh) or (None, None)
    """
    mesh = trimesh.load(str(mesh_path), process=False, force='mesh')

    # Try pyrender
    try:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        import pyrender

        scene = pyrender.Scene(
            bg_color=[255, 255, 255, 255],
            ambient_light=[0.4, 0.4, 0.4]
        )
        py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        scene.add(py_mesh)

        center = mesh.centroid
        cam = pyrender.PerspectiveCamera(yfov=np.radians(30))
        cam_pose = np.eye(4)
        cam_pose[2, 3] = 2.5
        cam_pose[1, 3] = center[1]
        scene.add(cam, pose=cam_pose)

        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
        scene.add(light, pose=cam_pose)

        r = pyrender.OffscreenRenderer(resolution, resolution)
        color, _ = r.render(scene)
        r.delete()
        return Image.fromarray(color), mesh
    except Exception:
        pass

    # Fallback: trimesh scene
    try:
        scene = mesh.scene()
        png = scene.save_image(resolution=[resolution, resolution])
        if png:
            import io
            return Image.open(io.BytesIO(png)), mesh
    except Exception:
        pass

    return None, None


def render_mesh_front_hq(mesh_path, resolution=4096):
    """Render high-resolution front view for HQ image conditions."""
    import pyrender

    mesh = trimesh.load(str(mesh_path), process=False, force='mesh')
    scene = pyrender.Scene(
        bg_color=[255, 255, 255, 255],
        ambient_light=[0.45, 0.45, 0.45],
    )

    py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(py_mesh)

    center = mesh.centroid
    bounds = mesh.bounds
    height = bounds[1, 1] - bounds[0, 1]

    cam = pyrender.PerspectiveCamera(yfov=np.radians(28))
    cam_pose = np.eye(4)
    cam_pose[0, 3] = center[0]
    cam_pose[1, 3] = center[1]
    cam_pose[2, 3] = max(2.5, height * 1.35)
    scene.add(cam, pose=cam_pose)

    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.5)
    scene.add(light, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    color, _ = renderer.render(scene)
    renderer.delete()

    return Image.fromarray(color[..., :3]), mesh


def detect_face_crop(image, output_size=512, expand=2.4):
    """Crop face from a high-resolution full-body render."""
    arr = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    faces_all = []
    for name in [
        'haarcascade_frontalface_alt2.xml',
        'haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_alt.xml',
    ]:
        path = cv2.data.haarcascades + name
        if not os.path.exists(path):
            continue
        cascade = cv2.CascadeClassifier(path)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(40, 40),
        )
        faces_all.extend(list(faces))

    h_img, w_img = arr.shape[:2]

    if faces_all:
        x, y, w, h = max(faces_all, key=lambda b: b[2] * b[3])
        cx = x + w / 2
        cy = y + h / 2
        size = max(w, h) * expand
        left = int(max(0, cx - size / 2))
        right = int(min(w_img, cx + size / 2))
        top = int(max(0, cy - size * 0.58))
        bottom = int(min(h_img, cy + size * 0.55))
    else:
        fg = (arr < 245).any(axis=2)
        rows = np.where(fg.any(axis=1))[0]
        cols = np.where(fg.any(axis=0))[0]
        if len(rows) == 0:
            raise RuntimeError('Cannot detect foreground or face.')

        fg_top, fg_bottom = rows.min(), rows.max()
        fg_left, fg_right = cols.min(), cols.max()
        body_h = fg_bottom - fg_top
        body_w = fg_right - fg_left
        cx = (fg_left + fg_right) // 2

        head_h = int(body_h * 0.22)
        half_w = int(body_w * 0.18)
        left = max(0, cx - half_w)
        right = min(w_img, cx + half_w)
        top = max(0, fg_top)
        bottom = min(h_img, fg_top + head_h)

    crop = image.crop((left, top, right, bottom))
    crop = crop.resize((output_size, output_size), Image.LANCZOS)

    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=8)
    return crop, vis


def crop_head_image(image, head_frac=0.2, output_size=(512, 512)):
    """
    Crop head from full-body rendering.
    Assumes person is centered, head is at top ~20% of image.
    """
    w, h = image.size
    # Head is at top portion
    crop_top = 0
    crop_bottom = int(h * head_frac * 1.5)
    # Center horizontally with some margin
    margin = int(w * 0.25)
    crop_left = margin
    crop_right = w - margin

    head = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    head = head.resize(output_size, Image.LANCZOS)
    return head


@torch.no_grad()
def preprocess_one_sample(
    sample_id, thuman_dir, vxz_dir, out_dir,
    encoder, pipeline, resolution, device,
    save_debug_images=False,
    hq_condition=False,
    render_res=4096,
    head_res=512,
    head_crop_expand=2.4,
    overwrite=False,
):
    """Process one THuman sample → .pt file."""
    # Paths
    mesh_path = Path(thuman_dir) / sample_id / f"{sample_id}.obj"
    vxz_path = Path(vxz_dir) / f"{sample_id}.vxz"
    out_path = Path(out_dir) / f"{sample_id}.pt"

    if out_path.exists() and not overwrite:
        return f"{sample_id}: exists, skip"

    if not mesh_path.exists():
        return f"{sample_id}: mesh not found at {mesh_path}"
    if not vxz_path.exists():
        return f"{sample_id}: vxz not found at {vxz_path}"

    # 1. SC-VAE encode → shape GT latent
    vertices, intersected = load_vxz(str(vxz_path))
    z = encoder(vertices.cuda(), intersected.cuda())
    shape_coords = z.coords[:, 1:].cpu().numpy().astype(np.uint8)
    shape_feats = z.feats.cpu().numpy().astype(np.float32)

    # 2. Render full-body image
    if hq_condition:
        fullbody_img, mesh = render_mesh_front_hq(mesh_path, resolution=render_res)
    else:
        fullbody_img, mesh = render_mesh_front(mesh_path)
    if fullbody_img is None:
        return f"{sample_id}: render failed"

    # 3. Mask head in full-body image → DINOv3 → base cond (body-only)
    fullbody_masked = mask_head_in_image(
        fullbody_img, mesh=mesh, head_ratio=0.15, expand=1.15, fill="black"
    )
    masked_processed = pipeline.preprocess_image(fullbody_masked)
    cond_resolution = 1024 if resolution == 1024 else 512
    cond_dict = pipeline.get_cond([masked_processed], resolution=cond_resolution)
    cond = cond_dict['cond'].cpu()          # [1, N_tok, 1024]
    neg_cond = cond_dict['neg_cond'].cpu()  # [1, N_tok, 1024]

    # 4. Crop head (from ORIGINAL full-body) → DINOv3 → mean-pool → id_emb
    crop_vis = None
    if hq_condition:
        head_img, crop_vis = detect_face_crop(
            fullbody_img,
            output_size=head_res,
            expand=head_crop_expand,
        )
    else:
        head_img = crop_head_image(fullbody_img)
    head_processed = pipeline.preprocess_image(head_img)
    head_cond_dict = pipeline.get_cond([head_processed], resolution=cond_resolution)
    head_cond = head_cond_dict['cond']       # [1, N_tok, 1024]
    id_emb = head_cond.mean(dim=1).cpu()     # [1, 1024]

    # 5. Save
    save_dict = {
        # Shape latent (GT target for shape flow model)
        'shape_coords': shape_coords,
        'shape_feats': shape_feats,
        # Conditions (base cond is from HEAD-MASKED full-body)
        'cond': cond.squeeze(0),         # [N_tok, 1024]
        'neg_cond': neg_cond.squeeze(0), # [N_tok, 1024]
        # Identity (from head crop of ORIGINAL full-body)
        'id_emb': id_emb.squeeze(0),     # [1024]
        # Images (for debugging/visualization)
        'fullbody_img_size': fullbody_img.size,
        'head_img_size': head_img.size,
        # Meta
        'sample_id': sample_id,
        'resolution': resolution,
        'head_masked': True,  # flag: base cond uses masked image
        'hq_condition': hq_condition,
    }
    torch.save(save_dict, str(out_path))

    # Optional: save debug images to visually verify mask
    if save_debug_images:
        debug_dir = Path(out_dir) / 'debug_images'
        debug_dir.mkdir(exist_ok=True)
        suffix = '_hq' if hq_condition else ''
        fullbody_img.save(debug_dir / f"{sample_id}_fullbody{suffix}.png")
        fullbody_masked.save(debug_dir / f"{sample_id}_masked{suffix}.png")
        head_img.save(debug_dir / f"{sample_id}_head{suffix}.png")
        if crop_vis is not None:
            crop_vis.save(debug_dir / f"{sample_id}_head_crop_bbox.png")

    n_tokens = shape_coords.shape[0]
    hq_msg = ', HQ condition' if hq_condition else ''
    return f"{sample_id}: {n_tokens} latent tokens → saved (head masked{hq_msg})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thuman_dir', type=str, required=True,
                        help='THuman2.1 root (contains 0000/, 0001/, ...)')
    parser.add_argument('--vxz_dir', type=str, required=True,
                        help='O-Voxel .vxz directory (from convert_thuman_to_vxz.py)')
    parser.add_argument('--out_dir', type=str, default='data/adapter_train',
                        help='Output directory for .pt files')
    parser.add_argument('--resolution', type=int, default=512,
                        help='O-Voxel resolution (must match vxz files)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Max samples to process (None = all)')
    parser.add_argument('--save_debug_images', action='store_true',
                        help='Save fullbody/masked/head PNGs for visual verification')
    parser.add_argument('--hq_condition', action='store_true',
                        help='Use high-resolution render + detected face crop for image conditions')
    parser.add_argument('--render_res', type=int, default=4096,
                        help='HQ render resolution when --hq_condition is set')
    parser.add_argument('--head_res', type=int, default=512,
                        help='Head crop resolution when --hq_condition is set')
    parser.add_argument('--head_crop_expand', type=float, default=2.4,
                        help='Face crop expansion when --hq_condition is set')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing .pt files instead of skipping them')
    parser.add_argument('--enc_model', type=str,
                        default='microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16',
                        help='Shape encoder model')
    parser.add_argument('--pipeline_model', type=str,
                        default='microsoft/TRELLIS.2-4B',
                        help='TRELLIS.2 pipeline for DINOv3')
    args = parser.parse_args()

    device = torch.device('cuda')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find samples
    thuman_dir = Path(args.thuman_dir)
    sample_ids = sorted([
        d.name for d in thuman_dir.iterdir()
        if d.is_dir() and (d / f"{d.name}.obj").exists()
    ])
    if args.num_samples:
        sample_ids = sample_ids[:args.num_samples]

    print(f"Found {len(sample_ids)} THuman samples")
    print(f"VXZ dir: {args.vxz_dir}")
    print(f"Output: {args.out_dir}")
    print(f"Head masking: ENABLED (base cond = body only)")
    print(f"HQ condition: {'ENABLED' if args.hq_condition else 'disabled'}")
    print(f"Overwrite: {'ENABLED' if args.overwrite else 'disabled'}")

    # Load models
    print("Loading shape encoder...")
    encoder = models.from_pretrained(args.enc_model).eval().cuda()

    print("Loading pipeline (for DINOv3)...")
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.pipeline_model)
    pipeline.cuda()

    # Process
    for sid in tqdm(sample_ids, desc="Preprocessing"):
        result = preprocess_one_sample(
            sid, args.thuman_dir, args.vxz_dir, args.out_dir,
            encoder, pipeline, args.resolution, device,
            save_debug_images=args.save_debug_images,
            hq_condition=args.hq_condition,
            render_res=args.render_res,
            head_res=args.head_res,
            head_crop_expand=args.head_crop_expand,
            overwrite=args.overwrite,
        )
        tqdm.write(f"  {result}")

    print(f"\nDone! {len(list(out_dir.glob('*.pt')))} files in {out_dir}")


if __name__ == '__main__':
    main()
