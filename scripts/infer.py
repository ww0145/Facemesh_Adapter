#!/usr/bin/env python
"""
FaceMesh-Adapter v2 inference.

Two-condition decoupled design:
  - Base cond: full-body image with HEAD MASKED → DINOv3 → cond tokens
    (provides body pose/shape, NO facial identity)
  - Adapter cond: head image → DINOv3 → mean-pool → id_emb
    (sole source of facial identity)

SS stage: frozen base. Shape stage: adapter. Texture stage: frozen base.

Usage:
    cd ~/TRELLIS.2
    CUDA_VISIBLE_DEVICES=2 python facemesh_adapter_v2/scripts/infer.py \
        --config facemesh_adapter_v2/configs/default.yaml \
        --adapter_ckpt runs/shape_xxx/adapter_final.pt \
        --head_image head.jpg \
        --fullbody_image body.jpg \
        --out_dir results/infer_test
"""
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn.functional as F
import imageio
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from trellis2.modules import sparse as sp

from facemesh_adapter_v2.trainer import load_config
from facemesh_adapter_v2.libs.adapters import IPAdapterSLatFlowModel
from facemesh_adapter_v2.libs.encoders.projections import ImageProjModel
from facemesh_adapter_v2.libs.utils.head_extract import mask_head_in_image
import trimesh


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def to_trimesh(mesh):
    return trimesh.Trimesh(
        vertices=to_numpy(mesh.vertices),
        faces=to_numpy(mesh.faces),
        process=False,
    )


def to_device_cond_dict(d, device):
    """Build a pipeline cond_dict directly from a preprocessed training .pt."""
    return {
        'cond': d['cond'].unsqueeze(0).to(device),
        'neg_cond': d['neg_cond'].unsqueeze(0).to(device),
    }


def build_gt_coords(d, device):
    coords = torch.from_numpy(d['shape_coords']).int().to(device)
    batch_col = torch.zeros(coords.shape[0], 1, dtype=torch.int32, device=device)
    return torch.cat([batch_col, coords], dim=1)


def normalize_shape_feats(feats, pipeline, device):
    norm = getattr(pipeline, 'shape_slat_normalization', None)
    if norm is None:
        return None
    mean = torch.tensor(norm['mean'], dtype=feats.dtype, device=device)
    std = torch.tensor(norm['std'], dtype=feats.dtype, device=device)
    return (feats - mean) / std


def align_feats_by_coords(pred_coords, pred_feats, gt_coords, gt_feats):
    """
    Align predicted and GT features by sparse coordinates.
    Fast path assumes identical order; fallback matches coordinate tuples.
    """
    pred_coords_cpu = pred_coords.detach().cpu()
    gt_coords_cpu = gt_coords.detach().cpu()

    if pred_coords_cpu.shape == gt_coords_cpu.shape and torch.equal(pred_coords_cpu, gt_coords_cpu):
        return pred_feats, gt_feats, pred_coords, {
            'pred_count': int(pred_coords_cpu.shape[0]),
            'gt_count': int(gt_coords_cpu.shape[0]),
            'matched_count': int(gt_coords_cpu.shape[0]),
            'missing_gt_count': 0,
            'extra_pred_count': 0,
            'coord_order_exact': True,
        }

    pred_index = {tuple(c.tolist()): i for i, c in enumerate(pred_coords_cpu)}
    gt_index = {tuple(c.tolist()): i for i, c in enumerate(gt_coords_cpu)}
    common = sorted(set(pred_index) & set(gt_index))
    if not common:
        raise ValueError("No overlapping sparse coordinates between prediction and GT.")

    pred_ids = torch.tensor([pred_index[c] for c in common], device=pred_feats.device)
    gt_ids = torch.tensor([gt_index[c] for c in common], device=gt_feats.device)
    stats = {
        'pred_count': int(pred_coords_cpu.shape[0]),
        'gt_count': int(gt_coords_cpu.shape[0]),
        'matched_count': int(len(common)),
        'missing_gt_count': int(len(set(gt_index) - set(pred_index))),
        'extra_pred_count': int(len(set(pred_index) - set(gt_index))),
        'coord_order_exact': False,
    }
    return (
        pred_feats.index_select(0, pred_ids),
        gt_feats.index_select(0, gt_ids),
        pred_coords.index_select(0, pred_ids),
        stats,
    )


def feature_metrics(pred_feats, gt_feats):
    diff = pred_feats.float() - gt_feats.float()
    return {
        'mse': float(diff.pow(2).mean().item()),
        'rmse': float(diff.pow(2).mean().sqrt().item()),
        'mae': float(diff.abs().mean().item()),
        'max_abs': float(diff.abs().max().item()),
        'cosine_mean': float(F.cosine_similarity(pred_feats.float(), gt_feats.float(), dim=-1).mean().item()),
        'pred_mean': float(pred_feats.float().mean().item()),
        'pred_std': float(pred_feats.float().std().item()),
        'gt_mean': float(gt_feats.float().mean().item()),
        'gt_std': float(gt_feats.float().std().item()),
    }


def topk_feature_errors(pred_feats, gt_feats, coords, topk):
    diff = pred_feats.float() - gt_feats.float()
    per_token_mse = diff.pow(2).mean(dim=1)
    per_token_rmse = per_token_mse.sqrt()
    per_token_mae = diff.abs().mean(dim=1)
    per_token_max_abs = diff.abs().max(dim=1).values
    per_token_cos = F.cosine_similarity(pred_feats.float(), gt_feats.float(), dim=-1)

    k = min(topk, pred_feats.shape[0])
    values, indices = torch.topk(per_token_mse, k=k, largest=True)
    top_tokens = []
    for rank, (value, idx) in enumerate(zip(values, indices), start=1):
        i = int(idx.item())
        token_diff = diff[i]
        ch_values, ch_indices = torch.topk(token_diff.abs(), k=min(5, token_diff.numel()))
        top_tokens.append({
            'rank': rank,
            'index': i,
            'coord': [int(v) for v in coords[i].detach().cpu().tolist()],
            'mse': float(value.item()),
            'rmse': float(per_token_rmse[i].item()),
            'mae': float(per_token_mae[i].item()),
            'max_abs': float(per_token_max_abs[i].item()),
            'cosine': float(per_token_cos[i].item()),
            'top_channels': [
                {
                    'channel': int(ch.item()),
                    'abs_error': float(abs_err.item()),
                    'signed_error': float(token_diff[int(ch.item())].item()),
                    'pred': float(pred_feats[i, int(ch.item())].float().item()),
                    'gt': float(gt_feats[i, int(ch.item())].float().item()),
                }
                for abs_err, ch in zip(ch_values, ch_indices)
            ],
        })

    per_channel_mse = diff.pow(2).mean(dim=0)
    ch_values, ch_indices = torch.topk(per_channel_mse, k=min(topk, pred_feats.shape[1]), largest=True)
    top_channels = []
    for rank, (value, ch) in enumerate(zip(ch_values, ch_indices), start=1):
        c = int(ch.item())
        top_channels.append({
            'rank': rank,
            'channel': c,
            'mse': float(value.item()),
            'rmse': float(value.sqrt().item()),
            'mae': float(diff[:, c].abs().mean().item()),
            'pred_mean': float(pred_feats[:, c].float().mean().item()),
            'gt_mean': float(gt_feats[:, c].float().mean().item()),
        })

    return {
        'top_tokens_by_mse': top_tokens,
        'top_channels_by_mse': top_channels,
    }


def spatial_error_summary(pred_feats, gt_feats, coords):
    """
    Summarize feature error in low/mid/high bands of each sparse coord axis.
    This does not assume which latent axis maps to head height.
    """
    diff = pred_feats.float() - gt_feats.float()
    per_token_mse = diff.pow(2).mean(dim=1)
    xyz = coords[:, 1:].detach().cpu()
    summary = {}

    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        values = xyz[:, axis_idx].float()
        low_q = torch.quantile(values, 0.2)
        high_q = torch.quantile(values, 0.8)
        masks = {
            'low_20pct': values <= low_q,
            'mid_60pct': (values > low_q) & (values < high_q),
            'high_20pct': values >= high_q,
        }
        axis_report = {}
        for band, mask_cpu in masks.items():
            mask = mask_cpu.to(per_token_mse.device)
            if not bool(mask.any()):
                continue
            band_mse = per_token_mse[mask]
            axis_report[band] = {
                'count': int(mask.sum().item()),
                'coord_min': float(values[mask_cpu].min().item()),
                'coord_max': float(values[mask_cpu].max().item()),
                'mse_mean': float(band_mse.mean().item()),
                'mse_max': float(band_mse.max().item()),
            }
        summary[axis_name] = axis_report

    return summary


def write_latent_error_report(shape_slat, gt_data, pipeline, out_dir, seed, device, topk):
    gt_coords = build_gt_coords(gt_data, device)
    gt_feats_raw = torch.from_numpy(gt_data['shape_feats']).float().to(device)
    gt_feats_norm = normalize_shape_feats(gt_feats_raw, pipeline, device)

    report = {}
    pred_raw, gt_raw, aligned_coords, coord_stats = align_feats_by_coords(
        shape_slat.coords, shape_slat.feats, gt_coords, gt_feats_raw
    )
    report['coords'] = coord_stats
    report['raw_gt_space'] = feature_metrics(pred_raw, gt_raw)
    report['raw_gt_space']['topk'] = topk_feature_errors(
        pred_raw, gt_raw, aligned_coords, topk
    )
    report['raw_gt_space']['spatial_error_summary'] = spatial_error_summary(
        pred_raw, gt_raw, aligned_coords
    )

    if gt_feats_norm is not None:
        pred_norm, gt_norm, _, _ = align_feats_by_coords(
            shape_slat.coords, shape_slat.feats, gt_coords, gt_feats_norm
        )
        report['normalized_gt_space'] = feature_metrics(pred_norm, gt_norm)
        raw_mse = report['raw_gt_space']['mse']
        norm_mse = report['normalized_gt_space']['mse']
        report['lower_mse_space'] = 'normalized_gt_space' if norm_mse < raw_mse else 'raw_gt_space'
    else:
        report['normalized_gt_space'] = None
        report['lower_mse_space'] = 'raw_gt_space'

    path = out_dir / f"latent_error_seed{seed}.json"
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  Saved latent error report: {path}")
    print(f"  Latent MSE raw={report['raw_gt_space']['mse']:.6f}")
    print("  Top raw-space token errors:")
    for item in report['raw_gt_space']['topk']['top_tokens_by_mse'][:min(5, topk)]:
        print(f"    #{item['rank']} idx={item['index']} coord={item['coord']} "
              f"mse={item['mse']:.6f} rmse={item['rmse']:.6f} cos={item['cosine']:.6f}")
    if report['normalized_gt_space'] is not None:
        print(f"  Latent MSE norm={report['normalized_gt_space']['mse']:.6f}")
    print(f"  Lower MSE space: {report['lower_mse_space']}")


def sample_mesh_points(tm, n_points, seed):
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, _ = trimesh.sample.sample_surface(tm, n_points)
    finally:
        np.random.set_state(rng_state)
    return points.astype(np.float32)


def nearest_distances(src_points, dst_points, chunk_size=4096):
    """
    Point-to-nearest-sampled-point distances. This is a robust fallback Chamfer
    approximation that avoids optional trimesh proximity dependencies.
    """
    src = torch.from_numpy(src_points).float()
    dst = torch.from_numpy(dst_points).float()
    mins = []
    for i in range(0, src.shape[0], chunk_size):
        d = torch.cdist(src[i:i + chunk_size], dst)
        mins.append(d.min(dim=1).values)
    return torch.cat(mins, dim=0).numpy()


def distance_stats(dist, unit_scale_cm):
    return {
        'mean': float(dist.mean()),
        'median': float(np.median(dist)),
        'rmse': float(np.sqrt((dist ** 2).mean())),
        'p90': float(np.percentile(dist, 90)),
        'p95': float(np.percentile(dist, 95)),
        'max': float(dist.max()),
        'mean_cm': float(dist.mean() * unit_scale_cm),
        'median_cm': float(np.median(dist) * unit_scale_cm),
        'rmse_cm': float(np.sqrt((dist ** 2).mean()) * unit_scale_cm),
        'p90_cm': float(np.percentile(dist, 90) * unit_scale_cm),
        'p95_cm': float(np.percentile(dist, 95) * unit_scale_cm),
        'max_cm': float(dist.max() * unit_scale_cm),
    }


def chamfer_report(pred_points, gt_points, unit_scale_cm):
    pred_to_gt = nearest_distances(pred_points, gt_points)
    gt_to_pred = nearest_distances(gt_points, pred_points)
    chamfer_l2 = float((pred_to_gt ** 2).mean() + (gt_to_pred ** 2).mean())
    chamfer_l1 = float(pred_to_gt.mean() + gt_to_pred.mean())
    return {
        'pred_to_gt': distance_stats(pred_to_gt, unit_scale_cm),
        'gt_to_pred': distance_stats(gt_to_pred, unit_scale_cm),
        'chamfer_l1': chamfer_l1,
        'chamfer_l2': chamfer_l2,
        'chamfer_l1_cm': chamfer_l1 * unit_scale_cm,
        'chamfer_l2_cm2': chamfer_l2 * (unit_scale_cm ** 2),
    }


def filter_head_points(points, y_threshold):
    mask = points[:, 1] >= y_threshold
    return points[mask]


def face_bbox_from_gt_vertices(vertices, x_ratio=0.18, y_min_q=0.70, y_max_q=0.98):
    """
    Approximate face/head-front-ish region from GT mesh coordinates.

    This intentionally starts from a central X slice before computing Y bounds,
    so raised arms at the side do not dominate the "head" region.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    x_center = float(np.median(x))
    x_radius = float((x.max() - x.min()) * x_ratio)
    central = np.abs(x - x_center) <= x_radius
    y_source = y[central] if central.any() else y
    y_min = float(np.percentile(y_source, 100.0 * y_min_q))
    y_max = float(np.percentile(y_source, 100.0 * y_max_q))
    return {
        'x_center': x_center,
        'x_radius': x_radius,
        'x_min': x_center - x_radius,
        'x_max': x_center + x_radius,
        'y_min': y_min,
        'y_max': y_max,
        'central_vertex_count': int(central.sum()),
    }


def filter_face_points(points, bbox):
    mask = (
        (points[:, 0] >= bbox['x_min']) &
        (points[:, 0] <= bbox['x_max']) &
        (points[:, 1] >= bbox['y_min']) &
        (points[:, 1] <= bbox['y_max'])
    )
    return points[mask]


def print_distance_line(label, metrics):
    if metrics is None:
        print(f"  {label}: no points")
        return
    p2g = metrics['pred_to_gt']
    g2p = metrics['gt_to_pred']
    print(
        f"  {label} p2g mean/p95/max: "
        f"{p2g['mean_cm']:.3f}/{p2g['p95_cm']:.3f}/{p2g['max_cm']:.3f} cm; "
        f"g2p mean/p95/max: "
        f"{g2p['mean_cm']:.3f}/{g2p['p95_cm']:.3f}/{g2p['max_cm']:.3f} cm"
    )


def write_mesh_error_report(pred_tm, gt_tm, gt_data, pipeline, out_dir, seed,
                            device, n_points, head_ratio, unit_scale_cm,
                            face_x_ratio, face_y_min_q, face_y_max_q):
    gt_coords = build_gt_coords(gt_data, device)
    gt_feats = torch.from_numpy(gt_data['shape_feats']).float().to(device)
    gt_slat = sp.SparseTensor(feats=gt_feats, coords=gt_coords)

    print("Debug: decoding GT latent reconstruction mesh...")
    gt_mesh = pipeline.models['shape_slat_decoder'](gt_slat)
    if isinstance(gt_mesh, tuple):
        gt_mesh = gt_mesh[0]
    if isinstance(gt_mesh, list):
        gt_mesh = gt_mesh[0]
    gt_tm = to_trimesh(gt_mesh)

    gt_mesh_path = out_dir / f"gt_recon_seed{seed}.obj"
    gt_tm.export(str(gt_mesh_path))

    pred_points = sample_mesh_points(pred_tm, n_points, seed + 17)
    gt_points = sample_mesh_points(gt_tm, n_points, seed + 29)
    full = chamfer_report(pred_points, gt_points, unit_scale_cm)

    y_threshold = float(np.percentile(gt_tm.vertices[:, 1], 100.0 * (1.0 - head_ratio)))
    pred_head = filter_head_points(pred_points, y_threshold)
    gt_head = filter_head_points(gt_points, y_threshold)
    head = None
    if len(pred_head) > 0 and len(gt_head) > 0:
        head = chamfer_report(pred_head, gt_head, unit_scale_cm)

    face_bbox = face_bbox_from_gt_vertices(
        gt_tm.vertices,
        x_ratio=face_x_ratio,
        y_min_q=face_y_min_q,
        y_max_q=face_y_max_q,
    )
    pred_face = filter_face_points(pred_points, face_bbox)
    gt_face = filter_face_points(gt_points, face_bbox)
    face = None
    if len(pred_face) > 0 and len(gt_face) > 0:
        face = chamfer_report(pred_face, gt_face, unit_scale_cm)

    report = {
        'reference': 'GT reconstruction decoded from GT shape_coords + GT shape_feats',
        'unit_scale_cm': unit_scale_cm,
        'n_surface_samples': int(n_points),
        'gt_recon_mesh': str(gt_mesh_path),
        'full_mesh': full,
        'head_region': {
            'note': 'Top-Y region; can include raised arms/hands and is not a strict face mask.',
            'head_ratio_by_gt_y': head_ratio,
            'gt_y_threshold': y_threshold,
            'pred_sample_count': int(len(pred_head)),
            'gt_sample_count': int(len(gt_head)),
            'metrics': head,
        },
        'face_region': {
            'note': 'Approximate central upper GT bbox; intended to exclude side raised arms.',
            'bbox': face_bbox,
            'pred_sample_count': int(len(pred_face)),
            'gt_sample_count': int(len(gt_face)),
            'metrics': face,
        },
    }

    path = out_dir / f"mesh_error_seed{seed}.json"
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  Saved GT reconstruction mesh: {gt_mesh_path}")
    print(f"  Saved mesh error report: {path}")
    print_distance_line("Full", full)
    print_distance_line("Top-Y head", head)
    print_distance_line("Central face", face)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--adapter_ckpt', type=str, required=True)
    parser.add_argument('--head_image', type=str, default=None,
                        help='Head/face image for identity (adapter condition)')
    parser.add_argument('--fullbody_image', type=str, default=None,
                        help='Full-body reference image for pose/body (base condition, head will be masked)')
    parser.add_argument('--out_dir', type=str, default='results/infer')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--head_mask_fill', type=str, default='black',
                        choices=['black', 'noise', 'mean'],
                        help='Fill method for head mask region')
    parser.add_argument('--pipeline_type', type=str, default='512',
                        choices=['512', '1024', '1024_cascade'])
    parser.add_argument('--gt_pt', type=str, default=None,
                    help='Use GT encoded shape_coords from .pt instead of SS sampling')
    parser.add_argument('--shape_steps', type=int, default=12,
                    help='Shape SLat sampling steps')
    parser.add_argument('--use_gt_cond', action='store_true',
                    help='Use cond/id_emb stored in --gt_pt instead of re-encoding images')
    parser.add_argument('--debug_latent_error', action='store_true',
                    help='Compare sampled shape_slat feats against GT shape_feats from --gt_pt')
    parser.add_argument('--debug_topk', type=int, default=20,
                    help='Number of largest-error tokens/channels to include in latent debug report')
    parser.add_argument('--debug_mesh_error', action='store_true',
                    help='Decode GT latent and report mesh-space Chamfer/nearest-point distances')
    parser.add_argument('--mesh_error_samples', type=int, default=20000,
                    help='Number of surface points sampled per mesh for mesh error metrics')
    parser.add_argument('--mesh_head_ratio', type=float, default=0.15,
                    help='Top GT-Y fraction used as head region for mesh error metrics')
    parser.add_argument('--mesh_unit_scale_cm', type=float, default=100.0,
                    help='Centimeters per mesh unit; use 100 if mesh units are meters')
    parser.add_argument('--mesh_face_x_ratio', type=float, default=0.18,
                    help='Central X half-width as fraction of GT mesh width for face-region metrics')
    parser.add_argument('--mesh_face_y_min_q', type=float, default=0.70,
                    help='Lower Y quantile within central GT slice for face-region metrics')
    parser.add_argument('--mesh_face_y_max_q', type=float, default=0.98,
                    help='Upper Y quantile within central GT slice for face-region metrics')
    args = parser.parse_args()

    if args.use_gt_cond and args.gt_pt is None:
        raise ValueError("--use_gt_cond requires --gt_pt")
    if args.debug_latent_error and args.gt_pt is None:
        raise ValueError("--debug_latent_error requires --gt_pt")
    if args.debug_mesh_error and args.gt_pt is None:
        raise ValueError("--debug_mesh_error requires --gt_pt")
    if not args.use_gt_cond and (args.head_image is None or args.fullbody_image is None):
        raise ValueError("--head_image and --fullbody_image are required unless --use_gt_cond is set")

    cfg = load_config(args.config)
    device = torch.device('cuda')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pipeline
    print("Loading pipeline...")
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(cfg.model.pipeline_path)
    pipeline.cuda()

    gt_data = None
    if args.gt_pt is not None:
        gt_data = torch.load(args.gt_pt, map_location='cpu', weights_only=False)

    # 2. Build shape adapter + load weights
    print("Building adapter...")
    flow_model_key = cfg.model.flow_model_key
    base_model = pipeline.models[flow_model_key]

    shared_proj = ImageProjModel(
        in_dim=cfg.model.id_channels,
        out_dim=base_model.model_channels,
        num_tokens=cfg.model.num_id_tokens,
    ).to(device)

    shape_adapter = IPAdapterSLatFlowModel(
        base_model,
        ip_scale=cfg.model.ip_scale,
        ip_interval=cfg.model.ip_interval,
        shared_id_proj=shared_proj,
    ).to(device)

    ckpt = torch.load(args.adapter_ckpt, map_location='cpu')
    shape_adapter.ip_layers.load_state_dict(ckpt['ip_layers'])
    if 'id_proj' in ckpt:
        shared_proj.load_state_dict(ckpt['id_proj'])
    print(f"Loaded adapter from {args.adapter_ckpt}")
    shape_adapter.eval()

    # 3. Encode conditions (decoupled), or reuse training-time conditions.
    if args.use_gt_cond:
        print("Encoding conditions: using cond/id_emb from GT .pt...")
        cond_dict = to_device_cond_dict(gt_data, device)
        id_emb = gt_data['id_emb'].unsqueeze(0).to(device)
    else:
        print("Encoding conditions...")

        # Base cond: full-body image with head masked → DINOv3 → cond tokens
        fullbody_img = Image.open(args.fullbody_image)
        fullbody_masked = mask_head_in_image(
            fullbody_img, mesh=None,  # no mesh at inference, use heuristic
            head_ratio=0.20, expand=1.3, fill=args.head_mask_fill,
        )
        fullbody_processed = pipeline.preprocess_image(fullbody_masked)
        cond_dict = pipeline.get_cond([fullbody_processed], resolution=512)

        # Save masked image for debugging
        fullbody_masked.save(out_dir / "debug_fullbody_masked.png")
        print(f"  Saved masked full-body to {out_dir / 'debug_fullbody_masked.png'}")

        # Adapter cond: head image → DINOv3 → mean-pool → id_emb
        head_img = Image.open(args.head_image)
        head_processed = pipeline.preprocess_image(head_img)
        head_cond_dict = pipeline.get_cond([head_processed], resolution=512)
        id_emb = head_cond_dict['cond'].mean(dim=1)  # [1, 1024]

    # Set cached id_cond so pipeline's sampler can use it
    shape_adapter.set_id_cond(id_emb)

    # 4. SS stage (frozen, no adapter — uses masked full-body cond)
    torch.manual_seed(args.seed)

    if args.gt_pt is not None:
        print("SS stage: using GT coords from preprocessed .pt...")
        coords = build_gt_coords(gt_data, device)
    else:
        print("SS stage (frozen)...")
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32}[args.pipeline_type]
        coords = pipeline.sample_sparse_structure(cond_dict, resolution=ss_res)

    print(f"  Sparse structure: {coords.shape[0]} voxels")

    # 5. Shape stage (with adapter, using pipeline's sampler with CFG)
    print("Shape stage (with adapter)...")
    shape_slat = pipeline.sample_shape_slat(
        cond_dict, shape_adapter, coords,
        sampler_params={"steps": args.shape_steps},
    )
    print(f"  Shape latent: {shape_slat.feats.shape}")

    if args.debug_latent_error:
        print("Debug: measuring sampled latent feature error...")
        write_latent_error_report(shape_slat, gt_data, pipeline, out_dir, args.seed, device, args.debug_topk)

    # 6. Decode shape → mesh → render normal
    print("Decoding shape...")
    pipeline.models['shape_slat_decoder'].to(device)
    pipeline.models['shape_slat_decoder'].set_resolution(
        {'512': 512, '1024': 1024, '1024_cascade': 1024}[args.pipeline_type]
    )
    mesh = pipeline.models['shape_slat_decoder'](shape_slat)
    if isinstance(mesh, tuple):
        mesh = mesh[0]
    if isinstance(mesh, list):
        mesh = mesh[0]

    mesh_path = out_dir / f"infer_seed{args.seed}.obj"
    tm = to_trimesh(mesh)
    tm.export(str(mesh_path))
    print(f"  Saved mesh: {mesh_path}")

    if args.debug_mesh_error:
        write_mesh_error_report(
            tm, None, gt_data, pipeline, out_dir, args.seed, device,
            args.mesh_error_samples, args.mesh_head_ratio, args.mesh_unit_scale_cm,
            args.mesh_face_x_ratio, args.mesh_face_y_min_q, args.mesh_face_y_max_q,
        )

    print("Rendering...")
    from trellis2.utils import render_utils
    frames = render_utils.render_video(mesh, bg_color=(0.5, 0.5, 0.5))
    for k, v in frames.items():
        video_path = out_dir / f"infer_seed{args.seed}_{k}.mp4"
        imageio.mimsave(str(video_path), v, fps=15)
        print(f"  Saved: {video_path}")

    # Cleanup
    shape_adapter.clear_id_cond()
    print(f"\n[Done] Results in {out_dir}/")


if __name__ == '__main__':
    main()
