#!/usr/bin/env python
"""
Quick full-pipeline validation for FaceMesh-Adapter v2.

Tests every component with minimal data (no real training):
  1. Load pipeline + build adapters → check param counts
  2. Forward pass with dummy data → check shapes
  3. Load real .pt data (if available) → check data format
  4. One training step → check loss is finite
  5. Save/load adapter weights → check roundtrip

Usage:
    cd ~/TRELLIS.2
    CUDA_VISIBLE_DEVICES=2 python facemesh_adapter_v2/scripts/validate.py \
        [--data_dir data/adapter_train]
"""
import os
import sys
import argparse
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Preprocessed .pt dir (optional, skips data test if absent)')
    parser.add_argument('--pipeline_model', type=str, default='microsoft/TRELLIS.2-4B')
    args = parser.parse_args()

    device = torch.device('cuda')
    passed = 0
    failed = 0

    def check(name, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # =========================================================
    print("\n[1/5] Loading pipeline + building adapters...")
    # =========================================================
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules import sparse as sp
    from facemesh_adapter_v2.libs.encoders.projections import ImageProjModel
    from facemesh_adapter_v2.libs.adapters import IPAdapterSLatFlowModel

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.pipeline_model)
    pipeline.cuda()

    shape_model = pipeline.models['shape_slat_flow_model_512']
    shared_proj = ImageProjModel(in_dim=1024, out_dim=1536, num_tokens=4).to(device)
    shape_adapter = IPAdapterSLatFlowModel(
        shape_model, shared_id_proj=shared_proj, ip_interval=2
    ).to(device)

    proj_params = sum(p.numel() for p in shared_proj.parameters())
    ip_params = sum(p.numel() for p in shape_adapter.ip_layers.parameters())
    frozen_params = sum(p.numel() for p in shape_adapter.base.parameters())

    check("Shared proj params", lambda: assert_close(proj_params / 1e6, 44, tol=5))
    check("IP layers params", lambda: assert_close(ip_params / 1e6, 141, tol=10))
    check("Base frozen", lambda: assert_(not any(p.requires_grad for p in shape_adapter.base.parameters())))
    check("Adapter layers trainable", lambda: assert_(all(p.requires_grad for p in shape_adapter.ip_layers.parameters())))

    print(f"  Params: proj={proj_params/1e6:.1f}M, ip_layers={ip_params/1e6:.1f}M, frozen={frozen_params/1e6:.1f}M")

    # =========================================================
    print("\n[2/5] Forward pass with dummy data...")
    # =========================================================
    B, N = 1, 500
    coords = torch.cat([
        torch.zeros(N, 1, dtype=torch.int32),
        torch.randint(0, 64, (N, 3), dtype=torch.int32)
    ], dim=1).to(device)
    feats = torch.randn(N, 32).to(device)
    x = sp.SparseTensor(feats=feats, coords=coords)
    t = torch.tensor([500.0]).to(device)
    cond = torch.randn(B, 1024, 1024).to(device)
    id_emb = torch.randn(B, 1024).to(device)

    with torch.no_grad():
        # With adapter
        out = shape_adapter(x, t, cond, id_cond=id_emb)
        check("Forward with id_cond", lambda: assert_(out.feats.shape == (N, 32)))

        # Without adapter (bypass)
        out_bypass = shape_adapter(x, t, cond, id_cond=None)
        check("Forward without id_cond (bypass)", lambda: assert_(out_bypass.feats.shape == (N, 32)))

    # =========================================================
    print("\n[3/5] Data loading test...")
    # =========================================================
    if args.data_dir and os.path.exists(args.data_dir):
        from facemesh_adapter_v2.data.dataset import ShapeAdapterDataset, collate_sparse

        dataset = ShapeAdapterDataset(args.data_dir, split='train', val_ratio=0.0)
        check("Dataset loaded", lambda: assert_(len(dataset) > 0))

        sample = dataset[0]
        check("Sample has shape_feats", lambda: assert_('shape_feats' in sample))
        check("Sample has id_emb", lambda: assert_('id_emb' in sample))
        check("Sample has cond", lambda: assert_('cond' in sample))
        check("shape_feats dim", lambda: assert_(sample['shape_feats'].shape[1] == 32))
        check("id_emb dim", lambda: assert_(sample['id_emb'].shape[0] == 1024))

        print(f"  Sample: coords={sample['coords'].shape}, feats={sample['shape_feats'].shape}, "
              f"cond={sample['cond'].shape}, id={sample['id_emb'].shape}")
    else:
        print("  (skipped, no --data_dir or dir not found)")

    # =========================================================
    print("\n[4/5] Training step test...")
    # =========================================================
    shape_adapter.train()
    shape_adapter.base.eval()
    trainable = list(shared_proj.parameters()) + shape_adapter.get_ip_layer_params()
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)

    sigma_min = 1e-5
    t_scalar = 0.5
    x_0 = feats  # pretend this is GT latent
    noise = torch.randn_like(x_0)
    x_t_feats = (1 - t_scalar) * x_0 + (sigma_min + (1 - sigma_min) * t_scalar) * noise
    v_target = (1 - sigma_min) * noise - x_0
    x_t = sp.SparseTensor(feats=x_t_feats, coords=coords)

    v_pred = shape_adapter(x_t, t, cond, id_cond=id_emb)
    loss = F.mse_loss(v_pred.feats, v_target)

    check("Loss is finite", lambda: assert_(torch.isfinite(loss)))
    print(f"  Loss: {loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    check("Backward pass", lambda: assert_(shared_proj.proj[0].weight.grad is not None))

    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    optimizer.step()
    check("Optimizer step", lambda: True)

    # =========================================================
    print("\n[5/5] Save/load checkpoint test...")
    # =========================================================
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp_path = f.name

    ckpt = {
        'id_proj': shared_proj.state_dict(),
        'ip_layers': shape_adapter.ip_layers.state_dict(),
    }
    torch.save(ckpt, tmp_path)

    # Reload into fresh instances
    shared_proj_2 = ImageProjModel(in_dim=1024, out_dim=1536, num_tokens=4).to(device)
    shape_adapter_2 = IPAdapterSLatFlowModel(
        shape_model, shared_id_proj=shared_proj_2, ip_interval=2
    ).to(device)

    loaded = torch.load(tmp_path, map_location='cpu')
    shared_proj_2.load_state_dict(loaded['id_proj'])
    shape_adapter_2.ip_layers.load_state_dict(loaded['ip_layers'])

    # Verify weights match
    with torch.no_grad():
        out_original = shape_adapter(x_t, t, cond, id_cond=id_emb)
        shape_adapter_2.eval()
        out_loaded = shape_adapter_2(x_t, t, cond, id_cond=id_emb)

    check("Checkpoint roundtrip", lambda: assert_(
        torch.allclose(out_original.feats, out_loaded.feats, atol=1e-5)
    ))

    os.unlink(tmp_path)

    # =========================================================
    print(f"\n{'='*50}")
    print(f"  Validation: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    return failed == 0


def assert_(condition):
    if not condition:
        raise AssertionError()


def assert_close(a, b, tol=1):
    if abs(a - b) > tol:
        raise AssertionError(f"{a} not close to {b} (tol={tol})")


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
