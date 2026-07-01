#!/usr/bin/env python
"""
FaceMesh-Adapter v2 training entry point.

Usage:
    cd ~/TRELLIS.2

    # Shape adapter (default)
    CUDA_VISIBLE_DEVICES=2 python facemesh_adapter_v2/scripts/train.py \
        --config facemesh_adapter_v2/configs/default.yaml

    # Overfit sanity check
    CUDA_VISIBLE_DEVICES=2 python facemesh_adapter_v2/scripts/train.py \
        --config facemesh_adapter_v2/configs/overfit_1sample.yaml

    # Override config values via CLI
    CUDA_VISIBLE_DEVICES=2 python facemesh_adapter_v2/scripts/train.py \
        --config facemesh_adapter_v2/configs/default.yaml \
        --train.steps 5000 --train.lr 2e-4
"""
import os
import sys
import argparse

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from facemesh_adapter_v2.trainer import ShapeTrainer, load_config


def apply_overrides(cfg, overrides):
    """Apply --key.subkey value CLI overrides to config."""
    i = 0
    while i < len(overrides):
        key = overrides[i].lstrip('-')
        val = overrides[i + 1]
        parts = key.split('.')
        d = cfg
        for p in parts[:-1]:
            d = d[p]
        # Auto-cast types
        old_val = d.get(parts[-1])
        if isinstance(old_val, bool):
            val = val.lower() in ('true', '1', 'yes')
        elif isinstance(old_val, int):
            val = int(val)
        elif isinstance(old_val, float):
            val = float(val)
        d[parts[-1]] = val
        i += 2


def main():
    parser = argparse.ArgumentParser(description="FaceMesh-Adapter v2 Training")
    parser.add_argument('--config', type=str, required=True, help='YAML config path')
    parser.add_argument('--stage', type=str, default='shape',
                        choices=['shape', 'texture'], help='Which stage to train')
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config)
    if overrides:
        apply_overrides(cfg, overrides)

    print(f"Stage: {args.stage}")
    print(f"Config: {args.config}")

    if args.stage == 'shape':
        trainer = ShapeTrainer(cfg)
    elif args.stage == 'texture':
        from facemesh_adapter_v2.trainer.texture_trainer import TextureTrainer
        trainer = TextureTrainer(cfg)

    trainer.run()


if __name__ == '__main__':
    main()
