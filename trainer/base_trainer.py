"""
Base trainer with shared training loop logic for TRELLIS.2.

Handles: config loading, optimizer setup, checkpointing, loss logging, CFG dropout.
Stage-specific trainers override `_build_model()`, `_load_dataset()`,
and `_train_step()`.
"""
import json
import random
import datetime
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from abc import ABC, abstractmethod
from easydict import EasyDict as edict


def load_config(config_path: str) -> edict:
    """Load YAML config and return as EasyDict."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return edict(cfg)

class BaseTrainer(ABC):
    """
    Shared training loop for Shape and Texture adapter training.

    Subclasses implement:
        _build_model(pipeline) → adapter model
        _load_dataset() → dataset, sigma_min
        _train_step(model, data, t_scalar, sigma_min) → loss tensor
    """

    def __init__(self, cfg: edict):
        self.cfg = cfg
        self.device = torch.device('cuda')
        random.seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        torch.manual_seed(cfg.train.seed)

    def run(self):
        """Full training procedure."""
        cfg = self.cfg
        ts = datetime.datetime.now().strftime("%m%d_%H%M%S")
        out_dir = Path(cfg.output.dir) / f"{self.run_prefix}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir

        # Save config
        yaml.dump(dict(cfg), open(out_dir / "config.yaml", 'w'), default_flow_style=False)

        # 1. Load pipeline
        print("[1/4] Loading pipeline...")
        pipeline = self._load_pipeline()

        # 2. Build adapter model
        print("[2/4] Building adapter...")
        model, shared_proj = self._build_model(pipeline)
        model = model.to(self.device)

        trainable = model.get_trainable_params()
        if shared_proj is not None:
            trainable = list(shared_proj.parameters()) + trainable
        n_params = sum(p.numel() for p in trainable)
        n_frozen = sum(p.numel() for p in model.base.parameters())
        print(f"  Trainable: {n_params / 1e6:.1f}M, Frozen: {n_frozen / 1e6:.1f}M")

        optimizer = torch.optim.AdamW(
            trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )

        # 3. Load dataset
        print("[3/4] Loading data...")
        dataset, sigma_min = self._load_dataset()

        # 4. Training loop
        print(f"[4/4] Training for {cfg.train.steps} steps...")
        model.train()
        model.base.eval()

        losses = []
        pbar = tqdm(range(cfg.train.steps), desc=f"{self.run_prefix} Train")

        for step in pbar:
            idx = np.random.randint(len(dataset))
            data = dataset[idx]

            # Move to device
            data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()}

            data = self._apply_cfg_dropout(data)

            t_scalar = torch.sigmoid(torch.randn(1) * 1.0 + 0.0).item()

            loss = self._train_step(model, data, t_scalar, sigma_min)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=cfg.train.grad_clip)
            optimizer.step()

            losses.append(loss.item())

            if (step + 1) % cfg.train.log_every == 0:
                avg_loss = np.mean(losses[-cfg.train.log_every:])
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

            if (step + 1) % cfg.train.save_every == 0:
                self._save_checkpoint(model, shared_proj, step + 1)

        # Final save
        self._save_checkpoint(model, shared_proj, cfg.train.steps, final=True)
        self._plot_loss(losses, out_dir / "loss.png")

        summary = {
            'stage': self.run_prefix,
            'steps': cfg.train.steps,
            'lr': cfg.train.lr,
            'n_trainable': n_params,
            'n_samples': len(dataset),
            'final_loss': float(np.mean(losses[-100:])),
        }
        json.dump(summary, open(out_dir / "summary.json", 'w'), indent=2)
        print(f"\n[Done] {out_dir}/")

    def _save_checkpoint(self, model, shared_proj, step, final=False):
        """Save adapter weights + shared proj."""
        suffix = "final" if final else f"step{step}"
        ckpt = {'ip_layers': model.ip_layers.state_dict()}
        if shared_proj is not None:
            ckpt['id_proj'] = shared_proj.state_dict()
        path = self.out_dir / f"adapter_{suffix}.pt"
        torch.save(ckpt, str(path))
        print(f"\n  Saved {path}")

    def _load_pipeline(self):
        """Load TRELLIS.2 pipeline."""
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(self.cfg.model.pipeline_path)
        pipeline.cuda()
        return pipeline

    @property
    @abstractmethod
    def run_prefix(self) -> str:
        ...

    @abstractmethod
    def _build_model(self, pipeline):
        """
        Build adapter model.

        Returns:
            model: adapter model
            shared_proj: ImageProjModel if shared, else None
        """
        ...

    @abstractmethod
    def _load_dataset(self):
        """Returns (dataset, sigma_min)."""
        ...

    @abstractmethod
    def _train_step(self, model, data, t_scalar, sigma_min) -> torch.Tensor:
        ...

    def _apply_cfg_dropout(self, data):
        """
        Classifier-free guidance dropout.

        cfg_drop_cond:  drop base cond → model relies on id_cond
        cfg_drop_id:    drop id_cond → base behavior
        cfg_drop_both:  drop both → unconditional
        """
        cfg = self.cfg.train
        drop = random.random()
        p1 = cfg.cfg_drop_cond
        p2 = p1 + cfg.cfg_drop_id
        p3 = p2 + cfg.cfg_drop_both

        if drop < p1:
            data['cond'] = data['neg_cond']
        elif drop < p2:
            data['id_emb'] = torch.zeros_like(data['id_emb'])
        elif drop < p3:
            data['cond'] = data['neg_cond']
            data['id_emb'] = torch.zeros_like(data['id_emb'])
        return data

    @staticmethod
    def _plot_loss(losses, path):
        """Save loss curve plot."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(losses, alpha=0.3, color='blue')
            w = min(100, len(losses) // 10 + 1)
            if w > 1:
                smooth = np.convolve(losses, np.ones(w) / w, mode='valid')
                ax.plot(range(w - 1, len(losses)), smooth, color='red',
                        label=f'smooth(w={w})')
            ax.set_xlabel('Step')
            ax.set_ylabel('MSE Loss')
            ax.set_title('Training Loss')
            ax.legend()
            fig.tight_layout()
            fig.savefig(str(path), dpi=150)
            plt.close()
        except Exception:
            pass
