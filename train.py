"""
train.py – StrokeSegNet 訓練スクリプト

使い方:
    python train.py                            # デフォルト設定
    python train.py --config configs/config.yaml
    python train.py --resume outputs/ckpt_ep10.pth
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import yaml

from datasets import LMDBStrokeDataset
from models import StrokeSegNet
from losses import StrokeLoss


# ------------------------------------------------------------------ #
#  設定読み込み                                                       #
# ------------------------------------------------------------------ #

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(model, lr: float, weight_decay: float):
    """BN パラメータと bias には weight_decay を適用しない。"""
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "bn" in name or "bias" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def build_scheduler(optimizer, warmup_epochs: int, max_epochs: int):
    """ウォームアップ + コサインアニーリング。"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------ #
#  訓練ループ                                                         #
# ------------------------------------------------------------------ #

def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device,
    epoch, cfg, writer
):
    model.train()
    log_every = cfg["TRAIN"]["log_every"]
    use_amp   = cfg["TRAIN"]["amp"]
    t0 = time.time()
    running = {}

    for step, batch in enumerate(loader):
        # --- データをデバイスへ ---
        image         = batch["image"].to(device)
        fg_mask       = batch["fg_mask"].to(device)
        stroke_id_map = batch["stroke_id_map"].to(device)
        orientation   = batch["orientation"].to(device)
        endpoint_hm   = batch["endpoint_heatmap"].to(device)
        junction_hm   = batch["junction_heatmap"].to(device)

        targets = {
            "fg_mask":           fg_mask,
            "stroke_id_map":     stroke_id_map,
            "orientation":       orientation,
            "endpoint_heatmap":  endpoint_hm,
            "junction_heatmap":  junction_hm,
        }

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            preds = model(image)
            loss, loss_dict = criterion(preds, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        # ログ集計
        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + v

        if (step + 1) % log_every == 0:
            global_step = epoch * len(loader) + step + 1
            avg = {k: v / log_every for k, v in running.items()}
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[ep{epoch:03d} step{step+1:05d}]  "
                f"loss={avg['total']:.4f}  "
                f"fg={avg['fg']:.4f}  emb={avg['emb']:.4f}  "
                f"ori={avg['ori']:.4f}  end={avg['end']:.4f}  junc={avg['junc']:.4f}  "
                f"lr={lr:.2e}  {elapsed:.1f}s"
            )
            for k, v in avg.items():
                writer.add_scalar(f"train/{k}", v, global_step)
            writer.add_scalar("train/lr", lr, global_step)
            running = {}
            t0 = time.time()


# ------------------------------------------------------------------ #
#  検証ループ (損失のみ計算)                                          #
# ------------------------------------------------------------------ #

@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer):
    model.eval()
    totals = {}
    for batch in loader:
        image         = batch["image"].to(device)
        targets = {
            "fg_mask":           batch["fg_mask"].to(device),
            "stroke_id_map":     batch["stroke_id_map"].to(device),
            "orientation":       batch["orientation"].to(device),
            "endpoint_heatmap":  batch["endpoint_heatmap"].to(device),
            "junction_heatmap":  batch["junction_heatmap"].to(device),
        }
        preds = model(image)
        _, loss_dict = criterion(preds, targets)
        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v

    avg = {k: v / max(len(loader), 1) for k, v in totals.items()}
    print(
        f"[val ep{epoch:03d}]  "
        + "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
    )
    for k, v in avg.items():
        writer.add_scalar(f"val/{k}", v, epoch)
    return avg["total"]


# ------------------------------------------------------------------ #
#  エントリポイント                                                   #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.resume:
        cfg["TRAIN"]["resume"] = args.resume

    out_dir = Path(cfg["TRAIN"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset & DataLoader ---
    max_samples = cfg["DATA"].get("max_samples", 0)
    train_ds = LMDBStrokeDataset(
        lmdb_path        = cfg["DATA"]["lmdb_train"],
        img_size         = cfg["DATA"]["img_size"],
        margin           = cfg["DATA"]["margin"],
        thickness_range  = (cfg["DATA"]["thickness_min"], cfg["DATA"]["thickness_max"]),
        thickness_fixed  = cfg["DATA"]["thickness_test"],
        endpoint_sigma   = cfg["DATA"]["endpoint_sigma"],
        junction_sigma   = cfg["DATA"]["junction_sigma"],
        augment          = cfg["DATA"]["augment"],
        max_samples      = max_samples,
    )
    val_ds = LMDBStrokeDataset(
        lmdb_path        = cfg["DATA"]["lmdb_test"],
        img_size         = cfg["DATA"]["img_size"],
        margin           = cfg["DATA"]["margin"],
        thickness_range  = (cfg["DATA"]["thickness_min"], cfg["DATA"]["thickness_max"]),
        thickness_fixed  = cfg["DATA"]["thickness_test"],
        endpoint_sigma   = cfg["DATA"]["endpoint_sigma"],
        junction_sigma   = cfg["DATA"]["junction_sigma"],
        augment          = False,
        max_samples      = max_samples // 10 if max_samples > 0 else 0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["TRAIN"]["batch_size"],
        shuffle     = True,
        num_workers = cfg["DATA"]["num_workers"],
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["TRAIN"]["batch_size"],
        shuffle     = False,
        num_workers = cfg["DATA"]["num_workers"],
        pin_memory  = True,
    )

    # --- Model ---
    model = StrokeSegNet(emb_dim=cfg["MODEL"]["emb_dim"]).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Loss / Optimizer / Scheduler ---
    L_cfg = cfg["LOSS"]
    criterion = StrokeLoss(
        lambda_fg    = L_cfg["lambda_fg"],
        lambda_emb   = L_cfg["lambda_emb"],
        lambda_ori   = L_cfg["lambda_ori"],
        lambda_end   = L_cfg["lambda_end"],
        lambda_junc  = L_cfg["lambda_junc"],
        delta_v      = L_cfg["delta_v"],
        delta_d      = L_cfg["delta_d"],
        lambda_reg   = L_cfg["lambda_reg"],
        focal_alpha  = L_cfg["focal_alpha"],
        focal_gamma  = L_cfg["focal_gamma"],
    )

    T_cfg = cfg["TRAIN"]
    optimizer = build_optimizer(model, T_cfg["lr"], T_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, T_cfg["warmup_epochs"], T_cfg["max_epochs"])
    scaler    = GradScaler(enabled=T_cfg["amp"])

    # --- Resume ---
    start_epoch = 0
    resume_path = T_cfg.get("resume", "")
    if resume_path and Path(resume_path).is_file():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch - 1}: {resume_path}")

    # --- Training ---
    best_val = float("inf")
    for epoch in range(start_epoch, T_cfg["max_epochs"]):
        train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, cfg, writer
        )
        val_loss = validate(model, val_loader, criterion, device, epoch, writer)
        scheduler.step()

        # チェックポイント保存
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
        if is_best or (epoch + 1) % T_cfg["save_every"] == 0:
            ckpt = {
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler":    scaler.state_dict(),
                "val_loss":  val_loss,
            }
            tag = "best" if is_best else f"ep{epoch:03d}"
            save_path = out_dir / f"ckpt_{tag}.pth"
            torch.save(ckpt, save_path)
            print(f"Saved: {save_path}")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
