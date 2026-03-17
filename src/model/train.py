"""
Usage:
    python src/model/train.py --config src/llm_generator_pipeline/config.yaml
    python src/model/train.py --config src/llm_generator_pipeline/config.yaml --resume
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.llm_generator_pipeline.dataloader import ANCEBatchSampler, build_infonce_dataloader
from src.llm_generator_pipeline.dataset import NameDataset
from src.model.config import ByteEncoderConfig
from src.model.encode_all import encode_all
from src.model.encoder import ByteLevelEncoder
from src.model.loss import build_loss


def get_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))  # cosine decay
    return LambdaLR(optimizer, lr_lambda)


def evaluate(model: ByteLevelEncoder, val_loader, loss_fn, device: str) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            anchor   = model(batch["anchor"].to(device),   batch["anchor_mask"].to(device))
            positive = model(batch["positive"].to(device), batch["positive_mask"].to(device))
            total_loss += loss_fn(anchor, positive).item()
            n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


def save_checkpoint(
    model: ByteLevelEncoder,
    optimizer: AdamW,
    scheduler: LambdaLR,
    global_step: int,
    best_val_loss: float,
    checkpoint_dir: str,
    name: str = "best",
) -> None:
    path = Path(checkpoint_dir) / name
    model.save_pretrained(str(path))
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "best_val_loss": best_val_loss,
        },
        str(path / "training_state.pt"),
    )


def load_checkpoint(
    model: ByteLevelEncoder,
    optimizer: AdamW,
    scheduler: LambdaLR,
    checkpoint_dir: str,
    name: str = "best",
) -> tuple[int, float]:
    path = Path(checkpoint_dir) / name
    state = torch.load(str(path / "training_state.pt"), map_location="cpu")
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    return state["global_step"], state["best_val_loss"]


def main(config_path: str, resume: bool = False) -> None:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_path = "data/pipeline/04_dataset.jsonl"
    train_ds = NameDataset(data_path, "train")
    val_ds   = NameDataset(data_path, "val")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_batch_size = int(config["train_batch_size"])

    train_sampler = ANCEBatchSampler(train_ds, config)
    val_sampler   = ANCEBatchSampler(val_ds,   config)
    train_loader  = build_infonce_dataloader(train_ds, train_sampler, config)
    val_loader    = build_infonce_dataloader(val_ds,   val_sampler,   config)

    enc_config = ByteEncoderConfig(
        hidden_dim=int(config["hidden_dim"]),
        n_layers=int(config["n_layers"]),
        n_heads=int(config["n_heads"]),
        ffn_dim=int(config["ffn_dim"]),
        dropout=float(config["dropout"]),
        max_len=int(config["max_len"]),
    )
    model = ByteLevelEncoder(enc_config).to(device)

    loss_fn   = build_loss(config)
    optimizer = AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))

    epochs           = int(config["epochs"])
    steps_per_epoch  = len(train_loader)
    total_steps      = epochs * steps_per_epoch
    scheduler        = get_scheduler(optimizer, int(config["warmup_steps"]), total_steps)

    global_step   = 0
    best_val_loss = float("inf")
    checkpoint_dir = config["checkpoint_dir"]
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if resume:
        print("Resuming from checkpoint...")
        model = ByteLevelEncoder.from_pretrained(str(Path(checkpoint_dir) / "best")).to(device)
        global_step, best_val_loss = load_checkpoint(model, optimizer, scheduler, checkpoint_dir)
        print(f"Resumed at step {global_step}, best_val_loss={best_val_loss:.4f}")

    model.train()
    refresh_every = int(config["refresh_every"])
    eval_every    = int(config["eval_every"])

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            anchor   = model(batch["anchor"].to(device),   batch["anchor_mask"].to(device))
            positive = model(batch["positive"].to(device), batch["positive_mask"].to(device))

            loss = loss_fn(anchor, positive)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            loss_val     = loss.item()
            epoch_loss  += loss_val
            global_step += 1
            lr_now       = scheduler.get_last_lr()[0]
            mix          = train_sampler._current_mix_ratio()
            pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{epoch_loss/global_step:.4f}", lr=f"{lr_now:.2e}", mix=f"{mix:.2f}", step=global_step)

            if global_step % refresh_every == 0:
                embs, ids = encode_all(model, train_ds, train_batch_size, device)
                train_sampler.update_index(embs, ids)

            if global_step % eval_every == 0:
                val_loss = evaluate(model, val_loader, loss_fn, device)
                print(f"  step={global_step}  val_loss={val_loss:.4f}  best={best_val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scheduler, global_step, best_val_loss, checkpoint_dir)
                    print(f"  Saved new best checkpoint")

        avg = epoch_loss / max(1, steps_per_epoch)
        print(f"Epoch {epoch + 1} avg train loss: {avg:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/llm_generator_pipeline/config.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.config, args.resume)
