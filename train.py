import os
import json
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from util.utils import get_device, set_seed, build_model
# from data.dataset import CriteoDataset  # 실제 데이터셋 있으면 주석 해제


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    log_interval: int = 100,
) -> int:
    model.train()
    running_loss = 0.0
    running_corrects = 0
    running_total = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", ncols=100)

    for batch_idx, (xi, xv, y) in enumerate(pbar):
        xi = xi.to(device=device, dtype=torch.long)
        xv = xv.to(device=device, dtype=torch.float)
        y = y.to(device=device, dtype=torch.float)  # (N,) 또는 (N,1)

        logits = model(xi, xv)          # DeepFM: (N,), AutoInt: (N,1)
        logits = logits.view(-1)        # (N,)
        loss = criterion(logits, y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct = (preds == y.view(-1)).sum().item()
            total = y.numel()

        running_loss += loss.item() * total
        running_corrects += correct
        running_total += total

        global_step += 1

        # tqdm status
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct / total:.4f}"
        })

        # wandb step logging
        if global_step % log_interval == 0:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/accuracy": correct / total,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }
            )

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = running_corrects / max(running_total, 1)

    wandb.log(
        {
            "train/epoch_loss": epoch_loss,
            "train/epoch_accuracy": epoch_acc,
            "train/epoch": epoch,
            "train/global_step": global_step,
        }
    )

    print(f"[Train] Epoch: {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    global_step: int,
    split: str = "val",
):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for xi, xv, y in loader:
        xi = xi.to(device=device, dtype=torch.long)
        xv = xv.to(device=device, dtype=torch.float)
        y = y.to(device=device, dtype=torch.float)

        logits = model(xi, xv)
        logits = logits.view(-1)
        loss = criterion(logits, y.view(-1))

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct = (preds == y.view(-1)).sum().item()
        total = y.numel()

        running_loss += loss.item() * total
        running_correct += correct
        running_total += total

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = running_correct / max(running_total, 1)

    wandb.log(
        {
            f"{split}/loss": epoch_loss,
            f"{split}/acc": epoch_acc,
            f"{split}/epoch": epoch,
            "global_step": global_step,
        }
    )

    print(f"[{split.capitalize()}] Epoch: {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def main():
    config_path = "config.json"
    with open(config_path, "r") as f:
        config: Dict[str, Any] = json.load(f)

    os.makedirs(config["save_dir"], exist_ok=True)

    set_seed(config["seed"])
    device = get_device(config["use_cuda"])
    print("Device:", device)

    wandb.init(
        project=config["project"],
        name=config["run_name"],
        config=config,
    )

    # TODO: 실제 데이터셋으로 교체
    # train_data = CriteoDataset(config["data_dir"], train=True)
    # val_data = CriteoDataset(config["data_dir"], train=False)
    train_data = ...
    val_data = ...

    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # feature_sizes 추출
    if hasattr(train_data, "feature_sizes"):
        feature_sizes = train_data.feature_sizes
    elif hasattr(train_data, "field_sizes"):
        feature_sizes = train_data.field_sizes
    else:
        raise ValueError("Dataset must have 'feature_sizes' or 'field_sizes' attribute.")

    # DeepFM / AutoInt 선택은 build_model 안에서 config["model_type"] 보고 처리한다고 가정
    model = build_model(config, feature_sizes, device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(config["epochs"] // 3, 1),
        gamma=0.1,
    )

    wandb.watch(model, log="all", log_freq=100)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, config["epochs"] + 1):
        print(f"\n===== Epoch [{epoch}/{config['epochs']}] =====")

        global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            global_step=global_step,
            log_interval=100,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            global_step=global_step,
            split="val",
        )

        scheduler.step()

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = os.path.join(
                config["save_dir"], f"best_{config['model_type']}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"★ New best model saved to {ckpt_path} (val_loss={best_val_loss:.4f})")

    print("Training finished.")
    wandb.finish()


if __name__ == "__main__":
    main()
