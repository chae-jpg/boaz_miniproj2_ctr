import os
import json
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import accuracy_score

from util.utils import get_device, set_seed, build_model
from data.split import split_exact_tvt_from_preprocessed, extract_dims
from data.dataset import create_v1_dataloaders, create_v2_dataloaders, create_v3_dataloaders

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

    for batch_idx, batch in enumerate(pbar):
        xi = batch["xi"].to(device=device, dtype=torch.long)
        xv = batch["xv"].to(device=device, dtype=torch.float)
        y = batch["label"].to(device=device, dtype=torch.long)  # (N,)
        
        seq = None
        if "seq" in batch:
             seq = batch["seq"].to(device=device, dtype=torch.long)

        logits = model(xi, xv, seq=seq)          # (N, num_classes)
        # logits = logits.view(-1)        # No need to view(-1) for CrossEntropy
        # preds = torch.sigmoid(logits)   # No sigmoid for CrossEntropy
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Calculate Accuracy
            preds = torch.argmax(logits, dim=1)
            correct = (preds == y).sum().item()
            total = y.numel()

        running_loss += loss.item() * total
        running_corrects += correct
        running_total += total

        global_step += 1

        # tqdm status
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct/total:.4f}"
        })

        # wandb step logging
        if global_step % log_interval == 0:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/acc": correct/total,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }
            )

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = running_corrects / max(running_total, 1)

    wandb.log(
        {
            "train/epoch_loss": epoch_loss,
            "train/epoch_acc": epoch_acc,
            "train/epoch": epoch,
            "train/global_step": global_step,
        }
    )

    print(f"[Train] Epoch: {epoch} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
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

    all_probs = []
    all_labels = []

    for batch in loader:
        xi = batch["xi"].to(device=device, dtype=torch.long)
        xv = batch["xv"].to(device=device, dtype=torch.float)
        y = batch["label"].to(device=device, dtype=torch.long)

        seq = None
        if "seq" in batch:
             seq = batch["seq"].to(device=device, dtype=torch.long)

        logits = model(xi, xv, seq=seq)
        # logits = logits.view(-1)
        # preds = torch.sigmoid(logits)
        loss = criterion(logits, y)

        # Calculate Accuracy
        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()
        total = y.numel()

        running_loss += loss.item() * total
        running_correct += correct
        running_total += total

        all_probs.append(preds.detach().cpu().numpy())
        all_labels.append(y.view(-1).detach().cpu().numpy())

    ## Calculate metrics
    all_preds = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = running_correct / max(running_total, 1)

    # from scipy.stats import pearsonr
    # correlation, _ = pearsonr(all_preds, all_labels) if len(all_preds) > 1 else (0, 1)

    wandb.log(
        {
            f"{split}/loss": epoch_loss,
            f"{split}/acc": epoch_acc,
            # f"{split}/correlation": correlation,
            f"{split}/epoch": epoch,
            "global_step": global_step,
        }
    )

    print(f"[{split.capitalize()}] Epoch: {epoch} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    return {"loss": epoch_loss, "acc": epoch_acc}


def main():
    config_path = "config.json"
    with open(config_path, "r") as f:
        config: Dict[str, Any] = json.load(f)

    os.makedirs(config["save_dir"], exist_ok=True)

    set_seed(config["seed"])
    device = config["use_cuda"]
    print("Device:", device)

    wandb.init(
        project=config["project"],
        entity=config["entity"],
        name=config["run_name"],
        config=config,
    )

    # Suppose using preprocessed data
    train_df, val_df, test_df = split_exact_tvt_from_preprocessed(
        parquet_path=config["data_dir"] + '/' + config["data_name"],
    )

    base_sparse_dims, v2_sparse_dims = extract_dims(train_df)

    if config["data_loader"] == "v1":
        train_loader, val_loader, test_loader = create_v1_dataloaders(train_df, val_df, test_df, base_sparse_dims)
    elif config["data_loader"] == "v2":
        train_loader, val_loader, test_loader = create_v2_dataloaders(train_df, val_df, test_df, v2_sparse_dims)
    elif config["data_loader"] == "v3":
        train_loader, val_loader, test_loader = create_v3_dataloaders(train_df, val_df, test_df, base_sparse_dims)
    else:
        raise ValueError(f"Unknown data loader: {config['data_loader']}")

    # feature_sizes 추출 (DataLoader의 dataset에서 가져오기)
    seq_vocab_size = None
    if hasattr(train_loader.dataset, "field_dims"):
        feature_sizes = train_loader.dataset.field_dims
        print(f"Feature sizes: {feature_sizes}")
        if hasattr(train_loader.dataset, "seq_vocab_size"):
             seq_vocab_size = train_loader.dataset.seq_vocab_size
             print(f"Sequence Vocab Size: {seq_vocab_size}")
    else:
        raise ValueError("Dataset must have 'field_dims' attribute.")

    # DeepFM / AutoInt 선택은 build_model 안에서 config["model_type"] 보고 처리한다고 가정
    model = build_model(config, feature_sizes, device, seq_vocab_size=seq_vocab_size)

    criterion = nn.CrossEntropyLoss()  # Using CrossEntropy for multi-class (0, 1, 2)
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
    
    # Early stopping setup
    patience = config.get("early_stopping_patience", 10)
    patience_counter = 0
    print(f"Early stopping enabled with patience={patience}")

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

        print(f'DEBUG: val_loss={val_metrics["loss"]:.6f}, best_val_loss={best_val_loss:.6f}')

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0  # Reset counter when improvement occurs
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
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                print(f"\n⚠️ Early stopping triggered! No improvement for {patience} epochs.")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break


    print("\n" + "="*50)
    print("Training finished.")
    print("="*50)
    
    # Load best model and evaluate on test set
    best_ckpt_path = os.path.join(config["save_dir"], f"best_{config['model_type']}.pth")
    
    if os.path.exists(best_ckpt_path):
        print(f"\nLoading best model from {best_ckpt_path}")
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Best model from epoch {checkpoint['epoch']} loaded (val_loss={checkpoint['best_val_loss']:.4f})")
        
        print("\n" + "="*50)
        print("Evaluating on Test Set")
        print("="*50)
        
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            epoch=checkpoint['epoch'],
            global_step=checkpoint['global_step'],
            split="test",
        )
        
        print(f"\n{'='*50}")
        print(f"Final Test Results:")
        print(f"  - Test Loss: {test_metrics['loss']:.4f}")
        print(f"  - Test Acc: {test_metrics['acc']:.4f}")
        print(f"{'='*50}\n")
    else:
        print(f"\nWarning: Best model checkpoint not found at {best_ckpt_path}")
        print("Skipping test evaluation.")
    
    wandb.finish()


if __name__ == "__main__":
    main()
