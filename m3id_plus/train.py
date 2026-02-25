"""
M3ID+ Unified Training Entry Point
===================================
Usage:
    python train.py --config config.yml

Edit config.yml to switch between phases, adjust learning rates,
number of samples, epochs, etc.
"""

import argparse
import json
import math
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from attention_m3id import HybridAttentionM3ID


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(
    train_losses: list,
    val_losses: list,
    title: str,
    ylabel: str,
    save_path: str,
) -> None:
    if not train_losses:
        print("[Warning] No losses to plot.")
        return

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss",
             color="steelblue", linewidth=2)

    if val_losses:
        plt.plot(epochs, val_losses, marker="s", label="Val Loss",
                 color="tomato", linewidth=2)
        best_epoch = int(np.argmin(val_losses)) + 1
        best_val   = min(val_losses)
        plt.axvline(x=best_epoch, color="green", linestyle="--",
                    label=f"Best val (epoch {best_epoch})")
        plt.annotate(
            f"Min val: {best_val:.4f}",
            xy=(best_epoch, best_val),
            xytext=(best_epoch + 0.3, best_val + 0.2),
            arrowprops=dict(arrowstyle="->", color="green"),
            color="green",
            fontsize=10,
        )

    plt.title(title, fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(list(epochs))
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n=> Loss curve saved to: {save_path}")
    plt.show()


def _parse_item(item: dict) -> tuple[str, str, str | None]:
    """Extract (prompt, chosen_response, rejected_response|None) from a dataset item."""
    text_data = item.get("text", item)
    if isinstance(text_data, str):
        text_data = json.loads(text_data)
    prompt    = text_data.get("question", "")
    chosen    = text_data.get("chosen", "")
    rejected  = text_data.get("rejected", None)
    return prompt, chosen, rejected


def build_dataset(cfg: dict):
    """Load, slice, shuffle and split the dataset according to config."""
    ds_cfg   = cfg["dataset"]
    tr_cfg   = cfg["training"]

    dataset  = load_dataset(ds_cfg["name"], split=ds_cfg["split"])
    n_total  = len(dataset) if ds_cfg["total_samples"] == -1 else min(ds_cfg["total_samples"], len(dataset))
    n_val    = max(1, int(n_total * tr_cfg["val_ratio"]))
    n_train  = n_total - n_val

    full_data  = dataset.select(range(n_total)).shuffle(seed=tr_cfg["seed"])
    train_data = full_data.select(range(n_train))
    val_data   = full_data.select(range(n_train, n_total))

    print(f"   Dataset : {ds_cfg['name']} [{ds_cfg['split']}]")
    print(f"   Total   : {n_total}  |  Train : {n_train}  |  Val : {n_val}")
    return train_data, val_data


def build_framework(cfg: dict, device: str) -> HybridAttentionM3ID:
    """Instantiate model + processor + framework from config."""
    m_cfg   = cfg["model"]
    lr_cfg  = cfg["lr"]

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    dtype = dtype_map[m_cfg["dtype"]]

    print("   Loading model weights...")
    processor = AutoProcessor.from_pretrained(m_cfg["name"])
    model     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        m_cfg["name"],
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=m_cfg["attn_implementation"],
    )

    lora_cfg = cfg["lora"]

    print("   Building framework...")
    framework = HybridAttentionM3ID(
        model=model,
        processor=processor,
        hidden_size=m_cfg["hidden_size"],
        num_regions=m_cfg["num_regions"],
        lora_r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        lora_target_modules=lora_cfg["target_modules"],
        lr_hybrid_encoder=lr_cfg["hybrid_encoder"],
        lr_projector=lr_cfg["projector"],
        lr_lora=lr_cfg["lora"],
        device=device,
    )
    return framework


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — validation
# ─────────────────────────────────────────────────────────────────────────────

def run_validation_phase1(framework: HybridAttentionM3ID, val_data) -> float:
    framework.model.eval()
    framework.hybrid_encoder.eval()

    total_loss  = 0.0
    valid_steps = 0

    pbar = tqdm(val_data, desc="  Validation", leave=False)
    for item in pbar:
        try:
            image = item["image"]
            prompt, chosen, _ = _parse_item(item)
            if not prompt or not chosen:
                continue

            step_loss = framework.eval_step_phase1(
                prompt=prompt, image=image, chosen_response=chosen
            )
            if math.isnan(step_loss):
                continue

            total_loss  += step_loss
            valid_steps += 1
            pbar.set_postfix({"val_loss": f"{total_loss / valid_steps:.4f}"})

        except Exception as exc:
            print(f"\n  [Val Warning] {exc}")
            traceback.print_exc()

    framework.model.train()
    framework.hybrid_encoder.train()
    return total_loss / max(1, valid_steps)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation_phase2(framework: HybridAttentionM3ID, val_data, margin: float) -> float:
    framework.model.eval()
    framework.hybrid_encoder.eval()
    framework.gamma_net.eval()

    total_loss  = 0.0
    valid_steps = 0

    pbar = tqdm(val_data, desc="  Validation", leave=False)
    for item in pbar:
        try:
            image = item["image"]
            prompt, chosen, rejected = _parse_item(item)
            if not prompt or not chosen or not rejected:
                continue

            image_pil = framework.load_image(image)

            def get_avg_gamma(response_text: str):
                full_text = prompt + " " + response_text
                inputs    = framework._prepare_inputs_with_image(full_text, image_pil)
                inputs    = framework._move_inputs_to_device(inputs, framework.device)

                prompt_inputs = framework._prepare_inputs_with_image(prompt, image_pil)
                prompt_len    = prompt_inputs["input_ids"].shape[1]

                if inputs["input_ids"].shape[1] <= prompt_len:
                    return None

                outputs = framework.model(**inputs, output_hidden_states=True, use_cache=False)
                full_hidden = outputs.hidden_states[-1]
                if torch.isnan(full_hidden).any():
                    return None

                response_hidden = full_hidden[:, prompt_len:, :]
                h_f32  = response_hidden.squeeze(0).to(dtype=torch.float32)
                gammas = framework.gamma_net(h_f32)
                return gammas.mean()

            gamma_chosen   = get_avg_gamma(chosen)
            gamma_rejected = get_avg_gamma(rejected)
            if gamma_chosen is None or gamma_rejected is None:
                continue

            loss_val = torch.clamp(
                torch.tensor(margin, dtype=torch.float32) - (gamma_chosen - gamma_rejected),
                min=0.0,
            ).item()

            if math.isnan(loss_val):
                continue

            total_loss  += loss_val
            valid_steps += 1
            pbar.set_postfix({"val_loss": f"{total_loss / valid_steps:.4f}"})

        except Exception as exc:
            print(f"\n  [Val Warning] {exc}")
            traceback.print_exc()

    framework.model.train()
    framework.hybrid_encoder.train()
    framework.gamma_net.train()
    return total_loss / max(1, valid_steps)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — train LLM (LoRA) + HybridVisualEncoder
# ─────────────────────────────────────────────────────────────────────────────

def train_phase1(cfg: dict) -> None:
    tr_cfg  = cfg["training"]
    out_cfg = cfg["output"]
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_cfg["dir"], exist_ok=True)

    print("\n=== Phase 1: LLM (LoRA) + HybridVisualEncoder ===")

    # ── 1. Build framework ────────────────────────────────────
    print("1. Loading model & building framework...")
    framework = build_framework(cfg, device)

    # ── 2. Dataset ────────────────────────────────────────────
    print("2. Loading dataset...")
    train_data, val_data = build_dataset(cfg)

    # ── 3. Training loop ──────────────────────────────────────
    max_epochs        = tr_cfg["epochs"]
    patience          = tr_cfg["patience"]
    min_delta         = tr_cfg["min_delta"]
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    train_history     = []
    val_history       = []

    print(f"3. Training (max {max_epochs} epochs, patience={patience})...")
    framework.model.train()
    framework.hybrid_encoder.train()

    for epoch in range(max_epochs):
        total_loss  = 0.0
        valid_steps = 0

        pbar = tqdm(train_data, desc=f"Epoch {epoch+1}/{max_epochs} [train]")
        for item in pbar:
            try:
                image = item["image"]
                prompt, chosen, _ = _parse_item(item)
                if not prompt or not chosen:
                    continue

                step_loss = framework.train_step_phase1(
                    prompt=prompt, image=image, chosen_response=chosen
                )
                if math.isnan(step_loss):
                    continue

                total_loss  += step_loss
                valid_steps += 1
                pbar.set_postfix({
                    "step": f"{step_loss:.4f}",
                    "avg":  f"{total_loss / valid_steps:.4f}",
                })

            except Exception as exc:
                print(f"\n[Warning] Skipping sample: {exc}")
                traceback.print_exc()

        train_avg = total_loss / max(1, valid_steps)
        train_history.append(train_avg)

        val_avg = run_validation_phase1(framework, val_data)
        val_history.append(val_avg)

        print(f"\n=> Epoch {epoch+1:02d} | Train: {train_avg:.4f} | "
              f"Val: {val_avg:.4f} | Steps: {valid_steps}")

        # ── Checkpoint ────────────────────────────────────────
        ckpt_prefix = os.path.join(out_cfg["dir"], "hybrid_encoder")
        torch.save(framework.hybrid_encoder.state_dict(),
                   f"{ckpt_prefix}_epoch_{epoch+1:02d}.pt")

        if val_avg < best_val_loss - min_delta:
            best_val_loss     = val_avg
            epochs_no_improve = 0
            best_path = os.path.join(out_cfg["dir"], "best_hybrid_encoder.pt")
            torch.save(framework.hybrid_encoder.state_dict(), best_path)
            print(f"   ✓ New best val {best_val_loss:.4f} — saved {best_path}")
        else:
            epochs_no_improve += 1
            print(f"   No improvement {epochs_no_improve}/{patience}")

        # ── Early stopping ────────────────────────────────────
        if epochs_no_improve >= patience:
            print(f"\n[Early Stop] Best val loss: {best_val_loss:.4f}")
            break

    plot_loss_curve(
        train_history, val_history,
        title="Phase-1 Training — Loss per Epoch",
        ylabel="Loss",
        save_path=os.path.join(out_cfg["dir"], "phase1_training_loss.png"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — train gamma_net with contrastive margin loss
# ─────────────────────────────────────────────────────────────────────────────

def train_phase2(cfg: dict) -> None:
    tr_cfg  = cfg["training"]
    p2_cfg  = cfg["phase2"]
    out_cfg = cfg["output"]
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_cfg["dir"], exist_ok=True)

    print("\n=== Phase 2: gamma_net (contrastive margin loss) ===")

    # ── 1. Build framework ────────────────────────────────────
    print("1. Loading model & building framework...")
    framework = build_framework(cfg, device)

    # ── 2. Load Phase-1 checkpoint ────────────────────────────
    ckpt_path = p2_cfg["checkpoint"]
    print(f"2. Loading Phase-1 checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    framework.hybrid_encoder.load_state_dict(state_dict)
    print("   ✓ hybrid_encoder weights restored.")

    # ── 3. Freeze LLM + encoder; unfreeze gamma_net ───────────
    for p in framework.model.parameters():
        p.requires_grad_(False)
    for p in framework.hybrid_encoder.parameters():
        p.requires_grad_(False)
    for p in framework.gamma_net.parameters():
        p.requires_grad_(True)

    framework.optimizer = torch.optim.AdamW(
        framework.gamma_net.parameters(), lr=cfg["lr"]["gamma"]
    )
    print("   ✓ Optimizer reset for gamma_net only.")

    # ── 4. Dataset ────────────────────────────────────────────
    print("3. Loading dataset...")
    train_data, val_data = build_dataset(cfg)

    # ── 5. Training loop ──────────────────────────────────────
    max_epochs        = tr_cfg["epochs"]
    patience          = tr_cfg["patience"]
    min_delta         = tr_cfg["min_delta"]
    margin            = p2_cfg["margin"]
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    train_history     = []
    val_history       = []

    print(f"4. Training (max {max_epochs} epochs, patience={patience}, margin={margin})...")
    framework.model.eval()           # LLM frozen — keep in eval
    framework.hybrid_encoder.eval()
    framework.gamma_net.train()

    for epoch in range(max_epochs):
        total_loss  = 0.0
        valid_steps = 0

        pbar = tqdm(train_data, desc=f"Epoch {epoch+1}/{max_epochs} [train]")
        for item in pbar:
            try:
                image = item["image"]
                prompt, chosen, rejected = _parse_item(item)
                if not prompt or not chosen or not rejected:
                    continue

                step_loss = framework.train_step(
                    prompt=prompt,
                    image_path=image,
                    chosen_response=chosen,
                    rejected_response=rejected,
                    margin=margin,
                )
                if math.isnan(step_loss):
                    continue

                total_loss  += step_loss
                valid_steps += 1
                pbar.set_postfix({
                    "step": f"{step_loss:.4f}",
                    "avg":  f"{total_loss / valid_steps:.4f}",
                })

            except Exception as exc:
                print(f"\n[Warning] Skipping sample: {exc}")
                traceback.print_exc()

        train_avg = total_loss / max(1, valid_steps)
        train_history.append(train_avg)

        val_avg = run_validation_phase2(framework, val_data, margin)
        val_history.append(val_avg)

        print(f"\n=> Epoch {epoch+1:02d} | Train: {train_avg:.4f} | "
              f"Val: {val_avg:.4f} | Steps: {valid_steps}")

        # ── Checkpoint ────────────────────────────────────────
        ckpt_prefix = os.path.join(out_cfg["dir"], "gamma_net")
        torch.save(framework.gamma_net.state_dict(),
                   f"{ckpt_prefix}_epoch_{epoch+1:02d}.pt")

        if val_avg < best_val_loss - min_delta:
            best_val_loss     = val_avg
            epochs_no_improve = 0
            best_path = os.path.join(out_cfg["dir"], "gamma_net_best.pt")
            torch.save(framework.gamma_net.state_dict(), best_path)
            print(f"   ✓ New best val {best_val_loss:.4f} — saved {best_path}")
        else:
            epochs_no_improve += 1
            print(f"   No improvement {epochs_no_improve}/{patience}")

        # ── Early stopping ────────────────────────────────────
        if epochs_no_improve >= patience:
            print(f"\n[Early Stop] Best val loss: {best_val_loss:.4f}")
            break

    plot_loss_curve(
        train_history, val_history,
        title="Phase-2 Training — Loss per Epoch",
        ylabel="Contrastive Margin Loss",
        save_path=os.path.join(out_cfg["dir"], "phase2_training_loss.png"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="M3ID+ Training",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the YAML config file (e.g. config.yml)",
    )
    args = parser.parse_args()

    cfg   = load_config(args.config)
    phase = cfg["training"]["phase"]

    print(f"Loaded config: {args.config}")
    print(f"Training phase: {phase}")

    if phase == 1:
        train_phase1(cfg)
    elif phase == 2:
        train_phase2(cfg)
    else:
        raise ValueError(f"Unknown training phase '{phase}'. Must be 1 or 2.")


if __name__ == "__main__":
    main()
