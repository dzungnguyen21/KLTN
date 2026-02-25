import torch
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from attention_m3id import HybridAttentionM3ID


def plot_loss_curve(train_losses, val_losses, save_path="phase1_training_loss.png"):
    """
    Plot per-epoch train and validation loss with best-epoch marker.
    """
    if not train_losses:
        print("[Warning] No losses to plot.")
        return

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss', color='steelblue', linewidth=2)
    if val_losses:
        plt.plot(epochs, val_losses, marker='s', label='Val Loss', color='tomato', linewidth=2)

        best_epoch = int(np.argmin(val_losses)) + 1
        plt.axvline(x=best_epoch, color='green', linestyle='--',
                    label=f'Best val (epoch {best_epoch})')
        plt.annotate(
            f'Min val: {min(val_losses):.4f}',
            xy=(best_epoch, min(val_losses)),
            xytext=(best_epoch + 0.3, min(val_losses) + 0.2),
            arrowprops=dict(arrowstyle='->', color='green'),
            color='green', fontsize=10
        )

    plt.title('Phase-1 Training \u2014 Loss per Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(list(epochs))
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n=> Loss curve saved to: {save_path}")
    plt.show()


def run_validation(framework, val_data):
    """Compute mean validation loss over val_data (no gradient updates)."""
    framework.model.eval()
    framework.hybrid_encoder.eval()

    total_loss  = 0.0
    valid_steps = 0

    pbar = tqdm(val_data, desc="  Validation", leave=False)
    for item in pbar:
        try:
            image = item['image']

            text_data = item.get('text', item)
            if isinstance(text_data, str):
                text_data = json.loads(text_data)

            prompt          = text_data.get('question', '')
            chosen_response = text_data.get('chosen', '')

            if not prompt or not chosen_response:
                continue

            step_loss = framework.eval_step_phase1(
                prompt=prompt,
                image=image,
                chosen_response=chosen_response,
            )

            if math.isnan(step_loss):
                continue

            total_loss  += step_loss
            valid_steps += 1
            pbar.set_postfix({"val_loss": f"{total_loss / valid_steps:.4f}"})

        except Exception as e:
            import traceback
            print(f"\n  [Val Warning] {e}")
            traceback.print_exc()
            continue

    # Restore train mode
    framework.model.train()
    framework.hybrid_encoder.train()

    return total_loss / max(1, valid_steps)


def train_phase1():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Load model ──────────────────────────────────────────
    print("1. Loading Model...")
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor  = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation='eager',
    )
    model.eval()

    # ── 2. Init framework ──────────────────────────────────────
    print("2. Initializing Framework...")
    framework = HybridAttentionM3ID(
        model=model,
        processor=processor,
        hidden_size=2048,
        num_regions=16,
        gamma_lr=1e-4,
        device=device,
    )

    # ── 3. Dataset \u2014 80 / 20 train-val split ──────────────────
    print("3. Loading Dataset...")
    dataset = load_dataset("openbmb/RLHF-V-Dataset", split="train")

    total_samples = len(dataset)        # adjust as needed
    val_ratio     = 0.2
    n_val         = max(1, int(total_samples * val_ratio))
    n_train       = total_samples - n_val

    full_data  = dataset.select(range(total_samples)).shuffle(seed=42)
    train_data = full_data.select(range(n_train))
    val_data   = full_data.select(range(n_train, total_samples))
    print(f"   Train: {n_train} samples  |  Val: {n_val} samples")

    # ── 4. Training config ─────────────────────────────────────
    epochs            = 20        # upper limit; early stopping cuts short
    patience          = 3         # epochs without improvement before stopping
    min_delta         = 1e-3      # minimum val-loss drop to count as improvement

    best_val_loss     = float('inf')
    epochs_no_improve = 0
    train_loss_history = []
    val_loss_history   = []

    print(f"4. Starting Phase-1 Training (max {epochs} epochs, patience={patience})...")
    model.train()
    framework.hybrid_encoder.train()

    for epoch in range(epochs):

        # ── 4a. Train ──────────────────────────────────────────
        total_loss  = 0.0
        valid_steps = 0

        pbar = tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs} [train]")
        for item in pbar:
            try:
                image = item['image']

                text_data = item.get('text', item)
                if isinstance(text_data, str):
                    text_data = json.loads(text_data)

                prompt          = text_data.get('question', '')
                chosen_response = text_data.get('chosen', '')

                if not prompt or not chosen_response:
                    continue

                step_loss = framework.train_step_phase1(
                    prompt=prompt,
                    image=image,
                    chosen_response=chosen_response,
                )

                if math.isnan(step_loss):
                    continue

                total_loss  += step_loss
                valid_steps += 1
                pbar.set_postfix({
                    "step_loss": f"{step_loss:.4f}",
                    "avg":       f"{total_loss / valid_steps:.4f}",
                })

            except Exception as e:
                import traceback
                print(f"\n[Warning] Skipping sample: {e}")
                traceback.print_exc()
                continue

        train_avg = total_loss / max(1, valid_steps)
        train_loss_history.append(train_avg)

        # ── 4b. Validation ─────────────────────────────────────
        val_avg = run_validation(framework, val_data)
        val_loss_history.append(val_avg)

        print(f"\n=> Epoch {epoch+1:02d} | "
              f"Train Loss: {train_avg:.4f} | "
              f"Val Loss: {val_avg:.4f} | "
              f"Valid train steps: {valid_steps}")

        # ── 4c. Best checkpoint ────────────────────────────────
        if val_avg < best_val_loss - min_delta:
            best_val_loss     = val_avg
            epochs_no_improve = 0
            torch.save(framework.hybrid_encoder.state_dict(), "best_hybrid_encoder.pt")
            print(f"   \u2713 New best val loss {best_val_loss:.4f} \u2014 saved best_hybrid_encoder.pt")
        else:
            epochs_no_improve += 1
            print(f"   No improvement for {epochs_no_improve}/{patience} epochs.")

        # Regular per-epoch checkpoint
        torch.save(framework.hybrid_encoder.state_dict(),
                   f"hybrid_encoder_epoch_{epoch+1:02d}.pt")

        # ── 4d. Early stopping ─────────────────────────────────
        if epochs_no_improve >= patience:
            print(f"\n[Early Stop] Val loss did not improve for {patience} consecutive epochs.")
            print(f"             Best val loss: {best_val_loss:.4f}")
            break

    # ── 5. Plot ────────────────────────────────────────────────
    print("\n5. Plotting loss curves...")
    plot_loss_curve(train_loss_history, val_loss_history,
                    save_path="phase1_training_loss.png")


if __name__ == "__main__":
    train_phase1()