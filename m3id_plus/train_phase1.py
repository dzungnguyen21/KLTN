import torch
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from attention_m3id import HybridAttentionM3ID


def plot_loss_curve(step_losses, save_path="training_loss_phase1.png"):
    """
    Vẽ biểu đồ loss theo từng bước + moving average trend.
    """
    if len(step_losses) == 0:
        print("[Warning] No losses to plot.")
        return

    plt.figure(figsize=(12, 6))

    # 1. Raw step losses (mờ)
    plt.plot(step_losses, label='Step Loss', color='blue', alpha=0.3, linewidth=1)

    # 2. Moving average (rõ nét)
    window_size = min(50, len(step_losses) // 2)
    if len(step_losses) >= window_size and window_size > 1:
        smoothed = np.convolve(
            step_losses,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        plt.plot(
            range(window_size - 1, len(step_losses)),
            smoothed,
            label=f'Trend (MA window={window_size})',
            color='red',
            linewidth=2
        )

    # 3. Min loss annotation
    min_loss = min(step_losses)
    min_idx  = step_losses.index(min_loss)
    plt.annotate(
        f'Min: {min_loss:.4f}',
        xy=(min_idx, min_loss),
        xytext=(min_idx + len(step_losses) * 0.02, min_loss + 0.05),
        arrowprops=dict(arrowstyle='->', color='green'),
        color='green', fontsize=10
    )

    plt.title('Phase-1 Training Loss (LM + HybridEncoder)', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n=> Loss curve saved to: {save_path}")
    plt.show()


def train_phase1():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Load model ──────────────────────────────────────────
    print("1. Loading Model...")
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation='eager',
    )
    model.eval()  # Initial eval; will switch to train() before the loop

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

    # ── 3. Dataset ─────────────────────────────────────────────
    print("3. Loading Dataset...")
    dataset   = load_dataset("openbmb/RLHF-V-Dataset", split="train")
    train_data = dataset.select(range(100))

    # ── 4. Training loop ───────────────────────────────────────
    epochs = 3
    history_step_losses = []   # ← tất cả step losses để plot

    print(f"4. Starting Phase-1 Training ({epochs} epochs)...")
    model.train()                        # ← switch to train mode before the loop
    framework.hybrid_encoder.train()

    for epoch in range(epochs):
        total_loss  = 0.0
        valid_steps = 0

        pbar = tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}")

        for item in pbar:
            try:
                image = item['image']

                # Parse text fields
                text_data = item.get('text', item)
                if isinstance(text_data, str):
                    text_data = json.loads(text_data)

                prompt = text_data.get('question', '')
                chosen_response = text_data.get('chosen', '')

                if not prompt or not chosen_response:
                    continue

                # ── forward + backward ──
                step_loss = framework.train_step_phase1(
                    prompt=prompt,
                    image=image,
                    chosen_response=chosen_response,
                )

                if math.isnan(step_loss):
                    continue

                # ── record ──
                history_step_losses.append(step_loss)
                total_loss  += step_loss
                valid_steps += 1

                pbar.set_postfix({
                    "step_loss": f"{step_loss:.4f}",
                    "avg_loss":  f"{total_loss / valid_steps:.4f}",
                })

            except Exception as e:
                print(f"\n[Warning] Skipping sample: {e}")
                continue

        epoch_avg = total_loss / max(1, valid_steps)
        print(f"\n=> Epoch {epoch+1} finished | Mean Loss: {epoch_avg:.4f} "
              f"| Valid steps: {valid_steps}")

        # Save checkpoint
        ckpt_path = f"hybrid_encoder_epoch_{epoch+1}.pt"
        torch.save(framework.hybrid_encoder.state_dict(), ckpt_path)
        print(f"=> Checkpoint saved: {ckpt_path}\n")

    # ── 5. Plot ────────────────────────────────────────────────
    print("5. Plotting loss curve...")
    plot_loss_curve(history_step_losses, save_path="phase1_training_loss.png")


if __name__ == "__main__":
    train_phase1()