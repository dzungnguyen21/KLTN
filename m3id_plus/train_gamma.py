import torch
import torch.nn as nn
import requests
from io import BytesIO
from gamma import LearnableGamma
from visual_encoder import HybridVisualEncoder
from attention_m3id import HybridAttentionM3ID
import json
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Giả sử framework M3ID của bạn được lưu trong file m3id.py
# from m3id import HybridAttentionM3ID 

def plot_loss_curve(step_losses, save_path="training_loss.png"):
    """
    Hàm vẽ biểu đồ loss.
    Vẽ loss theo từng bước (mờ) và đường trung bình trượt (rõ nét) để dễ quan sát xu hướng.
    """
    plt.figure(figsize=(12, 6))
    
    # 1. Vẽ loss gốc của từng bước (màu xanh nhạt)
    plt.plot(step_losses, label='Step Loss', color='blue', alpha=0.3, linewidth=1)
    
    # 2. Vẽ đường làm mượt (Moving Average) để nhìn rõ xu hướng
    window_size = 50
    if len(step_losses) >= window_size:
        # Tính trung bình trượt
        smoothed_losses = np.convolve(step_losses, np.ones(window_size)/window_size, mode='valid')
        # Căn chỉnh trục x cho khớp
        plt.plot(range(window_size-1, len(step_losses)), smoothed_losses, 
                 label=f'Trend (Moving Average window={window_size})', color='red', linewidth=2)
    
    plt.title('Training Loss of Learnable Gamma (M3ID Framework)', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Lưu ra file ảnh
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n=> Đã lưu biểu đồ loss tại: {save_path}")
    
    # Hiển thị biểu đồ ra màn hình (nếu chạy trên Jupyter Notebook/Colab)
    plt.show()

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("1. Loading Model...")
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation='eager' # Quan trọng để lấy được attentions
    )
    
    # BẮT BUỘC: Đóng băng toàn bộ LLM, chỉ train Gamma_net
    model.requires_grad_(False)
    model.eval()

    print("2. Initializing Framework...")
    framework = HybridAttentionM3ID(
        model=model,
        processor=processor,
        hidden_size=2048, # 2048 cho Qwen 3B
        num_regions=16,
        gamma_lr=1e-4,
        device=device
    )

    print("3. Loading RLHF-V Dataset...")
    dataset = load_dataset("openbmb/RLHF-V-Dataset", split="train")
    
    # Lấy 1000 mẫu để train nhanh nghiệm thu thuật toán
    train_data = dataset.select(range(100))
    
    epochs = 2
    
    # --- BIẾN LƯU TRỮ LOSS ĐỂ VẼ BIỂU ĐỒ ---
    history_step_losses = []
    
    print(f"4. Starting Training for {epochs} epochs...")
    framework.gamma_net.train() # Bật chế độ train cho mạng Gamma
    
    for epoch in range(epochs):
        total_loss = 0
        valid_steps = 0
        
        progress_bar = tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}")
        
        for item in progress_bar:
            try:
                image = item['image']
                
                # Bóc tách text_data
                text_data = item.get('text', item) # Đề phòng dataset format bị đổi
                if isinstance(text_data, str):
                    text_data = json.loads(text_data)
                
                prompt = text_data.get('question', '')
                chosen_response = text_data.get('chosen', '')
                rejected_response = text_data.get('rejected', '')
                
                if not prompt or not chosen_response or not rejected_response:
                    continue # Bỏ qua nếu data bị thiếu field
                
                # Contrastive step: push gamma_chosen > gamma_rejected
                step_loss = framework.train_step(
                    prompt=prompt,
                    image_path=image,
                    chosen_response=chosen_response,
                    rejected_response=rejected_response,
                )

                # Skip NaN steps (model emitted bad hidden states)
                import math
                if math.isnan(step_loss):
                    continue

                # LƯU LOSS VÀO LIST ĐỂ TÍ VẼ
                history_step_losses.append(step_loss)
                
                total_loss += step_loss
                valid_steps += 1
                
                progress_bar.set_postfix({"Avg Loss": f"{total_loss/valid_steps:.4f}"})
                
            except Exception as e:
                print(f"\n[Cảnh báo] Bỏ qua sample do lỗi: {str(e)}")
                continue
        
        epoch_avg = total_loss / max(1, valid_steps)
        print(f"\n=> End of Epoch {epoch+1} | Mean Epoch Loss: {epoch_avg:.4f}")
        
        # LƯU TRỌNG SỐ
        save_path = f"gamma_net_epoch_{epoch+1}.pt"
        torch.save(framework.gamma_net.state_dict(), save_path)
        print(f"=> Saved Checkpoint to {save_path}\n")

    # 5. KẾT THÚC HUẤN LUYỆN -> VẼ BIỂU ĐỒ
    print("5. Generating Loss Curve...")
    plot_loss_curve(history_step_losses, save_path="gamma_training_loss.png")

if __name__ == "__main__":
    train()