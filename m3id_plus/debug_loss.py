import torch
import json
import traceback
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from attention_m3id import HybridAttentionM3ID

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)
framework = HybridAttentionM3ID(
    model=model, processor=processor,
    hidden_size=2048, num_regions=16, gamma_lr=1e-4, device=device
)
model.train()
framework.hybrid_encoder.train()

dataset = load_dataset("openbmb/RLHF-V-Dataset", split="train")
item = dataset[0]
image = item["image"]
text_data = item.get("text", item)
if isinstance(text_data, str):
    text_data = json.loads(text_data)

prompt = text_data.get("question", "")
chosen = text_data.get("chosen", "")
print(f"prompt  : {repr(prompt[:100])}")
print(f"chosen  : {repr(chosen[:100])}")
print(f"img size: {image.size}")

try:
    # Manual diagnostic
    image_pil = framework.load_image(image)
    full_messages = [
        {"role": "user", "content": [{"type": "image", "image": image_pil}, {"type": "text", "text": prompt}]},
        {"role": "assistant", "content": chosen},
    ]
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[full_text], images=[image_pil], return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    print(f"input_ids dtype : {inputs['input_ids'].dtype}")
    print(f"pixel_values dtype: {inputs.get('pixel_values', torch.tensor(0)).dtype}")
    print(f"pixel_values shape: {inputs.get('pixel_values', torch.tensor(0)).shape}")

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=False, use_cache=False)
    logits = out.logits
    print(f"logits dtype : {logits.dtype}")
    print(f"logits shape : {logits.shape}")
    print(f"logits has NaN: {torch.isnan(logits).any().item()}")
    print(f"logits has Inf: {torch.isinf(logits).any().item()}")
    print(f"logits min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")

    loss = framework.train_step_phase1(prompt=prompt, image=image, chosen_response=chosen)
    print(f"\n✓ LOSS = {loss}")
except Exception as e:
    print(f"\n✗ EXCEPTION: {e}")
    traceback.print_exc()
