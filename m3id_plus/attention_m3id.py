import torch
import torch.nn as nn
import requests
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import Tuple
from gamma import LearnableGamma
from visual_encoder import HybridVisualEncoder
from peft import LoraConfig, get_peft_model

class HybridAttentionM3ID:
    """
    Full framework:
    1. Hybrid Patch + Region Encoder → richer visual representation
    2. Attention-aware scaling (αₜ) → correction chỉ khi attend image
    3. Learnable γₜ → adaptive visual trust

    Decode formula:
        l̂ₜ = lc + αₜ · ((1-γₜ)/γₜ) · (lc - lu)

    So sánh với M3ID gốc:
        l̂ₜ = lc + [indicator] · ((1-exp(-λt))/exp(-λt)) · (lc - lu)

    Thay thế:
    - exp(-λt) heuristic → γₜ = σ(W hₜ) learnable
    - binary indicator → αₜ = attention mass (continuous)
    """

    def __init__(
        self,
        model: Qwen2_5_VLForConditionalGeneration,
        processor: AutoProcessor,
        hidden_size: int = 2048,       # Qwen2VL-7B hidden size is 3584 and 3B is 2048
        num_regions: int = 16,          # số region tokens
        gamma_lr: float = 1e-4,         # learning rate cho γ network
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.hidden_size = hidden_size

        # === Learnable components ===
        # Keep in float32 to avoid float16 overflow/NaN during loss.backward()
        self.gamma_net = LearnableGamma(hidden_size).to(device=device, dtype=torch.float32)
        self.hybrid_encoder = HybridVisualEncoder(hidden_size, num_regions).to(device=device, dtype=torch.float32)

        # # Optimizer chỉ cho các learnable components (không train lại LLM)
        # self.optimizer = torch.optim.Adam(
        #     list(self.gamma_net.parameters()) +
        #     list(self.hybrid_encoder.parameters()),
        #     lr=gamma_lr
        # )

        # Attach LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )

        self.model.language_model = get_peft_model(
            self.model.language_model,
            lora_config
        )

        # Force trainable LoRA params onto CUDA so backward doesn't fail.
        # device_map="auto" can offload base layers to CPU; LoRA adapters for
        # those layers land on CPU too. Meta-device placeholders cannot be trained.
        for _name, param in self.model.language_model.named_parameters():
            if param.requires_grad:
                if param.device.type == 'cpu':
                    param.data = param.data.to(device)   # move CPU → GPU
                elif param.device.type == 'meta':
                    param.requires_grad_(False)           # meta tensor: skip

        # Freeze vision encoder
        for p in self.model.visual.parameters():
            p.requires_grad = False

        # Unfreeze projector
        for p in self.model.visual.merger.parameters():
            p.requires_grad = True

        # Freeze gamma in phase 1
        for p in self.gamma_net.parameters():
            p.requires_grad = False

        # Optimizer Phase 1
        # self.model.visual       → Qwen2.5-VL vision transformer
        # self.model.language_model → the LLM (with LoRA already applied)
        self.optimizer = torch.optim.AdamW([
            {"params": self.hybrid_encoder.parameters(), "lr": 1e-4},
            {"params": [p for p in self.model.visual.merger.parameters()
                        if p.requires_grad and p.device.type not in ('meta', 'cpu')],
             "lr": 5e-5},
            {"params": [p for p in self.model.language_model.parameters()
                        if p.requires_grad and p.device.type not in ('meta', 'cpu')],
             "lr": 2e-5},
        ])

    # ----------------------------------------------------------
    # Image loading
    # ----------------------------------------------------------
    def load_image(self, image_source, max_size=512) -> Image.Image:
        if isinstance(image_source, Image.Image):
            image = image_source.convert("RGB")
        elif isinstance(image_source, str) and image_source.startswith(('http://', 'https://')):
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(image_source, headers=headers, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_source).convert("RGB")
            
        # --- THÊM ĐOẠN NÀY ĐỂ CHỐNG OOM ---
        # Tự động thu nhỏ ảnh nếu quá lớn
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
        return image

    # ----------------------------------------------------------
    # Input preparation
    # ----------------------------------------------------------
    def _prepare_inputs_with_image(self, prompt: str, image: Image.Image) -> dict:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.processor(text=[text], images=[image], return_tensors="pt", padding=True)

    def _move_inputs_to_device(self, inputs: dict, device: str) -> dict:
        """Move tensor values in inputs dict to device, leave non-tensors as-is."""
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

    def _prepare_inputs_without_image(self, prompt: str) -> dict:
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.processor(text=[text], images=None, return_tensors="pt", padding=True)

    # ----------------------------------------------------------
    # COMPONENT: Tính attention mass αₜ
    # αₜ = tổng attention weight từ text token hiện tại → image tokens
    # ----------------------------------------------------------
    def _compute_attention_mass(
        self,
        attentions: Tuple,              # tuple of [batch, heads, seq, seq]
        num_image_tokens: int,
        layer_idx: int = -1             # dùng layer cuối (most semantic)
    ) -> torch.Tensor:
        """
        Tính αₜ = attention mass từ token cuối → image tokens.

        Dùng layer attention cuối cùng vì nó capture
        semantic-level dependencies (không phải syntactic).

        Returns:
            alpha_t: scalar tensor ∈ [0, 1]
        """
        if attentions is None:
            # Fallback: không có attention → assume moderate attention
            return torch.tensor(0.5, device=self.device)

        # Lọc ra các layer có attention weights thực sự (không phải None)
        valid_attentions = [a for a in attentions if a is not None]

        # Fallback 2: tất cả layers đều là None
        if len(valid_attentions) == 0:
            return torch.tensor(0.5, device=self.device)

        # Lấy layer hợp lệ theo layer_idx
        # Nếu layer_idx=-1 → lấy layer cuối trong danh sách hợp lệ
        attn_layer = valid_attentions[layer_idx]  # [batch, heads, seq_len, seq_len]

        # Attention từ token cuối (vị trí -1) đến tất cả positions
        # Average across heads
        attn_last_token = attn_layer[0, :, -1, :]   # [heads, seq_len]
        attn_avg = attn_last_token.mean(dim=0)       # [seq_len]

        # Image tokens nằm ở đầu sequence
        image_start = 1
        image_end = min(image_start + num_image_tokens, attn_avg.shape[0])

        # αₜ = tổng attention mass đến image tokens
        alpha_t = attn_avg[image_start:image_end].mean()

        return alpha_t.clamp(0.0, 1.0)

    # ----------------------------------------------------------
    # GENERATE: Main decode loop
    # ----------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        image_path: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        verbose: bool = True
    ) -> str:
        """
        Decode với full framework:

        Mỗi bước t:
        1. lc = log p(yₜ | x, image)
        2. lu = log p(yₜ | x)
        3. αₜ = attention mass → image tokens
        4. γₜ = σ(W hₜ) từ hidden state
        5. l̂ₜ = lc + αₜ · ((1-γₜ)/γₜ) · (lc - lu)
        6. Sample yₜ ~ softmax(l̂ₜ)
        """
        image = self.load_image(image_path)

        # Prepare inputs
        inputs_c = self._prepare_inputs_with_image(prompt, image)
        inputs_u = self._prepare_inputs_without_image(prompt)
        inputs_c = {k: v.to(self.device) for k, v in inputs_c.items()}
        inputs_u = {k: v.to(self.device) for k, v in inputs_u.items()}

        # === PREFILL: lần đầu chạy cả sequence, khởi tạo KV cache ===
        # output_attentions=True để lấy αₜ
        outputs_c = self.model(
            **inputs_c,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,   # cần hₜ cho γₜ
        )
        past_kv_c = outputs_c.past_key_values
        logits_c   = outputs_c.logits[:, -1, :]

        # Hidden state của layer cuối, token cuối → dùng cho γₜ
        # hidden_states: tuple of [batch, seq, hidden], lấy layer cuối
        h_t = outputs_c.hidden_states[-1][:, -1, :]  # [1, hidden_size]

        # Số image tokens trong conditioned input
        # Qwen2VL trả về image_grid_thw để tính số patch tokens
        num_image_tokens = 0
        if hasattr(outputs_c, 'image_grid_thw') or 'image_grid_thw' in inputs_c:
            grid = inputs_c.get('image_grid_thw', None)
            if grid is not None:
                # num_patches = T * H * W (với T=1 cho ảnh tĩnh)
                num_image_tokens = int(grid[0].prod().item())
        # Fallback estimate
        if num_image_tokens == 0:
            num_image_tokens = 256  # typical for 448px image

        # Tính αₜ từ prefill attention
        alpha_attention = self._compute_attention_mass(
            outputs_c.attentions,
            num_image_tokens
        )

        # Unconditioned prefill (không cần attention/hidden)
        outputs_u = self.model(**inputs_u, use_cache=True)
        past_kv_u = outputs_u.past_key_values
        logits_u  = outputs_u.logits[:, -1, :]

        seq_len_c = inputs_c['input_ids'].shape[1]
        seq_len_u = inputs_u['input_ids'].shape[1]

        inv_temp = 1.0 / temperature
        eos_token_id = self.processor.tokenizer.eos_token_id
        generated_ids = []

        if verbose:
            print(f"\n{'t':<4} {'α_attn':<10} {'γt':<8} {'w=(1-γ)/γ':<12} {'Token'}")
            print("-" * 60)

        # === DECODE LOOP ===
        for t in range(1, max_new_tokens + 1):

            # --- Step 4: Tính γₜ từ hidden state ---
            # Dùng gamma_net (có thể fine-tune sau)
            # inference_mode: tạm thời enable grad chỉ cho gamma_net
            with torch.enable_grad():
                # Always cast to float32 to avoid float16 NaN in gamma_net
                h_t_input = h_t.detach().to(dtype=torch.float32)
                gamma_t = self.gamma_net(h_t_input)  # [1, 1]
            gamma_t = gamma_t.detach().squeeze()         # scalar

            # --- Step 2-3: Log-probs ---
            lc = torch.log_softmax(logits_c * inv_temp, dim=-1)  # [1, vocab]
            lu = torch.log_softmax(logits_u * inv_temp, dim=-1)  # [1, vocab]

            # --- Step 5: Attention-aware M3ID formula ---
            # l̂ₜ = lc + αₜ · ((1-γₜ)/γₜ) · (lc - lu)
            # αₜ: continuous attention mass (thay cho binary indicator)
            # (1-γₜ)/γₜ: learnable correction weight (thay cho heuristic)

            eps = 1e-4  # 1e-6 underflows in float16; use 1e-4 which is safely representable
            correction_weight = ((1.0 - gamma_t) / (gamma_t + eps)).clamp(0.0, 5.0)

            # αₜ đóng vai trò gate: chỉ correct khi thực sự attend image
            l_star = lc + alpha_attention * correction_weight * (lc - lu)
            # [1, vocab]

            # --- Step 6: Top-p sampling ---
            probs = torch.softmax(l_star, dim=-1)

            # Nucleus sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)

            mask = cum_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False

            remove_mask = torch.zeros_like(probs, dtype=torch.bool)
            remove_mask.scatter_(-1, sorted_idx, mask)
            probs[remove_mask] = 0.0

            prob_sum = probs.sum(dim=-1, keepdim=True).clamp(min=eps)
            probs = probs / prob_sum

            next_token_id = torch.multinomial(probs, num_samples=1)  # [1, 1]

            if next_token_id.item() == eos_token_id:
                break

            generated_ids.append(next_token_id.item())

            if verbose:
                token_str = self.processor.tokenizer.decode([next_token_id.item()])
                w_val = correction_weight.item()
                print(f"{t:<4} {alpha_attention.item():<10.4f} {gamma_t.item():<8.4f} {w_val:<12.4f} {repr(token_str)}")

            # === DECODE STEP: feed 1 token mới với KV cache ===
            next_token_tensor = next_token_id.view(1, 1)

            # Conditioned: cần attention + hidden state cho bước sau
            seq_len_c += 1
            out_c = self.model(
                input_ids=next_token_tensor,
                attention_mask=torch.ones((1, seq_len_c), dtype=torch.long, device=self.device),
                past_key_values=past_kv_c,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True,
            )
            logits_c  = out_c.logits[:, -1, :]
            past_kv_c = out_c.past_key_values

            # Cập nhật hₜ và αₜ cho bước tiếp theo
            h_t = out_c.hidden_states[-1][:, -1, :]
            alpha_attention = self._compute_attention_mass(
                out_c.attentions,
                num_image_tokens
            )

            # Unconditioned (không cần hidden/attention)
            seq_len_u += 1
            out_u = self.model(
                input_ids=next_token_tensor,
                attention_mask=torch.ones((1, seq_len_u), dtype=torch.long, device=self.device),
                past_key_values=past_kv_u,
                use_cache=True,
            )
            logits_u  = out_u.logits[:, -1, :]
            past_kv_u = out_u.past_key_values

        return self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # ----------------------------------------------------------
    # TRAINING: Fine-tune γ network + hybrid encoder
    # Dùng khi có labeled data (response có/không hallucinate)
    # ----------------------------------------------------------
    def train_step(
        self,
        prompt: str,
        image_path: str,
        chosen_response: str,
        rejected_response: str,
        margin: float = 0.3,
    ) -> float:
        """
        Contrastive margin loss:
            loss = max(0, margin - (gamma_chosen - gamma_rejected))

        Pushes gamma_chosen > gamma_rejected by at least `margin`,
        instead of regressing to absolute targets (which cancel out).
        """
        image = self.load_image(image_path)

        self.optimizer.zero_grad()

        def get_avg_gamma(response_text: str) -> torch.Tensor:
            full_text = prompt + " " + response_text
            inputs = self._prepare_inputs_with_image(full_text, image)
            inputs = self._move_inputs_to_device(inputs, self.device)

            prompt_inputs = self._prepare_inputs_with_image(prompt, image)
            prompt_len = prompt_inputs['input_ids'].shape[1]

            if inputs['input_ids'].shape[1] <= prompt_len:
                raise ValueError(
                    f"Sequence length ({inputs['input_ids'].shape[1]}) <= prompt_len ({prompt_len})"
                )

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)

            full_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden]
            if torch.isnan(full_hidden).any():
                return None

            response_hidden = full_hidden[:, prompt_len:, :]
            del outputs, inputs
            torch.cuda.empty_cache()

            # float32 for stable gradients
            h_f32 = response_hidden.squeeze(0).to(dtype=torch.float32)
            gammas = self.gamma_net(h_f32)   # [response_len, 1]
            return gammas.mean()

        gamma_chosen   = get_avg_gamma(chosen_response)
        gamma_rejected = get_avg_gamma(rejected_response)

        if gamma_chosen is None or gamma_rejected is None:
            return float('nan')

        # Hinge / margin loss: push gamma_chosen - gamma_rejected >= margin
        loss = torch.clamp(
            torch.tensor(margin, device=self.device, dtype=torch.float32)
            - (gamma_chosen - gamma_rejected),
            min=0.0
        )

        if torch.isnan(loss):
            self.optimizer.zero_grad()
            return float('nan')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gamma_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()
    

    def train_step_phase1(
        self,
        prompt: str,
        image,
        chosen_response: str,
    ) -> float:
        """
        Phase-1 LM training:
        - Let Qwen handle vision internally (no manual vision forward)
        - Mask prompt tokens in labels so loss is only on chosen_response
        - Auxiliary MSE loss to train hybrid_encoder on patch tokens
        """
        self.optimizer.zero_grad()

        image = self.load_image(image)

        # === Build full input: user + assistant turns ===
        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": chosen_response},
        ]
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[full_text], images=[image], return_tensors="pt", padding=True
        )
        inputs = self._move_inputs_to_device(inputs, self.device)

        # === Label masking: find assistant turn start directly in token IDs ===
        # Scan input_ids for the last <|im_start|>assistant sequence and mask
        # everything before (and including) the header up to the response.
        tokenizer = self.processor.tokenizer
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_token_ids = tokenizer.encode("assistant", add_special_tokens=False)

        input_ids_list = inputs["input_ids"][0].tolist()
        split_pos = None  # position of first response token
        for i in range(len(input_ids_list) - 1, -1, -1):
            if input_ids_list[i] == im_start_id:
                tail = input_ids_list[i + 1: i + 1 + len(assistant_token_ids)]
                if tail == assistant_token_ids:
                    # skip: <|im_start|> assistant \n  (3 tokens)
                    split_pos = i + len(assistant_token_ids) + 2
                    break

        if split_pos is None or split_pos >= len(input_ids_list):
            print(f"[Warning] Could not locate assistant turn in token sequence "
                  f"(seq_len={len(input_ids_list)}). Skipping sample.")
            self.optimizer.zero_grad()
            return float('nan')

        labels = inputs["input_ids"].clone()
        labels[:, :split_pos] = -100  # mask everything before the response

        # === Full forward pass
        # Do NOT pass labels — let fp16 model do the forward, then compute
        # cross-entropy in fp32 to avoid fp16 overflow → NaN loss.
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

        # Shift logits/labels for next-token prediction, then compute in fp32
        shift_logits = outputs.logits[:, :-1, :].float()   # [1, seq-1, vocab]
        shift_labels = labels[:, 1:].contiguous()           # [1, seq-1]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            self.optimizer.zero_grad()
            return float('nan')

        # === Auxiliary loss: train hybrid_encoder on patch tokens ===
        # Patch token positions = where input_ids == image_token_id
        image_token_id = self.model.config.image_token_id
        visual_mask = (inputs["input_ids"] == image_token_id)  # [1, seq_len]

        if visual_mask.any():
            # hidden_states[0] = embedding layer output [1, seq_len, hidden]
            first_hidden = outputs.hidden_states[0]
            patch_tokens = first_hidden[visual_mask].unsqueeze(0)       # [1, n_patches, hidden]
            patch_tokens_f32 = patch_tokens.detach().to(dtype=torch.float32)

            # hybrid_tokens: [1, n_patches + n_regions, hidden]
            hybrid_tokens = self.hybrid_encoder(patch_tokens_f32)

            # Alignment loss: region tokens should be meaningful summaries
            # of patch tokens (compare region part to mean-pooled patches).
            # patch part == patch_tokens by construction, so compare region tokens
            # against global mean of patches to push them toward semantic summary.
            n_patches = patch_tokens_f32.shape[1]
            patch_mean = patch_tokens_f32.mean(dim=1, keepdim=True)             # [1, 1, H]
            region_tokens = hybrid_tokens[:, n_patches:, :]                     # [1, n_regions, H]
            aux_loss = torch.nn.functional.mse_loss(region_tokens, patch_mean.expand_as(region_tokens))
            loss = loss + 0.1 * aux_loss

        loss.backward()

        all_params = (
            list(self.hybrid_encoder.parameters()) +
            [p for p in self.model.language_model.parameters()
             if p.requires_grad and p.device.type not in ('meta', 'cpu')]
        )
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    # ----------------------------------------------------------
    # VALIDATION: same forward logic as train_step_phase1 but no backward
    # ----------------------------------------------------------
    @torch.no_grad()
    def eval_step_phase1(
        self,
        prompt: str,
        image,
        chosen_response: str,
    ) -> float:
        """
        Compute validation loss for one sample (no gradient update).
        Mirrors train_step_phase1 exactly, but wrapped in no_grad.
        """
        image = self.load_image(image)

        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": chosen_response},
        ]
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[full_text], images=[image], return_tensors="pt", padding=True
        )
        inputs = self._move_inputs_to_device(inputs, self.device)

        # Find the start of the assistant response in input_ids
        tokenizer = self.processor.tokenizer
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_token_ids = tokenizer.encode("assistant", add_special_tokens=False)

        input_ids_list = inputs["input_ids"][0].tolist()
        split_pos = None
        for i in range(len(input_ids_list) - 1, -1, -1):
            if input_ids_list[i] == im_start_id:
                tail = input_ids_list[i + 1: i + 1 + len(assistant_token_ids)]
                if tail == assistant_token_ids:
                    split_pos = i + len(assistant_token_ids) + 2
                    break

        if split_pos is None or split_pos >= len(input_ids_list):
            return float('nan')

        labels = inputs["input_ids"].clone()
        labels[:, :split_pos] = -100

        outputs = self.model(
            **inputs,
            output_hidden_states=False,
            use_cache=False,
        )

        shift_logits = outputs.logits[:, :-1, :].float()
        shift_labels = labels[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        if torch.isnan(loss) or torch.isinf(loss):
            return float('nan')

        return loss.item()
