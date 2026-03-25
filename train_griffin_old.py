#!/usr/bin/env python3
"""
Legacy simplified Griffin trainer preserved for old-vs-new benchmarking.

This keeps the original simplified backbone design:
- dense masked local attention
- naive gated recurrence token scan
- no recurrent cache
- no attention cache
- no FlashAttention backend abstraction
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from train_griffin_modern import (
    ExperimentConfig,
    GriffinModelConfig,
    RMSNorm,
    ShadowConfig,
    StreamingTokenLoader,
    WebArchaeologyShadow,
    apply_rope,
    benchmark_backbone,
    build_sentencepiece_luts,
    build_shadow_vocab_metadata,
    clamp_quant_bits,
    configure_model_for_mode,
    count_trainable_parameters,
    embedding_with_optional_qat,
    evaluate,
    flash_attn_available,
    get_default_dtype_for_device,
    get_device,
    linear_with_optional_qat,
    load_checkpoint,
    load_validation_tokens,
    log_device_info,
    parse_args,
    print_metrics,
    RotaryEmbedding,
    run_training_loop,
    seed_everything,
    setup_run_logging,
    supports_fast_attention,
    checkpoint_path_for_stage,
)


class LegacyGriffinLocalAttention(nn.Module):
    def __init__(self, config: GriffinModelConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.window = max(2, min(config.local_window, config.block_size))
        self.use_flash = config.use_flash
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, base=config.rope_base) if config.use_rope else None
        self.qat_enabled = config.qat_enabled
        self.qat_bits = clamp_quant_bits(config.qat_bits)

    def _local_mask(self, seq_len: int, device: torch.device) -> Tensor:
        positions = torch.arange(seq_len, device=device)
        rel = positions[:, None] - positions[None, :]
        allowed = (rel >= 0) & (rel < self.window)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32)
        return mask.masked_fill(allowed, 0.0)

    def _manual_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        scale = 1.0 / math.sqrt(q.size(-1))
        att = (q @ k.transpose(-2, -1)) * scale
        att = att + mask[None, None, :, :]
        att = F.softmax(att.float(), dim=-1).to(dtype=q.dtype)
        att = self.attn_dropout(att)
        return att @ v

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        qat_on = self.training and self.qat_enabled
        q = linear_with_optional_qat(x, self.q_proj, qat_on, self.qat_bits).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = linear_with_optional_qat(x, self.k_proj, qat_on, self.qat_bits).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = linear_with_optional_qat(x, self.v_proj, qat_on, self.qat_bits).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            cos, sin = self.rope(seq_len, q.device, q.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        mask = self._local_mask(seq_len, x.device)
        if supports_fast_attention(x.device, self.use_flash):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask[None, None, :, :],
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            y = self._manual_attention(q, k, v, mask)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_embd)
        return self.resid_dropout(linear_with_optional_qat(y, self.out_proj, qat_on, self.qat_bits))


class LegacyGriffinRecurrentPath(nn.Module):
    def __init__(self, config: GriffinModelConfig):
        super().__init__()
        self.in_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.decay_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.gate_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.decay_bias = nn.Parameter(torch.full((config.n_embd,), float(config.recurrent_decay_bias)))
        self.qat_enabled = config.qat_enabled
        self.qat_bits = clamp_quant_bits(config.qat_bits)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, width = x.shape
        qat_on = self.training and self.qat_enabled
        inp = F.silu(linear_with_optional_qat(x, self.in_proj, qat_on, self.qat_bits)).float()
        decay = torch.sigmoid(linear_with_optional_qat(x, self.decay_proj, qat_on, self.qat_bits).float() + self.decay_bias)
        gate = torch.sigmoid(linear_with_optional_qat(x, self.gate_proj, qat_on, self.qat_bits).float())
        state = torch.zeros((bsz, width), device=x.device, dtype=torch.float32)
        outputs: list[Tensor] = []
        for t in range(seq_len):
            state = decay[:, t, :] * state + (1.0 - decay[:, t, :]) * inp[:, t, :]
            outputs.append(gate[:, t, :] * state)
        y = torch.stack(outputs, dim=1).to(dtype=x.dtype)
        y = linear_with_optional_qat(y, self.out_proj, qat_on, self.qat_bits)
        return self.resid_dropout(y)


class LegacyGriffinMLP(nn.Module):
    def __init__(self, config: GriffinModelConfig):
        super().__init__()
        hidden = max(config.n_embd, int(round(config.n_embd * config.mlp_mult)))
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.qat_enabled = config.qat_enabled
        self.qat_bits = clamp_quant_bits(config.qat_bits)

    def forward(self, x: Tensor) -> Tensor:
        qat_on = self.training and self.qat_enabled
        gate = F.silu(linear_with_optional_qat(x, self.gate_proj, qat_on, self.qat_bits))
        value = linear_with_optional_qat(x, self.up_proj, qat_on, self.qat_bits)
        out = linear_with_optional_qat(gate * value, self.down_proj, qat_on, self.qat_bits)
        return self.dropout(out)


class LegacyGriffinBlock(nn.Module):
    def __init__(self, config: GriffinModelConfig):
        super().__init__()
        self.temporal_norm = RMSNorm(config.n_embd)
        self.local_attn = LegacyGriffinLocalAttention(config)
        self.recurrent = LegacyGriffinRecurrentPath(config)
        self.temporal_mix = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.temporal_dropout = nn.Dropout(config.dropout)
        self.channel_norm = RMSNorm(config.n_embd)
        self.channel_mlp = LegacyGriffinMLP(config)
        nn.init.zeros_(self.temporal_mix.weight)
        nn.init.zeros_(self.temporal_mix.bias)

    def forward(self, x: Tensor) -> Tensor:
        h = self.temporal_norm(x)
        recurrent_out = self.recurrent(h)
        attn_out = self.local_attn(h)
        mix = torch.sigmoid(self.temporal_mix(h))
        temporal_out = mix * attn_out + (1.0 - mix) * recurrent_out
        x = x + self.temporal_dropout(temporal_out)
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class LegacyGriffinBackbone(nn.Module):
    def __init__(self, config: GriffinModelConfig):
        super().__init__()
        self.config = config
        self.qat_enabled = config.qat_enabled
        self.qat_bits = clamp_quant_bits(config.qat_bits)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = None if config.use_rope else nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([LegacyGriffinBlock(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_qat(self, enabled: bool, bits: int | None = None) -> None:
        self.qat_enabled = enabled
        if bits is not None:
            self.qat_bits = clamp_quant_bits(bits)
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "qat_enabled"):
                module.qat_enabled = enabled
            if bits is not None and hasattr(module, "qat_bits"):
                module.qat_bits = self.qat_bits

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        return_hidden: bool = True,
        control_bias: Tensor | None = None,
        token_scale: Tensor | None = None,
        pos_bias: Tensor | None = None,
        adapter_bias: Tensor | None = None,
        adapter_layers: int = 0,
        ln_gain: Tensor | None = None,
        ln_bias: Tensor | None = None,
        cache=None,
        use_cache: bool = False,
    ) -> dict[str, Tensor | None]:
        del cache, use_cache
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.config.block_size}")
        qat_on = self.training and self.qat_enabled
        tok = embedding_with_optional_qat(input_ids, self.wte, qat_on, max(self.qat_bits, 6))
        if self.wpe is not None:
            pos = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
            pos_emb = embedding_with_optional_qat(pos, self.wpe, qat_on, max(self.qat_bits, 6))
            x = tok + pos_emb[None, :, :]
        else:
            x = tok
        if pos_bias is not None:
            x = x + pos_bias
        if control_bias is not None:
            x = x + control_bias
        if token_scale is not None:
            x = x * token_scale.unsqueeze(-1)
        x = self.drop(x)
        apply_adapter_from = max(len(self.blocks) - max(adapter_layers, 0), 0)
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            if adapter_bias is not None and block_idx >= apply_adapter_from:
                x = x + adapter_bias
        hidden = self.ln_f(x)
        if ln_gain is not None:
            hidden = hidden * (1.0 + ln_gain)
        if ln_bias is not None:
            hidden = hidden + ln_bias
        logits = linear_with_optional_qat(hidden, self.lm_head, qat_on, max(self.qat_bits, 6))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="mean")
        return {"logits": logits, "loss": loss, "hidden_states": hidden if return_hidden else None, "cache": None}

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> Tensor:
        out = input_ids
        for _ in range(max_new_tokens):
            idx = out[:, -self.config.block_size :]
            logits = self(idx, return_hidden=False)["logits"][:, -1, :]
            logits = logits / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_token], dim=1)
        return out


class LegacyHybridGriffinLanguageModel(nn.Module):
    def __init__(self, model_config: GriffinModelConfig, shadow_config: ShadowConfig, shadow_metadata: dict[str, Tensor] | None = None):
        super().__init__()
        self.backbone = LegacyGriffinBackbone(model_config)
        self.model_config = model_config
        self.shadow_config = shadow_config
        self.shadow_stream = WebArchaeologyShadow(model_config.n_embd, model_config.vocab_size, shadow_config, shadow_metadata)

    def forward_backbone(self, input_ids: Tensor, targets: Tensor | None = None) -> dict[str, Tensor | None]:
        out = self.backbone(input_ids, targets=targets, return_hidden=True)
        return {"logits": out["logits"], "loss": out["loss"], "hidden_states": out["hidden_states"], "gpt_logits": out["logits"], "gpt_loss": out["loss"]}

    def forward_shadow(self, input_ids: Tensor, targets: Tensor | None = None) -> dict[str, Tensor | None]:
        with torch.no_grad():
            base_out = self.backbone(input_ids, targets=None, return_hidden=True)
            gpt_logits = base_out["logits"]
            hidden = base_out["hidden_states"]
        assert hidden is not None
        shadow_out = self.shadow_stream(input_ids, gpt_logits.detach(), hidden.detach())
        controlled_out = self.backbone(
            shadow_out["augmented_input_ids"],
            targets=None,
            return_hidden=True,
            control_bias=shadow_out["control_bias"],
            token_scale=shadow_out["token_scale"],
            pos_bias=shadow_out["pos_bias"],
            adapter_bias=shadow_out["adapter_bias"],
            adapter_layers=self.shadow_config.adapter_layers,
            ln_gain=shadow_out["ln_gain"],
            ln_bias=shadow_out["ln_bias"],
        )
        logits = controlled_out["logits"] + shadow_out["logit_bias"]
        outputs: dict[str, Tensor | None] = {
            "logits": logits,
            "loss": None,
            "gpt_logits": gpt_logits,
            "gpt_loss": None,
            "hidden_states": controlled_out["hidden_states"],
            "shadow_gate": shadow_out["gate"],
            "shadow_gate_raw_mean": shadow_out["gate_raw_mean"],
            "shadow_active_frac": shadow_out["active_frac"],
            "shadow_bias_norm": shadow_out["bias_norm"],
            "shadow_boundary_rate": shadow_out["boundary_rate"],
            "shadow_hard_boundary_rate": shadow_out["hard_boundary_rate"],
            "shadow_suppression_mean": shadow_out["suppression_mean"],
            "shadow_token_scale_mean": shadow_out["token_scale"].mean(dim=1),
            "shadow_dropout_rate": shadow_out["dropout_rate"],
            "shadow_corruption_rate": shadow_out["corruption_rate"],
            "shadow_fingerprint_diversity": shadow_out["fingerprint_diversity"],
            "shadow_recurrence_tag_mean": shadow_out["recurrence_tag_mean"],
            "shadow_template_slot_norm": shadow_out["template_slot_norm"],
            "shadow_template_usage_entropy": shadow_out["template_usage_entropy"],
            "shadow_adapter_norm": shadow_out["adapter_norm"],
            "shadow_logit_bias_norm": shadow_out["logit_bias_norm"],
            "shadow_ln_mod_norm": shadow_out["ln_mod_norm"],
            "shadow_regime_probs": shadow_out["regime_probs"],
            "shadow_consistency_logits": shadow_out["consistency_logits"],
            "shadow_ancestry_logits": shadow_out["ancestry_logits"],
        }
        if targets is not None:
            outputs["loss"] = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="mean")
            outputs["gpt_loss"] = F.cross_entropy(gpt_logits.reshape(-1, gpt_logits.size(-1)).float(), targets.reshape(-1), reduction="mean")
            outputs["shadow_target_logprobs"] = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            outputs["gpt_target_logprobs"] = F.log_softmax(gpt_logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            outputs["shadow_consistency_loss"] = F.cross_entropy(
                shadow_out["consistency_logits"][:, :-1, :].reshape(-1, shadow_out["consistency_logits"].size(-1)).float(),
                shadow_out["regime_probs"][:, 1:, :].argmax(dim=-1).reshape(-1),
                reduction="mean",
            )
            outputs["shadow_ancestry_loss"] = F.cross_entropy(
                shadow_out["ancestry_logits"].reshape(-1, shadow_out["ancestry_logits"].size(-1)).float(),
                shadow_out["ancestry_targets"].reshape(-1),
                reduction="mean",
            )
        return outputs

    def forward(self, input_ids: Tensor, targets: Tensor | None = None, mode: str = "backbone") -> dict[str, Tensor | None]:
        if mode == "shadow":
            return self.forward_shadow(input_ids, targets=targets)
        return self.forward_backbone(input_ids, targets=targets)


def run_sanity_checks_old(model: LegacyHybridGriffinLanguageModel, config: ExperimentConfig, device: torch.device) -> None:
    seq_len = min(8, int(config.model.block_size))
    batch_size = 2
    with torch.no_grad():
        model.eval()
        sample_ids = torch.randint(0, int(config.model.vocab_size), (batch_size, seq_len), device=device)
        sample_targets = torch.randint(0, int(config.model.vocab_size), (batch_size, seq_len), device=device)
        backbone_out = model(sample_ids, targets=sample_targets, mode="backbone")
        assert backbone_out["logits"] is not None and backbone_out["hidden_states"] is not None
        assert backbone_out["logits"].shape == (batch_size, seq_len, int(config.model.vocab_size))
        assert backbone_out["hidden_states"].shape == (batch_size, seq_len, int(config.model.n_embd))
        if config.shadow.enabled:
            configure_model_for_mode(model, "shadow")
            assert count_trainable_parameters(model.backbone) == 0
            assert count_trainable_parameters(model.shadow_stream) > 0
            shadow_out = model(sample_ids, targets=sample_targets, mode="shadow")
            assert shadow_out["logits"] is not None and shadow_out["hidden_states"] is not None
        configure_model_for_mode(model, "backbone")
        print(
            f"legacy_sanity_checks:ok seq_len:{seq_len} "
            f"logits_shape:{tuple(backbone_out['logits'].shape)} "
            f"hidden_shape:{tuple(backbone_out['hidden_states'].shape)}"
        )


def main() -> None:
    config = parse_args()
    config.model.backbone_type = "griffin_legacy_simple"
    config.model.local_attn_backend = "sdpa"
    config.model.recurrent_variant = "legacy_simple"
    config.model.use_recurrent_cache = False
    config.model.use_attention_cache = False
    with setup_run_logging(config.train.run_id):
        code = Path(__file__).read_text(encoding="utf-8")
        print(code)
        print("=" * 100)
        print(f"Running Python {sys.version}")
        print(f"Running PyTorch {torch.__version__}")
        gpu_report = ""
        try:
            if torch.cuda.is_available():
                gpu_report = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout
        except Exception as exc:
            gpu_report = f"nvidia_smi_unavailable:{exc}"
        if gpu_report:
            print(gpu_report)
        print("=" * 100)

        seed_everything(config.train.seed)
        device = get_device(config.train.device_preference)
        amp_dtype = get_default_dtype_for_device(device)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        tokenizer_luts = build_sentencepiece_luts(config.train.tokenizer_path, config.model.vocab_size) if config.train.report_bpb else None
        shadow_metadata = build_shadow_vocab_metadata(config.train.tokenizer_path, config.model.vocab_size)
        log_device_info(device, amp_dtype, supports_fast_attention(device, config.model.use_flash))
        print(
            f"griffin_config backbone_type:{config.model.backbone_type} "
            f"local_attn_backend:{config.model.local_attn_backend} "
            f"flash_attn_available:{flash_attn_available()} "
            f"recurrent_variant:{config.model.recurrent_variant} "
            f"use_recurrent_cache:{config.model.use_recurrent_cache} "
            f"use_attention_cache:{config.model.use_attention_cache}"
        )
        print(json.dumps(asdict(config), indent=2))

        train_loader = StreamingTokenLoader(config.train.resolved_train_pattern())
        val_tokens = load_validation_tokens(config.train.resolved_val_pattern(), config.model.block_size)
        model = LegacyHybridGriffinLanguageModel(config.model, config.shadow, shadow_metadata)
        print(
            f"model_params backbone:{sum(p.numel() for p in model.backbone.parameters())} "
            f"shadow:{sum(p.numel() for p in model.shadow_stream.parameters())} "
            f"total:{sum(p.numel() for p in model.parameters())}"
        )
        model.to(device)
        if not getattr(config.train, "skip_sanity_checks", False):
            run_sanity_checks_old(model, config, device)
        benchmark_backbone(model, config, device, amp_dtype, int(getattr(config.train, "benchmark_steps", 0)))

        if config.train.eval_only:
            if config.train.mode == "all":
                raise ValueError("--eval-only is only supported with --mode backbone or --mode shadow, not --mode all.")
            if config.train.load_checkpoint:
                load_checkpoint(config.train.load_checkpoint, model, map_location="cpu")
            model.to(device)
            metrics = evaluate(model, val_tokens, config, device, amp_dtype, tokenizer_luts)
            print_metrics("eval_only", metrics)
            return

        if config.train.mode == "all":
            backbone_train = replace(
                config.train,
                mode="backbone",
                load_checkpoint=config.train.load_checkpoint,
                checkpoint_path=checkpoint_path_for_stage(config.train.checkpoint_path, "backbone"),
                legacy_export_path="",
            )
            backbone_config = replace(config, train=backbone_train)
            print("pipeline_stage:backbone")
            run_training_loop(model, train_loader, val_tokens, backbone_config, device, amp_dtype, tokenizer_luts)

            shadow_train = replace(
                config.train,
                mode="shadow",
                load_checkpoint="",
                checkpoint_path=checkpoint_path_for_stage(config.train.checkpoint_path, "shadow"),
            )
            shadow_config = replace(config, train=shadow_train)
            print("pipeline_stage:shadow")
            run_training_loop(model, train_loader, val_tokens, shadow_config, device, amp_dtype, tokenizer_luts)
        else:
            run_training_loop(model, train_loader, val_tokens, config, device, amp_dtype, tokenizer_luts)


if __name__ == "__main__":
    main()
