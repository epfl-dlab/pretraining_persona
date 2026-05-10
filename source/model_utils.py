import os, re, json, torch
from typing import Optional
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import socket

try:
    from peft import PeftConfig, PeftModel
    _PEFT_AVAILABLE = True
except ImportError:
    PeftConfig = None
    PeftModel = None
    _PEFT_AVAILABLE = False





def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  
        return s.getsockname()[1]


# ---------- 工具 ----------
_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")

def _pick_latest_checkpoint(model_path: str) -> str:
    ckpts = [(int(m.group(1)), p) for p in Path(model_path).iterdir()
             if (m := _CHECKPOINT_RE.fullmatch(p.name)) and p.is_dir()]
    return str(max(ckpts, key=lambda x: x[0])[1]) if ckpts else model_path

def _is_lora(path: str) -> bool:
    return Path(path, "adapter_config.json").exists()

def _load_and_merge_lora(lora_path: str, dtype, device_map):
    if not _PEFT_AVAILABLE:
        raise ImportError("peft is required to load LoRA checkpoints.")
    cfg = PeftConfig.from_pretrained(lora_path)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path, torch_dtype=dtype, device_map=device_map
    )
    return PeftModel.from_pretrained(base, lora_path).merge_and_unload()

def _load_tokenizer(path_or_id: str):
    tok = AutoTokenizer.from_pretrained(path_or_id)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok

def _default_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def _is_mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _hub_load_kwargs(dtype: torch.dtype, revision: Optional[str] = None) -> dict:
    kwargs = {
        "torch_dtype": dtype,
        "revision": revision,
    }
    if _is_mps_available():
        # Avoid accelerate auto-offload on MPS: some hub checkpoints keep bfloat16
        # metadata, and accelerate later attempts unsupported bfloat16 -> MPS moves.
        return kwargs
    kwargs["device_map"] = "auto"
    return kwargs


def _finalize_model_device(model):
    if _is_mps_available():
        return model.to("mps")
    return model


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def _vllm_load_kwargs(model_path: str, revision: Optional[str] = None) -> dict:
    kwargs = {
        "model": model_path,
        "enable_prefix_caching": os.getenv("VLLM_ENABLE_PREFIX_CACHING", "true").lower() not in {"0", "false", "no"},
        "enable_lora": False,
        "tensor_parallel_size": _env_int("VLLM_TENSOR_PARALLEL_SIZE", max(torch.cuda.device_count(), 1)),
        "max_num_seqs": _env_int("VLLM_MAX_NUM_SEQS", 32),
        "gpu_memory_utilization": _env_float("VLLM_GPU_MEMORY_UTILIZATION", 0.9),
        "max_model_len": _env_int("VLLM_MAX_MODEL_LEN", 4096),
        "max_lora_rank": _env_int("VLLM_MAX_LORA_RANK", 128),
    }
    if revision:
        kwargs["revision"] = revision
    if os.getenv("VLLM_ENFORCE_EAGER", "").lower() in {"1", "true", "yes"}:
        kwargs["enforce_eager"] = True
    return kwargs


def load_model(model_path: str, dtype: Optional[torch.dtype] = None, revision: str = None):
    if dtype is None:
        dtype = _default_torch_dtype()
    if not os.path.exists(model_path):               # ---- Hub ----
        model = AutoModelForCausalLM.from_pretrained(
            model_path, **_hub_load_kwargs(dtype, revision=revision)
        )
        model = _finalize_model_device(model)
        tok = _load_tokenizer(model_path)
        return model, tok

    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    if _is_lora(resolved):
        device_map = None if _is_mps_available() else "auto"
        model = _load_and_merge_lora(resolved, dtype, device_map)
        model = _finalize_model_device(model)
        tok = _load_tokenizer(model.config._name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=dtype,
            **({"device_map": "auto"} if not _is_mps_available() else {})
        )
        model = _finalize_model_device(model)
        tok = _load_tokenizer(resolved)
    return model, tok

def load_vllm_model(model_path: str, revision: str = None):
    from vllm import LLM

    if not os.path.exists(model_path):               # ---- Hub ----
        llm = LLM(**_vllm_load_kwargs(model_path, revision=revision))
        tok = llm.get_tokenizer()
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"
        return llm, tok, None

    # ---- 本地 ----
    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    is_lora = _is_lora(resolved)
    if is_lora and not _PEFT_AVAILABLE:
        raise ImportError("peft is required to load LoRA checkpoints.")

    base_path = (PeftConfig.from_pretrained(resolved).base_model_name_or_path
                 if is_lora else resolved)

    llm = LLM(**_vllm_load_kwargs(base_path))

    if is_lora:
        lora_path = resolved
    else:
        lora_path = None

    tok = llm.get_tokenizer()
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return llm, tok, lora_path
