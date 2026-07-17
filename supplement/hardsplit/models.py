# -*- coding: utf-8 -*-
"""Model presets and CLI namespace helpers (package module).

Prefer:
  from hardsplit.models import resolve_model_spec, apply_model_namespace
Backward-compatible:
  from model_registry import resolve_model_spec
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HF_ID = "EleutherAI/pythia-1b"

# Historical Pythia-1B paths (still the default CLI values)
PYTHIA1B_RUN_DIR = "mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2"
PYTHIA1B_RUN_DIR_RESULTS = f"results/{PYTHIA1B_RUN_DIR}"
PYTHIA1B_FEATURES_ROOT = "attention_features_mimir_hardsplit"
DEFAULT_EXP1_DIR = "results/exp1_layer_head_stats"
DEFAULT_EVAL_DIR = "results/strict_fixed20_unified"
DEFAULT_LORA_LEAK_DIR = "results/lora_leak_official_mimir_hardsplit"
DEFAULT_ATTENMIA_DIR = "results/attenmia_official_mimir_hardsplit"

PYTHIA_LORA_MODULES = [
    "query_key_value",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
]
GPT_NEO_LORA_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "c_fc",
    "c_proj",
]
DEFAULT_PYTHIA_LORA_CSV = ",".join(PYTHIA_LORA_MODULES)

MODEL_CLI_HELP = "Preset: pythia-1b | pythia-410m | gpt-neo-2.7b"


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str  # gpt_neox | gpt_neo
    lora_target_modules: Sequence[str]
    short_name: str
    label: str
    default_run_dir: str
    default_features_root: str
    notes: str = ""

    def lora_target_modules_csv(self) -> str:
        return ",".join(self.lora_target_modules)

    @property
    def default_lora_root(self) -> str:
        return f"results/lora_leak_{self.short_name}"

    @property
    def default_attenmia_root(self) -> str:
        return f"results/attenmia_{self.short_name}"

    @property
    def default_exp1_dir(self) -> str:
        return f"results/exp1_layer_head_stats_{self.short_name}"

    @property
    def default_eval_dir(self) -> str:
        return f"results/strict_fixed20_{self.short_name}"

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["lora_target_modules"] = list(self.lora_target_modules)
        d["default_lora_root"] = self.default_lora_root
        d["default_attenmia_root"] = self.default_attenmia_root
        return d


MODEL_PRESETS: Dict[str, ModelSpec] = {
    "pythia-1b": ModelSpec(
        key="pythia-1b",
        hf_id="EleutherAI/pythia-1b",
        family="gpt_neox",
        lora_target_modules=PYTHIA_LORA_MODULES,
        short_name="pythia1b",
        label="Pythia-1B",
        default_run_dir=PYTHIA1B_RUN_DIR,
        default_features_root=PYTHIA1B_FEATURES_ROOT,
        notes="Main paper model (GPT-NeoX).",
    ),
    "pythia-410m": ModelSpec(
        key="pythia-410m",
        hf_id="EleutherAI/pythia-410m",
        family="gpt_neox",
        lora_target_modules=PYTHIA_LORA_MODULES,
        short_name="pythia410m",
        label="Pythia-410M",
        default_run_dir="mimir_lora_pythia410m",
        default_features_root="attention_features_pythia410m",
        notes="Supplementary size comparison; same LoRA modules as Pythia-1B.",
    ),
    "gpt-neo-2.7b": ModelSpec(
        key="gpt-neo-2.7b",
        hf_id="EleutherAI/gpt-neo-2.7B",
        family="gpt_neo",
        lora_target_modules=GPT_NEO_LORA_MODULES,
        short_name="gptneo27b",
        label="GPT-Neo-2.7B",
        default_run_dir="mimir_lora_gptneo27b",
        default_features_root="attention_features_gptneo27b",
        notes="Appendix model; architecture-specific LoRA modules.",
    ),
}

_ALIASES: Dict[str, str] = {
    "pythia1b": "pythia-1b",
    "pythia-1b": "pythia-1b",
    "eleutherai/pythia-1b": "pythia-1b",
    "pythia410m": "pythia-410m",
    "pythia-410m": "pythia-410m",
    "eleutherai/pythia-410m": "pythia-410m",
    "gptneo": "gpt-neo-2.7b",
    "gpt-neo": "gpt-neo-2.7b",
    "gptneo27b": "gpt-neo-2.7b",
    "gpt-neo-2.7b": "gpt-neo-2.7b",
    "gpt-neo-2.7B": "gpt-neo-2.7b",
    "eleutherai/gpt-neo-2.7b": "gpt-neo-2.7b",
    "eleutherai/gpt-neo-2.7B": "gpt-neo-2.7b",
}


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------

def list_model_keys() -> List[str]:
    return list(MODEL_PRESETS.keys())


def list_eval_keys() -> List[str]:
    return [s.short_name for s in MODEL_PRESETS.values()]


def normalize_model_key(name: str) -> str:
    raw = name.strip()
    key = _ALIASES.get(raw, _ALIASES.get(raw.lower(), raw.lower()))
    if key in MODEL_PRESETS:
        return key
    for k, spec in MODEL_PRESETS.items():
        if raw == spec.hf_id or raw.lower() == spec.hf_id.lower():
            return k
    raise KeyError(
        f"Unknown model '{name}'. Known: {list_model_keys()} "
        f"(aliases: {sorted(set(_ALIASES) - set(MODEL_PRESETS))})"
    )


def resolve_model_spec(name: str) -> ModelSpec:
    return MODEL_PRESETS[normalize_model_key(name)]


def try_resolve_model_spec(name: Optional[str]) -> Optional[ModelSpec]:
    if not name:
        return None
    try:
        return resolve_model_spec(name)
    except KeyError:
        return None


def infer_family_from_hf_id(hf_id: str) -> str:
    low = hf_id.lower()
    if "gpt-neo" in low and "neox" not in low:
        return "gpt_neo"
    if "pythia" in low or "neox" in low:
        return "gpt_neox"
    return "gpt_neox"


def lora_modules_for_hf_id(hf_id: str) -> List[str]:
    try:
        return list(resolve_model_spec(hf_id).lora_target_modules)
    except KeyError:
        fam = infer_family_from_hf_id(hf_id)
        return list(GPT_NEO_LORA_MODULES if fam == "gpt_neo" else PYTHIA_LORA_MODULES)


def model_spec_or_custom(name: str) -> ModelSpec:
    """Return preset or a best-effort custom ModelSpec for arbitrary HF ids."""
    try:
        return resolve_model_spec(name)
    except KeyError:
        fam = infer_family_from_hf_id(name)
        modules = GPT_NEO_LORA_MODULES if fam == "gpt_neo" else PYTHIA_LORA_MODULES
        slug = name.replace("/", "_").replace(".", "").lower()
        return ModelSpec(
            key=slug,
            hf_id=name,
            family=fam,
            lora_target_modules=modules,
            short_name=slug,
            label=name,
            default_run_dir=f"mimir_lora_{slug}",
            default_features_root=f"attention_features_{slug}",
            notes="Custom HF model (auto family/modules).",
        )


def eval_key_from_model(name: Optional[str], default: str = "pythia1b") -> str:
    """Map --model / HF id / short name → strict-eval key (e.g. pythia410m)."""
    if not name:
        return default
    try:
        return resolve_model_spec(name).short_name
    except KeyError:
        return default


def strict_eval_model_configs() -> Dict[str, Dict[str, str]]:
    """Build MODEL_CONFIGS for run_strict_fixed20_comparison_10runs."""
    return {
        spec.short_name: {
            "label": spec.label,
            "proposed_root": spec.default_features_root,
            "lora_root": "",
            "attenmia_root": "",
        }
        for spec in MODEL_PRESETS.values()
    }


# ---------------------------------------------------------------------------
# Adapter / path helpers
# ---------------------------------------------------------------------------

def _normalize_path_str(path: str) -> str:
    return path.strip().rstrip("/")


def is_default_run_dir(path: Optional[str]) -> bool:
    if path is None:
        return True
    p = _normalize_path_str(path)
    return p in {"", PYTHIA1B_RUN_DIR, PYTHIA1B_RUN_DIR_RESULTS}


def is_default_features_root(path: Optional[str]) -> bool:
    if path is None:
        return True
    return _normalize_path_str(path) in {"", PYTHIA1B_FEATURES_ROOT}


def is_default_lora_csv(value: Optional[str]) -> bool:
    if value is None:
        return True
    return value.strip() in {"", DEFAULT_PYTHIA_LORA_CSV}


def is_default_model_name(value: Optional[str]) -> bool:
    if not value:
        return True
    return value.strip() in {"", DEFAULT_HF_ID}


def read_base_model_from_adapter(adapter_dir: str | Path) -> Optional[str]:
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.exists():
        alt = Path(adapter_dir) / "adapter" / "adapter_config.json"
        cfg_path = alt if alt.exists() else cfg_path
    if not cfg_path.exists():
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data.get("base_model_name_or_path")


def resolve_model_name(
    explicit: Optional[str] = None,
    adapter_dir: Optional[str | Path] = None,
    default: str = DEFAULT_HF_ID,
) -> str:
    """Resolve HF model id: explicit preset/HF id > adapter_config > default."""
    if explicit:
        try:
            return resolve_model_spec(explicit).hf_id
        except KeyError:
            return explicit  # raw HF id allowed
    if adapter_dir is not None:
        inferred = read_base_model_from_adapter(adapter_dir)
        if inferred:
            return inferred
    return default


def resolve_adapter_dir(
    path_like: str | Path,
    *,
    run_dir: Optional[str | Path] = None,
) -> Path:
    """Return directory that contains adapter_config.json.

    Accepts either the adapter dir itself or a run dir containing ``adapter/``.
    """
    candidates: List[Path] = []
    if path_like:
        path = Path(path_like).expanduser()
        candidates.extend([path, path / "adapter"])
        stripped = Path(str(path_like).replace("results/", "", 1))
        if stripped != path:
            candidates.extend([stripped, stripped / "adapter"])
        candidates.extend([Path(path.name), Path(path.name) / "adapter"])
    if run_dir:
        run = Path(run_dir).expanduser()
        candidates.extend([run / "adapter", run])
        stripped = Path(str(run_dir).replace("results/", "", 1))
        candidates.extend([stripped / "adapter", stripped])

    seen: Set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if (cand / "adapter_config.json").exists():
            return cand
    raise FileNotFoundError(
        f"adapter_config.json not found (path_like={path_like!r}, run_dir={run_dir!r})"
    )


# ---------------------------------------------------------------------------
# CLI namespace defaults (shared by train / extract / baselines / orchestrate)
# ---------------------------------------------------------------------------

LogFn = Optional[Callable[[str], None]]

# Profiles select which fields get filled from a ModelSpec.
_PROFILE_FIELDS: Dict[str, Sequence[str]] = {
    # train_mimir_wikipedia_hardsplit_lora.py
    "train": ("model_name", "target_modules", "output_dir"),
    # extract / orchestrate extract
    "extract": ("model_name", "run_dir", "features_root"),
    # orchestrate / unified pipeline
    "pipeline": (
        "model_name",
        "run_dir",
        "features_root",
        "exp1_output_dir",
        "eval_output_dir",
        "lora_root",
        "attenmia_root",
        "model_key",
    ),
    # LoRA-Leak baseline
    "lora_leak": ("model_name", "run_dir", "adapter_dir", "output_dir"),
    # AttenMIA baseline
    "attenmia": ("model_name", "run_dir", "adapter_dir", "output_dir"),
}


def _spec_value(spec: ModelSpec, field: str, profile: str) -> Optional[str]:
    if field == "model_name":
        return spec.hf_id
    if field == "target_modules":
        return spec.lora_target_modules_csv()
    if field in {"run_dir", "output_dir"} and profile == "train":
        # train uses output_dir as run dir
        return spec.default_run_dir
    if field == "run_dir":
        return spec.default_run_dir
    if field == "adapter_dir":
        return spec.default_run_dir
    if field == "features_root":
        return spec.default_features_root
    if field == "exp1_output_dir":
        return spec.default_exp1_dir
    if field == "eval_output_dir":
        return spec.default_eval_dir
    if field == "lora_root":
        return spec.default_lora_root
    if field == "attenmia_root":
        return spec.default_attenmia_root
    if field == "model_key":
        return spec.short_name
    if field == "output_dir" and profile == "lora_leak":
        return spec.default_lora_root
    if field == "output_dir" and profile == "attenmia":
        return spec.default_attenmia_root
    if field == "output_dir":
        return spec.default_run_dir
    return None


def _default_values_for_field(field: str, profile: str) -> Set[str]:
    """Values treated as 'still the CLI default' → safe to overwrite from preset."""
    if field == "model_name":
        return {"", DEFAULT_HF_ID}
    if field == "target_modules":
        return {"", DEFAULT_PYTHIA_LORA_CSV}
    if field in {"run_dir", "adapter_dir"}:
        return {"", PYTHIA1B_RUN_DIR, PYTHIA1B_RUN_DIR_RESULTS}
    if field == "output_dir" and profile == "train":
        return {"", PYTHIA1B_RUN_DIR, PYTHIA1B_RUN_DIR_RESULTS}
    if field == "output_dir" and profile == "lora_leak":
        return {"", DEFAULT_LORA_LEAK_DIR}
    if field == "output_dir" and profile == "attenmia":
        return {"", DEFAULT_ATTENMIA_DIR}
    if field == "output_dir":
        return {"", PYTHIA1B_RUN_DIR, PYTHIA1B_RUN_DIR_RESULTS}
    if field == "features_root":
        return {"", PYTHIA1B_FEATURES_ROOT}
    if field == "exp1_output_dir":
        return {"", DEFAULT_EXP1_DIR}
    if field == "eval_output_dir":
        return {"", DEFAULT_EVAL_DIR}
    if field in {"lora_root", "attenmia_root", "model_key"}:
        return {"", None}  # type: ignore[list-item]
    return {""}


def _set_if_default(args: Any, field: str, new_value: str, profile: str, force: bool = False) -> None:
    if not hasattr(args, field):
        # Inject optional derived attrs (e.g. model_key for eval).
        if field in {"model_key"}:
            setattr(args, field, new_value)
        return
    cur = getattr(args, field)
    if cur is None:
        cur = ""
    defaults = _default_values_for_field(field, profile)
    if force or cur in defaults:
        setattr(args, field, new_value)


def apply_model_namespace(
    args: Any,
    *,
    profile: str = "pipeline",
    model_flag: Optional[str] = None,
    force: bool = False,
    fill_baselines: bool = False,
    log: LogFn = None,
) -> Any:
    """Fill argparse Namespace fields from ``--model`` preset.

    Parameters
    ----------
    profile:
      One of: train | extract | pipeline | lora_leak | attenmia
    model_flag:
      Override for ``args.model`` (optional).
    force:
      Overwrite even non-default field values.
    fill_baselines:
      For pipeline profile, also set lora_root / attenmia_root when empty.
    """
    if profile not in _PROFILE_FIELDS:
        raise ValueError(f"Unknown profile '{profile}'. Known: {list(_PROFILE_FIELDS)}")

    flag = model_flag if model_flag is not None else getattr(args, "model", "") or ""
    log_fn = log or (lambda _m: None)

    if not flag:
        # train special-case: only fix LoRA modules from model_name
        if profile == "train" and hasattr(args, "target_modules") and hasattr(args, "model_name"):
            if is_default_lora_csv(getattr(args, "target_modules", None)):
                name = getattr(args, "model_name", "") or DEFAULT_HF_ID
                args.target_modules = ",".join(lora_modules_for_hf_id(name))
        return args

    spec = resolve_model_spec(flag)
    fields = list(_PROFILE_FIELDS[profile])
    if profile == "pipeline" and not fill_baselines:
        fields = [f for f in fields if f not in {"lora_root", "attenmia_root"}]

    for field in fields:
        value = _spec_value(spec, field, profile)
        if value is None:
            continue
        # model_name / target_modules always set from preset when --model given
        always = field in {"model_name", "target_modules", "model_key"}
        _set_if_default(args, field, value, profile, force=force or always)

    log_fn(f"model preset={spec.key} hf_id={spec.hf_id}")
    if hasattr(args, "run_dir"):
        log_fn(f"  run_dir={args.run_dir}")
    if hasattr(args, "features_root"):
        log_fn(f"  features_root={args.features_root}")
    if hasattr(args, "output_dir"):
        log_fn(f"  output_dir={args.output_dir}")
    return args


def add_model_arguments(
    parser: Any,
    *,
    include_model: bool = True,
    include_model_name: bool = True,
    model_name_default: str = "",
) -> Any:
    """Attach standard --model / --model-name flags."""
    if include_model:
        parser.add_argument("--model", default="", help=MODEL_CLI_HELP)
    if include_model_name:
        parser.add_argument(
            "--model-name",
            default=model_name_default,
            help="HF id or preset key (default: from --model or adapter_config)",
        )
    return parser


def resolve_from_args(
    args: Any,
    *,
    adapter_dir: Optional[str | Path] = None,
    default: str = DEFAULT_HF_ID,
) -> str:
    """Convenience: resolve HF id from argparse args (+ optional adapter)."""
    explicit = getattr(args, "model_name", None) or getattr(args, "model", None) or ""
    adapter = adapter_dir
    if adapter is None:
        for attr in ("adapter_dir", "run_dir", "output_dir"):
            val = getattr(args, attr, None)
            if val:
                try:
                    adapter = resolve_adapter_dir(val)
                    break
                except FileNotFoundError:
                    adapter = val
    return resolve_model_name(explicit=explicit or None, adapter_dir=adapter, default=default)
