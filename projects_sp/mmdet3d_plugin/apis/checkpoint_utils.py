# Copyright (c) OpenMMLab / AlignDrive. Selective load_from helpers.
"""Allowlist-based checkpoint loading with audit logs.

Used by the open-source retrain protocol: load at most backbone/neck/det/map
weights; never load planning heads (longitudinal / motion_plan / fusion).
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from mmcv.runner import load_state_dict

# Hard deny list for the open-source retrain path (must stay random-init).
PLANNING_DENY_PREFIXES: Tuple[str, ...] = (
    "head.longitudinal_head",
    "head.odo_head",
    "head.motion_plan_head",
    "head.fusion_model",
)


def strip_module_prefix(key: str) -> str:
    if key.startswith("module."):
        return key[len("module.") :]
    return key


def key_matches_prefixes(bare_key: str, prefixes: Sequence[str]) -> bool:
    for prefix in prefixes:
        if bare_key == prefix or bare_key.startswith(prefix + "."):
            return True
    return False


def summarize_key_prefixes(keys: Iterable[str], depth: int = 2) -> Counter:
    counts: Counter = Counter()
    for key in keys:
        parts = key.split(".")
        if parts and parts[0] == "head" and len(parts) > 1:
            counts[".".join(parts[:2])] += 1
        else:
            counts[".".join(parts[:depth])] += 1
    return counts


def filter_state_dict_by_modules(
    state_dict: dict,
    allow_modules: Optional[Sequence[str]],
    deny_modules: Optional[Sequence[str]] = None,
) -> Tuple[OrderedDict, List[str], List[str]]:
    """Filter checkpoint tensors by module-prefix allow/deny lists.

    Args:
        state_dict: raw checkpoint state_dict (may include ``module.`` prefix).
        allow_modules: if set, only keys matching these prefixes are kept.
        deny_modules: if set, matching keys are always skipped (after allow).

    Returns:
        filtered_state_dict, loaded_bare_keys, skipped_bare_keys
    """
    deny_modules = list(deny_modules) if deny_modules else []
    allow_modules = list(allow_modules) if allow_modules is not None else None

    filtered = OrderedDict()
    loaded: List[str] = []
    skipped: List[str] = []

    for key, value in state_dict.items():
        bare = strip_module_prefix(key)
        keep = True
        if allow_modules is not None:
            keep = key_matches_prefixes(bare, allow_modules)
        if keep and deny_modules and key_matches_prefixes(bare, deny_modules):
            keep = False
        if keep:
            filtered[key] = value
            loaded.append(bare)
        else:
            skipped.append(bare)
    return filtered, loaded, skipped


def audit_load(
    loaded: Sequence[str],
    skipped: Sequence[str],
    allow_modules: Optional[Sequence[str]],
    logger=None,
    deny_check_prefixes: Sequence[str] = PLANNING_DENY_PREFIXES,
) -> dict:
    """Print / return load audit stats. Raises if denied prefixes were loaded."""
    loaded_counts = summarize_key_prefixes(loaded)
    skipped_counts = summarize_key_prefixes(skipped)
    denied_loaded = [
        k for k in loaded if key_matches_prefixes(k, deny_check_prefixes)
    ]

    lines = [
        "=" * 72,
        "[load_from_modules] selective checkpoint audit",
        f"  allow_modules = {list(allow_modules) if allow_modules is not None else None}",
        f"  loaded keys   = {len(loaded)}",
        f"  skipped keys  = {len(skipped)}",
        "  loaded by prefix:",
    ]
    for prefix, count in sorted(loaded_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {count:5d}  {prefix}")
    lines.append("  skipped by prefix (top 20):")
    for prefix, count in sorted(skipped_counts.items(), key=lambda x: -x[1])[:20]:
        lines.append(f"    {count:5d}  {prefix}")

    planning_loaded = {
        p: sum(1 for k in loaded if key_matches_prefixes(k, [p]))
        for p in deny_check_prefixes
    }
    lines.append("  planning-prefix loaded counts (must be 0 for retrain):")
    for prefix, count in planning_loaded.items():
        lines.append(f"    {count:5d}  {prefix}")
    lines.append("=" * 72)
    msg = "\n".join(lines)

    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

    report = {
        "num_loaded": len(loaded),
        "num_skipped": len(skipped),
        "loaded_prefixes": dict(loaded_counts),
        "skipped_prefixes": dict(skipped_counts),
        "planning_loaded": planning_loaded,
        "denied_loaded_keys": denied_loaded[:50],
        "loaded_keys": list(loaded),
        "skipped_keys": list(skipped),
    }
    if denied_loaded:
        raise RuntimeError(
            "load_from_modules audit failed: planning weights were loaded: "
            f"{denied_loaded[:10]} (total {len(denied_loaded)})"
        )
    return report


def load_checkpoint_allowlist(
    model,
    filename: str,
    allow_modules: Sequence[str],
    logger=None,
    map_location: str = "cpu",
    strict: bool = False,
    deny_modules: Optional[Sequence[str]] = None,
    revise_keys: Optional[list] = None,
) -> dict:
    """Load checkpoint into ``model`` keeping only allowlisted module prefixes."""
    if revise_keys is None:
        revise_keys = [(r"^module\.", "")]

    checkpoint = torch.load(filename, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Apply the same revise_keys rewrite mmcv uses before filtering.
    import re

    revised = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        for pattern, repl in revise_keys:
            nk = re.sub(pattern, repl, nk)
        revised[nk] = v

    if deny_modules is None:
        # Always hard-deny planning heads when an allowlist is active.
        deny_modules = PLANNING_DENY_PREFIXES

    filtered, loaded, skipped = filter_state_dict_by_modules(
        revised, allow_modules=allow_modules, deny_modules=deny_modules
    )
    report = audit_load(
        loaded, skipped, allow_modules=allow_modules, logger=logger
    )

    load_state_dict(model, filtered, strict=strict, logger=logger)
    if logger is not None:
        logger.info(
            f"[load_from_modules] finished loading from {filename} "
            f"(strict={strict})"
        )
    else:
        print(
            f"[load_from_modules] finished loading from {filename} "
            f"(strict={strict})"
        )
    report["filename"] = filename
    return report
