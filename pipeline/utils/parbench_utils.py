#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Any, Optional

def get_parbench_root() -> Path:
    """Get the ParBench root directory."""
    # 1. Check environment variable
    env_root = os.environ.get("PARBENCH_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # 2. Check candidate locations
    candidates = [
        "/root/parbench",
        "/root/codex_baseline/parbench",
        str(Path(__file__).parent.parent.parent / "parbench")
    ]
    
    for cand in candidates:
        cand_path = Path(cand).resolve()
        if (cand_path / "config" / "paths.json").exists():
            return cand_path

    # Fallback to the most likely legacy location
    return Path("/root/codex_baseline/parbench").resolve()

def load_parbench_config() -> dict[str, Any]:
    """Load config/paths.json from ParBench root."""
    root = get_parbench_root()
    config_path = root / "config" / "paths.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def resolve_source_dir(spec_data: dict[str, Any], parbench_spec_path: Optional[str] = None) -> Optional[str]:
    """
    Resolve the source directory for a ParBench spec using the project's config.
    """
    config = load_parbench_config()
    downloads_root = Path(config.get("downloads_root", "/root/codex_baseline")).resolve()
    
    provenance = spec_data.get("provenance", {})
    repo_root_rel = provenance.get("repo_root", "")
    source_path_rel = provenance.get("source_path", "")
    
    if not source_path_rel:
        return None
    
    # If absolute in spec, use it
    if Path(source_path_rel).is_absolute():
        return source_path_rel

    # Collect possible roots from config
    possible_roots = []
    
    # If the spec is from HeCBench, use hecbench_root from config if available
    if "hecbench" in repo_root_rel.lower() and config.get("hecbench_root"):
        possible_roots.append(Path(config["hecbench_root"]).resolve())
        
    # If the spec is from Rodinia, use rodinia_root from config if available
    if "rodinia" in repo_root_rel.lower() and config.get("rodinia_root"):
        possible_roots.append(Path(config["rodinia_root"]).resolve())

    # Always fallback to the standard downloads_root approach
    possible_roots.append((downloads_root / repo_root_rel).resolve())
    
    # Also add downloads_root directly in case source_path_rel is relative to it
    possible_roots.append(downloads_root)

    # Search through possible roots
    for root in possible_roots:
        candidate = (root / source_path_rel).resolve()
        if candidate.exists():
            return str(candidate)

    # Fallback to parbench spec relative (legacy search)
    if parbench_spec_path:
        spec_path = Path(parbench_spec_path).resolve()
        parbench_root = spec_path.parent.parent
        candidate = (parbench_root / source_path_rel).resolve()
        if candidate.exists():
            return str(candidate)

    return None
