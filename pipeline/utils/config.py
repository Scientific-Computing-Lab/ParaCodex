"""
Centralized configuration management for the pipeline.

Replaces hard-coded paths with configurable values via environment variables
or configuration files. Provides sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional


class PipelineConfig:
    """Centralized configuration for the pipeline."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # Auto-detect base directory: go up from utils/ to codex_baseline/
        # This file is in pipeline_refactored/utils/, so parent.parent = codex_baseline
        _this_file = Path(__file__).resolve()
        _default_base = _this_file.parent.parent  # pipeline_refactored/utils -> pipeline_refactored -> codex_baseline
        
        # Base paths - can be overridden via environment variables
        self._base_dir = Path(
            os.environ.get("CODEX_BASELINE_DIR", str(_default_base))
        ).resolve()
        
        # Pipeline root (pipeline_refactored directory)
        self._pipeline_root = _this_file.parent.parent.resolve()
        
        # Codex workdir - main working directory
        self._codex_workdir = Path(
            os.environ.get("CODEX_WORKDIR", str(self._base_dir / "cuda_omp_workdir"))
        ).resolve()
        
        # Gate SDK directory
        self._gate_sdk_dir = Path(
            os.environ.get("GATE_SDK_DIR", str(self._codex_workdir / "gate_sdk"))
        ).resolve()
        
        # NAS identifier for special handling
        self.nas_identifier = os.environ.get("NAS_IDENTIFIER", "serial_omp_nas_workdir")
        
        # Compiler settings
        self.nas_cc = os.environ.get("NAS_CC", "nvc++")
        self.nas_class = os.environ.get("NAS_CLASS", "B")
        
        # GPU timeout settings
        self.gpu_timeout_seconds = int(
            os.environ.get("GPU_TIMEOUT_SECONDS", "300")
        )
        
        # Performance gate factor
        self.performance_gate_factor = float(
            os.environ.get("PERFORMANCE_GATE_FACTOR", "1.1")
        )
        
        # Default JSONL filename
        self.default_jsonl_filename = os.environ.get(
            "DEFAULT_JSONL_FILENAME", "paratrans_serial.jsonl"
        )
    
    @property
    def base_dir(self) -> Path:
        """Base directory for the codex baseline project."""
        return self._base_dir
    
    @property
    def pipeline_root(self) -> Path:
        """Pipeline refactored directory (pipeline_refactored/)."""
        return self._pipeline_root
    
    @property
    def codex_workdir(self) -> Path:
        """Codex working directory."""
        return self._codex_workdir
    
    @codex_workdir.setter
    def codex_workdir(self, path: Path | str):
        """Set codex workdir and update environment variable."""
        resolved = Path(path).expanduser().resolve()
        self._codex_workdir = resolved
        os.environ["CODEX_WORKDIR"] = str(resolved)
    
    @property
    def gate_sdk_dir(self) -> Path:
        """Gate SDK directory."""
        return self._gate_sdk_dir
    
    @gate_sdk_dir.setter
    def gate_sdk_dir(self, path: Path | str):
        """Set gate SDK directory."""
        self._gate_sdk_dir = Path(path).expanduser().resolve()
        os.environ["GATE_SDK_DIR"] = str(self._gate_sdk_dir)
    
    def data_src(self) -> Path:
        """Data source directory where translated code goes (data/src)."""
        return self.codex_workdir / "data" / "src"
    
    def golden_root(self) -> Path:
        """Golden labels root under the current workdir."""
        return self.codex_workdir / "golden_labels" / "src"
    
    def default_jsonl(self, filename: Optional[str] = None) -> Path:
        """Default JSONL file path.
        
        Args:
            filename: Optional filename override
            
        Returns:
            Path to JSONL file
        """
        if filename is None:
            filename = self.default_jsonl_filename
        return self.codex_workdir / filename
    
    def is_nas_workdir(self) -> bool:
        """Check if current workdir is a NAS workdir."""
        return self.nas_identifier in str(self.codex_workdir)


# Global configuration instance
_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """Get the global configuration instance (singleton pattern).
    
    Returns:
        PipelineConfig instance
    """
    global _config
    if _config is None:
        _config = PipelineConfig()
    return _config


def reset_config():
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
