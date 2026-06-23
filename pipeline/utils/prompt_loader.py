"""
Utility functions for loading prompts from markdown files.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Iterable

from .config import get_config

# PROMPTS_DIR is relative to the pipeline_refactored directory, not utils
PROMPTS_DIR = get_config().pipeline_root / "prompts"
SKILL_PROMPTS_DIR = Path(
    os.environ.get("PARACODEX_SKILLS_DIR", str(Path.home() / ".codex" / "skills" / "paracodex"))
)


def normalize_api_name(api: str) -> str:
    """Normalize API names for consistency.
    
    Args:
        api: API name (e.g., 'ocl', 'opencl', 'omp', 'cuda')
        
    Returns:
        Normalized API name ('ocl' for 'ocl'/'opencl', others unchanged)
    """
    api_lower = api.lower()
    if api_lower in ['ocl', 'opencl']:
        return 'ocl'
    return api_lower


def load_prompt_from_file(prompt_file: str) -> str:
    """Load prompt content from a markdown file in the prompts directory.
    
    Args:
        prompt_file: Name of the prompt file (e.g., 'serial_omp_step1.md')
        
    Returns:
        The content of the prompt file as a string
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    skill_suffix = Path(prompt_file).stem.replace("_", "-")
    skill_name = f"paracodex-{skill_suffix}"
    skill_path = SKILL_PROMPTS_DIR / skill_name / "SKILL.md"
    if skill_path.exists():
        text = skill_path.read_text()
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                return parts[2].lstrip("\n")
        return text

    skill_prompt_path = SKILL_PROMPTS_DIR / skill_name / "references" / "prompt.md"
    if skill_prompt_path.exists():
        return skill_prompt_path.read_text()

    prompt_path = PROMPTS_DIR / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text()


def _format_var_lines(variables: Dict[str, str]) -> str:
    lines = []
    for key, value in variables.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def build_skill_trigger_prompt(
    skill_name: str,
    task: str,
    workdir: Path,
    source_dir: Optional[Path],
    target_dir: Optional[Path],
    file_listing: str,
    variables: Dict[str, str],
    notes: Optional[Iterable[str]] = None,
) -> str:
    """Build a minimal outer prompt that triggers a skill and supplies variables."""
    skill_root = SKILL_PROMPTS_DIR
    skill_path = skill_root / skill_name / "SKILL.md"
    lines = [
        f"Task: {task}",
        "",
        f"Workdir: {workdir}",
        f"Skill root: {skill_root}",
        f"Skill file: {skill_path}",
    ]
    if source_dir is not None:
        lines.append(f"Source dir: {source_dir}")
    if target_dir is not None:
        lines.append(f"Target dir: {target_dir}")
    lines.append(f"Files: {file_listing}")
    lines.append("")
    if notes:
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    if variables:
        lines.append("Variables:")
        lines.append(_format_var_lines(variables))
        lines.append("")
    lines.append(f"${{{skill_name}}}")
    return "\n".join(lines)


def get_prompt_filename(source_api: str, target_api: str, step: str) -> str:
    """Get the prompt filename based on source/target API and step.
    
    Args:
        source_api: Source API (e.g., 'serial', 'cuda', 'opencl', 'ocl')
        target_api: Target API (e.g., 'omp', 'cuda', 'opencl', 'ocl')
        step: Step identifier (e.g., 'analysis', 'step1', 'step2')
        
    Returns:
        Prompt filename (e.g., 'serial_omp_step1.md')
    """
    source_normalized = normalize_api_name(source_api)
    target_normalized = normalize_api_name(target_api)
    return f"{source_normalized}_{target_normalized}_{step}.md"


def load_translation_prompt(source_api: str, target_api: str) -> str:
    """Load the initial translation/analysis prompt.
    
    Args:
        source_api: Source API (e.g., 'serial', 'cuda', 'opencl', 'ocl')
        target_api: Target API (e.g., 'omp', 'cuda', 'opencl', 'ocl')
        
    Returns:
        The analysis/translation prompt content
    """
    prompt_file = get_prompt_filename(source_api, target_api, 'analysis')
    return load_prompt_from_file(prompt_file)


def load_optimization_prompt(source_api: str, target_api: str, step: int) -> str:
    """Load an optimization step prompt.
    
    Args:
        source_api: Source API (e.g., 'serial', 'cuda', 'opencl', 'ocl')
        target_api: Target API (e.g., 'omp', 'cuda', 'opencl', 'ocl')
        step: Step number (1 or 2)
        
    Returns:
        The optimization step prompt content
    """
    prompt_file = get_prompt_filename(source_api, target_api, f'step{step}')
    return load_prompt_from_file(prompt_file)
