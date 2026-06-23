"""
Bridge to invoke opencode CLI non-interactively for the pipeline.

Replaces the TypeScript-based run_codex.ts / @openai/codex-sdk approach.
opencode is invoked as:
    opencode run --model <model> --agent build --format json --dir <workdir>
with the prompt piped via stdin.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


def run_opencode(
    prompt: str,
    workdir: str,
    model: Optional[str] = None,
    timeout: int = 6000,
    allowed_tools: Optional[List[str]] = None,  # reserved, not used by opencode CLI
    traces_dir: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Invoke opencode CLI non-interactively with the given prompt.

    Args:
        prompt: Full prompt text. Passed via stdin so there is no length limit.
        workdir: Working directory opencode will operate in (--dir flag).
        model: Model string in opencode format 'provider/model'
               e.g. 'anthropic/claude-sonnet-4-5', 'google/gemini-1.5-pro-002'.
               Falls back to OPENCODE_MODEL env var, then CODEX_MODEL env var.
        timeout: Timeout in seconds (default: 6000).
        allowed_tools: Not used — opencode uses the 'build' agent which enables
                       all development tools (bash, file read/write, etc.).
        traces_dir: If set, write the raw NDJSON event stream to this directory.

    Returns:
        Dict with 'combined' (full transcript) and 'summary' (final text response)
        on success, or None on failure.
    """
    # Resolve model: explicit arg > OPENCODE_MODEL env > CODEX_MODEL env
    resolved_model = (
        model
        or os.environ.get("OPENCODE_MODEL")
        or os.environ.get("CODEX_MODEL")
    )

    cmd = ["opencode", "run"]
    if resolved_model:
        cmd += ["--model", resolved_model]
    # 'build' agent enables all development tools (bash, file edit, etc.) without
    # interactive confirmation — equivalent to codex's approvalPolicy='never'.
    cmd += ["--agent", "build", "--format", "json", "--dir", str(workdir)]

    env = os.environ.copy()

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        # opencode binary not found on PATH
        raise RuntimeError(
            "opencode binary not found. Install it with: "
            "curl -fsSL https://opencode.ai/install | bash"
        )
    except Exception:
        return None

    raw_stdout = result.stdout or ""
    raw_stderr = result.stderr or ""

    # Optionally save raw NDJSON event stream for analysis
    if traces_dir and raw_stdout:
        try:
            Path(traces_dir).mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            trace_path = Path(traces_dir) / f"opencode-trace-{ts}.jsonl"
            trace_path.write_text(raw_stdout, encoding="utf-8")
        except Exception:
            pass  # non-fatal

    # Parse NDJSON event stream into a human-readable transcript
    transcript_parts: List[str] = []
    last_text: str = ""
    total_tokens: Optional[int] = None

    for line in raw_stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            transcript_parts.append(line)
            continue

        etype = event.get("type", "")
        part = event.get("part", {}) or {}

        if etype == "text":
            text = part.get("text", "") if isinstance(part, dict) else ""
            if text:
                transcript_parts.append(text)
                last_text = text

        elif etype == "tool_use":
            if isinstance(part, dict):
                tool_name = part.get("tool", "tool")
                state = part.get("state") or {}
                inp = state.get("input", {})
                out = state.get("output", "")
                # Extract the primary command/input string
                if isinstance(inp, dict):
                    cmd_str = (
                        inp.get("command")
                        or inp.get("input")
                        or inp.get("path")
                        or str(inp)
                    )
                else:
                    cmd_str = str(inp) if inp else ""
                if cmd_str:
                    transcript_parts.append(f"[{tool_name}] $ {cmd_str}")
                if out:
                    transcript_parts.append(str(out))

        elif etype == "reasoning":
            text = part.get("text", "") if isinstance(part, dict) else ""
            if text:
                transcript_parts.append(f"Thinking: {text}")

        elif etype == "error":
            error = event.get("error") or {}
            msg = (
                error.get("message", str(error))
                if isinstance(error, dict)
                else str(error)
            )
            transcript_parts.append(f"Error: {msg}")

        elif etype == "step_finish":
            # step_finish may carry token usage information
            usage = (part or {}).get("usage") if isinstance(part, dict) else None
            if isinstance(usage, dict):
                for key in ("totalTokens", "total_tokens"):
                    if isinstance(usage.get(key), (int, float)):
                        total_tokens = int(usage[key])
                        break
                if total_tokens is None:
                    inp_t = usage.get("inputTokens") or usage.get("input_tokens") or 0
                    out_t = usage.get("outputTokens") or usage.get("output_tokens") or 0
                    if inp_t or out_t:
                        total_tokens = int(inp_t) + int(out_t)

        # step_start and other events are silently skipped

    combined = "\n".join(transcript_parts)

    # Fallback: if no events parsed, use raw stderr/stdout
    if not combined:
        combined = raw_stderr or raw_stdout

    # Compute summary before appending token count so the count doesn't
    # pollute the summary passed to the next pipeline step.
    summary = last_text or (combined[-2000:] if combined else "")

    # Append token count to combined transcript for logging/analysis only
    if total_tokens is not None:
        combined = combined + "\ntokens used\n" + f"{total_tokens:,}"

    # Treat a completely empty result as failure only when exit code is non-zero
    if result.returncode != 0 and not combined.strip():
        return None

    return {"combined": combined, "summary": summary}
