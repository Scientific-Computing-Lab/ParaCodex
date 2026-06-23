"""
Bridge to invoke the OpenAI Codex TypeScript SDK non-interactively.

Calls: npx ts-node --esm utils/run_codex.ts <prompt_file> <workdir>
with the prompt written to a temporary file.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional


def run_codex_ts(
    prompt: str,
    workdir: str,
    model: Optional[str] = None,
    timeout: int = 6000,
    allowed_tools: Optional[List[str]] = None,  # reserved, unused
    traces_dir: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Invoke the OpenAI Codex TypeScript SDK non-interactively.

    Args:
        prompt: Full prompt text. Written to a temp file and passed to the TS bridge.
        workdir: Working directory for the Codex agent.
        model: Model name in Codex format (bare, e.g. 'o3', 'gpt-4o').
               Falls back to CODEX_MODEL env var.
        timeout: Timeout in seconds (default: 6000).
        allowed_tools: Not used by this bridge (Codex tool set is fixed).
        traces_dir: If set, traces are written here via CODEX_TRACE_OUTPUT.

    Returns:
        Dict with 'combined' (full transcript) and 'summary' (final response)
        on success, or None on failure.
    """
    pipeline_root = Path(__file__).parent.parent
    ts_bridge = pipeline_root / "utils" / "run_codex.ts"

    if not ts_bridge.exists():
        raise RuntimeError(
            f"Codex TypeScript bridge not found: {ts_bridge}\n"
            "Ensure utils/run_codex.ts is present and run: npm install"
        )

    env = os.environ.copy()

    # Resolve model
    resolved_model = model or env.get("CODEX_MODEL")
    if resolved_model:
        env["CODEX_MODEL"] = resolved_model

    # Set trace output directory
    if traces_dir:
        env["CODEX_TRACE_OUTPUT"] = str(traces_dir)

    # Write prompt to a temp file (avoids shell escaping / length limits)
    prompt_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(prompt)
            prompt_file = f.name

        cmd = [
            "npx", "ts-node", "--esm",
            str(ts_bridge),
            prompt_file,
            str(workdir),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(pipeline_root),
        )
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        raise RuntimeError(
            "npx not found. Install Node.js and run: npm install\n"
            "(in the pipeline_refactored directory)"
        )
    except Exception:
        return None
    finally:
        if prompt_file:
            try:
                os.unlink(prompt_file)
            except Exception:
                pass

    raw_stdout = result.stdout or ""
    raw_stderr = result.stderr or ""

    # Optionally save raw output for analysis
    if traces_dir and raw_stdout:
        try:
            Path(traces_dir).mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            trace_path = Path(traces_dir) / f"codex-trace-{ts}.json"
            trace_path.write_text(raw_stdout, encoding="utf-8")
        except Exception:
            pass

    # Parse Codex SDK JSON output into a human-readable transcript
    transcript_parts: List[str] = []
    last_text = ""
    total_tokens: Optional[int] = None

    try:
        data = json.loads(raw_stdout)

        # The SDK result may nest items under various keys
        output_items = (
            data.get("output_items")
            or data.get("items")
            or data.get("messages")
            or []
        )

        for item in output_items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")

            # Codex SDK native types
            if item_type == "agent_message":
                text = item.get("text", "")
                if text:
                    transcript_parts.append(text)
                    last_text = text

            elif item_type == "command_execution":
                cmd = item.get("command", "")
                out = item.get("aggregated_output", "") or item.get("output", "")
                if cmd:
                    transcript_parts.append(f"[shell] $ {cmd}")
                if out:
                    transcript_parts.append(str(out))

            elif item_type == "file_change":
                changes = item.get("changes", [])
                for ch in changes:
                    path = ch.get("path", "")
                    kind = ch.get("kind", "change")
                    if path:
                        transcript_parts.append(f"[file_{kind}] {path}")

            # Legacy / opencode types
            elif item_type == "message":
                for part in item.get("content", []):
                    if isinstance(part, dict):
                        text = (
                            part.get("text")
                            or part.get("output_text")
                            or part.get("value", "")
                        )
                    else:
                        text = str(part)
                    if text:
                        transcript_parts.append(text)
                        last_text = text

            elif item_type in ("function_call", "tool_call", "tool_use"):
                name = item.get("name") or item.get("call_type", "tool")
                args = item.get("arguments") or item.get("input") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                if isinstance(args, dict):
                    cmd_str = (
                        args.get("command")
                        or args.get("input")
                        or args.get("path")
                        or str(args)
                    )
                else:
                    cmd_str = str(args) if args else ""
                out = item.get("output", "")
                if cmd_str:
                    transcript_parts.append(f"[{name}] $ {cmd_str}")
                if out:
                    transcript_parts.append(str(out))

            elif item_type in ("reasoning", "thinking"):
                text = item.get("text") or item.get("summary", "")
                if isinstance(text, list):
                    text = " ".join(
                        p.get("text", "") for p in text if isinstance(p, dict)
                    )
                if text:
                    transcript_parts.append(f"Thinking: {text}")

        # finalResponse is a top-level summary string from the Codex SDK
        final_response = data.get("finalResponse", "")
        if final_response and isinstance(final_response, str):
            last_text = final_response

        # Token usage
        usage = data.get("usage") or {}
        if isinstance(usage, dict):
            total_tokens = usage.get("total_tokens") or usage.get("totalTokens")
            if total_tokens is None:
                inp_t = usage.get("input_tokens") or usage.get("inputTokens") or 0
                out_t = usage.get("output_tokens") or usage.get("outputTokens") or 0
                if inp_t or out_t:
                    total_tokens = int(inp_t) + int(out_t)

    except (json.JSONDecodeError, AttributeError):
        # Not JSON — fall back to raw output
        if raw_stdout.strip():
            transcript_parts = [raw_stdout]

    combined = "\n".join(transcript_parts)

    if not combined:
        combined = raw_stderr or raw_stdout

    # Compute summary before appending token count so the count doesn't
    # pollute the summary passed to the next pipeline step.
    summary = last_text or (combined[-2000:] if combined else "")

    # Append token count to combined transcript for logging/analysis only
    if total_tokens is not None:
        combined = combined + "\ntokens used\n" + f"{total_tokens:,}"

    if result.returncode != 0 and not combined.strip():
        return None

    return {"combined": combined, "summary": summary}
