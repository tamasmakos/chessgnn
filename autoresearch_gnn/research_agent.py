"""
autoresearch_gnn/research_agent.py

Autonomous research agent that runs the autoresearch_gnn improvement loop.

The agent uses Groq's tool-use API (stateless per iteration) to:
  1. Read the current train_gnn.py and results.tsv
  2. Propose and apply one change
  3. Run the 10-minute experiment
  4. Record the result and decide keep/revert
  5. Repeat forever (or until --max-experiments is reached)

Usage:
    cd /workspaces/chessgnn
    export GROQ_API_KEY=...   # or put in .env
    python autoresearch_gnn/research_agent.py
    python autoresearch_gnn/research_agent.py --max-experiments 5
    python autoresearch_gnn/research_agent.py --model moonsong-labs/mistral-nemo
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from groq import Groq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_WORKSPACE  = Path(__file__).parent.parent
_TRAIN      = _WORKSPACE / "autoresearch_gnn" / "train_gnn.py"
_PREPARE    = _WORKSPACE / "autoresearch_gnn" / "prepare_gnn.py"
_RESULTS    = _WORKSPACE / "autoresearch_gnn" / "results.tsv"
_PROGRAM    = _WORKSPACE / "autoresearch_gnn" / "program_gnn.md"

# ---------------------------------------------------------------------------
# Tool implementations (pure Python, no LLM)
# ---------------------------------------------------------------------------

def _tool_read_train_file() -> str:
    return _TRAIN.read_text()


def _tool_write_train_file(content: str) -> str:
    _TRAIN.write_text(content)
    return "train_gnn.py written successfully."


def _tool_read_results() -> str:
    if not _RESULTS.exists():
        return "results.tsv does not exist yet."
    return _RESULTS.read_text()


def _tool_append_result(row: str) -> str:
    """Append one TSV row (no newline needed — we add it)."""
    with open(_RESULTS, "a") as f:
        f.write(row.rstrip("\n") + "\n")
    return "Row appended to results.tsv."


def _tool_run_experiment() -> str:
    """Run train_gnn.py, return only the summary block (last ~10 lines after ---)."""
    print("  [agent] Running experiment — this takes ~10 minutes…", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "autoresearch_gnn/train_gnn.py"],
        capture_output=True, text=True, cwd=str(_WORKSPACE),
    )
    elapsed = time.time() - t0
    print(f"  [agent] Experiment finished in {elapsed:.0f}s  (exit={result.returncode})", flush=True)

    if result.returncode != 0:
        stderr_tail = result.stderr[-1000:] if result.stderr else ""
        return f"CRASH (exit {result.returncode})\nstderr tail:\n{stderr_tail}"

    # Return only the --- summary block so we don't flood the context
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "---":
            return "\n".join(lines[i:])
    # Fallback: last 1500 chars
    return result.stdout[-1500:]


def _tool_git_commit(message: str) -> str:
    subprocess.run(["git", "add", str(_TRAIN)], cwd=str(_WORKSPACE), capture_output=True)
    r = subprocess.run(
        ["git", "commit", "-m", message],
        capture_output=True, text=True, cwd=str(_WORKSPACE),
    )
    short = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, cwd=str(_WORKSPACE),
    ).stdout.strip()
    return f"Committed: {short}\n{r.stdout}{r.stderr}"


def _tool_git_revert() -> str:
    r = subprocess.run(
        ["git", "revert", "HEAD", "--no-edit"],
        capture_output=True, text=True, cwd=str(_WORKSPACE),
    )
    return r.stdout + r.stderr


def _tool_git_log() -> str:
    r = subprocess.run(
        ["git", "log", "--oneline", "-10"],
        capture_output=True, text=True, cwd=str(_WORKSPACE),
    )
    return r.stdout


# ---------------------------------------------------------------------------
# Tool registry — Groq function-calling schema
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_train_file",
            "description": "Read the current contents of train_gnn.py (the editable model file).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_train_file",
            "description": "Overwrite train_gnn.py with new content. Use this to apply your proposed change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The complete new content of train_gnn.py.",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_results",
            "description": "Read results.tsv — the history of all experiments with their top1_agreement scores.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_result",
            "description": "Append one tab-separated row to results.tsv. Format: commit<TAB>top1_agreement<TAB>memory_gb<TAB>status<TAB>description",
            "parameters": {
                "type": "object",
                "properties": {
                    "row": {
                        "type": "string",
                        "description": "One TSV row, e.g. 'a1b2c3d\\t0.349000\\t0.4\\tkeep\\tincrease LR to 0.01'",
                    }
                },
                "required": ["row"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_experiment",
            "description": "Run train_gnn.py for the full 10-minute budget. Returns the summary block with top1_agreement and peak_vram_mb.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Stage train_gnn.py and create a git commit with the given message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Short commit message describing the experiment.",
                    }
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_revert",
            "description": "Revert the last commit (undo the last change to train_gnn.py) if the experiment did not improve top1_agreement.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show the last 10 git commits for context.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

TOOL_DISPATCH = {
    "read_train_file":  lambda args: _tool_read_train_file(),
    "write_train_file": lambda args: _tool_write_train_file(args["content"]),
    "read_results":     lambda args: _tool_read_results(),
    "append_result":    lambda args: _tool_append_result(args["row"]),
    "run_experiment":   lambda args: _tool_run_experiment(),
    "git_commit":       lambda args: _tool_git_commit(args["message"]),
    "git_revert":       lambda args: _tool_git_revert(),
    "git_log":          lambda args: _tool_git_log(),
}

# ---------------------------------------------------------------------------
# System prompt — injects program_gnn.md + stateless iteration instructions
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""\
{_PROGRAM.read_text()}

---

## Agent operating instructions

You run ONE experiment per reply. The loop is:

1. Call read_results to see the history and find the current best top1_agreement.
2. Call read_train_file to see the current code.
3. Decide on ONE change (architecture, optimizer, hyperparams, training loop).
   - Reason briefly about WHY this change might help.
   - Prefer simple, targeted changes over large rewrites.
4. Call write_train_file with the complete modified file.
5. Call git_commit with a short description of the change.
6. Call run_experiment. Wait for the result.
7. Parse top1_agreement from the summary block.
8. Compare to the previous best in results.tsv:
   - If improved: status = "keep". Append the result row.
   - If not improved: status = "discard". Call git_revert. Append the result row.
9. Call append_result to log the row.
10. End your reply with a one-sentence plan for the next iteration.

CRITICAL RULES:
- Never modify prepare_gnn.py.
- Never change the output format (the --- summary block keys).
- The experiment takes ~10 minutes — be patient after calling run_experiment.
- After git_revert, the file is restored; future reads will see the reverted code.
- top1_agreement is HIGHER IS BETTER (unlike val_bpb in autoresearch).
- Always include the 7-char commit hash in the TSV row.
"""

# ---------------------------------------------------------------------------
# Agentic loop — stateless per iteration
# ---------------------------------------------------------------------------

def run_one_iteration(client: Groq, model: str, iteration: int) -> str:
    """Run a single research iteration: propose → edit → run → record.

    Returns the assistant's final text reply for that iteration.
    """
    print(f"\n{'='*60}")
    print(f"  ITERATION {iteration}")
    print(f"{'='*60}\n")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"This is iteration {iteration}. "
                "Run one full experiment cycle: read the current state, "
                "propose and apply one improvement to train_gnn.py, "
                "run the experiment, and record the result. "
                "Follow the loop in the instructions exactly."
            ),
        },
    ]

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=1.0,
            max_tokens=8192,
        )

        msg = response.choices[0].message
        finish = response.choices[0].finish_reason

        # Append assistant turn to history
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (msg.tool_calls or [])
        ] or None})

        if finish == "stop" or not msg.tool_calls:
            # Agent finished the iteration
            return msg.content or ""

        # Dispatch tool calls
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}

            arg_summary = ', '.join(f'{k}={repr(v)[:60]}' for k, v in args.items()) if args else ""
            print(f"  [tool] {name}({arg_summary})", flush=True)

            try:
                result = TOOL_DISPATCH[name](args)
            except KeyError:
                result = f"Unknown tool: {name}"
            except Exception as e:
                result = f"Tool error: {e}"

            # Truncate very large results to avoid filling the context window
            if name == "read_train_file" and len(result) > 6000:
                result = result[:6000] + "\n... [truncated]"

            print(f"    → {str(result)[:200]}", flush=True)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })


def main():
    parser = argparse.ArgumentParser(description="Autonomous GNN research agent")
    parser.add_argument("--max-experiments", type=int, default=0,
                        help="Stop after N experiments (0 = run forever)")
    parser.add_argument("--model", default="llama-3.3-70b-versatile",
                        help="Groq model name")
    args = parser.parse_args()

    # Load .env if present
    env_file = _WORKSPACE / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"'))

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    client = Groq(api_key=api_key)

    print(f"ChessGNN Autoresearch Agent")
    print(f"Model:           {args.model}")
    print(f"Max experiments: {args.max_experiments or 'unlimited'}")
    print(f"Results TSV:     {_RESULTS}")
    print()

    iteration = 1
    while True:
        reply = run_one_iteration(client, args.model, iteration)
        print(f"\n[Agent reply]\n{reply}\n")

        # Plot after each iteration if matplotlib is available
        try:
            from autoresearch_gnn.plot_progress import plot
            plot(str(_RESULTS), str(_WORKSPACE / "autoresearch_gnn" / "progress.png"))
        except Exception:
            pass

        if args.max_experiments and iteration >= args.max_experiments:
            print(f"Reached max experiments ({args.max_experiments}). Stopping.")
            break
        iteration += 1


if __name__ == "__main__":
    main()
