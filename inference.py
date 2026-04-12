"""
Baseline inference script for DataCleaningEnv.

Stdout contract per episode:
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
from typing import Any, Optional

try:
    from dotenv import load_dotenv
except Exception as exc:  # pragma: no cover - optional dependency fallback
    DOTENV_IMPORT_ERROR = str(exc)

    def load_dotenv() -> bool:
        return False
else:
    DOTENV_IMPORT_ERROR = None

try:
    from client import DataCleaningEnvClient
    from models import CleaningAction, CleaningObservation
except Exception as exc:  # pragma: no cover - optional dependency fallback
    DataCleaningEnvClient = None
    CleaningAction = None
    CleaningObservation = Any
    CLIENT_IMPORT_ERROR = str(exc)
else:
    CLIENT_IMPORT_ERROR = None

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - optional dependency fallback
    OpenAI = None
    OPENAI_IMPORT_ERROR = str(exc)
else:
    OPENAI_IMPORT_ERROR = None

# Load environment variables
load_dotenv()

# Required hackathon environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional runtime controls
BENCHMARK_NAME = os.getenv("BENCHMARK_NAME", "data-cleaning-env")
ENV_URL = (
    os.getenv("ENV_URL")
    or os.getenv("OPENENV_ENV_URL")
    or os.getenv("OPENENV_URL")
    or "http://localhost:7860"
)
TASK_IDS = [
    task.strip()
    for task in os.getenv("TASK_IDS", "easy_nulls,medium_formats,hard_multitable").split(",")
    if task.strip()
]

def _safe_int(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


MAX_STEPS = _safe_int(os.getenv("MAX_STEPS"), 20)


def _init_llm_client() -> tuple[Optional[Any], Optional[str]]:
    if OpenAI is None:
        return None, f"openai import failed: {OPENAI_IMPORT_ERROR}"
    if not HF_TOKEN:
        return None, "HF_TOKEN environment variable is required"
    try:
        return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN), None
    except Exception as exc:
        return None, str(exc)


llm_client, llm_init_error = _init_llm_client()
dependency_error = "; ".join(
    error for error in [DOTENV_IMPORT_ERROR, CLIENT_IMPORT_ERROR, llm_init_error] if error
) or None

# System prompt for the LLM
SYSTEM_PROMPT = """You are a data cleaning agent. You receive a dataset preview and a list of data quality issues.

Your task is to output a JSON CleaningAction to fix ONE issue at a time.

Available operations:
- fill_missing: Fill missing values in a column (params: column, value)
- drop_duplicates: Remove duplicate rows (no params)
- normalize_dates: Convert dates to YYYY-MM-DD format (params: column, date_format)
- fix_types: Convert column to correct data type (params: column)
- remove_outliers: Remove rows with outliers (params: column, bounds=[min, max])
- standardize_categorical: Standardize category values (params: column, mapping={old: new})
- done: Signal that cleaning is complete (no params)

Output ONLY valid JSON matching this schema:
{
  "operation": "fill_missing",
  "column": "age",
  "value": "30"
}

Be strategic: tackle the most impactful issues first."""


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _sanitize_single_line(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ").strip()


def _format_error(value: Optional[str]) -> str:
    if not value:
        return "null"
    return _sanitize_single_line(value)


def _done_action() -> Any:
    if CleaningAction is None:
        return {"operation": "done"}
    return CleaningAction(operation="done")


def _action_to_str(action: Any) -> str:
    if hasattr(action, "model_dump"):
        payload = action.model_dump(exclude_none=True)
    elif isinstance(action, dict):
        payload = action
    else:
        payload = {"operation": "done"}
    return json.dumps(payload, separators=(",", ":"))


def _message_to_error(message: str) -> Optional[str]:
    lowered = message.lower()
    if lowered.startswith(("invalid", "failed", "error")):
        return message
    return None


def _build_user_prompt(obs: Any) -> str:
    issues_remaining = getattr(obs, "issues_remaining", []) or []
    issues = "\n".join(f"- {issue}" for issue in issues_remaining)
    dataset_preview = getattr(obs, "dataset_preview", "")
    column_stats = getattr(obs, "column_stats", {})
    cumulative_reward = float(getattr(obs, "cumulative_reward", 0.0))
    return (
        f"Dataset Preview:\n{dataset_preview}\n\n"
        f"Column Statistics:\n{json.dumps(column_stats, indent=2)}\n\n"
        f"Issues Remaining:\n{issues}\n\n"
        f"Current Score: {cumulative_reward:.3f}\n\n"
        "What cleaning action should I take next? Output JSON only."
    )


def _choose_action(obs: Any) -> tuple[Any, str, Optional[str]]:
    if llm_client is None:
        fallback_action = _done_action()
        return fallback_action, _action_to_str(fallback_action), llm_init_error

    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(obs)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=200,
        )
        raw_action = response.choices[0].message.content or '{"operation":"done"}'
        if CleaningAction is None:
            action = json.loads(raw_action)
            if not isinstance(action, dict) or "operation" not in action:
                action = {"operation": "done"}
        else:
            action = CleaningAction.model_validate_json(raw_action)
        return action, _action_to_str(action), None
    except Exception as exc:
        fallback_action = _done_action()
        return fallback_action, _action_to_str(fallback_action), str(exc)


async def run_task(task_id: str, env_url: str) -> dict:
    rewards: list[float] = []
    steps = 0
    done = False
    success = False

    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={MODEL_NAME}")

    try:
        if DataCleaningEnvClient is None:
            raise RuntimeError(dependency_error or "DataCleaningEnvClient import failed")

        async with DataCleaningEnvClient(env_url) as env:
            obs = await env.reset(task_id=task_id)
            done = obs.done

            while not done and steps < MAX_STEPS:
                action, action_str, action_error = _choose_action(obs)
                obs = await env.step(action)

                steps += 1
                done = obs.done
                rewards.append(obs.step_reward)

                last_action_error = action_error or _message_to_error(obs.message)
                print(
                    f"[STEP] step={steps} action={action_str} "
                    f"reward={obs.step_reward:.2f} done={_bool_str(done)} "
                    f"error={_format_error(last_action_error)}"
                )

            success = done
    except Exception as exc:
        print(f"inference_error task={task_id} error={_sanitize_single_line(str(exc))}", file=sys.stderr)
        success = False
    finally:
        rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
        print(f"[END] success={_bool_str(success)} steps={steps} rewards={rewards_str}")

    return {
        "task_id": task_id,
        "success": success,
        "steps": steps,
        "rewards": rewards,
    }


async def main() -> None:
    if dependency_error:
        print(f"inference_dependency_error error={_sanitize_single_line(dependency_error)}", file=sys.stderr)
    for task_id in TASK_IDS:
        await run_task(task_id=task_id, env_url=ENV_URL)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BaseException as exc:
        print(f"inference_fatal error={_sanitize_single_line(str(exc))}", file=sys.stderr)
    finally:
        sys.exit(0)
