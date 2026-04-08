"""
Baseline inference script for DataCleaningEnv.

CRITICAL: This script must follow the exact logging format required by the hackathon:
- [START] task_id=<task> model=<model> timestamp=<iso8601>
- [STEP] step=<n> action=<json> reward=<float> score=<float> done=<bool>
- [END] task_id=<task> final_score=<float> steps=<int> duration=<float>
"""

import os
import json
import asyncio
from datetime import datetime
from openai import OpenAI
from client import DataCleaningEnvClient
from models import CleaningAction
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment variables (CRITICAL: exact names, defaults only for API_BASE_URL and MODEL_NAME)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # NO DEFAULT - must be provided

# Validate HF_TOKEN
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable must be set")

# Initialize OpenAI client with HF Inference API
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

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


async def run_task(task_id: str, difficulty: str, env_url: str = "http://localhost:7860") -> dict:
    """
    Run a single task episode.
    
    Args:
        task_id: Task identifier
        difficulty: Difficulty level
        env_url: Environment server URL
        
    Returns:
        Dictionary with task results
    """
    start_time = datetime.now()
    
    # [START] log - EXACT FORMAT REQUIRED
    print(f"[START] task_id={task_id} model={MODEL_NAME} timestamp={start_time.isoformat()}")
    
    try:
        async with DataCleaningEnvClient(env_url) as env:
            # Reset environment
            obs = await env.reset(task_id=task_id)
            
            step_num = 0
            max_steps = 20  # Limit to avoid timeout
            
            while not obs.done and step_num < max_steps:
                # Prepare prompt for LLM
                user_prompt = f"""Dataset Preview:
{obs.dataset_preview}

Column Statistics:
{json.dumps(obs.column_stats, indent=2)}

Issues Remaining:
{chr(10).join(f'- {issue}' for issue in obs.issues_remaining)}

Current Score: {obs.cumulative_reward:.3f}

What cleaning action should I take next? Output JSON only."""

                # Call LLM
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.7,
                        max_tokens=200
                    )
                    
                    action_json = response.choices[0].message.content
                    action = CleaningAction.model_validate_json(action_json)
                
                except Exception as e:
                    # Fallback: signal done if LLM fails
                    print(f"Warning: LLM call failed ({str(e)}), signaling done")
                    action = CleaningAction(operation="done")
                    action_json = json.dumps(action.model_dump())
                
                # Take step
                obs = await env.step(action)
                
                # [STEP] log - EXACT FORMAT REQUIRED
                print(f"[STEP] step={step_num} action={action_json} reward={obs.step_reward:.3f} score={obs.cumulative_reward:.3f} done={obs.done}")
                
                step_num += 1
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            final_score = obs.cumulative_reward
            
            # [END] log - EXACT FORMAT REQUIRED
            print(f"[END] task_id={task_id} final_score={final_score:.3f} steps={step_num} duration={duration:.2f}")
            
            return {
                "task_id": task_id,
                "difficulty": difficulty,
                "final_score": final_score,
                "steps": step_num,
                "duration": duration
            }
    
    except Exception as e:
        # Log error and continue
        duration = (datetime.now() - start_time).total_seconds()
        print(f"[END] task_id={task_id} final_score=0.000 steps=0 duration={duration:.2f}")
        print(f"Error running task {task_id}: {str(e)}")
        return {
            "task_id": task_id,
            "difficulty": difficulty,
            "final_score": 0.0,
            "steps": 0,
            "duration": duration,
            "error": str(e)
        }


async def main():
    """Run inference on all tasks."""
    print("=" * 80)
    print("DataCleaningEnv - Baseline Inference")
    print("=" * 80)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"HF Token: {'***' if HF_TOKEN else 'NOT SET'}")
    print("=" * 80)
    print()
    
    # Define tasks
    tasks = [
        ("easy_nulls", "easy"),
        ("medium_formats", "medium"),
        ("hard_multitable", "hard"),
    ]
    
    results = []
    
    # Run each task
    for task_id, difficulty in tasks:
        print()
        result = await run_task(task_id, difficulty)
        results.append(result)
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for result in results:
        print(f"{result['difficulty'].upper():8} | {result['task_id']:20} | Score: {result['final_score']:6.3f} | Steps: {result['steps']:2} | Time: {result['duration']:6.2f}s")
    
    # Calculate average
    avg_score = sum(r['final_score'] for r in results) / len(results)
    total_time = sum(r['duration'] for r in results)
    print("=" * 80)
    print(f"Average Score: {avg_score:.3f}")
    print(f"Total Time: {total_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
