"""FastAPI server for DataCleaningEnv."""

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import ValidationError

from models import (
    CleaningAction,
    CleaningObservation,
    CleaningState,
    EnvResponse,
    ResetRequest,
)
from server.environment import DataCleaningEnvironment


# Global environment instance
env_instance: DataCleaningEnvironment | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    print("DataCleaningEnv server starting...")
    yield
    # Shutdown
    print("DataCleaningEnv server shutting down...")


app = FastAPI(
    title="DataCleaningEnv",
    description="Real-world data cleaning environment for agent training",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "DataCleaningEnv",
        "version": "1.0.0",
        "tasks": ["easy_nulls", "medium_formats", "hard_multitable"]
    }


@app.post("/reset", response_model=EnvResponse)
def reset(request: ResetRequest | None = Body(default=None)):
    """
    Reset the environment with a new task.
    
    Args:
        request: Optional reset request with task_id and optional difficulty
        
    Returns:
        Initial observation wrapped in OpenEnv response shape
    """
    global env_instance
    
    try:
        req = request or ResetRequest()
        env_instance = DataCleaningEnvironment(
            task_id=req.task_id,
            difficulty=req.difficulty,
            max_steps=20
        )
        observation = env_instance.reset()
        return EnvResponse(
            observation=observation.model_dump(),
            reward=None,
            done=observation.done
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to reset environment: {str(e)}")


@app.post("/step", response_model=EnvResponse)
def step(payload: dict = Body(...)):
    """
    Take a step in the environment.
    
    Args:
        payload: Either direct action payload or OpenEnv style {"action": {...}}
        
    Returns:
        Observation wrapped in OpenEnv response shape
    """
    global env_instance
    
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        # Support both direct body {"operation": ...} and wrapped {"action": {...}}
        action_payload = payload.get("action", payload)
        action = CleaningAction.model_validate(action_payload)
        observation = env_instance.step(action)
        return EnvResponse(
            observation=observation.model_dump(),
            reward=observation.step_reward,
            done=observation.done
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to step environment: {str(e)}")


@app.get("/state", response_model=CleaningState)
def get_state():
    """
    Get the current environment state.
    
    Returns:
        Current state
    """
    global env_instance
    
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        return env_instance.state
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get state: {str(e)}")


@app.get("/tasks")
def list_tasks():
    """
    List available tasks.
    
    Returns:
        Dictionary of available tasks with metadata
    """
    return {
        "tasks": [
            {
                "id": "easy_nulls",
                "difficulty": "easy",
                "description": "Fix missing values and duplicates in a 50-row dataset",
                "expected_score_range": [0.85, 1.0],
                "max_steps": 20
            },
            {
                "id": "medium_formats",
                "difficulty": "medium",
                "description": "Normalize dates, phones, and categories in a 200-row dataset",
                "expected_score_range": [0.6, 0.85],
                "max_steps": 20
            },
            {
                "id": "hard_multitable",
                "difficulty": "hard",
                "description": "Fix types, outliers, and hidden nulls in a 500-row dataset",
                "expected_score_range": [0.3, 0.65],
                "max_steps": 20
            }
        ]
    }


@app.get("/schema")
def get_schema():
    """Return action/observation/state JSON schemas."""
    return {
        "action": CleaningAction.model_json_schema(),
        "observation": CleaningObservation.model_json_schema(),
        "state": CleaningState.model_json_schema()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
