"""FastAPI server for DataCleaningEnv."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from models import CleaningAction, CleaningObservation, CleaningState, ResetRequest
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


@app.post("/reset", response_model=CleaningObservation)
def reset(request: ResetRequest):
    """
    Reset the environment with a new task.
    
    Args:
        request: Reset request with task_id and optional difficulty
        
    Returns:
        Initial observation
    """
    global env_instance
    
    try:
        env_instance = DataCleaningEnvironment(
            task_id=request.task_id,
            difficulty=request.difficulty,
            max_steps=20
        )
        observation = env_instance.reset()
        return observation
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to reset environment: {str(e)}")


@app.post("/step", response_model=CleaningObservation)
def step(action: CleaningAction):
    """
    Take a step in the environment.
    
    Args:
        action: Cleaning action to apply
        
    Returns:
        Observation after applying the action
    """
    global env_instance
    
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        observation = env_instance.step(action)
        return observation
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
