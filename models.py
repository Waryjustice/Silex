"""Pydantic models for DataCleaningEnv."""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List


class CleaningAction(BaseModel):
    """Action that the agent can take to clean the dataset."""
    
    operation: Literal[
        "fill_missing",
        "drop_duplicates",
        "normalize_dates",
        "fix_types",
        "remove_outliers",
        "standardize_categorical",
        "done"
    ] = Field(..., description="The cleaning operation to perform")
    
    column: Optional[str] = Field(None, description="Target column name for the operation")
    value: Optional[str] = Field(None, description="Fill value for fill_missing operation")
    date_format: Optional[str] = Field(None, description="Target date format for normalize_dates (default: %Y-%m-%d)")
    bounds: Optional[List[float]] = Field(None, description="[min, max] bounds for remove_outliers operation")
    mapping: Optional[dict[str, str]] = Field(None, description="Category mapping for standardize_categorical")
    
    class Config:
        json_schema_extra = {
            "example": {
                "operation": "fill_missing",
                "column": "age",
                "value": "30"
            }
        }


class CleaningObservation(BaseModel):
    """Observation returned after each step."""
    
    dataset_preview: str = Field(..., description="First 10 rows of the dataset as a markdown table")
    column_stats: dict = Field(..., description="Statistics for each column (nulls, dtype, unique count)")
    issues_remaining: List[str] = Field(..., description="Human-readable list of data quality issues")
    step_reward: float = Field(..., description="Reward received for the last action")
    cumulative_reward: float = Field(..., description="Total reward accumulated in this episode")
    done: bool = Field(..., description="Whether the episode is finished")
    message: str = Field(..., description="Feedback message about the last action")
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset_preview": "| id | name | age |\n|---|---|---|\n| 1 | Alice | 30 |",
                "column_stats": {"age": {"nulls": 0, "dtype": "int64", "unique": 45}},
                "issues_remaining": ["2 duplicate rows found"],
                "step_reward": 0.15,
                "cumulative_reward": 0.45,
                "done": False,
                "message": "Filled 10 missing values in 'age' column"
            }
        }


class CleaningState(BaseModel):
    """Current state of the environment episode."""
    
    task_id: str = Field(..., description="Identifier of the current task")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Difficulty level of the task")
    steps_taken: int = Field(..., description="Number of steps taken in this episode")
    max_steps: int = Field(..., description="Maximum number of steps allowed")
    score: float = Field(..., description="Current score (same as cumulative_reward)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "easy_nulls",
                "difficulty": "easy",
                "steps_taken": 5,
                "max_steps": 20,
                "score": 0.45
            }
        }


class ResetRequest(BaseModel):
    """Request to reset the environment."""
    
    task_id: str = Field(..., description="Task ID to load (e.g., 'easy_nulls', 'medium_formats', 'hard_multitable')")
    difficulty: Optional[Literal["easy", "medium", "hard"]] = Field(None, description="Difficulty level (inferred from task_id if not provided)")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
