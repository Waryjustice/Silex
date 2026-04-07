"""Core DataCleaningEnvironment class."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import re

from models import CleaningAction, CleaningObservation, CleaningState
from server.graders import get_grader


class DataCleaningEnvironment:
    """
    Environment for data cleaning tasks.
    
    Agents interact with dirty CSV datasets and apply cleaning operations
    to maximize the score computed by comparing against ground truth.
    """
    
    def __init__(self, task_id: str, difficulty: Optional[str] = None, max_steps: int = 20):
        """
        Initialize the environment.
        
        Args:
            task_id: Task identifier (e.g., 'easy_nulls', 'medium_formats', 'hard_multitable')
            difficulty: Difficulty level (inferred from task_id if not provided)
            max_steps: Maximum number of steps allowed per episode
        """
        self.task_id = task_id
        self.max_steps = max_steps
        
        # Infer difficulty if not provided
        if difficulty is None:
            if 'easy' in task_id:
                self.difficulty = 'easy'
            elif 'medium' in task_id:
                self.difficulty = 'medium'
            elif 'hard' in task_id:
                self.difficulty = 'hard'
            else:
                raise ValueError(f"Cannot infer difficulty from task_id: {task_id}")
        else:
            self.difficulty = difficulty
        
        # Load datasets
        data_dir = Path(__file__).parent.parent / 'data'
        self.dirty_df_original = pd.read_csv(data_dir / f'{task_id}_dirty.csv')
        self.ground_truth_df = pd.read_csv(data_dir / f'{task_id}_clean.csv')
        
        # Get grader function
        self.grader = get_grader(task_id)
        
        # Episode state
        self.df: Optional[pd.DataFrame] = None
        self.steps_taken = 0
        self.cumulative_reward = 0.0
        self.done_flag = False
        
    def reset(self) -> CleaningObservation:
        """
        Reset the environment to start a new episode.
        
        Returns:
            Initial observation
        """
        # Reset episode state
        self.df = self.dirty_df_original.copy()
        self.steps_taken = 0
        self.cumulative_reward = 0.0
        self.done_flag = False
        
        return self._observe(step_reward=0.0, message="Episode started. Begin cleaning the dataset.")
    
    def step(self, action: CleaningAction) -> CleaningObservation:
        """
        Apply a cleaning action and return the new observation.
        
        Args:
            action: Cleaning action to apply
            
        Returns:
            Observation with reward and feedback
        """
        if self.done_flag:
            return self._observe(
                step_reward=0.0,
                message="Episode already finished. Call reset() to start a new episode.",
                done=True
            )
        
        # Compute score before action
        score_before = self.grader(self.df, self.ground_truth_df)
        
        # Apply the action
        reward, message = self._apply_action(action)
        
        # Compute score after action
        score_after = self.grader(self.df, self.ground_truth_df)
        
        # Adjust reward based on score delta
        delta = score_after - score_before
        if delta > 0:
            reward = delta  # Positive progress
        elif delta == 0:
            reward = -0.02  # No progress, small penalty
        else:
            reward = -0.1  # Made things worse
        
        # Check if done
        self.steps_taken += 1
        if action.operation == "done":
            self.done_flag = True
            # Penalty for finishing with low score
            if score_after < 0.5:
                reward -= 0.3
                message += " Warning: Finished with low score."
        elif self.steps_taken >= self.max_steps:
            self.done_flag = True
            message += " Max steps reached."
        
        self.cumulative_reward += reward
        
        return self._observe(step_reward=reward, message=message, done=self.done_flag)
    
    @property
    def state(self) -> CleaningState:
        """
        Get the current environment state.
        
        Returns:
            CleaningState object
        """
        return CleaningState(
            task_id=self.task_id,
            difficulty=self.difficulty,
            steps_taken=self.steps_taken,
            max_steps=self.max_steps,
            score=self.cumulative_reward
        )
    
    def _apply_action(self, action: CleaningAction) -> tuple[float, str]:
        """
        Apply a cleaning operation to the dataset.
        
        Args:
            action: CleaningAction to apply
            
        Returns:
            Tuple of (reward, message)
        """
        try:
            if action.operation == "fill_missing":
                if action.column and action.column in self.df.columns:
                    null_count = self.df[action.column].isnull().sum()
                    if null_count > 0:
                        self.df[action.column].fillna(action.value or "", inplace=True)
                        return 0.0, f"Filled {null_count} missing values in '{action.column}'"
                    else:
                        return 0.0, f"No missing values in '{action.column}'"
                else:
                    return -0.05, f"Invalid column: {action.column}"
            
            elif action.operation == "drop_duplicates":
                dup_count = self.df.duplicated().sum()
                if dup_count > 0:
                    self.df.drop_duplicates(inplace=True)
                    self.df.reset_index(drop=True, inplace=True)
                    return 0.0, f"Removed {dup_count} duplicate rows"
                else:
                    return 0.0, "No duplicate rows found"
            
            elif action.operation == "normalize_dates":
                if action.column and action.column in self.df.columns:
                    try:
                        self.df[action.column] = pd.to_datetime(
                            self.df[action.column], 
                            infer_datetime_format=True,
                            errors='coerce'
                        ).dt.strftime(action.date_format or '%Y-%m-%d')
                        return 0.0, f"Normalized dates in '{action.column}'"
                    except Exception as e:
                        return -0.05, f"Failed to normalize dates: {str(e)}"
                else:
                    return -0.05, f"Invalid column: {action.column}"
            
            elif action.operation == "fix_types":
                if action.column and action.column in self.df.columns:
                    try:
                        # Remove currency symbols and clean
                        if self.df[action.column].dtype == object:
                            self.df[action.column] = self.df[action.column].astype(str).str.replace('$', '', regex=False)
                            self.df[action.column] = self.df[action.column].str.replace(',', '', regex=False)
                        
                        # Try to convert to numeric
                        self.df[action.column] = pd.to_numeric(self.df[action.column], errors='coerce')
                        return 0.0, f"Fixed types in '{action.column}'"
                    except Exception as e:
                        return -0.05, f"Failed to fix types: {str(e)}"
                else:
                    return -0.05, f"Invalid column: {action.column}"
            
            elif action.operation == "remove_outliers":
                if action.column and action.column in self.df.columns and action.bounds and len(action.bounds) == 2:
                    try:
                        min_val, max_val = action.bounds
                        col = pd.to_numeric(self.df[action.column], errors='coerce')
                        mask = (col >= min_val) & (col <= max_val)
                        removed = (~mask).sum()
                        self.df = self.df[mask].reset_index(drop=True)
                        return 0.0, f"Removed {removed} outliers from '{action.column}'"
                    except Exception as e:
                        return -0.05, f"Failed to remove outliers: {str(e)}"
                else:
                    return -0.05, "Invalid parameters for remove_outliers"
            
            elif action.operation == "standardize_categorical":
                if action.column and action.column in self.df.columns:
                    try:
                        if action.mapping:
                            self.df[action.column] = self.df[action.column].map(action.mapping).fillna(self.df[action.column])
                        else:
                            # Auto-standardize common patterns
                            col = self.df[action.column].astype(str)
                            # Gender standardization
                            if action.column.lower() in ['gender', 'sex']:
                                mapping = {
                                    'male': 'Male', 'm': 'Male', 'M': 'Male',
                                    'female': 'Female', 'f': 'Female', 'F': 'Female',
                                    'other': 'Other', 'o': 'Other', 'O': 'Other',
                                    'N/A': 'Other', 'n/a': 'Other', 'NA': 'Other'
                                }
                                self.df[action.column] = col.map(mapping).fillna(col)
                        return 0.0, f"Standardized categories in '{action.column}'"
                    except Exception as e:
                        return -0.05, f"Failed to standardize: {str(e)}"
                else:
                    return -0.05, f"Invalid column: {action.column}"
            
            elif action.operation == "done":
                return 0.0, "Agent signaled completion"
            
            else:
                return -0.05, f"Unknown operation: {action.operation}"
        
        except Exception as e:
            return -0.1, f"Error applying action: {str(e)}"
    
    def _observe(self, step_reward: float, message: str, done: bool = False) -> CleaningObservation:
        """
        Generate an observation of the current state.
        
        Args:
            step_reward: Reward for the last action
            message: Feedback message
            done: Whether episode is finished
            
        Returns:
            CleaningObservation
        """
        # Generate dataset preview (first 10 rows as markdown table)
        preview_df = self.df.head(10)
        preview = preview_df.to_markdown(index=False)
        
        # Compute column statistics
        column_stats = {}
        for col in self.df.columns:
            stats = {
                'nulls': int(self.df[col].isnull().sum()),
                'dtype': str(self.df[col].dtype),
                'unique': int(self.df[col].nunique())
            }
            column_stats[col] = stats
        
        # Detect remaining issues
        issues = self._detect_issues()
        
        return CleaningObservation(
            dataset_preview=preview,
            column_stats=column_stats,
            issues_remaining=issues,
            step_reward=round(step_reward, 3),
            cumulative_reward=round(self.cumulative_reward, 3),
            done=done,
            message=message
        )
    
    def _detect_issues(self) -> List[str]:
        """
        Detect data quality issues in the current dataset.
        
        Returns:
            List of human-readable issue descriptions
        """
        issues = []
        
        # Check for nulls
        null_counts = self.df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                issues.append(f"{count} missing values in '{col}'")
        
        # Check for duplicates
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            issues.append(f"{dup_count} duplicate rows")
        
        # Check for hidden nulls (N/A, n/a, NA strings)
        for col in self.df.columns:
            if self.df[col].dtype == object:
                hidden_nulls = self.df[col].isin(['N/A', 'n/a', 'NA', 'na']).sum()
                if hidden_nulls > 0:
                    issues.append(f"{hidden_nulls} hidden null values in '{col}'")
        
        # Check for type inconsistencies
        for col in self.df.columns:
            if self.df[col].dtype == object:
                # Try to detect if it should be numeric
                numeric_count = pd.to_numeric(self.df[col], errors='coerce').notna().sum()
                if numeric_count > len(self.df) * 0.5:  # More than 50% are numeric
                    issues.append(f"Column '{col}' appears numeric but stored as text")
        
        if not issues:
            issues.append("No obvious issues detected")
        
        return issues
