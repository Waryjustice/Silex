"""Grading functions for data cleaning tasks."""

import pandas as pd
import numpy as np
from typing import Tuple


def _to_open_unit_interval(value: float) -> float:
    """
    Clamp score to strict open interval (0, 1) with 3-decimal output.
    """
    if not np.isfinite(value):
        value = 0.001
    value = float(value)
    value = min(0.999, max(0.001, value))
    return round(value, 3)


def grade_easy(cleaned_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    """
    Grade the easy_nulls task.
    
    Criteria:
    - No null values (33%)
    - No duplicate rows (33%)
    - Correct shape (34%)
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    scores = []
    
    # Check for null values
    null_count = cleaned_df.isnull().sum().sum()
    if null_count == 0:
        null_score = 1.0
    else:
        # Partial credit: reduce score by percentage of nulls remaining
        total_cells = cleaned_df.shape[0] * cleaned_df.shape[1]
        null_score = max(0.0, 1.0 - (null_count / total_cells))
    scores.append(null_score)
    
    # Check for duplicates
    dup_count = cleaned_df.duplicated().sum()
    if dup_count == 0:
        dup_score = 1.0
    else:
        # Partial credit
        dup_score = max(0.0, 1.0 - (dup_count / len(cleaned_df)))
    scores.append(dup_score)
    
    # Check shape (should match ground truth after duplicates removed)
    if cleaned_df.shape == ground_truth_df.shape:
        shape_score = 1.0
    else:
        # Partial credit based on row count difference
        row_diff = abs(cleaned_df.shape[0] - ground_truth_df.shape[0])
        shape_score = max(0.0, 1.0 - (row_diff / ground_truth_df.shape[0]))
    scores.append(shape_score)
    
    return _to_open_unit_interval(sum(scores) / len(scores))


def grade_medium(cleaned_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    """
    Grade the medium_formats task.
    
    Criteria:
    - Date format consistency (30%)
    - Duplicate removal (30%)
    - Phone format standardization (20%)
    - Category standardization (20%)
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    scores = []
    
    # Date format consistency - should all be YYYY-MM-DD
    try:
        date_col = cleaned_df['date']
        # Try to parse all dates with strict format
        parsed = pd.to_datetime(date_col, format='%Y-%m-%d', errors='coerce')
        valid_dates = parsed.notna().sum()
        date_score = valid_dates / len(date_col)
    except Exception:
        date_score = 0.0
    scores.append(date_score)
    
    # Duplicate removal
    dup_count = cleaned_df.duplicated().sum()
    expected_dups = 15  # We know there are 15 duplicates in the dirty data
    if dup_count == 0:
        dup_score = 1.0
    else:
        dup_score = max(0.0, 1.0 - (dup_count / expected_dups))
    scores.append(dup_score)
    
    # Phone format - should start with +1
    try:
        phone_col = cleaned_df['phone'].astype(str)
        correct_format = phone_col.str.startswith('+1').sum()
        phone_score = correct_format / len(phone_col)
    except Exception:
        phone_score = 0.0
    scores.append(phone_score)
    
    # Category standardization - should only be Male/Female
    try:
        cat_col = cleaned_df['category']
        valid_categories = {'Male', 'Female'}
        valid_count = cat_col.isin(valid_categories).sum()
        cat_score = valid_count / len(cat_col)
    except Exception:
        cat_score = 0.0
    scores.append(cat_score)
    
    # Weighted average
    weights = [0.3, 0.3, 0.2, 0.2]
    weighted_score = sum(s * w for s, w in zip(scores, weights))
    
    return _to_open_unit_interval(weighted_score)


def grade_hard(cleaned_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    """
    Grade the hard_multitable task.
    
    Criteria:
    - Type correctness (25%)
    - Outlier removal (25%)
    - Categorical standardization (25%)
    - Hidden null detection and handling (25%)
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    scores = []
    
    # Type correctness - age and revenue should be numeric
    try:
        age_numeric = pd.to_numeric(cleaned_df['age'], errors='coerce').notna().sum()
        age_type_score = age_numeric / len(cleaned_df)
    except Exception:
        age_type_score = 0.0
    
    try:
        revenue_numeric = pd.to_numeric(cleaned_df['revenue'], errors='coerce').notna().sum()
        revenue_type_score = revenue_numeric / len(cleaned_df)
    except Exception:
        revenue_type_score = 0.0
    
    type_score = (age_type_score + revenue_type_score) / 2
    scores.append(type_score)
    
    # Outlier removal - age should be 18-120, revenue should be positive
    try:
        age_col = pd.to_numeric(cleaned_df['age'], errors='coerce')
        valid_age = ((age_col >= 18) & (age_col <= 120)).sum()
        age_outlier_score = valid_age / age_col.notna().sum() if age_col.notna().sum() > 0 else 0
    except Exception:
        age_outlier_score = 0.0
    
    try:
        revenue_col = pd.to_numeric(cleaned_df['revenue'], errors='coerce')
        valid_revenue = (revenue_col >= 0).sum()
        revenue_outlier_score = valid_revenue / revenue_col.notna().sum() if revenue_col.notna().sum() > 0 else 0
    except Exception:
        revenue_outlier_score = 0.0
    
    outlier_score = (age_outlier_score + revenue_outlier_score) / 2
    scores.append(outlier_score)
    
    # Categorical standardization - gender should be Male/Female/Other only
    try:
        gender_col = cleaned_df['gender']
        valid_genders = {'Male', 'Female', 'Other'}
        valid_count = gender_col.isin(valid_genders).sum()
        cat_score = valid_count / len(gender_col)
    except Exception:
        cat_score = 0.0
    scores.append(cat_score)
    
    # Hidden null detection - should not contain N/A, n/a, NA strings
    try:
        hidden_nulls = 0
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == object:
                hidden_nulls += cleaned_df[col].isin(['N/A', 'n/a', 'NA', 'na']).sum()
        
        total_cells = cleaned_df.shape[0] * cleaned_df.shape[1]
        hidden_null_score = 1.0 - (hidden_nulls / total_cells)
        hidden_null_score = max(0.0, hidden_null_score)
    except Exception:
        hidden_null_score = 0.0
    scores.append(hidden_null_score)
    
    return _to_open_unit_interval(sum(scores) / len(scores))


def get_grader(task_id: str):
    """
    Get the appropriate grader function for a task.
    
    Args:
        task_id: Task identifier (e.g., 'easy_nulls', 'medium_formats', 'hard_multitable')
        
    Returns:
        callable: Grading function that takes (cleaned_df, ground_truth_df) -> float
    """
    graders = {
        'easy_nulls': grade_easy,
        'medium_formats': grade_medium,
        'hard_multitable': grade_hard
    }
    
    if task_id not in graders:
        raise ValueError(f"Unknown task_id: {task_id}. Valid options: {list(graders.keys())}")
    
    return graders[task_id]
