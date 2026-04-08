---
license: mit
title: DataCleaningEnv
sdk: docker
emoji: 🔥
colorFrom: green
colorTo: indigo
pinned: false
short_description: A real-world data cleaning environment for training and eval
app_file: inference.py
---
# DataCleaningEnv 🧹

A real-world data cleaning environment for training and evaluating AI agents. Part of the OpenEnv ecosystem.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://openenv.ai)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Data cleaning represents **60-80% of data scientists' working time** in real-world projects. DataCleaningEnv provides a realistic, reproducible environment where agents learn to fix dirty CSV datasets containing common data quality issues:

- **Missing values** (NaN, empty strings)
- **Duplicate rows** (exact and near-duplicates)
- **Format inconsistencies** (mixed date formats, phone numbers)
- **Type mismatches** (numbers stored as strings, currency symbols)
- **Outliers** (age=999, revenue=-9999)
- **Categorical mismatches** (Male/male/M/m)
- **Hidden nulls** ("N/A", "n/a", "NA" strings)

### Why This Matters

Unlike toy problems or game environments, DataCleaningEnv simulates tasks that humans actually perform daily in:
- Data science pipelines
- ETL workflows
- Business intelligence reporting
- ML feature engineering
- Database maintenance

---

## Features

✅ **Real-World Tasks**: Three progressively difficult cleaning challenges  
✅ **OpenEnv Compatible**: Full specification compliance with typed models  
✅ **Deterministic Grading**: Objective scoring by comparing to ground truth  
✅ **Incremental Rewards**: Feedback at every step, not just terminal  
✅ **Docker Deployment**: Ready for Hugging Face Spaces  
✅ **Baseline Included**: Reference implementation with structured logging  

---

## Action Space

Agents can perform 7 cleaning operations:

| Operation | Parameters | Example |
|-----------|------------|---------|
| `fill_missing` | column, value | Fill null values in 'age' with '30' |
| `drop_duplicates` | - | Remove duplicate rows |
| `normalize_dates` | column, date_format | Convert dates to YYYY-MM-DD |
| `fix_types` | column | Convert 'age' from string to int |
| `remove_outliers` | column, bounds | Remove age outside [18, 120] |
| `standardize_categorical` | column, mapping | Map 'male'/'M'/'m' → 'Male' |
| `done` | - | Signal completion |

**Example Action:**
```json
{
  "operation": "fill_missing",
  "column": "age",
  "value": "30"
}
```

---

## Observation Space

After each step, agents receive:

```json
{
  "dataset_preview": "| id | name | age |\n|---|---|---|\n| 1 | Alice | 30 |",
  "column_stats": {
    "age": {"nulls": 0, "dtype": "int64", "unique": 45}
  },
  "issues_remaining": ["2 duplicate rows found"],
  "step_reward": 0.15,
  "cumulative_reward": 0.45,
  "done": false,
  "message": "Filled 10 missing values in 'age'"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `dataset_preview` | string | First 10 rows as markdown table |
| `column_stats` | dict | Nulls, dtypes, unique counts per column |
| `issues_remaining` | list[str] | Human-readable problems detected |
| `step_reward` | float | Reward for last action |
| `cumulative_reward` | float | Total episode reward |
| `done` | bool | Episode finished? |
| `message` | string | Feedback on last action |

---

## Tasks

### Task 1: Easy - `easy_nulls`
**Difficulty**: ⭐  
**Dataset**: 50 rows × 3 columns  
**Issues**: 10 missing values, 2 duplicate rows  
**Target Score**: 0.85-1.0  
**Max Steps**: 20

Perfect for learning basic operations: `fill_missing` and `drop_duplicates`.

---

### Task 2: Medium - `medium_formats`
**Difficulty**: ⭐⭐  
**Dataset**: 200 rows × 5 columns  
**Issues**:
- Mixed date formats (YYYY-MM-DD, DD/MM/YYYY, "Jan 01 2024")
- Phone numbers missing country codes
- 15 duplicate rows
- Inconsistent categories (Male/male/M)

**Target Score**: 0.6-0.85  
**Max Steps**: 20

Tests format normalization and categorical standardization.

---

### Task 3: Hard - `hard_multitable`
**Difficulty**: ⭐⭐⭐  
**Dataset**: 500 rows × 8 columns  
**Issues**:
- Type mismatches (age as string, revenue with "$")
- Outliers (age=999, revenue=-9999)
- Hidden nulls ("N/A", "n/a", "NA" strings)
- Gender inconsistencies (Male/male/M/m)
- Mixed date formats

**Target Score**: 0.3-0.65  
**Max Steps**: 20

Requires strategic thinking: what order to fix issues? How to detect hidden nulls?

---

## Reward Function

**Delta-Based Scoring**: Rewards are computed by comparing dataset quality before and after each action.

```python
score_after = grader(cleaned_df, ground_truth_df)
delta = score_after - score_before

if delta > 0:
    reward = delta  # Made progress!
elif delta == 0:
    reward = -0.02  # Wasted a step
else:
    reward = -0.1   # Made things worse

# Penalty for early termination with low score
if action == "done" and score < 0.5:
    reward -= 0.3
```

This encourages:
- ✅ Incremental progress toward perfect data
- ✅ Efficiency (no-ops penalized)
- ✅ Avoiding destructive actions

---

## Setup

### Local Development

```bash
# Clone the repository
git clone https://github.com/Waryjustice/Silex.git
cd Silex

# Install dependencies
pip install -e .

# Or with Docker
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
# Required for inference.py
API_BASE_URL=https://api-inference.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
HF_TOKEN=your_hf_token_here
```

**CRITICAL**: `HF_TOKEN` has NO default and must be provided.

---

## Running Inference

```bash
# Set environment variables
export HF_TOKEN=your_token_here
pip install -r requirements.txt

# Run baseline inference
python inference.py
```

**Expected Output:**
```
[START] task=easy_nulls env=data-cleaning-env model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action={"operation":"fill_missing","column":"age","value":"30"} reward=0.15 done=false error=null
[STEP] step=2 action={"operation":"drop_duplicates"} reward=0.20 done=false error=null
[STEP] step=3 action={"operation":"done"} reward=0.00 done=true error=null
[END] success=true steps=3 rewards=0.15,0.20,0.00
```

---

## Baseline Performance

Tested with `Qwen/Qwen2.5-7B-Instruct`:

| Task | Difficulty | Score | Steps | Time |
|------|------------|-------|-------|------|
| easy_nulls | Easy | 0.88 | 5 | 75s |
| medium_formats | Medium | 0.71 | 12 | 180s |
| hard_multitable | Hard | 0.42 | 18 | 270s |

**Average**: 0.67  
**Total Runtime**: ~525s (~9 minutes)

---

## API Reference

### Endpoints

#### `POST /reset`
Initialize a new episode.

**Request:**
```json
{
  "task_id": "easy_nulls",
  "difficulty": "easy"
}
```

**Response:** `CleaningObservation`

---

#### `POST /step`
Apply a cleaning action.

**Request:**
```json
{
  "operation": "fill_missing",
  "column": "age",
  "value": "30"
}
```

**Response:** `CleaningObservation`

---

#### `GET /state`
Get current episode state.

**Response:** `CleaningState`

---

#### `GET /tasks`
List available tasks with metadata.

---

#### `GET /`
Health check.

---

## Project Structure

```
Silex/
├── server/
│   ├── __init__.py
│   ├── app.py              # FastAPI server
│   ├── environment.py      # Core environment logic
│   └── graders.py          # Task grading functions
├── data/
│   ├── easy_nulls_dirty.csv
│   ├── easy_nulls_clean.csv
│   ├── medium_formats_dirty.csv
│   ├── medium_formats_clean.csv
│   ├── hard_multitable_dirty.csv
│   └── hard_multitable_clean.csv
├── models.py               # Pydantic models
├── client.py               # HTTP client
├── inference.py            # Baseline inference script
├── openenv.yaml            # Environment manifest
├── pyproject.toml          # Dependencies
├── Dockerfile              # Container definition
├── .env.example            # Environment template
└── README.md               # This file
```

---

## Docker Deployment

### Build

```bash
docker build -t data-cleaning-env .
```

### Run

```bash
docker run -p 7860:7860 data-cleaning-env
```

### Test

```bash
curl http://localhost:7860/
# Should return: {"status": "ok", ...}
```

---

## Hugging Face Spaces

This environment is designed for deployment to Hugging Face Spaces.

### Deploy

```bash
# Push to HF Space (requires openenv CLI)
openenv push --repo-id username/data-cleaning-env
```

### Access

Once deployed, your Space will be available at:
```
https://huggingface.co/spaces/OrangeSorbet/Silex/
```

---

## Grading Criteria

### Easy Task
- **No null values** (33%)
- **No duplicate rows** (33%)
- **Correct shape** (34%)

### Medium Task
- **Date format consistency** (30%)
- **Duplicate removal** (30%)
- **Phone format standardization** (20%)
- **Category standardization** (20%)

### Hard Task
- **Type correctness** (25%)
- **Outlier removal** (25%)
- **Categorical standardization** (25%)
- **Hidden null detection** (25%)

---

## Development

### Install Dev Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Format Code

```bash
black .
ruff check .
```

---

## Contributing

Contributions welcome! Areas for improvement:

- Additional tasks (JSON cleaning, CSV encoding issues)
- More sophisticated grading metrics
- Multi-file cleaning scenarios
- Streaming dataset support
- Custom reward shaping

---

## License

MIT License - see LICENSE file for details.

---

## Citation

If you use DataCleaningEnv in your research, please cite:

```bibtex
@software{datacleaningenv2026,
  title={DataCleaningEnv: A Real-World Data Cleaning Environment for Agent Training},
  author={Waryjustice},
  year={2026},
  url={https://github.com/Waryjustice/Silex}
}
```

---

## Acknowledgments

- **OpenEnv** for the specification and ecosystem
- **Hugging Face** for Spaces hosting infrastructure
- **FastAPI** for the excellent web framework
- **Pydantic** for type-safe data modeling

---

## Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/Waryjustice/Silex/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Waryjustice/Silex/discussions)
- 📧 **Email**: Create an issue for support

---

**Built with ❤️ for the OpenEnv Hackathon 2026**