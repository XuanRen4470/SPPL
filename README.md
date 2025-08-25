# Efficiently Selecting Response Generation Strategy by Self-Aligned Perplexity for Fine-Tuning LLMs

This repository provides the source code for our paper *Efficiently Selecting Response Generation Strategy by Self-Aligned Perplexity for Fine-Tuning LLMs*.

## Table of Contents

* [Project Structure](#project-structure)
* [Setup Configuration](#setup-configuration)
* [Quick Start (Core Path)](#quick-start-core-path)
* [Recompute Prerequisites (Only if needed)](#recompute-prerequisites-only-if-needed)

  * [Part 1: Training and Evaluation (heavy; generally not recommended)](#part-1-training-and-evaluation-heavy-generally-not-recommended)
  * [Part 2: Metrics Computation](#part-2-metrics-computation)

    * [Step 1: Initial predictions](#step-1-initial-predictions)
    * [Step 2: Compute SPPL](#step-2-compute-sppl)
    * [Step 3: Compute other metrics](#step-3-compute-other-metrics)
* [Utilities](#utilities)
* [Notes](#notes)

---

## Project Structure

⚠️ **Important:** This GitHub repository only contains the `SPPL/` folder.
To run our code correctly, you need to **manually create additional folders outside `SPPL/`** and set their paths in [`SPPL/config/config.py`](SPPL/config/config.py).

### Required Workspace Layout

```
YOUR_WORKSPACE/
│
├── SPPL/                          # This GitHub repository (core code)
│   ├── config/
│   ├── script/
│   ├── Mix_Score_Ranking_Calculation/   # Efficient ranking code (SPPL core)
|   |── LLaMA-Factory-ACL-2025/        # Downloaded from the official LLaMA-Factory GitHub
│   |                                     # - Required to obtain training outcomes(training + evaluation)
│   |                                     # - Required to obtain initial predictions
│   └── ...
│
├── model/                         # Target models for training & prediction
│
├── output_record/output              # Not sure if useful, but leave it here
│

│
└── llama_factory_temp/delete_later/
    └── ...                        # Temporary copies of LLaMA-Factory
                                   # - Each task runs in an isolated copy
                                   # - Prevents conflicts when training multiple tasks
```

Quick setup:

```bash
mkdir -p model output_record/output llama_factory_temp/delete_later
```

Then download LLaMA-Factory from github and name it as LLaMA-Factory-ACL-2025

---

## Setup Configuration

Edit [`SPPL/config/config.py`](SPPL/config/config.py) to match your environment:

```python
# Path to your local model
MODEL_DIRECTORY = os.path.dirname(parent_dir) + '/model'

# Just leave this path here
OUTPUT_RECORD_DIRECTORY = os.path.dirname(parent_dir) + '/output_record'

# Path to LLaMA-Factory (external dependency). Just download LLaMA-Facotry and place it here.
LLAMA_FACTORY_DIRECTORY = 'SPPL/LLaMA-Factory-ACL-2025'

# Temporary directory for isolated LLaMA-Factory copies
LLAMA_FACTORY_TEMP_DIRECTORY = '/gpfs/users/a1796450/llama_factory_temp/delete_later'

# Home directory of the project
HOME_DIRECTORY = parent_dir

# API key for GPT access (if needed)
YOUR_API_KEY = os.getenv('GPT_API')
GPT_API = YOUR_API_KEY
```

```bash
# If needed for synthetic data generation:
# SPPL/Mix_Score_Ranking_Calculation/synthetic_data_generator.py
export GPT_API="your_api_key_here"
```

---

## Quick Start (Core Code)

**The core of this project is to *rank and select the best response generation strategy*.**

This part of code is all under SPPL/Mix_Score_Ranking_Calculation

The **core comparison scripts** rely on the following **precomputed artifacts**. These are bundled with the project’s release download, so you do **not** need to compute or generate them yourself.

1. Training outcomes (from [Part 1](#part-1-training-and-evaluation-heavy-generally-not-recommended))
2. Initial predictions of the target model (from [Step 1 in Part 2](#step-1-initial-predictions))
3. SPPL scores (from [Step 2 in Part 2](#step-2-compute-sppl))
4. Other metrics (from [Step 3 in Part 2](#step-3-compute-other-metrics))

> ℹ️ **In principle**, synthetic data should be generated first.
> **However, we have already generated the synthetic data and computed the above artifacts.**
> With the available artifacts, you can **run the comparisons below directly**:

```bash
# Core comparisons (run these to reproduce the main results)

# SPPL vs. other metrics
python SPPL/Mix_Score_Ranking_Calculation/icppl_vs_other_metrics.py

# SPPL vs. different response generation strategies
python SPPL/Mix_Score_Ranking_Calculation/icppl_vs_other_metrics.py

# SPPL vs. train-then-test approach
python SPPL/Mix_Score_Ranking_Calculation/spearman_correlation_and_acc_train_then_test.py
```

All experimental outputs are saved as LaTeX (`.tex`) tables in `SPPL/Mix_Score_Ranking_Calculation/experiment_result`. Each script generates a ready-to-paste comparison table. To visualize the tables, open the relevant `.tex` file and copy–paste the table into your Overleaf project.

---

## Recompute Prerequisites (Only if needed)

Only follow this section if you wish to generate the precomputed artifacts required by the [Core Path](#quick-start-core-path) on your own.

### Part 1: Training and Evaluation (heavy; generally not recommended)

Generates training outcomes across datasets using LLaMA-Factory:

```bash
python SPPL/script/main_experiment/run_template_script.py
```

* Auto-generates and executes `.sh` files.
* Adjustable via comments inside the script.
* Runs evaluation automatically after training.

The training outcomes for part 1 are saved in /SPPL/log_total/experiment_data_recorder. These outcomes report the accuracy achieved after training with data generated by each response generation strategy, evaluated across different target models and tasks.

### Part 2: Metrics Computation

> **Note:** Running [Step 1](#step-1-initial-predictions) (initial predictions) on all tasks with all target models time-consuming. If you already download it, we recommend you to skip this part.

#### Step 1: Initial predictions

```bash
python SPPL/Mix_Score_Ranking_Calculation/initial_prediction.py
```

#### Step 2: Compute SPPL

```bash
python SPPL/Mix_Score_Ranking_Calculation/icppl_calculation.py
```

#### Step 3: Compute Skywork score (required for CAR)

`CAR` uses the **Skywork score** inside its formula. Compute Skywork first:

```bash
python SPPL/Mix_Score_Ranking_Calculation/sky_work_calculation.py
```

#### Step 4: Compute other metrics (including CAR)

Calculates additional metrics used in the comparisons, **including CAR** (now that Skywork scores are available).

```bash
python SPPL/Mix_Score_Ranking_Calculation/other_metrics_calculation.py
```

The computed results for step 2 and step 4 are saved in SPPL/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/ 
The skywork score for step 3 are saved in SPPL/Mix_Score_Ranking_Calculation/Mix_Score_record/skywork_reward_record
The initial predictions for step 1 are saved in SPPL/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/
The correct initial predictions are saved in SPPL/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_correct_examples/
Once the initial predictions are generated and the scores for all metrics are computed, return to the [Core Path](#quick-start-core-path).

---

## Utilities

Helper scripts for data generation and report formatting:

* **`synthetic_data_generator.py`** — create synthetic data for experiments.
  Although synthetic generation is *conceptually first*, the datasets used in our comparisons are **already generated** under the SPPL/dataset folder.
  Run this only if you want to customize or regenerate synthetic data.

  ```bash
  python SPPL/Mix_Score_Ranking_Calculation/synthetic_data_generator.py
  ```

* **`conver_data_to_latex_table.py`** — convert **computed ranking results** into LaTeX tables.
  ```bash
  python SPPL/Mix_Score_Ranking_Calculation/conver_data_to_latex_table.py
  ```

---
