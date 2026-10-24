# PowerGrid Utility Intelligence

> Predicting power grid failures before they happen using machine learning.

---

## What This Project Does

Every year, unexpected power grid failures cost utilities billions of dollars - through emergency repairs, regulatory fines, and compensation to affected customers. The traditional approach is reactive: wait for something to break, then fix it.

This project flips that. We built a machine learning model that looks at the current condition of each grid asset - things like equipment age, health scores, weather exposure, maintenance history, and load levels - and estimates the probability that it will fail. Every asset in the network gets a score and a risk label: **HIGH**, **MEDIUM**, or **LOW**.

The maintenance team can then focus their limited time and budget on the assets most likely to fail, before they actually do.

---

## Business Problem

PowerGrid Utilities Corporation operates 50,500 assets - substations, transformers, transmission lines, switchgear, and circuit breakers - spread across 5 regions. The infrastructure is aging, demand keeps rising, and climate events are becoming more frequent and severe.

The problem is simple: the current approach is almost entirely reactive. Teams only intervene after a failure has already caused an outage, triggered regulatory penalties, or frustrated customers. By that point, the damage is done.

We set out to answer one question: **can we reliably predict which assets are at risk before they fail?**

---

## The Data

We worked with a dataset of 50,500 asset records, each with 32 attributes describing the asset's physical condition, operating environment, and maintenance history.

| Property | Value |
|----------|-------|
| Total records | 50,500 |
| Features per asset | 32 |
| What we're predicting | `grid_failure_flag` - whether the asset will fail (Yes/No) |
| Failure rate in data | ~43% of assets had failures |
| Asset types | Transmission Line, Transformer, Switchgear, Circuit Breaker, Substation |
| Regions | Central, North, South, East, West |
| Manufacturers | ABB, Schneider, GE, Siemens, Hitachi |

The 32 features cover four broad areas:
- **Physical condition** - equipment health score, oil quality, vibration levels, transformer temperature
- **Operating stress** - power load, voltage fluctuations, load utilization
- **Maintenance history** - overdue days, number of inspections, previous outages
- **Environmental exposure** - storm risk, rainfall, wind speed, humidity

---

## How We Built It

We followed the **CRISP-DM** framework - a standard industry approach for data science projects that keeps the business goal front and center at every step.

### Step 1 - Data Cleaning
The raw data had duplicates, inconsistent category names, and missing values. We removed 500 duplicate records, standardized asset type names, filled missing values using the median for each asset type (so a transformer's missing health score is filled from other transformers, not the whole dataset), and capped extreme outliers so they don't distort the model.

### Step 2 - Feature Engineering
Raw data alone doesn't always tell the full story. We created new features that capture relationships the model might otherwise miss:
- **Interaction features** - e.g., an old asset with overdue maintenance is worse than either condition alone
- **Composite risk score** - a single weighted score combining the six most failure-predictive attributes (health score, storm risk, maintenance overdue days, inspection score, voltage fluctuation, oil quality)

### Step 3 - Model Development
We trained and compared three industry-standard machine learning models:
- **Random Forest** - builds many decision trees and averages their predictions
- **XGBoost** - a powerful gradient boosting algorithm widely used in industry
- **LightGBM** - a faster, more efficient variant of gradient boosting, especially good on large datasets

Each model was evaluated using cross-validation to ensure results are reliable and not just lucky on one data split.

### Step 4 - Hyperparameter Tuning
Each model has settings (hyperparameters) that control how it learns. We used **Optuna** - an automated tuning library - to run 50 intelligent experiments per model, finding the combination of settings that performs best. This is like running 50 experiments automatically rather than guessing by hand.

### Step 5 - Evaluation
In a failure prediction problem, **missing a real failure is much worse than a false alarm**. A false alarm means an unnecessary inspection (wasted time and money). A missed failure means an unplanned outage (safety risk, regulatory penalty, customer impact). So we tuned the model to prioritize catching failures - we'd rather flag too many assets than miss ones that are genuinely at risk.

The key metric we optimized for is **Recall** - what percentage of actual failures did the model catch?

### Step 6 - Explainability
A model that just outputs a score without explanation is hard to trust, especially for operational decisions. We used **SHAP** (SHapley Additive exPlanations) to understand exactly which factors are driving each prediction. This lets the team see not just *that* an asset is high risk, but *why* - for example, "this transformer is flagged because its health score is low and it's overdue for maintenance during a high storm-risk period."

### Step 7 - Risk Tiering
Finally, every asset in the network was scored and assigned a risk tier:

| Risk Tier | Failure Probability | What it means |
|-----------|--------------------|-|
| HIGH | 70% or above | Inspect as soon as possible |
| MEDIUM | 45% - 69% | Schedule for inspection soon |
| LOW | Below 45% | Monitor on normal schedule |

---

## Results

**Winning model: LightGBM** - it delivered the best overall performance and ran significantly faster than Random Forest during training.

| Model | Recall | Precision | F1 Score | AUC-ROC | AUC-PR |
|-------|--------|-----------|----------|---------|--------|
| Random Forest | 0.814 | 0.760 | 0.786 | 0.897 | 0.870 |
| XGBoost (tuned) | 0.852 | 0.768 | 0.808 | 0.913 | 0.895 |
| **LightGBM (tuned)** | **0.850** | **0.770** | **0.810** | **0.9134** | **0.8953** |

**What these numbers mean in plain terms:**
- **Recall 0.85** - the model correctly identifies 85 out of every 100 assets that will actually fail
- **Precision 0.77** - when the model flags an asset as likely to fail, it's right 77% of the time
- **AUC-ROC 0.91** - on a scale of 0.5 (random guessing) to 1.0 (perfect), the model scores 0.91 - excellent discrimination between failing and healthy assets

**Top factors driving failure (from SHAP analysis):**
1. `equipment_health_score` - low health score is the single strongest predictor of failure
2. `composite_risk_score` - the combined weighted risk measure we engineered
3. `storm_risk_index` - assets in high storm-exposure areas are significantly more likely to fail

**Business impact:**
- **15,715 assets** (31.4% of the network) classified as HIGH risk
- HIGH-risk assets represent **$9.62 billion** in revenue exposure
- Inspecting all HIGH-risk assets costs approximately **$23.6M** (15,715 × $1,500 per inspection)
- Projected savings from preventing outages: **$1.44B - $2.89B** (assuming 15-30% outage reduction)
- That's a potential **60× return on the inspection investment**

---

## How to Run

### 1. Set up the environment

You'll need [Anaconda](https://www.anaconda.com/download) or Miniconda installed. Then run:

```bash
conda create -n capstone-1 python=3.11 -y
conda activate capstone-1
conda install -c conda-forge xgboost -y
pip install -r requirements.txt
python -m ipykernel install --user --name capstone-1 --display-name "Python (capstone-1)"
```

> **macOS note:** If XGBoost fails to import, run `brew install libomp` first.

### 2. Place the raw data

The dataset goes here:
```
data/raw/PowerGrid_Utility_Intelligence.csv
```

> **Which path to choose?** Use the scripts (Steps 3-4) to quickly reproduce the trained model and verify the metrics. Use the notebooks (Step 5) if you want to follow the full analysis step by step, see all visualizations, or understand how decisions were made at each stage.

### 3. Run the full pipeline via scripts

> **Important:** All scripts use relative paths - run them from the project root directory (where this README lives), not from inside `src/`.

```bash
python src/data_cleaning.py
python src/feature_engineering.py
python src/train.py
python src/evaluate.py
```

Each script prints progress to the terminal. `evaluate.py` will print the final metrics - these should match the numbers in the Results table above.

**What each script produces:**
- `data_cleaning.py` - saves `data/processed/clean.csv` (50,000 rows after deduplication)
- `feature_engineering.py` - saves `data/processed/features.csv` (41 columns after feature creation)
- `train.py` - saves `models/best_model.pkl` and `models/optimal_threshold.pkl`
- `evaluate.py` - prints metrics to terminal (no file output)

### 4. Score new assets (optional)

Once the model is trained, you can score any new batch of assets:

```bash
python src/predict.py --input data/raw/PowerGrid_Utility_Intelligence.csv --output outputs/predictions.csv
```

The input CSV must have the same columns as the raw training data. The output CSV adds three new columns to each asset:
- `failure_probability` - a score between 0 and 1 (higher = more likely to fail)
- `predicted_failure` - binary flag: 1 = predicted failure, 0 = predicted no failure
- `risk_tier` - HIGH, MEDIUM, or LOW based on the probability score

### 5. Or run notebooks in order

For a full walkthrough with visualizations, open JupyterLab and run these notebooks in sequence:

```
notebooks/01_eda.ipynb                  - Exploratory data analysis and charts
notebooks/02_data_cleaning.ipynb        - Data cleaning steps with before/after comparisons
notebooks/03_feature_engineering.ipynb  - Feature creation and validation
notebooks/04_modelling.ipynb            - Training all three models
notebooks/05_hyperparameter_tuning.ipynb - Optuna optimization runs
notebooks/06_evaluation_shap.ipynb      - Model evaluation and SHAP explainability charts
notebooks/07_business_insights.ipynb    - Risk tiering and business impact calculations
```

Select the `Python (capstone-1)` kernel in JupyterLab before running.

---

## Repository Structure

```
├── data/
│   ├── raw/                        - original dataset (input)
│   └── processed/                  - cleaned and feature-engineered CSVs (generated by pipeline)
├── notebooks/                      - Jupyter notebooks, numbered in execution order
├── src/                            - standalone Python scripts (the pipeline)
├── models/                         - trained model files (generated by pipeline)
├── outputs/                        - evaluation plots, risk scores, SHAP charts
├── report/                         - final technical report PDF
├── documents/                      - original problem statement and brief
├── requirements.txt                - Python package dependencies
└── README.md
```

---

## Acknowledgements

- XGBoost: Chen & Guestrin (2016) - https://arxiv.org/abs/1603.02754
- LightGBM: Ke et al. (2017) - https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- SHAP: Lundberg & Lee (2017) - https://arxiv.org/abs/1705.07874
- SMOTE: Chawla et al. (2002) - https://arxiv.org/abs/1106.1813
- Optuna: Akiba et al. (2019) - https://optuna.org
