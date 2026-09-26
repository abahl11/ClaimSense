# Workers' Compensation Claim Cost Prediction

> Work in progress. The full README (results, figures, approach diagram) is written in the final phase.

Predict the ultimate cost of a workers' compensation claim, flag likely high-cost claims for early
escalation, and explain the drivers of each prediction, using only information available when the
claim is reported.

## Data

Kaggle [Actuarial Loss Estimation](https://www.kaggle.com/competitions/actuarial-loss-estimation)
`train.csv` (synthetic workers' compensation claims). Download it manually and place it at
`data/raw/train.csv`. The file is gitignored and is not redistributed in this repository.

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or run `.\run.ps1 setup`, which does the same without activating the environment.

## Run

```powershell
python -m src.pipeline --stage validate   # one stage
python -m src.pipeline --stage all        # every stage in order
python -m pytest                          # tests
```

Stages: `validate`, `clean`, `eda`, `features`, `train_reg`, `train_clf`, `explain`, `report`, `all`.
