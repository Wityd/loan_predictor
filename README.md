
# Loan Default Prediction (Zindi / Data Science Nigeria Challenge)

Predicting whether a loan will be repaid ("Good") or not ("Bad") for SuperLender, a digital lending company, using the applicant's current loan, demographics and repayment history.

- **Task:** binary classification (Good = 1, Bad = 0)
- **Metric:** percentage of loans predicted incorrectly (lower is better)
- **Competition:** Data Science Nigeria Challenge #1: Loan Default Prediction on [Zindi][The direct link is here](https://zindi.africa/) (](https://zindi.world/competitions/data-science-nigeria-challenge-1-loan-default-prediction))
- **Author:** Witness Lubisi

## Results

| Approach | Error |
|---|---|
| Always predict "Good" (baseline, on training data) | 21.8% |
| 9 features, default model (5-fold CV) | 21.57% |
| 10 features, adding `loannumber` (5-fold CV) | 21.93% |
| **5 features, constrained model, threshold 0.5 (5-fold CV)** | **20.67%** |
| **Zindi public leaderboard** | **20.90%** |

The leaderboard score landed within 0.25 points of the cross-validation estimate, which suggests the train and test pipelines are consistent. The improvement over the baseline is real but modest. The signal in this dataset is weak, and I report it as it is.

## Approach

The brief frames credit risk around two drivers, **willingness to pay** and **ability to pay**. The features follow that split.

**Willingness to pay (from previous loans)**
For each past loan I measured `days_late` as the first repayment date minus the first due date. I then collapsed the history to one row per customer:

- `mean_late`: average days late
- `max_late`: worst case days late
- `late_share`: share of past loans repaid late
- `n_prev`: number of past loans (tested, later dropped)

**Ability to pay (from the current loan)**

- `interest_ratio` = `totaldue / loanamount`
- `loannumber`

**Model**
`HistGradientBoostingClassifier`, deliberately constrained because the signal is weak:

```python
HistGradientBoostingClassifier(
    max_depth=3, learning_rate=0.05, max_iter=150,
    min_samples_leaf=50, l2_regularization=1.0, random_state=0)
```

Predictions are made from the predicted probability with a 0.5 cut-off (0.4 was worse: 21.2% error).

## What I learned

- **More features made it worse.** A flexible model with 10 features scored 21.93% because it fit noise. A constrained model with 5 features scored 20.67%.
- **Permutation importance on training data was misleading.** `age` ranked 3rd when measured on the data the model was trained on, but fell to roughly zero when measured on held-out data. Always measure on data the model has not seen.
- **Missing age carried no signal.** About a quarter of customers have no demographics, but their Good rate (78.3%) is nearly identical to customers with demographics (78.2%).
- **Data quirks matter:**
  - `traindemographics.csv` has 12 duplicate customers. Merging without deduplicating would duplicate rows and misalign labels.
  - The date columns in `testperf.csv` are corrupted, so I did not use them.
  - About 8 test customers have no previous loans. The model handles these as missing values, so no test rows are dropped.

## Project structure

```
.
├── README.md
├── requirements.txt
├── loan_default.ipynb      # full workflow: explore, features, CV, test predictions
├── .gitignore              # excludes data/ and submission files
└── data/                   # not included, see below
```

## How to run

1. Download the competition files from the Zindi competition page into a `data/` folder. They are not included here because of the competition's data rules.
2. Unzip `trainprevloans.zip` and `testprevloans.zip` inside `data/`.
3. Install dependencies and open the notebook:

```bash
pip install -r requirements.txt
jupyter notebook loan_default.ipynb
```

4. Run all cells. The last cell writes `submission.csv` in the format Zindi expects (`customerid`, `Good_Bad_flag`).

## Next steps

- Add a repayment-closing feature: how long after the expected end date (`approveddate` + `termdays`) each past loan was closed, using `closeddate`.
- Wrap the history feature steps in one function applied to both train and test.
- Tune the threshold and try other models (LightGBM, XGBoost, logistic regression as a sanity check).
- Compare a "new customer" model against a "repeat customer" model, as the brief describes.

## Tools

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn.
