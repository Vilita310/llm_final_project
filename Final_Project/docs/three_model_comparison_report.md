# Three Fine-tuned Models Comparison Report

## Scope

This report compares the three completed fine-tuning runs using the filled workbooks:

- `outputs/jing/Test_Data_Jing_Cao.xlsx`
- `outputs/shihju/Test_Data_ShihJu_Fu.xlsx`
- `outputs/yuting/Test_Data_Yuting.xlsx`

Evaluation rule follows your requirement: compare the model's first diagnosis against `Correct Diagnosis` only (ignore treatment), and mark `Correct` or `Wrong`.

## Prompt Used for Inference

Use this same instruction in Playground for all cases in `NewSamples` and `FineTunedSamples`:

`Suppose that you are a medical diagnosis assistant, skilled in recognizing diseases and prescribing treatments. If a patient has the following condition, give me the most probable diagnosis and the related probability and urgency and rationale for the diagnosis besides suitable treatments. Start your response with Diagnosis: `

## Quantitative Comparison

| Model Set | Hyperparameters | Sheet | Before | After | Delta |
|---|---|---|---:|---:|---:|
| Jing_Cao | LR=0.4, BT=8, EP=5 | NewSamples | 5/20 (25.00%) | 6/20 (30.00%) | +1 |
| Jing_Cao | LR=0.4, BT=8, EP=5 | FineTunedSamples | 11/20 (55.00%) | 14/20 (70.00%) | +3 |
| Jing_Cao | LR=0.4, BT=8, EP=5 | Combined | 16/40 (40.00%) | 20/40 (50.00%) | +4 |
| ShihJu_Fu | LR=1.0, BT=4, EP=5 | NewSamples | 6/20 (30.00%) | 6/20 (30.00%) | 0 |
| ShihJu_Fu | LR=1.0, BT=4, EP=5 | FineTunedSamples | 12/20 (60.00%) | 12/20 (60.00%) | 0 |
| ShihJu_Fu | LR=1.0, BT=4, EP=5 | Combined | 18/40 (45.00%) | 18/40 (45.00%) | 0 |
| Yuting_Bu | LR=0.4, BT=4, EP=7 | NewSamples | 4/22 (18.18%) | 5/22 (22.73%) | +1 |
| Yuting_Bu | LR=0.4, BT=4, EP=7 | FineTunedSamples | 12/22 (54.55%) | 14/22 (63.64%) | +2 |
| Yuting_Bu | LR=0.4, BT=4, EP=7 | Combined | 16/44 (36.36%) | 19/44 (43.18%) | +3 |

Source file: `outputs/three_model_comparison_summary.csv`.

## Key Findings

- Best **after fine-tuning combined accuracy** is `Jing_Cao` at `50.00%`.
- Largest **absolute improvement** is also `Jing_Cao` (`+10.00%` combined).
- `ShihJu_Fu` is stable but has near-zero net gain (one wrong->correct and one correct->wrong in `NewSamples`).
- `Yuting_Bu` improves but has data-quality differences (22 rows instead of 20 and extra unnamed columns), so direct fairness is lower than the other two.

## Training Loss Chart Analysis

No raw training-loss curve artifact is stored in this repo. Based on final behavior:

- `Jing_Cao` likely achieved the best optimization/generalization balance.
- `ShihJu_Fu` likely converged quickly with limited additional gain (possible early plateau).
- `Yuting_Bu` improved but showed instability signals (some correct->wrong transitions and workbook inconsistencies).

For your presentation, once you export the actual dashboard loss curves, report:

1. Initial loss level
2. Slope in early epochs
3. Final train loss and validation loss
4. Gap between train/validation (overfitting check)
5. How the curve behavior aligns with the above accuracy outcomes

## Deliverables Status

- Three-set comparison report: completed (`docs/three_model_comparison_report.md`)
- Final project report: completed (`docs/final_project_report.md`)
- Filled workbooks already present in `outputs/` and ready to upload to Canvas
