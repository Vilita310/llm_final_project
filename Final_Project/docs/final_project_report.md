# Final Project Report - Medical Diagnosis Fine-tuning

## Project Goal

Fine-tune `gpt-4o-mini-2024-07-18` for medical diagnosis tasks, then compare model diagnosis quality before and after fine-tuning using the provided `Test_Data.xlsx` framework.

## Data and Pipeline

- Training data: `data/raw/FineTuning_Data.csv`
- Validation data: `data/raw/Validation_Data.csv`
- Test workbook template: `data/raw/Test_Data.xlsx`
- Core pipeline: `src/pipeline.py`

Main workflow in the pipeline:

1. Prefix all input cases with a diagnosis instruction
2. Convert CSV to JSONL for fine-tuning
3. Run fine-tuning jobs with different hyperparameters
4. Evaluate `NewSamples` and `FineTunedSamples`
5. Extract first diagnosis from model output
6. Compare to `Correct Diagnosis` and mark `Correct/Wrong`
7. Fill `Overall` with before/after counts and transitions

## Inference Prompt Used

`Suppose that you are a medical diagnosis assistant, skilled in recognizing diseases and prescribing treatments. If a patient has the following condition, give me the most probable diagnosis and the related probability and urgency and rationale for the diagnosis besides suitable treatments. Start your response with Diagnosis: `

This prompt is applied consistently to all test cases.

## Model Configurations Compared

- `Jing_Cao`: LR=0.4, Batch=8, Epochs=5
- `ShihJu_Fu`: LR=1.0, Batch=4, Epochs=5
- `Yuting_Bu`: LR=0.4, Batch=4, Epochs=7

## Results Summary

Combined before/after diagnosis accuracy:

- `Jing_Cao`: 40.00% -> 50.00% (`+10.00%`, best)
- `ShihJu_Fu`: 45.00% -> 45.00% (`+0.00%`)
- `Yuting_Bu`: 36.36% -> 43.18% (`+6.82%`)

Per-sheet pattern:

- `FineTunedSamples` improves more than `NewSamples` for `Jing_Cao` and `Yuting_Bu`
- `ShihJu_Fu` shows no net gain, suggesting under-generalized tuning impact

## Interpretation

- Moderate learning rate with moderate epochs and larger batch (`Jing_Cao`) produced the most reliable improvement.
- Higher learning rate (`ShihJu_Fu`, LR=1.0) may have reduced stability or harmed transfer to test evaluation quality.
- More epochs with smaller batch (`Yuting_Bu`) helped fit fine-tuned-style cases, but variability remains.

## Training Loss Discussion

Training-loss chart data is not saved in this repository, so final reporting currently uses post-training evaluation outcomes.

When dashboard loss curves are available, include:

- Train vs validation loss trajectory
- Epoch of stabilization
- Overfitting signs (widening train-valid gap)
- Correlation with `NewSamples` generalization performance

## Data Quality / Risk Notes

- `outputs/yuting/Test_Data_Yuting.xlsx` has 22 rows and extra unnamed columns in some sheets, while the other two workbooks have 20 rows and clean schema.
- This makes strict cross-model fairness weaker; normalize row counts/columns before final presentation if possible.

## Final Conclusion

- Best performing fine-tuned configuration in this run: `Jing_Cao (LR=0.4, BT=8, EP=5)`.
- Recommended candidate for deployment/demo: `Jing_Cao` model output style and prompt template.
- Next improvement steps: diagnosis synonym normalization, stricter extraction of first diagnosis phrase, and consistent test workbook schema.
