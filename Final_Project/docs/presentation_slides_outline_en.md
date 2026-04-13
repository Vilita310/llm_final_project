# Presentation Slide Outline

## Slide 1 - Project Title

- Medical Diagnosis Fine-tuning with `gpt-4o-mini-2024-07-18`
- INFO7374 Final Project
- Team comparison across three fine-tuning settings
- Objective: improve diagnosis accuracy on structured test cases

## Slide 2 - Project Objective

- Fine-tune a base model for medical diagnosis tasks
- Evaluate model outputs on `NewSamples` and `FineTunedSamples`
- Compare first diagnosis with `Correct Diagnosis` only
- Mark each case as `Correct` or `Wrong` and summarize in `Overall`

## Slide 3 - Data and Workflow

- Training set: `FineTuning_Data.csv`
- Validation set: `Validation_Data.csv`
- Evaluation workbook: `Test_Data.xlsx`
- Pipeline steps: preprocess -> JSONL -> fine-tune -> evaluate -> summarize

## Slide 4 - Inference Prompt (Playground)

- We used one consistent prompt for all test cases:
- "Suppose that you are a medical diagnosis assistant... Start your response with Diagnosis:"
- Same instruction applied to both `NewSamples` and `FineTunedSamples`
- Purpose: reduce output variance and enforce comparable formatting

## Slide 5 - Hyperparameter Sets Compared

- **Jing_Cao:** LR=0.4, Batch=8, Epochs=5
- **ShihJu_Fu:** LR=1.0, Batch=4, Epochs=5
- **Yuting_Bu:** LR=0.4, Batch=4, Epochs=7
- Base model for all runs: `gpt-4o-mini-2024-07-18`

## Slide 6 - Evaluation Rule

- Extract the first diagnosis from each model response
- Compare to `Correct Diagnosis` (ignore treatment)
- If matched -> `Correct`; otherwise -> `Wrong`
- Track transitions: `Wrong->Correct` and `Correct->Wrong`

## Slide 7 - Results by Set (Combined Accuracy)

- **Jing_Cao:** 40.00% -> 50.00% (**+10.00%**)
- **ShihJu_Fu:** 45.00% -> 45.00% (**+0.00%**)
- **Yuting_Bu:** 36.36% -> 43.18% (**+6.82%**)
- Best final and best improvement: **Jing_Cao**

## Slide 8 - Sheet-level Observations

- `FineTunedSamples` improved more than `NewSamples` for Jing_Cao and Yuting_Bu
- ShihJu_Fu remained stable with nearly zero net gain
- Jing_Cao: strong gains with no `Correct->Wrong` on combined set
- Suggests balanced learning behavior under moderate LR and batch size

## Slide 9 - Training Loss Chart Analysis

- Insert dashboard plot: training loss vs validation loss per epoch
- Report: initial loss, slope, final loss, and train-valid gap
- Interpretation template:
- Small final gap + smooth decline -> better generalization
- Diverging gap -> potential overfitting

## Slide 10 - Why Jing_Cao Performed Best

- Moderate LR avoids unstable updates
- Batch size 8 likely gives smoother gradient estimates
- 5 epochs appears sufficient for convergence without obvious drift
- Best practical trade-off between adaptation and generalization

## Slide 11 - Risks and Data Quality Notes

- Yuting workbook has 22 rows and extra unnamed columns
- Not fully schema-aligned with other two outputs (20 rows each)
- This may affect strict fairness in cross-model comparison
- Recommendation: normalize workbook schema before final benchmarking

## Slide 12 - Final Conclusion and Next Steps

- Selected best setting: **LR=0.4, Batch=8, Epochs=5 (Jing_Cao)**
- Fine-tuning improves diagnosis quality, especially on seen-style cases
- Next steps: diagnosis synonym normalization and stricter diagnosis extraction
- Deliverables submitted: filled test workbooks + comparison report + final report
