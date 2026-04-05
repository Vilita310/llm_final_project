# INFO7374 LLM Final Project

## Project Structure

- `src/`
  - `pipeline.py`: data prep, fine-tuning, and evaluation script.
- `data/raw/`
  - `FineTuning_Data.csv`
  - `Validation_Data.csv`
  - `Test_Data.xlsx`
  - `Settings.xlsx`
- `data/processed/`
  - `FineTuning_Data.jsonl`
  - `Validation_Data.jsonl`
  - (generated) `FineTuning_Data_prefixed.csv`
  - (generated) `Validation_Data_prefixed.csv`
- `outputs/jing/`
  - `Test_Data_Jing_Cao.xlsx` (final evaluated workbook)
- `docs/`
  - optional presentation notes/materials.

## Common Commands

Run from project root:

```bash
python3 src/pipeline.py prepare
```

```bash
python3 src/pipeline.py finetune --lr 0.4 --epochs 5 --batch-size 8 --suffix group3-jing-cao
```

```bash
python3 src/pipeline.py evaluate-before-after \
  --workbook data/raw/Test_Data.xlsx \
  --base-model gpt-4o-mini-2024-07-18 \
  --finetuned-model <your-finetuned-model-id> \
  --settings-label "LR=0.4, BT=8, EP=5" \
  --output outputs/jing/Test_Data_Jing_Cao.xlsx
```
