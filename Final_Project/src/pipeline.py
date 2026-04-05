import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


INSTRUCTION_PREFIX = (
    "Suppose that you are a medical diagnosis assistant, skilled in recognizing diseases "
    "and prescribing treatments. If a patient has the following condition, give me the most "
    "probable diagnosis and the related probability and urgency and rationale for the diagnosis "
    "besides suitable treatments. Start your response with Diagnosis: "
)


@dataclass
class HyperParams:
    learning_rate_multiplier: float
    n_epochs: int
    batch_size: int


PRESET_STUDENT_CONFIGS: Dict[str, HyperParams] = {
    "jing_cao": HyperParams(learning_rate_multiplier=0.4, n_epochs=5, batch_size=8),
    "shih_ju_fu": HyperParams(learning_rate_multiplier=1.0, n_epochs=5, batch_size=4),
    "yuting_bu": HyperParams(learning_rate_multiplier=0.4, n_epochs=7, batch_size=4),
}


def _read_csv_with_fallback(path: str) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _ensure_prefixed_query(text: str, prefix: str) -> str:
    text = "" if pd.isna(text) else str(text)
    if text.startswith(prefix):
        return text
    return f"{prefix}{text}"


def preprocess_csv(input_csv: str, output_csv: str, prefix: str = INSTRUCTION_PREFIX) -> None:
    df = _read_csv_with_fallback(input_csv)
    if "Query" not in df.columns or "Response" not in df.columns:
        raise ValueError(f"{input_csv} must contain Query and Response columns.")
    df["Query"] = df["Query"].apply(lambda x: _ensure_prefixed_query(x, prefix))
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")


def save_to_jsonl(data: pd.DataFrame, output_file_path: str, query_title: str = "user", response_title: str = "assistant") -> None:
    with open(output_file_path, "w", encoding="utf-8") as f:
        for _, row in data.iterrows():
            item = {
                "messages": [
                    {"role": query_title, "content": str(row["Query"])},
                    {"role": response_title, "content": f"\"{str(row['Response'])}\""},
                ]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def preprocess_and_convert(input_csv: str, output_prefixed_csv: str, output_jsonl: str, prefix: str = INSTRUCTION_PREFIX) -> None:
    preprocess_csv(input_csv, output_prefixed_csv, prefix=prefix)
    df = _read_csv_with_fallback(output_prefixed_csv)
    save_to_jsonl(df, output_jsonl)


def _normalize_diagnosis(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = text.lower()
    text = re.sub(r"[\*\-:;,\.\(\)\[\]\{\}\"']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_first_diagnosis(model_output: str) -> str:
    if model_output is None:
        return ""
    text = str(model_output).strip()
    match = re.search(r"diagnosis\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        remainder = match.group(1).strip()
    else:
        remainder = text
    first_line = remainder.splitlines()[0].strip()
    first_line = re.split(r"\s{2,}|\* probability|\* emergency|\* rationale|suitable treatments", first_line, flags=re.IGNORECASE)[0].strip()
    first_line = first_line.rstrip(" .,:;-")
    return first_line


def _invoke_model(client: OpenAI, model: str, query: str, system_prompt: Optional[str], temperature: float = 0.0) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=90,
    )
    return resp.choices[0].message.content or ""


def evaluate_test_workbook(
    workbook_path: str,
    model: str,
    output_path: str,
    system_prompt: Optional[str] = None,
    with_instruction_prefix: bool = True,
    response_column_name: str = "Response of Engine Before Fine-tuning",
    result_column_name: str = "Result of Diagnosis (Correct /Wrong ) Before Fine-tuning",
) -> Dict[str, Tuple[int, int, float]]:
    if OpenAI is None:
        raise ImportError("openai package is required. Install with: pip install openai")
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    xls = pd.ExcelFile(workbook_path)
    sheet_names = xls.sheet_names
    targets = [s for s in sheet_names if s.strip().lower() in {"newsamples", "finetunedsamples", "new samples", "finetuned samples"}]
    if not targets:
        raise ValueError("Workbook must include sheets named 'New Samples' and 'Finetuned Samples'.")

    summary: Dict[str, Tuple[int, int, float]] = {}
    output_book: Dict[str, pd.DataFrame] = {}
    for name in sheet_names:
        output_book[name] = pd.read_excel(workbook_path, sheet_name=name)

    for sheet_name in targets:
        df = output_book[sheet_name]
        needed = {"Patient Conditions", "Correct Diagnosis"}
        if not needed.issubset(df.columns):
            raise ValueError(f"{sheet_name} missing columns: {needed - set(df.columns)}")

        results: List[str] = []
        model_outputs: List[str] = []
        correct_count = 0
        total_count = 0

        total_rows = len(df)
        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            print(f"[{sheet_name}] {idx}/{total_rows} -> querying {model}")
            query = "" if pd.isna(row["Patient Conditions"]) else str(row["Patient Conditions"])
            if not query.strip():
                model_outputs.append("")
                results.append("")
                continue

            final_query = _ensure_prefixed_query(query, INSTRUCTION_PREFIX) if with_instruction_prefix else query
            output = _invoke_model(client, model=model, query=final_query, system_prompt=system_prompt)
            first_diag = extract_first_diagnosis(output)

            expected = "" if pd.isna(row["Correct Diagnosis"]) else str(row["Correct Diagnosis"])
            is_correct = _normalize_diagnosis(first_diag) == _normalize_diagnosis(expected)

            results.append("Correct" if is_correct else "Wrong")
            model_outputs.append(output)

            total_count += 1
            if is_correct:
                correct_count += 1

        accuracy = (correct_count / total_count * 100.0) if total_count else 0.0
        summary[sheet_name] = (correct_count, total_count, accuracy)
        df[response_column_name] = model_outputs
        df[result_column_name] = results
        output_book[sheet_name] = df

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in output_book.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return summary


def _count_transitions(before_series: pd.Series, after_series: pd.Series) -> Tuple[int, int]:
    changed_wrong_to_correct = 0
    changed_correct_to_wrong = 0
    for b, a in zip(before_series.fillna(""), after_series.fillna("")):
        b_clean = str(b).strip().lower()
        a_clean = str(a).strip().lower()
        if b_clean == "wrong" and a_clean == "correct":
            changed_wrong_to_correct += 1
        elif b_clean == "correct" and a_clean == "wrong":
            changed_correct_to_wrong += 1
    return changed_wrong_to_correct, changed_correct_to_wrong


def fill_overall_sheet(workbook_path: str, output_path: str, settings_label: str) -> None:
    sheets = pd.read_excel(workbook_path, sheet_name=None)
    if "NewSamples" not in sheets or "FineTunedSamples" not in sheets:
        raise ValueError("Workbook must include NewSamples and FineTunedSamples sheets.")

    new_df = sheets["NewSamples"]
    ft_df = sheets["FineTunedSamples"]
    overall_cols = [
        "Settings (LR=X, BT=Y, EP=Z)",
        "Correct Responses in New Samples (Before Fine-tuning)",
        "Correct Responses in New Samples (After Fine-tuning)",
        "Responses Changed from Wrong to Correct After Fine-tuning (New Samples)",
        "Responses Changed from Correct to Wrong  After Fine-tuning (New Samples)",
        "Correct Responses in FineTuned Samples  (Before Fine-tuning)",
        "Correct Responses in FineTuned Samples  (After Fine-tuning)",
        "Responses Changed from Wrong to Correct After Fine-tuning (FineTuned Samples)",
        "Responses Changed from Correct to Wrong  After Fine-tuning (FineTuned Samples)",
    ]

    new_before = (new_df["Result of Diagnosis (Correct /Wrong ) Before Fine-tuning"].fillna("").str.strip().str.lower() == "correct").sum()
    new_after = (new_df["Result of Diagnosis (Correct /Wrong ) After Fine-tuning"].fillna("").str.strip().str.lower() == "correct").sum()
    new_w2c, new_c2w = _count_transitions(
        new_df["Result of Diagnosis (Correct /Wrong ) Before Fine-tuning"],
        new_df["Result of Diagnosis (Correct /Wrong ) After Fine-tuning"],
    )

    ft_before = (ft_df["Result of Diagnosis (Correct /Wrong ) Before Fine-tuning"].fillna("").str.strip().str.lower() == "correct").sum()
    ft_after = (ft_df["Result of Diagnosis (Correct /Wrong ) After Fine-tuning"].fillna("").str.strip().str.lower() == "correct").sum()
    ft_w2c, ft_c2w = _count_transitions(
        ft_df["Result of Diagnosis (Correct /Wrong ) Before Fine-tuning"],
        ft_df["Result of Diagnosis (Correct /Wrong ) After Fine-tuning"],
    )

    overall_row = pd.DataFrame(
        [[settings_label, new_before, new_after, new_w2c, new_c2w, ft_before, ft_after, ft_w2c, ft_c2w]],
        columns=overall_cols,
    )
    sheets["Overall"] = overall_row

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


def evaluate_before_after(
    workbook_path: str,
    base_model: str,
    finetuned_model: str,
    output_path: str,
    settings_label: str,
    system_prompt: Optional[str] = None,
) -> None:
    temp_after_before = output_path.replace(".xlsx", "_tmp_before.xlsx")

    evaluate_test_workbook(
        workbook_path=workbook_path,
        model=base_model,
        output_path=temp_after_before,
        system_prompt=system_prompt,
        with_instruction_prefix=True,
        response_column_name="Response of Engine Before Fine-tuning",
        result_column_name="Result of Diagnosis (Correct /Wrong ) Before Fine-tuning",
    )
    evaluate_test_workbook(
        workbook_path=temp_after_before,
        model=finetuned_model,
        output_path=output_path,
        system_prompt=system_prompt,
        with_instruction_prefix=True,
        response_column_name="Response of Engine After Fine-tuning",
        result_column_name="Result of Diagnosis (Correct /Wrong ) After Fine-tuning",
    )
    fill_overall_sheet(output_path, output_path, settings_label=settings_label)
    if os.path.exists(temp_after_before):
        os.remove(temp_after_before)


def launch_finetune(
    training_jsonl: str,
    validation_jsonl: str,
    base_model: str,
    suffix: str,
    hp: HyperParams,
) -> str:
    if OpenAI is None:
        raise ImportError("openai package is required. Install with: pip install openai")
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    with open(training_jsonl, "rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    with open(validation_jsonl, "rb") as f:
        val_file = client.files.create(file=f, purpose="fine-tune")

    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=val_file.id,
        model=base_model,
        suffix=suffix,
        hyperparameters={
            "learning_rate_multiplier": hp.learning_rate_multiplier,
            "n_epochs": hp.n_epochs,
            "batch_size": hp.batch_size,
        },
    )
    return job.id


def wait_for_finetune_job(job_id: str, poll_seconds: int = 30) -> Dict[str, Optional[str]]:
    if OpenAI is None:
        raise ImportError("openai package is required. Install with: pip install openai")
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    client = OpenAI()

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = getattr(job, "status", None)
        if status in {"succeeded", "failed", "cancelled"}:
            return {
                "status": status,
                "fine_tuned_model": getattr(job, "fine_tuned_model", None),
                "trained_tokens": getattr(job, "trained_tokens", None),
            }
        print(f"Job {job_id} status: {status}. Waiting {poll_seconds}s...")
        time.sleep(poll_seconds)


def run_three_student_sets(
    workbook_path: str,
    train_jsonl: str,
    val_jsonl: str,
    base_model: str,
    output_dir: str,
    system_prompt: Optional[str] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    comparison_rows = []

    for student_key, hp in PRESET_STUDENT_CONFIGS.items():
        suffix = f"group3-{student_key.replace('_', '-')}"
        print(f"\n=== Running set for {student_key} with LR={hp.learning_rate_multiplier}, EP={hp.n_epochs}, BS={hp.batch_size} ===")
        job_id = launch_finetune(
            training_jsonl=train_jsonl,
            validation_jsonl=val_jsonl,
            base_model=base_model,
            suffix=suffix,
            hp=hp,
        )
        print(f"Created fine-tune job: {job_id}")
        result = wait_for_finetune_job(job_id)
        if result["status"] != "succeeded" or not result["fine_tuned_model"]:
            comparison_rows.append(
                {
                    "student": student_key,
                    "lr": hp.learning_rate_multiplier,
                    "epochs": hp.n_epochs,
                    "batch_size": hp.batch_size,
                    "job_id": job_id,
                    "status": result["status"],
                    "fine_tuned_model": result["fine_tuned_model"],
                    "output_file": "",
                }
            )
            continue

        model_id = str(result["fine_tuned_model"])
        output_file = os.path.join(output_dir, f"Test_Data_{student_key}.xlsx")
        settings_label = f"LR={hp.learning_rate_multiplier}, BT={hp.batch_size}, EP={hp.n_epochs}"
        evaluate_before_after(
            workbook_path=workbook_path,
            base_model=base_model,
            finetuned_model=model_id,
            output_path=output_file,
            settings_label=settings_label,
            system_prompt=system_prompt,
        )
        comparison_rows.append(
            {
                "student": student_key,
                "lr": hp.learning_rate_multiplier,
                "epochs": hp.n_epochs,
                "batch_size": hp.batch_size,
                "job_id": job_id,
                "status": result["status"],
                "fine_tuned_model": model_id,
                "output_file": output_file,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv = os.path.join(output_dir, "three_set_comparison.csv")
    comparison_df.to_csv(comparison_csv, index=False, encoding="utf-8-sig")
    return comparison_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning and evaluation workflow helper.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Prefix instruction and convert CSV -> JSONL.")
    prep.add_argument("--train-csv", default="data/raw/FineTuning_Data.csv")
    prep.add_argument("--val-csv", default="data/raw/Validation_Data.csv")
    prep.add_argument("--train-prefixed", default="data/processed/FineTuning_Data_prefixed.csv")
    prep.add_argument("--val-prefixed", default="data/processed/Validation_Data_prefixed.csv")
    prep.add_argument("--train-jsonl", default="data/processed/FineTuning_Data.jsonl")
    prep.add_argument("--val-jsonl", default="data/processed/Validation_Data.jsonl")

    ft = sub.add_parser("finetune", help="Launch one fine-tune job.")
    ft.add_argument("--train-jsonl", default="data/processed/FineTuning_Data.jsonl")
    ft.add_argument("--val-jsonl", default="data/processed/Validation_Data.jsonl")
    ft.add_argument("--base-model", default="gpt-4o-mini-2024-07-18")
    ft.add_argument("--suffix", default="group3-jingcao-ft1")
    ft.add_argument("--lr", type=float, required=True)
    ft.add_argument("--epochs", type=int, required=True)
    ft.add_argument("--batch-size", type=int, required=True)

    ev = sub.add_parser("evaluate", help="Evaluate workbook sheets.")
    ev.add_argument("--workbook", required=True)
    ev.add_argument("--model", required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--result-column", default="Result of Diagnosis (Correct /Wrong ) Before Fine-tuning")
    ev.add_argument("--response-column", default="Response of Engine Before Fine-tuning")
    ev.add_argument("--system-prompt", default=None)

    ev2 = sub.add_parser("evaluate-before-after", help="Run base + fine-tuned evaluation and fill Overall.")
    ev2.add_argument("--workbook", required=True)
    ev2.add_argument("--base-model", required=True)
    ev2.add_argument("--finetuned-model", required=True)
    ev2.add_argument("--output", required=True)
    ev2.add_argument("--settings-label", required=True)
    ev2.add_argument("--system-prompt", default=None)

    batch = sub.add_parser("run-three-sets", help="Run Jing/Shih-Ju/Yuting sets end-to-end.")
    batch.add_argument("--workbook", default="data/raw/Test_Data.xlsx")
    batch.add_argument("--train-jsonl", default="data/processed/FineTuning_Data.jsonl")
    batch.add_argument("--val-jsonl", default="data/processed/Validation_Data.jsonl")
    batch.add_argument("--base-model", default="gpt-4o-mini-2024-07-18")
    batch.add_argument("--output-dir", default="outputs/three_sets")
    batch.add_argument("--system-prompt", default=None)

    args = parser.parse_args()

    if args.cmd == "prepare":
        preprocess_and_convert(args.train_csv, args.train_prefixed, args.train_jsonl)
        preprocess_and_convert(args.val_csv, args.val_prefixed, args.val_jsonl)
        print("Prepared prefixed CSV and JSONL files for training and validation.")
    elif args.cmd == "finetune":
        hp = HyperParams(args.lr, args.epochs, args.batch_size)
        job_id = launch_finetune(
            training_jsonl=args.train_jsonl,
            validation_jsonl=args.val_jsonl,
            base_model=args.base_model,
            suffix=args.suffix,
            hp=hp,
        )
        print(f"Fine-tune job created: {job_id}")
    elif args.cmd == "evaluate":
        summary = evaluate_test_workbook(
            workbook_path=args.workbook,
            model=args.model,
            output_path=args.output,
            system_prompt=args.system_prompt,
            with_instruction_prefix=True,
            response_column_name=args.response_column,
            result_column_name=args.result_column,
        )
        for sheet, (c, t, a) in summary.items():
            print(f"{sheet}: {c}/{t} correct ({a:.2f}%)")
    elif args.cmd == "evaluate-before-after":
        evaluate_before_after(
            workbook_path=args.workbook,
            base_model=args.base_model,
            finetuned_model=args.finetuned_model,
            output_path=args.output,
            settings_label=args.settings_label,
            system_prompt=args.system_prompt,
        )
        print(f"Completed before/after evaluation: {args.output}")
    elif args.cmd == "run-three-sets":
        comparison_csv = run_three_student_sets(
            workbook_path=args.workbook,
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            base_model=args.base_model,
            output_dir=args.output_dir,
            system_prompt=args.system_prompt,
        )
        print(f"Completed three-set run. Comparison file: {comparison_csv}")


if __name__ == "__main__":
    main()
