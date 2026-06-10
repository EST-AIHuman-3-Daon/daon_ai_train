import glob
import argparse
import pandas as pd
from evaluators.style_rules import validate_row

aug_files = sorted(
    glob.glob("data/processed/sft_augmented_dataset_*.csv")
)

if not aug_files:
    raise FileNotFoundError(
        "augmentation 결과 CSV가 없습니다."
    )

DEFAULT_INPUT_CSV = aug_files[-1]
from datetime import datetime

run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

DEFAULT_OUTPUT_CSV = (
    f"experiments/qa_results/style_validation_{run_tag}.csv"
)


def main(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results = [validate_row(row) for row in df.to_dict("records")]

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(result_df.groupby("persona")["score"].describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    main(args.input_csv, args.output_csv)