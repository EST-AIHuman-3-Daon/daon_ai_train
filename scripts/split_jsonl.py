import random
from pathlib import Path


# =========================================================
# JSONL 데이터셋을 train / valid 로 분리하는 함수
# =========================================================
#
# 목적:
# - Qwen / Llama / Gemma 등 LLM 학습용 데이터 분리
# - train.jsonl / valid.jsonl 자동 생성
#
# 예시 결과:
#
# data/
# └── pressure_ict/
#     ├── train.jsonl
#     └── valid.jsonl
#
# =========================================================
def split_jsonl_dataset(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.9,
    seed: int = 42,
):
    """
    JSONL 파일을 train / valid 데이터셋으로 분리

    Args:
        input_path (str):
            원본 JSONL 파일 경로

        output_dir (str):
            train.jsonl / valid.jsonl 저장 폴더

        train_ratio (float):
            train 데이터 비율
            기본값: 0.9 (90%)

        seed (int):
            랜덤 시드
            팀원 간 동일 결과 재현용
    """

    # 랜덤 시드 고정
    random.seed(seed)

    # pathlib 객체 생성
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # output 폴더 자동 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # JSONL 전체 읽기
    # =====================================================
    with open(input_path, "r", encoding="utf-8") as f:

        # 빈 줄 제외
        lines = [
            line
            for line in f
            if line.strip()
        ]

    # =====================================================
    # 데이터 랜덤 셔플
    # =====================================================
    random.shuffle(lines)

    # =====================================================
    # train / valid 개수 계산
    # =====================================================
    train_size = int(len(lines) * train_ratio)

    # train / valid 분리
    train_lines = lines[:train_size]
    valid_lines = lines[train_size:]

    # =====================================================
    # 저장 경로 생성
    # =====================================================
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    # =====================================================
    # train 저장
    # =====================================================
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    # =====================================================
    # valid 저장
    # =====================================================
    with open(valid_path, "w", encoding="utf-8") as f:
        f.writelines(valid_lines)

    # =====================================================
    # 결과 출력
    # =====================================================
    print("=" * 50)
    print("[데이터셋 분리 완료]")
    print(f"원본 파일: {input_path}")
    print(f"저장 폴더: {output_dir}")
    print("-" * 50)
    print(f"train 개수: {len(train_lines)}")
    print(f"valid 개수: {len(valid_lines)}")
    print("=" * 50)


# =========================================================
# 실행 예시
# =========================================================
if __name__ == "__main__":

    # qwen_pressure_ict.jsonl 을
    # pressure_ict 폴더 내부의
    # train.jsonl / valid.jsonl 로 분리

    split_jsonl_dataset(
        input_path="./data/qwen_friendly_ict.jsonl",
        output_dir="./data/friendly_ict",
    )