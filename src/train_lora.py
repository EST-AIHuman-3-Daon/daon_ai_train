"""
train_lora.py
- pressure / friendly LoRA 어댑터 순차 학습
- 학습용 간소화 프롬프트 사용
- A100 환경 최적화 (bfloat16, load_in_4bit=False)

실행: cd daon_project && python src/train_lora.py
"""

import os
import json
import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from transformers import TrainerCallback

# ───────────────────────────────────────────
# 0. 경로 설정
# ───────────────────────────────────────────

MODEL_VERSION = "Qwen3.5-9B"

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data" / "processed"

BASE_ADAPTER_DIR = BASE_DIR /"adapters"/ MODEL_VERSION
BASE_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH   = DATA_DIR / "val.csv"

# ───────────────────────────────────────────
# 1. 공통 설정
# ───────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen3.5-9B"
MAX_SEQ_LENGTH = 2048
PERSONAS       = ["pressure", "friendly"]

# ───────────────────────────────────────────
# 2. 학습용 간소화 프롬프트
#    (규칙 암기 대신 페르소나 스타일 내재화 목적)
# ───────────────────────────────────────────
TRAIN_PROMPTS = {
    "pressure": (
        "당신은 대기업 압박 면접관입니다. "
        "지원자 답변의 모호한 부분을 지적하고 "
        "근거와 사례를 요구하는 꼬리질문을 생성하세요."
    ),
    "friendly": (
        "당신은 코칭형 친절 면접관입니다. "
        "지원자를 배려하며 답변을 구체화하도록 "
        "유도하는 꼬리질문을 생성하세요."
    )
}

# ───────────────────────────────────────────
# 3. ChatML 포맷 변환
# ───────────────────────────────────────────
def to_chatml(row: dict) -> str:
    """
    학습 데이터를 Qwen ChatML 포맷으로 변환
    <|im_start|>system ... <|im_end|>
    <|im_start|>user   ... <|im_end|>
    <|im_start|>assistant ... <|im_end|>
    """
    persona = row["persona"].replace("_interviewer", "")

    inp = json.loads(row["input"])
    question         = inp.get("question", "")
    candidate_answer = inp.get("candidate_answer", "")

    system_msg = (
        f"직무: {row['job_role']}\n\n"
        f"{TRAIN_PROMPTS[persona]}"
    )
    user_msg = (
        f"면접 질문: {question}\n"
        f"지원자 답변: {candidate_answer}"
    )
    asst_msg = row["output"]

    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{asst_msg}<|im_end|>"
    )

# ───────────────────────────────────────────
# 4. 데이터 로드
# ───────────────────────────────────────────

def load_dataset(persona: str, path: Path) -> Dataset:
    df = pd.read_csv(path, encoding="utf-8-sig")
    
    # 필요한 컬럼만 선택
    df = df[["persona", "job_role", "input", "output"]]
    
    df = df[df["persona"] == f"{persona}_interviewer"].reset_index(drop=True)
    df["text"] = df.apply(to_chatml, axis=1)
    print(f"  [{persona}] {path.name}: {len(df)}개")
    return Dataset.from_pandas(df[["text"]])


# ───────────────────────────────────────────
# 5. 페르소나별 학습 함수
# ───────────────────────────────────────────
class LossLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            train_loss = logs.get("loss", None)
            eval_loss  = logs.get("eval_loss", None)
            if train_loss:
                print(f"  Step {step} | train_loss: {train_loss:.4f}")
            if eval_loss:
                print(f"  Step {step} | eval_loss:  {eval_loss:.4f}")


def train_persona(persona: str):
    print(f"\n{'='*55}")
    print(f"  [{persona.upper()}] LoRA 학습 시작")
    print(f"{'='*55}")

    # 베이스 모델 로드 (매번 새로 — 페르소나 간 오염 방지)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = torch.bfloat16,  # A100 최적화
        load_in_4bit   = True,
    )

    # LoRA 어댑터 설정
    model = FastLanguageModel.get_peft_model(
        model,
        r              = 16,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha               = 32,    # 2*r 권장
        lora_dropout             = 0.05,
        bias                     = "none",
        use_gradient_checkpointing = "unsloth",
        random_state             = 3407,
        use_rslora               = False,
        loftq_config             = None,
    )

    # 데이터셋 준비
    train_ds = load_dataset(persona, TRAIN_PATH)
    val_ds   = load_dataset(persona, VAL_PATH)

    # 체크포인트 / 어댑터 저장 경로
    ckpt_dir = str(BASE_ADAPTER_DIR / f"{persona}_checkpoints")
    save_dir = str(BASE_ADAPTER_DIR / f"{persona}_lora")

    # 학습 설정
    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        args = SFTConfig(
            dataset_text_field          = "text",
            max_seq_length              = MAX_SEQ_LENGTH,

            # 배치 설정 (A100 40GB 기준)
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,   # 유효 배치 = 16

            # 학습 스케줄
            num_train_epochs            = 3,
            learning_rate               = 2e-4,
            warmup_ratio                = 0.05,
            lr_scheduler_type           = "cosine",

            # 정밀도
            fp16                        = False,
            bf16                        = True,  # A100 최적화

            # 로깅 / 저장
            logging_steps               = 10,
            eval_strategy               = "steps",
            eval_steps                  = 50,
            save_strategy               = "steps",
            save_steps                  = 100,
            save_total_limit            = 2,     # 최근 2개만 유지
            load_best_model_at_end      = True,
            metric_for_best_model       = "eval_loss",
            greater_is_better           = False,

            output_dir                  = ckpt_dir,
            report_to                   = "none",
        ),
        callbacks = [   
            EarlyStoppingCallback(
                early_stopping_patience=3
            ),
            LossLogCallback(),
        ],
    )

    trainer.train()

    # 어댑터 저장
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\n✅ [{persona.upper()}] 저장 완료: {save_dir}")

    # GPU 메모리 해제 (다음 페르소나 학습 전 필수)
    del model, tokenizer, trainer
    torch.cuda.empty_cache()
    print(f"🧹 [{persona.upper()}] GPU 메모리 해제 완료")

# ───────────────────────────────────────────
# 6. 실행
# ───────────────────────────────────────────
if __name__ == "__main__":
    print(f"BASE_DIR  : {BASE_DIR}")
    print(f"TRAIN_PATH: {TRAIN_PATH}")
    print(f"VAL_PATH  : {VAL_PATH}")
    print(f"ADAPTER_DIR: {BASE_ADAPTER_DIR}")

    for persona in PERSONAS:
        train_persona(persona)

    print("\n" + "="*55)
    print("✅ 전체 학습 완료")
    print(f"저장 위치: {BASE_ADAPTER_DIR}")
    print("  pressure_lora/")
    print("  friendly_lora/")
    print("="*55)