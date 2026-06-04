"""
train_lora.py
- pressure / friendly LoRA 어댑터 순차 학습
- 학습용 간소화 프롬프트 사용
- A100 환경 최적화 (bfloat16, load_in_4bit=True)
"""

import json
import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, TrainerCallback

# ───────────────────────────────────────────
# 0. 경로 및 모델 설정
# ───────────────────────────────────────────

# [설정 1: Qwen3.5-9B (현재 활성화)]
MODEL_VERSION = "Qwen3.5-9B"
MODEL_NAME    = "Qwen/Qwen3.5-9B"
LORA_R        = 8
LORA_LR       = 2e-4

# [설정 2: Qwen2.5-7B (주석 처리됨)]
# MODEL_VERSION = "Qwen2.5-7B-Instruct"
# MODEL_NAME    = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
# LORA_R        = 32
# LORA_LR       = 1e-4

# 공통 설정
LORA_ALPHA    = LORA_R
LORA_DROPOUT  = 0.0

BASE_DIR         = Path(__file__).resolve().parent.parent
DATA_DIR         = BASE_DIR / "data" / "processed"
BASE_ADAPTER_DIR = BASE_DIR / "adapters" / MODEL_VERSION
BASE_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH   = DATA_DIR / "val.csv"
MAX_SEQ_LENGTH = 2048
PERSONAS       = ["pressure", "friendly"]

# ───────────────────────────────────────────
# 1. 학습용 간소화 프롬프트
# ───────────────────────────────────────────
TRAIN_PROMPTS = {
    "pressure": "당신은 대기업 압박 면접관입니다. 지원자 답변의 모호한 부분을 지적하고 근거와 사례를 요구하는 꼬리질문을 생성하세요.",
    "friendly": "당신은 코칭형 친절 면접관입니다. 지원자를 배려하며 답변을 구체화하도록 유도하는 꼬리질문을 생성하세요."
}

# ───────────────────────────────────────────
# 2. ChatML 포맷 변환
# ───────────────────────────────────────────
def to_chatml(row: dict) -> str:
    persona = row["persona"].replace("_interviewer", "")
    inp = json.loads(row["input"])
    system_msg = f"직무: {row['job_role']}\n\n{TRAIN_PROMPTS[persona]}"
    user_msg = f"면접 질문: {inp.get('question', '')}\n지원자 답변: {inp.get('candidate_answer', '')}"
    return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{row['output']}<|im_end|>"

def load_dataset(persona: str, path: Path) -> Dataset:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[["persona", "job_role", "input", "output"]]
    df = df[df["persona"] == f"{persona}_interviewer"].reset_index(drop=True)
    df["text"] = df.apply(to_chatml, axis=1)
    return Dataset.from_pandas(df[["text"]])

# ───────────────────────────────────────────
# 3. 로깅 콜백
# ───────────────────────────────────────────
class LossLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            if "loss" in logs: print(f"  Step {step} | train_loss: {logs['loss']:.4f}")
            if "eval_loss" in logs: print(f"  Step {step} | eval_loss:  {logs['eval_loss']:.4f}")

# ───────────────────────────────────────────
# 4. 페르소나별 학습 함수
# ───────────────────────────────────────────
def train_persona(persona: str):
    print(f"\n>>> [{persona.upper()}] 학습 시작 (Model: {MODEL_VERSION})")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = torch.bfloat16,
        load_in_4bit   = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r              = LORA_R,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 3407,
    )

    train_ds = load_dataset(persona, TRAIN_PATH)
    val_ds   = load_dataset(persona, VAL_PATH)

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        args = SFTConfig(
            dataset_text_field = "text",
            max_seq_length     = MAX_SEQ_LENGTH,
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4, # 유효 배치 16
            num_train_epochs   = 3,
            learning_rate      = LORA_LR,
            warmup_steps = 20,
            lr_scheduler_type  = "cosine",
            bf16               = True,
            logging_steps      = 10,
            eval_strategy      = "steps",
            eval_steps         = 50,
            save_strategy      = "steps",
            save_steps         = 100,
            save_total_limit   = 2,
            load_best_model_at_end = True,
            metric_for_best_model  = "eval_loss",
            greater_is_better      = False,
            output_dir         = str(BASE_ADAPTER_DIR / f"{persona}_checkpoints"),
            report_to          = "none",
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3), LossLogCallback()],
    )

    trainer.train()
    
    save_dir = str(BASE_ADAPTER_DIR / f"{persona}_lora")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    del model, tokenizer, trainer
    torch.cuda.empty_cache()

# ───────────────────────────────────────────
# 5. 실행
# ───────────────────────────────────────────
if __name__ == "__main__":
    for persona in PERSONAS:
        train_persona(persona)
    print("\n✅ 전체 학습 완료!")


# qwen 2.5-7B-Instruction : Trainable parameters = 80,740,352 of 7,696,356,864 (1.05% trained)
# qwen 3.5-9B             : Trainable parameters = 14,548,992 of 9,424,362,736 (0.15% trained)