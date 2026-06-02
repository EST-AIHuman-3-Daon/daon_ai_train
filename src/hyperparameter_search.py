import json
import torch
import optuna
import pandas as pd
import gc
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, AutoTokenizer # AutoTokenizer 추가

# ───────────────────────────────────────────
# 0. 경로 및 설정
# ───────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
RESULT_DIR = BASE_DIR / "experiments"
RESULT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048
SAMPLE_SIZE = 200
MAX_STEPS = 80 
PERSONA = "pressure"

# ───────────────────────────────────────────
# 1. 토크나이징 및 데이터 로드 (사전 수행)
# ───────────────────────────────────────────
def to_chatml(row: dict) -> str:
    persona = row["persona"].replace("_interviewer", "")
    inp = json.loads(row["input"])
    system_msg = f"직무: {row['job_role']}\n\n{TRAIN_PROMPTS[persona]}"
    user_msg = f"면접 질문: {inp.get('question', '')}\n지원자 답변: {inp.get('candidate_answer', '')}"
    return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{row['output']}<|im_end|>"

TRAIN_PROMPTS = {"pressure": "당신은 대기업 압박 면접관입니다.", "friendly": "당신은 코칭형 친절 면접관입니다."}

# [수정] 모델 전체를 로드하지 않고 토크나이저만 가볍게 로드
print("사전 토크나이징을 위한 토크나이저 로드 중...")
temp_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
temp_tokenizer.pad_token = temp_tokenizer.eos_token # 필수 설정

def get_preprocessed_datasets():
    train_df = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")
    train_df = train_df[train_df["persona"] == f"{PERSONA}_interviewer"].sample(n=SAMPLE_SIZE, random_state=42)
    train_df["text"] = train_df.apply(to_chatml, axis=1)
    
    val_df = pd.read_csv(DATA_DIR / "val.csv", encoding="utf-8-sig")
    val_df = val_df[val_df["persona"] == f"{PERSONA}_interviewer"]
    val_df["text"] = val_df.apply(to_chatml, axis=1)
    
    train_ds = Dataset.from_pandas(train_df[["text"]])
    val_ds = Dataset.from_pandas(val_df[["text"]])
    
    # 토크나이징 수행
    tokenize_func = lambda x: temp_tokenizer(x["text"], truncation=True, max_length=MAX_SEQ_LENGTH)
    return train_ds.map(tokenize_func, batched=True), val_ds.map(tokenize_func, batched=True)

print("데이터 사전 토크나이징 중...")
train_ds, val_ds = get_preprocessed_datasets()
print("사전 토크나이징 완료.")

# 메모리 확보
del temp_tokenizer
gc.collect()
torch.cuda.empty_cache()

# ───────────────────────────────────────────
# 2. Optuna Objective
# ───────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_categorical("lr", [1e-4, 2e-4, 3e-4])
    r  = trial.suggest_categorical("r",  [8, 16, 32])

    print(f"\n[Trial {trial.number}] lr={lr}, r={r}")

    # 여기서 Unsloth 모델 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True,
        dtype = torch.bfloat16,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = r,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_ds,
        eval_dataset = val_ds,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=1)],
        args = SFTConfig(
            dataset_text_field = "text",
            max_seq_length = MAX_SEQ_LENGTH,
            max_steps = MAX_STEPS,
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,
            learning_rate = lr,
            warmup_steps = 5,
            lr_scheduler_type = "cosine",
            bf16 = True,
            logging_steps = 20,
            eval_strategy = "steps",
            eval_steps = 20,
            load_best_model_at_end = True,
            metric_for_best_model = "eval_loss",
            greater_is_better = False,
            output_dir = str(RESULT_DIR / f"trial_{trial.number}"),
            report_to = "none",
        ),
    )

    trainer.train()
    eval_loss = trainer.evaluate()["eval_loss"]

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return eval_loss

# ───────────────────────────────────────────
# 3. 실행
# ───────────────────────────────────────────
if __name__ == "__main__":
    sampler = optuna.samplers.GridSampler({"lr": [1e-4, 2e-4, 3e-4], "r": [8, 16, 32]})
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=9)
    study.trials_dataframe().to_csv(RESULT_DIR / "hyperparam_results.csv", index=False)