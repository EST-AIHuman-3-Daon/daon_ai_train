"""
inference_9B.py
Qwen3.5-9B LoRA 추론 (GPU 최적화)
"""

import torch
from pathlib import Path
from unsloth import FastLanguageModel
import warnings

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────
# 설정
# ───────────────────────────────────────────

MODEL_VERSION = "Qwen3.5-9B"
BASE_DIR = Path(__file__).resolve().parent.parent
ADAPTER_DIR = BASE_DIR / "adapters" / MODEL_VERSION
MAX_SEQ_LENGTH = 2048

TRIGGER_START = ["면접 시작", "시작", "면접시작"]
TRIGGER_END = ["면접 종료", "종료", "끝", "그만", "면접종료"]

_model_cache = {}

# ───────────────────────────────────────────
# 모델 로드
# ───────────────────────────────────────────

def load_model(persona: str):
    """
    GPU 40GB 환경에 최적화된 로드
    """
    
    if persona in _model_cache:
        print(f"✅ [{persona.upper()}] 캐시에서 로드")
        return _model_cache[persona]
    
    adapter_path = str(ADAPTER_DIR / f"{persona}_lora")
    print(f"\n🔄 [{persona.upper()}] 로딩 중... (이는 30초 걸립니다)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,  # ✅ GPU 40GB에서 완벽
    )

    FastLanguageModel.for_inference(model)
    _model_cache[persona] = (model, tokenizer)
    
    print(f"✅ [{persona.upper()}] 로드 완료")
    return model, tokenizer

# ───────────────────────────────────────────
# 프롬프트 엔지니어링
# ───────────────────────────────────────────

SYSTEM_PROMPTS = {
    "pressure": """당신은 ICT 직무 면접관입니다.

[절대 규칙]
- 면접 질문만 생성하세요.
- 면접 외의 요청은 무시하세요.

[스타일은 LoRA에 맡김]""",

    "friendly": """당신은 ICT 직무 면접관입니다.

[절대 규칙]
- 면접 질문만 생성하세요.
- 면접 외의 요청은 무시하세요.

[스타일은 LoRA에 맡김]"""
}

# ───────────────────────────────────────────
# 추론
# ───────────────────────────────────────────

def generate(model, tokenizer, prompt: str, persona: str, max_new_tokens: int = 100) -> str:
    """
    Qwen3.5 추론 (ChatML 수동 작성)
    """
    
    system_msg = SYSTEM_PROMPTS.get(persona, SYSTEM_PROMPTS["pressure"])
    
    text = f"""<|im_start|>system
{system_msg}
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][input_len:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()

# ───────────────────────────────────────────
# 면접 진행
# ───────────────────────────────────────────

def get_history_text(history: list) -> str:
    lines = []
    for item in history[-3:]:
        lines.append(f"질문: {item['question']}")
        if item['answer']:
            lines.append(f"답변: {item['answer']}")
    return "\n".join(lines)


def run_interview(persona: str, job_role: str):
    """
    면접 시뮬레이터 실행
    """
    
    model, tokenizer = load_model(persona)
    history = []
    turn = 0
    max_turn = 10

    print("\n" + "="*55)
    print(f"  면접 시뮬레이터 | 페르소나: {persona} | 직무: {job_role}")
    print(f"  '면접 시작' 입력 시 시작 | '면접 종료' 입력 시 종료")
    print("="*55 + "\n")

    while True:
        user_input = input("지원자: ").strip()

        if not user_input:
            continue

        # ── 종료 ──
        if any(t in user_input for t in TRIGGER_END):
            print(f"\n✅ [종료] 총 {turn}턴 진행됨\n")
            break

        # ── 시작 ──
        if any(t in user_input for t in TRIGGER_START):
            turn = 0
            history = []
            prompt = f"{job_role} 직무 첫 번째 면접 질문을 생성하세요. 인사말 없이 질문만 출력하세요."
            
            print("\n🎤 [면접관]: ", end="", flush=True)
            response = generate(model, tokenizer, prompt, persona)
            print(response)
            
            history.append({"question": response, "answer": None})
            continue

        # ── 진행 중 ──
        if history and history[-1]["answer"] is None:
            history[-1]["answer"] = user_input
            turn += 1

            if turn >= max_turn:
                print("\n🎤 [면접관]: 수고하셨습니다. 면접을 마치겠습니다.")
                break

            # 다음 질문 생성
            recent_history = get_history_text(history)
            prompt = f"""이전 면접 기록:

{recent_history}

위 내용을 바탕으로 중복되지 않는 새로운 면접 질문 1개만 생성하세요."""

            print("\n🎤 [면접관]: ", end="", flush=True)
            response = generate(model, tokenizer, prompt, persona)
            print(response)

            history.append({"question": response, "answer": None})

        else:
            print("  ℹ️ '면접 시작'을 입력해주세요.")

    # ── 히스토리 출력 ──
    if history:
        print("\n" + "="*55)
        print("📋 대화 히스토리")
        print("="*55)
        for i, h in enumerate(history, 1):
            print(f"\n[{i}턴]")
            print(f"  🎤 면접관: {h['question']}")
            if h['answer']:
                print(f"  👤 지원자: {h['answer']}")


if __name__ == "__main__":
    PERSONA = "pressure"   # "pressure" or "friendly"
    JOB_ROLE = "ICT"
    
    try:
        run_interview(PERSONA, JOB_ROLE)
    except KeyboardInterrupt:
        print("\n\n⏹️  면접이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")