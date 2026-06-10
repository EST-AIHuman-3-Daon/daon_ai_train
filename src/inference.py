"""
inference.py
- pressure / friendly LoRA 어댑터 스왑 추론
- 강한 프롬프트 (C-2 버전) 적용
- 면접 시작/종료 트리거 처리
- 멀티턴 대화 지원

실행: cd daon_project && python src/inference.py
"""

import json
import torch
from pathlib import Path
from unsloth import FastLanguageModel
import warnings

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────
# 0. 경로 설정
# ───────────────────────────────────────────

MODEL_VERSION = "Qwen2.5-7B-Instruct"

BASE_DIR    = Path(__file__).resolve().parent.parent
ADAPTER_DIR = BASE_DIR / "adapters" / MODEL_VERSION

MAX_SEQ_LENGTH = 2048

# ───────────────────────────────────────────
# 1. 추론용 강한 프롬프트 (C-2 버전)
# ───────────────────────────────────────────
INFERENCE_PROMPTS = {
    "pressure": """[공통 규칙]
- 반드시 존댓말을 사용하세요.
- 마지막 문장은 질문형으로 끝내세요.
- 무례한 인신공격, 비하, 차별 표현은 사용하지 마세요.
- 예시 문장을 그대로 복사하지 말고 입력 답변에 맞게 표현을 바꾸세요.
- 지원자가 답변을 더 이상 이어가기 어렵다고 하거나, 주제 전환을 요청하면 존중하고 다음 질문으로 넘어가세요.
- "지원자가 구체적인 실무 사례가 없다고 답변했습니다. 이 경우, 무리하게 사례를 요구하지 말고 '그 기술을 어떻게 학습하고 있는지', '관련하여 어떤 기술적 호기심을 가지고 공부하고 있는지' 학습 태도나 기초 역량을 검증하는 방향으로 전환하여 질문하세요."

[압박 면접관 규칙]
당신은 대기업 실무 압박 면접관입니다.

[필수 규칙]
1. 첫 문장에는 지원자 답변의 핵심 키워드 1개를 포함하고,
   "모호", "정확히", "구체적이지", "명확하지" 중 1개를 함께 사용하여
   답변의 부족한 지점을 지적하세요.
2. 긍정 표현 및 이모지를 사용하지 마세요.
   예: "좋습니다", "훌륭합니다", "인상적입니다"
3. 발화 길이는 3문장 이하로 끝내세요.
4. 마지막 질문 문장에는 "왜", "어떤", "어떻게", "구체적으로",
   "근거", "수치", "사례", "경험" 중 1개를 포함하여
   근거나 실제 사례를 요구하세요.
5. 인정·완충 표현을 사용하지 마세요.
   예: "괜찮습니다", "편하게", "천천히", "부담 없이", "네, 이해했습니다" """,

    "friendly": """[공통 규칙]
- 반드시 존댓말을 사용하세요.
- 마지막 문장은 질문형으로 끝내세요.
- 무례한 인신공격, 비하, 차별 표현은 사용하지 마세요.
- 예시 문장을 그대로 복사하지 말고 입력 답변에 맞게 표현을 바꾸세요.

[친절 면접관 규칙]
당신은 코칭형 친절 면접관입니다.

[필수 규칙]
1. 인정 또는 완충 표현을 발화 전체에 1회 이상 포함하세요.
   예: "좋습니다", "좋아요", "괜찮습니다", "네, 이해했습니다"
2. 답변을 구체화하도록 유도하는 표현을 1개 이상 포함하세요.
   예: "구체적으로", "조금 더", "자세히", "설명해", "말씀해"
3. 답변 방향을 잡아주는 단어를 1개 이상 포함하세요.
   예: "상황", "역할", "행동", "결과", "이유", "근거", "사례"
4. 발화 길이는 4문장 이하로 끝내세요.
5. 직접적인 부정 평가 표현은 사용하지 마세요.
   예: "부족", "아쉽", "미흡", "틀렸", "문제"
6. "STAR 방식"이라는 표현은 사용하지 마세요."""
}

# ───────────────────────────────────────────
# 2. 트리거 키워드
# ───────────────────────────────────────────
TRIGGER_START = ["면접 시작", "시작", "면접시작"]
TRIGGER_END   = ["면접 종료", "종료", "끝", "그만", "면접종료"]

# ───────────────────────────────────────────
# 3. 모델 로드
# ───────────────────────────────────────────
def load_model(persona: str):
    adapter_path = str(ADAPTER_DIR / f"{persona}_lora")
    print(f"[{persona.upper()}] 어댑터 로드 중: {adapter_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = adapter_path,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = torch.bfloat16,
        load_in_4bit   = True,
    )
    FastLanguageModel.for_inference(model)
    print(f"[{persona.upper()}] 로드 완료")
    return model, tokenizer

# ───────────────────────────────────────────
# 4. ChatML 프롬프트 빌더
# ───────────────────────────────────────────
def build_prompt(
    persona: str,
    job_role: str,
    question: str,
    candidate_answer: str,
) -> str:
    system_msg = (
        f"직무: {job_role}\n\n"
        f"{INFERENCE_PROMPTS[persona]}"
    )
    user_msg = (
        f"면접 질문: {question}\n"
        f"지원자 답변: {candidate_answer}"
    )
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def build_first_prompt(persona: str, job_role: str) -> str:
    """면접 시작 시 첫 질문 생성용 프롬프트"""
    system_msg = (
        f"직무: {job_role}\n\n"
        f"{INFERENCE_PROMPTS[persona]}"
    )
    user_msg = (
        f"지금부터 {job_role} 직무 면접을 시작합니다. "
        f"첫 번째 면접 질문을 생성하세요. "
        f"인사말 없이 바로 질문으로 시작하세요."
    )
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def history_to_text(history):
    lines = []

    for item in history:
        q = item.get("question", "")
        a = item.get("answer", "")

        lines.append(f"질문: {q}")

        if a:
            lines.append(f"답변: {a}")

    return "\n".join(lines)

def build_new_question_prompt(
    persona: str,
    job_role: str,
    history_text: str,
) -> str:

    system_msg = (
        f"직무: {job_role}\n\n"
        f"{INFERENCE_PROMPTS[persona]}"
    )

    user_msg = f"""
이전 면접 기록:

{history_text}

위 질문들과 중복되지 않는
새로운 면접 질문 1개를 생성하세요.

꼬리질문이 아니라
새로운 평가 항목의 질문이어야 합니다.

인사말 없이 질문만 출력하세요.

"""

    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ───────────────────────────────────────────
# 5. 추론 함수
# ───────────────────────────────────────────
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            temperature    = 0.7,
            top_p          = 0.9,
            do_sample      = True,
            repetition_penalty = 1.1,
            pad_token_id   = tokenizer.eos_token_id,
        )

    generated = outputs[0][input_len:]
    response  = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()

# ───────────────────────────────────────────
# 6. 세션 관리
# ───────────────────────────────────────────
def create_session(persona: str, job_role: str, max_turn: int = 10) -> dict:
    return {
        "persona": persona,
        "job_role": job_role,
        "max_turn": max_turn,
        "turn": 0,
        "status": "idle",
        "history": [],
        "last_question": None,
        "phase": "new_question",
        "followup_count": 0,
    }


# ───────────────────────────────────────────
# 7. 메인 대화 루프
# ───────────────────────────────────────────
def run_interview(persona: str = "pressure", job_role: str = "ICT"):
    model, tokenizer = load_model(persona)
    session = create_session(persona, job_role)

    print("\n" + "="*55)
    print(f"  면접 시뮬레이터 | 페르소나: {persona} | 직무: {job_role}")
    print(f"  '면접 시작' 입력 시 면접 시작 | '면접 종료' 입력 시 종료")
    print("="*55 + "\n")

    while True:
        user_input = input("지원자: ").strip()

        if not user_input:
            continue

        # ── 종료 트리거 ──────────────────────────
        if any(t in user_input for t in TRIGGER_END):
            print("\n[면접 종료]")
            print(f"총 {session['turn']}턴 진행됨")
            return session

        # ── 시작 트리거 ──────────────────────────
        if any(t in user_input for t in TRIGGER_START):
            session["status"] = "active"
            session["turn"] = 0

            print("\n[면접관]: ", end="", flush=True)
            prompt = build_first_prompt(persona, job_role)
            response = generate(model, tokenizer, prompt)
            print(response)

            session["last_question"] = response
            session["history"].append({"question": response, "answer": None})
            continue

        # ── 면접 진행 중 ──────────────────────────
        if session["status"] == "active":
            
            # 1. 답변 저장
            if session["history"] and session["history"][-1]["answer"] is None:
                session["history"][-1]["answer"] = user_input

            # 2. 턴 증가 및 종료 체크
            session["turn"] += 1
            if session["turn"] >= session["max_turn"]:
                session["status"] = "done"
                print("\n[면접관]: 수고하셨습니다. 면접을 마치겠습니다.")
                return session

            # 3. 질문 생성 로직
            if "다음 질문" in user_input or "다른 질문" in user_input:
                print("\n[면접관]: 알겠습니다. 다음 항목으로 넘어가겠습니다.")
                prompt = build_new_question_prompt(persona, job_role, history_to_text(session["history"][-5:]))
                session["followup_count"] = 0
            
            elif session["followup_count"] < 2:
                prompt = build_prompt(persona, job_role, session["last_question"], user_input)
                session["followup_count"] += 1
            
            else:
                prompt = build_new_question_prompt(persona, job_role, history_to_text(session["history"][-5:]))
                session["followup_count"] = 0

            # 4. 응답 생성
            print("\n[면접관]: ", end="", flush=True)
            response = generate(model, tokenizer, prompt)
            print(response)

            session["last_question"] = response
            session["history"].append({"question": response, "answer": None})

        else:
            print("  '면접 시작'을 입력해주세요.")

    return session

# ───────────────────────────────────────────
# 8. 실행
# ───────────────────────────────────────────
if __name__ == "__main__":
    PERSONA  = "pressure"   # "pressure" or "friendly"
    JOB_ROLE = "ICT"

    try:
        session = run_interview(persona=PERSONA, job_role=JOB_ROLE)
    except Exception as e:
        print(f"\n[오류 발생] 면접 중 예기치 못한 문제가 발생했습니다: {e}")
        session = None

    # 대화 히스토리 출력
    if session and "history" in session:
        print("\n=== 대화 히스토리 ===")
        for i, turn in enumerate(session["history"], 1):
            print(f"\n[{i}턴]")
            print(f"  면접관: {turn['question']}")
            if turn["answer"]:
                print(f"  지원자: {turn['answer']}")
    
    else:
        print("\n세션 정보가 없거나 면접이 정상적으로 초기화되지 않았습니다.")