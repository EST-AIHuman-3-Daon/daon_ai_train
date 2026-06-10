import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from getpass import getpass
from dotenv import load_dotenv


import pandas as pd
from openai import OpenAI


# =========================================================
# 0. 설정
# =========================================================



PROJECT_DIR = "/smhrd/DaonTeam/Siyeong/daon_project"

DATA_DIR = f"{PROJECT_DIR}/data"
OUTPUT_DIR = f"{PROJECT_DIR}/data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT_VERSION = "promptC2_final"
run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

# 입력 CSV는 상황에 맞게 하나만 사용하세요.
# 1. 고정 샘플 150개
# INPUT_CSV = f"{PROJECT_DIR}/data/samples/prompt_tuning_sample_150.csv"

# 2. 전체 추출 데이터

import glob

extract_files = sorted(
    glob.glob(f"{OUTPUT_DIR}/interview_extracted_*.csv")
)

if not extract_files:
    raise FileNotFoundError(
        "data/processed 안에 interview_extracted CSV가 없습니다."
    )

INPUT_CSV = extract_files[-1]

OUTPUT_CSV = f"{OUTPUT_DIR}/sft_augmented_dataset_{PROMPT_VERSION}_{run_tag}.csv"

MODEL = "gpt-4o-mini"
MAX_WORKERS = 3
SLEEP_SEC = 0.15

# 전체 증강이면 None, 테스트면 150
SAMPLE_SIZE = None
# SAMPLE_SIZE = 150


# =========================================================
# 1. OpenAI API Key 설정
# =========================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# =========================================================
# 2. 페르소나 프롬프트
# =========================================================

COMMON_RULES = """
[공통 규칙]
- 반드시 존댓말을 사용하세요.
- 마지막 문장은 질문형으로 끝내세요.
- 무례한 인신공격, 비하, 차별 표현은 사용하지 마세요.
- 예시 문장을 그대로 복사하지 말고 입력 답변에 맞게 표현을 바꾸세요.
"""

PRESSURE_RULES = """
[압박 면접관 규칙]
당신은 대기업 실무 압박 면접관입니다. 아래 규칙을 반드시 준수하세요.

[필수 규칙]
1. 첫 문장에는 지원자 답변의 핵심 키워드 1개를 포함하고,
   "모호", "정확히", "구체적이지", "명확하지" 중 1개를 함께 사용하여 답변의 부족한 지점을 지적하세요.

2. 긍정 표현 및 이모지를 사용하지 마세요.
   예: "좋습니다", "훌륭합니다", "인상적입니다", "😊", "👍"

3. 발화 길이는 3문장 이하로 끝내세요.

4. 마지막 질문 문장에는 "왜", "어떤", "어떻게", "구체적으로", "근거", "수치", "사례", "경험" 중 1개를 포함하여 근거나 실제 사례를 요구하세요.

5. 인정·완충 표현을 사용하지 마세요.
   예: "괜찮습니다", "편하게", "천천히", "부담 없이", "네, 이해했습니다", "말씀 감사합니다"

[좋은 예시]
면접 질문: RAG를 사용해본 경험이 있나요?
지원자 답변: LangChain과 FAISS를 사용했습니다.
output: FAISS를 선택한 이유가 정확히 드러나지 않습니다. 검색 성능을 판단한 근거와 실제 사례를 구체적으로 설명해 주시겠습니까?

면접 질문: 팀 프로젝트에서 갈등이 생기면 어떻게 해결하시나요?
지원자 답변: 팀원들과 대화하면서 원만하게 해결했습니다.
output: 팀원들과 대화했다는 설명은 구체적이지 않습니다. 어떤 갈등이 있었고 본인이 어떤 근거로 해결 방식을 선택했는지 말씀해 주시겠습니까?

면접 질문: 성능 최적화를 해본 경험이 있나요?
지원자 답변: 쿼리를 개선해서 속도를 높인 적이 있습니다.
output: 쿼리 개선 경험이 명확하지 않습니다. 개선 전후 수치나 실제 사례를 근거로 설명해 주시겠습니까?

[나쁜 예시]
output: 좋습니다. FAISS를 잘 사용하셨네요. 조금 더 설명해 주세요.
이유: 긍정 표현 사용

output: 왜 그렇게 했죠?
이유: 키워드 재사용 없음, 근거 요구 없음

output: 그건 너무 부족한 답변입니다. 제대로 다시 말해보세요.
이유: 비하 표현 사용
"""

FRIENDLY_RULES = """
[친절 면접관 규칙]
당신은 코칭형 친절 면접관입니다. 아래 규칙을 반드시 준수하세요.

[필수 규칙]
1. 인정 또는 완충 표현을 발화 전체에 1회 이상 포함하세요.
   예: "좋습니다", "좋아요", "괜찮습니다", "네, 이해했습니다", "말씀 감사합니다"

2. 답변을 구체화하도록 유도하는 표현을 1개 이상 포함하세요.
   예: "구체적으로", "조금 더", "자세히", "설명해", "말씀해"

3. 답변 방향을 잡아주는 단어를 1개 이상 포함하세요.
   예: "상황", "역할", "행동", "결과", "이유", "근거", "사례", "배운 점"

4. 발화 길이는 4문장 이하로 끝내세요.

5. 직접적인 부정 평가 표현은 사용하지 마세요.
   예: "부족", "아쉽", "미흡", "틀렸", "문제"

6. "STAR 방식"이라는 표현은 사용하지 마세요.

[좋은 예시]
면접 질문: 팀 프로젝트에서 갈등이 생기면 어떻게 해결하시나요?
지원자 답변: 팀원들과 대화하면서 원만하게 해결했습니다.
output: 좋습니다. 그 상황에서 본인의 역할과 대화 방식, 그리고 결과를 조금 더 구체적으로 말씀해 주실 수 있을까요?

면접 질문: RAG를 사용해본 경험이 있나요?
지원자 답변: LangChain과 FAISS를 사용했습니다.
output: 네, 이해했습니다. LangChain과 FAISS를 활용한 사례에서 어떤 이유로 그 방식을 선택했는지 조금 더 자세히 설명해 주실 수 있을까요?

면접 질문: 실패를 겪은 경험이 있나요?
지원자 답변: 프로젝트 일정이 밀려서 힘들었던 적이 있습니다.
output: 괜찮습니다. 그 경험에서 일정이 밀린 상황과 이후에 배운 점을 조금 더 구체적으로 말씀해 주세요.

면접 질문: 본인의 강점은 무엇인가요?
지원자 답변: 저는 책임감이 강하다고 생각합니다.
output: 좋습니다. 책임감을 보여준 구체적인 사례와 그 결과를 함께 말씀해 주실 수 있을까요?

[나쁜 예시]
output: 부족한 답변입니다. 다시 말해보세요.
이유: 직접적인 부정 평가 표현 사용

output: STAR 방식으로 상황, 행동, 결과를 말해주세요.
이유: STAR 표현 직접 사용

output: 왜 그렇게 했습니까? 근거가 뭡니까?
이유: 압박형 추궁 표현 사용
"""

PERSONA_RULES = {
    "pressure": f"{COMMON_RULES}\n\n{PRESSURE_RULES}",
    "friendly": f"{COMMON_RULES}\n\n{FRIENDLY_RULES}",
}


# =========================================================
# 3. instruction 템플릿
# =========================================================

INSTRUCTIONS = {
    "pressure": (
        "압박 면접관으로서 지원자 답변의 모호한 부분을 짚고, "
        "근거, 수치, 실제 사례 중 하나를 요구하는 존댓말 꼬리질문을 생성하세요."
    ),
    "friendly": (
        "친절한 면접관으로서 지원자의 답변을 짧게 인정하고, "
        "상황, 역할, 행동, 결과, 이유, 근거, 사례 중 하나를 자연스럽게 묻는 존댓말 꼬리질문을 생성하세요."
    ),
}


# =========================================================
# 4. 입력 데이터 로드
# =========================================================

def load_input_csv(input_csv):
    df = pd.read_csv(input_csv)

    rename_map = {
        "job_category": "job",
        "job_role": "job",
        "experience": "career",
    }
    df = df.rename(columns=rename_map)

    if "career" not in df.columns:
        df["career"] = "NEW"

    required_cols = ["job", "career", "question"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"입력 CSV에 필요한 컬럼이 없습니다: {missing_cols}")

    df = df[required_cols].dropna()

    if SAMPLE_SIZE is not None:
        df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

    return df.reset_index(drop=True)


# =========================================================
# 5. GPT 프롬프트
# =========================================================

def build_persona_prompt(question, job, career):
    career_str = "신입" if str(career).upper() == "NEW" else "경력"

    return f"""직무: {job}
경력: {career_str}
면접 질문: {question}

[Task]
1. candidate_answer: 위 질문에 대한 자연스러운 지원자 답변을 2~3문장으로 생성하세요.
   - 답변 유형은 아래 중 하나를 자연스럽게 선택하세요.
     1) vague: 모호하고 구체성이 부족한 답변
     2) logical: 논리적이고 비교적 잘 정리된 답변
     3) off_topic: 질문 의도에서 약간 벗어난 답변
     4) experience_based: 실제 경험 중심 답변
     5) knowledge_based: 개념이나 지식 중심 답변
     6) short: 짧고 단정적인 답변
   - 질문과 완전히 무관한 답변은 피하세요.
   - 직무와 경력 수준에 맞게 작성하세요.

2. output: candidate_answer를 들은 뒤, 시스템 규칙을 모두 만족하는 꼬리질문 1개를 생성하세요.
   - output은 반드시 질문 형태여야 합니다.
   - candidate_answer의 내용과 답변 유형을 반영해야 합니다.
   - 처음 면접 질문을 다시 쓰지 말고, 후속 질문만 작성하세요.

JSON만 출력:
{{"answer_type": "...", "candidate_answer": "...", "output": "..."}}"""


# =========================================================
# 6. GPT 호출
# =========================================================

def call_gpt(persona, question, job, career, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": PERSONA_RULES[persona]},
                    {
                        "role": "user",
                        "content": build_persona_prompt(
                            question=question,
                            job=job,
                            career=career,
                        ),
                    },
                ],
                temperature=0.8,
                max_tokens=350,
                response_format={"type": "json_object"},
            )

            parsed = json.loads(response.choices[0].message.content)

            if all(key in parsed for key in ["answer_type", "candidate_answer", "output"]):
                return parsed

            print(f"[WARN] 키 누락: {parsed.keys()}")

        except Exception as e:
            if "429" in str(e):
                wait = 2 * (attempt + 1)
                print(f"[429] {wait}초 대기 후 재시도")
                time.sleep(wait)
            else:
                wait = 1.5 ** attempt
                print(f"[ERROR] attempt {attempt + 1}: {e}")
                time.sleep(wait)

    return None


# =========================================================
# 7. 작업 함수
# =========================================================

def process_task(base_idx, row, persona):
    time.sleep(SLEEP_SEC)

    result = call_gpt(
        persona=persona,
        question=row["question"],
        job=row["job"],
        career=row["career"],
    )

    if not result:
        return None

    return {
        "id": f"{persona}_{base_idx:06d}",
        "persona": f"{persona}_interviewer",
        "job_role": row["job"],
        "instruction": INSTRUCTIONS[persona],
        "input": json.dumps(
            {
                "question": row["question"],
                "candidate_answer": result["candidate_answer"],
            },
            ensure_ascii=False,
        ),
        "output": result["output"],
        "answer_type": result.get("answer_type", ""),
        "generation_method": PROMPT_VERSION,
    }


# =========================================================
# 8. 병렬 실행
# =========================================================

def run_augmentation():
    df = load_input_csv(INPUT_CSV)

    print(f"입력 파일: {INPUT_CSV}")
    print(f"총 처리 대상: {len(df)}개 × 2 페르소나 = {len(df) * 2}회 호출")

    results = []
    futures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for base_idx, (_, row) in enumerate(df.iterrows(), start=1):
            for persona in ["pressure", "friendly"]:
                futures.append(
                    executor.submit(
                        process_task,
                        base_idx,
                        row.to_dict(),
                        persona,
                    )
                )

        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()

            if result:
                results.append(result)

            if i % 50 == 0:
                print(f"진행: {i}/{len(futures)} | 성공: {len(results)}")
                pd.DataFrame(results).to_csv(
                    OUTPUT_CSV,
                    index=False,
                    encoding="utf-8-sig",
                )

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n========== 완료 ==========")
    print(f"총 생성: {len(result_df)}행")
    print(result_df["persona"].value_counts().to_string())
    print(f"저장 위치: {OUTPUT_CSV}")

    return result_df


# =========================================================
# 9. 실행
# =========================================================

if __name__ == "__main__":
    run_augmentation()