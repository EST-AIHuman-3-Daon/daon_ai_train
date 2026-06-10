# =========================================================
# 1. 기본 설정
# =========================================================

import os
import json
import glob
import time
import uuid
import pandas as pd
from tqdm.auto import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import random

from datetime import datetime
date_tag = datetime.now().strftime("%Y%m%d")


PROJECT_DIR = "/smhrd/DaonTeam/Siyeong/daon_project"
DATA_DIR = f"{PROJECT_DIR}/data/raw"

OUTPUT_DIR = f"{PROJECT_DIR}/data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_EXTRACT_CSV = f"{OUTPUT_DIR}/interview_extracted_{date_tag}.csv"
AUGMENTED_JSONL = f"{OUTPUT_DIR}/sft_followup_neutral_{date_tag}.jsonl"
AUGMENTED_CSV = f"{OUTPUT_DIR}/sft_followup_neutral_{date_tag}.csv"

MODEL = "gpt-4.1-mini"  # 비용/속도 균형용. 필요하면 변경.
BATCH_SIZE = 15         # 10~20 권장. 에러 나면 8~10으로 낮추기.

# =========================================================
# 2. OpenAI API Key 설정
# =========================================================
from getpass import getpass
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# =========================================================
# 3. 원본 JSON에서 필요한 속성 추출 
# =========================================================

def build_neutral_prompt(batch):
    payload = []
    for row in batch:
        payload.append({
            "base_id": row["base_id"],
            "job_role": row["job_role"],
            "question": row["question"],
            "candidate_answer": row["candidate_answer"],
        })

    return json.dumps(payload, ensure_ascii=False)


def safe_get(d, path, default=""):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur is not None else default


def make_base_id(file_path):
    name = os.path.splitext(os.path.basename(file_path))[0]
    return name.replace(" ", "_")


def extract_record(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    occupation = safe_get(obj, ["dataSet", "info", "occupation"], "")
    question = safe_get(obj, ["dataSet", "question", "raw", "text"], "")
    answer = safe_get(obj, ["dataSet", "answer", "raw", "text"], "")
    answer_summary = safe_get(obj, ["dataSet", "answer", "summary", "text"], "")
    experience = safe_get(obj, ["dataSet", "info", "experience"], "")

    # answer raw가 비어 있으면 summary라도 사용
    candidate_answer = answer.strip() if answer.strip() else answer_summary.strip()

    return {
        "source_file": file_path,
        "base_id": make_base_id(file_path),
        "job_role": occupation,
        "experience": experience,
        "question": question.strip(),
        "candidate_answer": candidate_answer,
        "has_real_answer": bool(candidate_answer),
    }


json_files = []
for sub in ["ICT", "RND", "Management"]:
    json_files.extend(glob.glob(f"{DATA_DIR}/{sub}/**/*.json", recursive=True))

SAMPLE_SIZE = None  # 전체 증강
# SAMPLE_SIZE = 150 # 샘플 테스트

if SAMPLE_SIZE is not None and len(json_files) > SAMPLE_SIZE:
    random.seed(42)
    json_files = random.sample(json_files, SAMPLE_SIZE)

print("json files (추출 대상):", len(json_files))

records = []

bad_files = []

for fp in tqdm(json_files):
    try:
        rec = extract_record(fp)
        if rec["question"]:
            records.append(rec)
    except Exception as e:
        bad_files.append({"file": fp, "error": str(e)})

df = pd.DataFrame(records)
df.to_csv(RAW_EXTRACT_CSV, index=False, encoding="utf-8-sig")

print("추출 완료:", len(df))
print("오류 파일:", len(bad_files))
df.head()

# =========================================================
# 4. 지원자 답변이 없는 경우 fake answer 보완
#    - 우선 간단 템플릿으로 채움
#    - 나중에 필요하면 이 부분도 GPT로 고도화 가능
# =========================================================
FAKE_ANSWERS = [
    "관련 경험이 많지는 않지만, 유사한 프로젝트에서 문제를 분석하고 해결 방향을 제안한 경험이 있습니다.",
    "아직 실무 경험은 부족하지만, 학습 과정에서 비슷한 상황을 접했고 기본 개념은 이해하고 있습니다.",
    "팀 프로젝트에서 해당 주제와 관련된 역할을 맡은 적이 있으며, 당시 자료 조사와 구현 일부를 담당했습니다.",
    "정확히 같은 경험은 없지만, 비슷한 문제를 해결하기 위해 여러 방법을 비교하고 적용해본 적이 있습니다.",
    "그 부분은 깊게 경험해보지는 못했지만, 프로젝트를 진행하면서 필요성을 느끼고 학습한 경험이 있습니다.",
]

def fill_fake_answer(row):
    if row["candidate_answer"]:
        return row["candidate_answer"], "real"

    idx = abs(hash(row["question"])) % len(FAKE_ANSWERS)
    return FAKE_ANSWERS[idx], "synthetic_template"


filled = df.apply(fill_fake_answer, axis=1)
df["candidate_answer"] = [x[0] for x in filled]
df["answer_type"] = [x[1] for x in filled]

df["answer_type"].value_counts()

# =========================================================
# 5. GPT 배치 호출: 중립 꼬리질문 생성
# =========================================================
FOLLOWUP_SCHEMA = {
    "type": "json_schema",
    "name": "followup_batch",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "base_id": {"type": "string"},
                        "followup_question": {"type": "string"}
                    },
                    "required": ["base_id", "followup_question"]
                }
            }
        },
        "required": ["items"]
    }
}
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
def generate_followups(batch):
    prompt = build_neutral_prompt(batch)

    response = client.responses.create(
        model=MODEL,
        instructions="당신은 한국어 면접 데이터셋을 만드는 데이터 증강 전문가입니다.",
        input=prompt,
        text={
            "format": FOLLOWUP_SCHEMA
        },
        temperature=0.4,
    )

    return json.loads(response.output_text)

# =========================================================
# 6. 체크포인트 이어쓰기용 로드
# =========================================================
done = {}

if os.path.exists(AUGMENTED_JSONL):
    with open(AUGMENTED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                done[obj["id"]] = obj

print("이미 생성된 개수:", len(done))

# =========================================================
# 7. 팀장 정의서 형태로 저장
# =========================================================
def to_sft_record(row, followup_question):
    return {
        "id": f"neutral_{row['base_id']}",
        "persona": "neutral_interviewer",
        "job_role": row["job_role"],
        "instruction": "면접관으로서 지원자 답변을 바탕으로 자연스러운 꼬리질문을 생성하세요.",
        "input": {
            "question": row["question"],
            "candidate_answer": row["candidate_answer"],
        },
        "output": followup_question,
        "generation_method": f"gpt_followup_{row['answer_type']}",
    }


rows = df.to_dict("records")
pending = [r for r in rows if f"neutral_{r['base_id']}" not in done]

print("전체:", len(rows))
print("남은 작업:", len(pending))

with open(AUGMENTED_JSONL, "a", encoding="utf-8") as out:
    for i in tqdm(range(0, len(pending), BATCH_SIZE)):
        batch = pending[i:i + BATCH_SIZE]

        result = generate_followups(batch)

        result_map = {
            item["base_id"]: item["followup_question"]
            for item in result["items"]
        }

        for row in batch:
            followup = result_map.get(row["base_id"], "").strip()
            if not followup:
                continue

            sft_obj = to_sft_record(row, followup)
            out.write(json.dumps(sft_obj, ensure_ascii=False) + "\n")
            out.flush()

        time.sleep(0.2)

print("저장 완료:", AUGMENTED_JSONL)

# =========================================================
# 8. JSONL -> CSV 변환
# =========================================================
augmented = []

with open(AUGMENTED_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            augmented.append({
                "id": obj["id"],
                "persona": obj["persona"],
                "job_role": obj["job_role"],
                "instruction": obj["instruction"],
                "input": json.dumps(obj["input"], ensure_ascii=False),
                "output": obj["output"],
                "generation_method": obj["generation_method"],
            })

aug_df = pd.DataFrame(augmented)
aug_df.to_csv(AUGMENTED_CSV, index=False, encoding="utf-8-sig")

print("CSV 저장:", AUGMENTED_CSV)
print("개수:", len(aug_df))