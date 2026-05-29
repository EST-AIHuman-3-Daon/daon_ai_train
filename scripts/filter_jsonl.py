import json
from pathlib import Path


# =========================================================
# JSONL 데이터 필터링 함수
# =========================================================
# job_role / persona 조건에 맞는 데이터만 추출하여 저장
#
# 예시:
# - ICT 직무만 추출
# - pressure_interviewer 페르소나만 추출
# =========================================================
def filter_jsonl_by_role_persona(
    input_path: str,
    output_path: str,
    job_role: str | None = None,
    persona: str | None = None,
):
    # 경로 객체 생성
    input_path = Path(input_path)
    output_path = Path(output_path)

    # output 폴더 자동 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_count = 0

    # 원본 파일 읽기 + 결과 파일 쓰기
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        # JSONL은 한 줄당 JSON 1개
        for line in infile:
            line = line.strip()

            # 빈 줄은 무시
            if not line:
                continue

            # JSON 파싱
            data = json.loads(line)

            # 직무 조건 체크
            role_match = (
                job_role is None
                or data.get("job_role") == job_role
            )

            # 페르소나 조건 체크
            persona_match = (
                persona is None
                or data.get("persona") == persona
            )

            # 조건 만족 시 저장
            if role_match and persona_match:
                outfile.write(
                    json.dumps(data, ensure_ascii=False) + "\n"
                )
                filtered_count += 1

    print(f"[필터링 완료] {output_path}")
    print(f"[데이터 수] {filtered_count}")

    return output_path


# =========================================================
# 단일 데이터를 Qwen messages 포맷으로 변환
# =========================================================
#
# 기존 구조:
# {
#   "instruction": "...",
#   "input": {
#       "question": "...",
#       "candidate_answer": "..."
#   },
#   "output": "..."
# }
#
# 변환 구조:
# {
#   "messages": [
#       {"role": "system", ...},
#       {"role": "user", ...},
#       {"role": "assistant", ...}
#   ]
# }
#
# Qwen / Llama / Gemma 등 Chat 모델 학습용 포맷
# =========================================================
def convert_item_to_qwen_messages(data: dict) -> dict:

    # 시스템 프롬프트 정보
    instruction = data["instruction"]
    persona = data.get("persona", "")
    job_role = data.get("job_role", "")

    # 면접 질문 / 지원자 답변
    question = data["input"]["question"]
    candidate_answer = data["input"]["candidate_answer"]

    # 정답(모델이 생성해야 할 꼬리질문)
    output = data["output"]

    # system prompt 구성
    # 페르소나 + 직무 정보를 같이 넣어줌
    system_prompt = (
        f"{instruction}\n"
        f"페르소나: {persona}\n"
        f"직무: {job_role}"
    )

    # Qwen messages 포맷 반환
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    f"면접 질문:\n{question}\n\n"
                    f"지원자 답변:\n{candidate_answer}"
                )
            },
            {
                "role": "assistant",
                "content": output
            }
        ]
    }


# =========================================================
# 전체 JSONL 파일을 Qwen messages 포맷으로 변환
# =========================================================
def convert_to_qwen_messages(
    input_path: str,
    output_path: str,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    # output 폴더 자동 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted_count = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()

            if not line:
                continue

            # JSON 파싱
            data = json.loads(line)

            # Qwen messages 포맷 변환
            converted = convert_item_to_qwen_messages(data)

            # 저장
            outfile.write(
                json.dumps(converted, ensure_ascii=False) + "\n"
            )

            converted_count += 1

    print(f"[Qwen 변환 완료] {output_path}")
    print(f"[변환된 데이터 수] {converted_count}")

    return output_path


# =========================================================
# 필터링 + Qwen 포맷 변환을 한 번에 수행
# =========================================================
#
# 실무에서 가장 많이 사용할 함수
#
# 예시:
# - ICT 직무
# - friendly_interviewer 페르소나
#
# 조건에 맞는 데이터만 추출 후
# 바로 Qwen 학습용 messages 포맷으로 저장
# =========================================================
def make_dataset_for_qwen(
    input_path: str,
    output_path: str,
    job_role: str | None = None,
    persona: str | None = None,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    # output 폴더 자동 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()

            if not line:
                continue

            # JSON 파싱
            data = json.loads(line)

            # 필터 조건 확인
            role_match = (
                job_role is None
                or data.get("job_role") == job_role
            )

            persona_match = (
                persona is None
                or data.get("persona") == persona
            )

            # 조건 만족 시 Qwen 포맷 변환
            if role_match and persona_match:

                converted = convert_item_to_qwen_messages(data)

                # 저장
                outfile.write(
                    json.dumps(converted, ensure_ascii=False) + "\n"
                )

                saved_count += 1

    print(f"[Qwen 학습 데이터 생성 완료] {output_path}")
    print(f"[저장된 데이터 수] {saved_count}")

    return output_path


# =========================================================
# 실행 예시
# =========================================================
if __name__ == "__main__":

    # ICT + friendly_interviewer 데이터만 추출 후
    # Qwen messages 포맷으로 변환하여 저장

    make_dataset_for_qwen(
        input_path="./data/sft_persona_augmented.jsonl",
        output_path="./data/qwen_friendly_ict.jsonl",
        job_role="ICT",
        persona="friendly_interviewer",
    )