import json
import re

PRESSURE_POSITIVE_WORDS = [
    "좋습니다", "좋아요", "훌륭합니다", "잘하셨습니다", "인상적", "😊", "👍"
]

CUSHION_WORDS = [
    "괜찮습니다", "편하게", "천천히", "부담 없이", "네, 이해했습니다", "말씀 감사합니다"
]

AMBIGUITY_WORDS = [
    "모호", "구체적이지", "정확히", "명확하지", "불분명", "구체적으로"
]

EVIDENCE_WORDS = [
    "왜", "어떤", "어떻게", "구체적으로", "근거", "수치", "사례", "경험"
]

FRIENDLY_ACK_WORDS = [
    "좋습니다", "좋아요", "괜찮습니다", "네", "이해했습니다", "말씀 감사합니다"
]

ELABORATION_WORDS = [
    "구체적으로", "조금 더", "자세히", "설명해", "말씀해", "구체적인"
]

STRUCTURE_WORDS = [
    "상황", "역할", "행동", "결과", "이유", "근거", "사례", "배운 점",
    "경험", "과정", "방법", "예시"
]

BAD_CRITICISM_PATTERNS = [
    "부족한 답변", "답변이 부족", "부족합니다", "아쉽습니다",
    "미흡합니다", "틀렸", "문제있는 답변", "문제 있는 답변"
]


def contains_any(text, keywords):
    return any(k in str(text) for k in keywords)


def count_sentences(text):
    parts = re.split(r"[.!?。！？\n]+|(?<=[가-힣])\?", str(text).strip())
    parts = [p.strip() for p in parts if p.strip()]
    return max(1, len(parts))


def get_input_obj(row):
    inp = row.get("input", {})
    if isinstance(inp, str):
        try:
            return json.loads(inp)
        except Exception:
            return {}
    return inp if isinstance(inp, dict) else {}


def keyword_overlap(candidate_answer, output):
    stopwords = {
        "저는", "제가", "그리고", "하지만", "그래서", "그때", "당시",
        "것", "수", "등", "좀", "더", "잘", "많이", "관련", "경험"
    }

    tokens = re.findall(r"[가-힣A-Za-z0-9+#.]{2,}", str(candidate_answer))
    tokens = [t for t in tokens if t not in stopwords]

    reused = [t for t in set(tokens) if t in str(output)]
    return len(reused) >= 1, reused[:5]


def validate_common(row):
    output = row["output"]

    return {
        "c1_honorific_question": contains_any(
            output,
            ["습니까?", "인가요?", "나요?", "까요?", "해주세요", "주실 수 있을까요", "좋겠습니다"]
        ),
        "c2_no_abusive_expression": not contains_any(
            output,
            ["멍청", "한심", "쓸모없", "비하", "차별", "제대로 못"]
        ),
    }


def validate_pressure(row):
    output = row["output"]
    inp = get_input_obj(row)
    candidate_answer = inp.get("candidate_answer", "")

    overlap_ok, reused_keywords = keyword_overlap(candidate_answer, output)

    checks = {
        "p1_keyword_and_ambiguity": overlap_ok and contains_any(output, AMBIGUITY_WORDS),
        "p2_no_positive_emotion": not contains_any(output, PRESSURE_POSITIVE_WORDS),
        "p3_length_under_3_sentences": count_sentences(output) <= 3,
        "p4_requires_evidence": contains_any(output, EVIDENCE_WORDS),
        "p5_no_cushion_expression": not contains_any(output, CUSHION_WORDS),
    }

    return checks, reused_keywords


def validate_friendly(row):
    output = row["output"]

    checks = {
        "f1_acknowledgement": contains_any(output, FRIENDLY_ACK_WORDS),
        "f2_elaboration": contains_any(output, ELABORATION_WORDS),
        "f3_structure_or_reason": contains_any(output, STRUCTURE_WORDS),
        "f4_length_under_4_sentences": count_sentences(output) <= 4,
        "f5_no_direct_criticism": not contains_any(output, BAD_CRITICISM_PATTERNS),
    }

    return checks, []


def validate_row(row):
    persona = row["persona"]

    if persona == "pressure_interviewer":
        checks, reused_keywords = validate_pressure(row)
    elif persona == "friendly_interviewer":
        checks, reused_keywords = validate_friendly(row)
    else:
        checks, reused_keywords = {}, []

    common_checks = validate_common(row)

    total = len(checks)
    passed = sum(1 for v in checks.values())
    score = round(passed / total * 100, 2) if total else 0

    common_total = len(common_checks)
    common_passed = sum(1 for v in common_checks.values())
    common_score = round(common_passed / common_total * 100, 2) if common_total else 0

    result = {
        "id": row["id"],
        "persona": persona,
        "job_role": row.get("job_role", ""),
        "score": score,
        "passed": passed,
        "total": total,
        "common_score": common_score,
        "common_passed": common_passed,
        "common_total": common_total,
        "failed_rules": ", ".join([k for k, v in checks.items() if not v]),
        "failed_common_rules": ", ".join([k for k, v in common_checks.items() if not v]),
        "reused_keywords": ", ".join(reused_keywords),
        "output": row["output"],
    }

    result.update(checks)
    result.update(common_checks)
    return result