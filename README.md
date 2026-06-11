# AI 면접 시뮬레이터

ICT 직무 면접 시뮬레이터의 LLM 학습 및 평가 파이프라인입니다.
Qwen2.5-7B-Instruct / Qwen3.5-9B 모델에 압박(Pressure) / 친화(Friendly) 면접관 페르소나 LoRA 어댑터를 학습하고, 프롬프트 엔지니어링 및 평가를 수행하였습니다.

> 팀 협업 및 개발 환경 설정은 [CONTRIBUTING.md](./CONTRIBUTING.md)를 참고하세요.

---

## LoRA 어댑터

허깅페이스에서 다운로드할 수 있습니다.

레포지토리 주소 : **[teems79/interview-lora](https://huggingface.co/teems79/interview-lora)**

| 어댑터 | 설명 |
|--------|------|
| `qwen2.5/pressure` | Qwen2.5-7B-Instruct 압박 면접관 LoRA |
| `qwen2.5/friendly` | Qwen2.5-7B-Instruct 친화 면접관 LoRA |
| `qwen3.5/pressure` | Qwen3.5-9B 압박 면접관 LoRA |
| `qwen3.5/friendly` | Qwen3.5-9B 친화 면접관 LoRA |

---

## 실행 환경

| 구분 | 환경 | 용도 |
|------|------|------|
| 학습 / 추론 | A100 40GB (SSH, MIG 3g.40gb) | LoRA 학습, 하이퍼파라미터 탐색, 추론 |
| 평가 / 분석 | Google Colab (L4 GPU) | A/B 테스트, 프롬프트 엔지니어링, LLM-as-a-Judge, 시각화 |

> SSH 환경은 GPU 공유로 인한 OOM 발생 가능성이 있어, 평가 및 분석 작업은 Colab에서 진행하였습니다.

---

## 프로젝트 구조

```
daon_ai_train/
│
├── src/                             # 핵심 학습 및 평가 코드
│   ├── train_lora.py                # LoRA 파인튜닝 (Pressure / Friendly 순차 학습)
│   ├── hyperparameter_search.py     # Optuna TPE 하이퍼파라미터 탐색
│   ├── inference.py                 # Qwen2.5-7B 추론 (Multi-LoRA 스와핑)
│   ├── inference_9B.py              # Qwen3.5-9B 추론
│   ├── inference_cpu.py             # CPU 환경 추론
│   ├── convert_to_gguf.py           # GGUF 변환
│   ├── prompt_engineering.ipynb     # 프롬프트 엔지니어링 v1~v12 [Colab]
│   ├── AB_Test.ipynb                # A/B 강약 프롬프트 테스트 [Colab]
│   ├── AB_Test_evaluation.ipynb     # Rule Pass Rate / Style Score 평가 [Colab]
│   ├── llm_as_a_judge.ipynb         # LLM-as-a-Judge Pairwise Win Rate 분석 [Colab]
│   └── four_quadrants.ipynb         # 4사분면 모델 성능 시각화 [Colab]
│
├── scripts/                         # 데이터 전처리 및 증강
│   ├── augment_persona.py           # GPT 기반 페르소나 데이터 증강
│   ├── extract_neutral_followups.py # 중립 꼬리질문 추출
│   └── run_style_qa.py              # 스타일 QA 자동 평가 파이프라인
│
├── evaluators/
│   └── style_rules.py               # Pressure / Friendly 룰 기반 평가 기준 정의
│
├── experiments/                     # 하이퍼파라미터 탐색 및 프롬프트 실험 결과
│   ├── hyperparam_results.csv       # Optuna 탐색 결과
│   ├── prompt_scores.csv            # 프롬프트 버전별 평가 점수
│   ├── prompt_evolution_summary.txt # 프롬프트 개선 이력 요약
│   ├── qwen2.5_hyperparameter.txt   # Qwen2.5 최적 하이퍼파라미터
│   └── qwen3.5_hyperparameter.txt   # Qwen3.5 최적 하이퍼파라미터
│
├── prompts/                         # 프롬프트 버전 관리
│   ├── final_pressure.txt           # 최종 압박 페르소나 프롬프트
│   ├── final_friendly.txt           # 최종 친화 페르소나 프롬프트
│   ├── prompt_versions_pressure.txt # 압박 페르소나 프롬프트 변경 이력
│   ├── prompt_versions_friendly.txt # 친화 페르소나 프롬프트 변경 이력
│   ├── prompt_baseline_summary.txt  # 프롬프트 엔지니어링 baseline 총정리
│   └── prompt_changelog.txt         # 프롬프트 엔지니어링 변경점 총정리
│
├── data/
│   ├── processed/                   # 전처리 및 증강 완료 데이터
│   ├── raw/                         # 원본 데이터
│   ├── samples/                     # 샘플 데이터
│   └── splits/                      # train / val / test split
│
├── training/                        # 학습 설정 및 Unsloth 관련
│   ├── configs/
│   └── unsloth/
│
├── embedding.py                     # OpenAI 임베딩 + 클러스터링 기반 데이터 분할
├── requirements.txt
├── CONTRIBUTING.md                  # 개발 환경 및 협업 규칙
└── README.md
```

---

## 파이프라인

### 1. 데이터 증강

```bash
python scripts/augment_persona.py
```

GPT-4o-mini를 활용하여 ICT 직무 면접 질문에 대한 지원자 답변 및 페르소나별 꼬리질문을 ThreadPoolExecutor 기반 병렬 처리로 생성합니다.
Pressure / Friendly 페르소나 각각에 대해 vague, logical, experience_based 등 6가지 답변 유형을 생성합니다.

### 2. 데이터 분할

```bash
python embedding.py
```

OpenAI 임베딩 + AgglomerativeClustering + GroupShuffleSplit을 사용하여
의미적으로 유사한 데이터가 동일 split에 집중되지 않도록 분할합니다.

### 3. LoRA 학습

```bash
python src/train_lora.py
```

Unsloth QLoRA를 사용하여 Pressure / Friendly 페르소나 어댑터를 순차 학습합니다.

| 모델 | LoRA Rank | Learning Rate | Trainable Params |
|------|-----------|---------------|-----------------|
| Qwen2.5-7B-Instruct | 32 | 1e-4 | 80.7M (1.05%) |
| Qwen3.5-9B | 8 | 2e-4 | 14.5M (0.15%) |

- Optimizer: AdamW (cosine scheduler)
- Effective Batch Size: 16 (batch 4 × gradient accumulation 4)
- Early Stopping: patience=3 (eval_loss 기준)

### 4. 하이퍼파라미터 탐색

```bash
python src/hyperparameter_search.py
```

Optuna TPE 탐색으로 LoRA rank, learning rate, batch size 최적값을 탐색합니다.
탐색 결과는 `experiments/` 디렉토리에 저장됩니다.

### 5. 추론

```bash
# Qwen2.5-7B (Multi-LoRA 스와핑)
python src/inference.py

# Qwen3.5-9B
python src/inference_9B.py
```

### 6. 스타일 QA 평가

```bash
python scripts/run_style_qa.py
```

룰 기반 검증기(Style Rules)를 통해 모델 출력의 규칙 준수율을 자동 평가합니다.

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| Style Score | Pressure / Friendly 룰 준수 종합 점수 |
| Rule Pass Rate | sentence / question / forbidden / end / korean 규칙별 통과율 |
| Win Rate | LLM-as-a-Judge Pairwise 양방향 평가 승률 (position bias 보정) |
| Hard Fail Rate | 중국어 혼입 / 프롬프트 유출 / 한국어 비율 미달 발생률 |
| Reference Similarity | 모델 출력과 reference output 간 코사인 유사도 |

---

## 주요 결과

### LLM-as-a-Judge Pairwise (표현 품질)
GPT-5 양방향 평가 (position bias 보정)

| 순위 | 모델 | Win Rate |
|------|------|----------|
| 1 | Qwen3.5-9B-LoRA | 81.09% |
| 2 | Qwen2.5-7B-Base | 48.40% |
| 3 | Qwen2.5-7B-LoRA | 37.50% |
| 4 | Qwen3.5-9B-Base | 33.01% |

### Rule Pass Rate (형식 준수율)
시스템이 요구하는 출력 규칙 준수 여부 측정

| 모델 | Style Score | sentence | question | end | korean |
|------|------------|----------|----------|-----|--------|
| Qwen3.5-9B-LoRA | 83.77% | 100% | 84.21% | 52.64% | 53.38% |
| Qwen2.5-7B-LoRA | 66.82% | 98.87% | 73.31% | 18.42% | 78.38% |
| Qwen2.5-7B-Base | 61.68% | 96.99% | 71.43% | 20.11% | 94.74% |
| Qwen3.5-9B-Base | 54.92% | 92.30% | 63.72% | 22.56% | 44.74% |

> Win Rate는 출력의 자연스러움과 표현 품질을, Rule Pass Rate는 시스템이 요구하는 출력 형식 준수 여부를 측정하며 두 지표는 상호 보완적으로 해석해야 합니다.
---

## 의존성 설치

```bash
pip install -r requirements.txt
```
