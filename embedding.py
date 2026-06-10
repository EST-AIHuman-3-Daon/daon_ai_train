import re
import json
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from openai import OpenAI
import os
import pickle
from dotenv import load_dotenv

# 1. 인코딩 설정 (가장 먼저!)
os.environ["PYTHONIOENCODING"] = "utf-8"

# 2. .env 파일 경로 지정 및 로드
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, '.env')
load_dotenv(dotenv_path=env_path)

# 3. API 키 가져오기 및 정제
raw_api_key = os.getenv("OPENAI_API_KEY")

if not raw_api_key:
    print(f"❌ API 키를 찾을 수 없습니다! .env 파일 경로: {env_path}")
    sys.exit(1) # 에러 시 여기서 강제 종료

# ASCII 정제 (오류 방지)
api_key = raw_api_key.encode('ascii', 'ignore').decode('ascii').strip()

# 4. 클라이언트 생성 (전역 변수로 확실하게 선언)
try:
    client = OpenAI(api_key=api_key)
    print("✅ OpenAI 클라이언트 생성 성공!")
except Exception as e:
    print(f"❌ 클라이언트 생성 중 오류 발생: {e}")
    sys.exit(1)


# =========================
# 1. 텍스트 정규화
# =========================

# 프로젝트 루트 경로 설정 (현재 스크립트 위치 기준)
# 만약 실행 위치가 daon_project 폴더 내부라면 아래처럼 설정합니다.
BASE_DIR = os.getcwd() 
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'sft_augmented_dataset_promptC2_final_20260529_152034.csv')

# 데이터 불러오기
df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def extract_question(input_json):
    data = json.loads(input_json)
    return normalize_text(data.get("question", ""))


df["normalized_q"] = df["input"].apply(extract_question)


# =========================
# 2. unique question 추출
# =========================
unique_qs = df["normalized_q"].unique().tolist()


# =========================
# 3. embedding cache 로드/저장
# =========================
cache_path = "embeddings_cache.pkl"

if os.path.exists(cache_path):
    print("Loading cached embeddings...")
    with open(cache_path, "rb") as f:
        embeddings = pickle.load(f)

else:
    print("Generating embeddings...")

    # ✔ batch embedding (핵심 개선)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=unique_qs
    )

    embeddings = np.array([d.embedding for d in response.data])

    # cache 저장 (재현성)
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)


# =========================
# 4. clustering (semantic grouping)
# =========================

# cosine distance = 1 - similarity
distance_threshold = 0.20  # ≈ similarity 0.80

clustering = AgglomerativeClustering(
    n_clusters=None,
    metric="cosine",
    linkage="average",
    distance_threshold=distance_threshold
)

group_labels = clustering.fit_predict(embeddings)


# =========================
# 5. mapping 생성
# =========================
q_to_group = dict(zip(unique_qs, group_labels))

df["group_id"] = df["normalized_q"].map(q_to_group)


# =========================
# 6. 그룹 품질 체크
# =========================
group_stats = df.groupby("group_id").size()

print("\n=== Group Statistics ===")
print(group_stats.describe())

print("\n=== Top 10 largest groups ===")
print(group_stats.sort_values(ascending=False).head(10))

# =====================================================
# =====================================================

from sklearn.model_selection import GroupShuffleSplit

# 1. 8 : 2 분할 (전체 -> Train 80% / Temp 20%)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, temp_idx = next(splitter.split(df, groups=df["group_id"]))

train_df = df.iloc[train_idx].copy()
temp_df = df.iloc[temp_idx].copy()

# 2. Temp를 5:5 분할 (Val 10% / Test 10% -> 전체 대비 8:1:1)
splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(splitter2.split(temp_df, groups=temp_df["group_id"]))

val_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()

# 3. Sanity Check (중복 없이 딱 한 번만 실행)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# 그룹 누수(Leakage) 확인: 결과가 empty set([]) 이어야 완벽함
overlap = set(train_df.group_id) & set(test_df.group_id)
print(f"\nGroup overlap check (Train vs Test): {overlap}")
assert len(overlap) == 0, "데이터 누수 발생! 분할 로직 확인 필요"

print("전체:", len(df))
print("Train+Val+Test:", len(train_df)+len(val_df)+len(test_df))
print("누락:", len(df) - (len(train_df)+len(val_df)+len(test_df)))

print(df["persona"].value_counts())
print(df["job_role"].value_counts())
print(df["generation_method"].value_counts())

# =====================================================
# 8. CSV 저장
# =====================================================
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False, encoding='utf-8-sig')
val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'),   index=False, encoding='utf-8-sig')
test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False, encoding='utf-8-sig')

print(f"\n✅ 저장 완료")
print(f"  train.csv: {len(train_df)}행")
print(f"  val.csv:   {len(val_df)}행")
print(f"  test.csv:  {len(test_df)}행")
print(f"  저장 위치: {OUTPUT_DIR}")