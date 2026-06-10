from unsloth import FastLanguageModel
import os


MODEL_VERSION = "Qwen2.5-7B-Instruct"
# MODEL_VERSION = "Qwen3.5-9B"

BASE_PATH = f"/smhrd/DaonTeam/Siyeong/daon_project/adapters/{MODEL_VERSION}"
PERSONAS = ["friendly", "pressure"]

def convert_to_gguf(persona):
    adapter_path = os.path.join(BASE_PATH, f"{persona}_lora")
    print(f"\n>>> [{persona}] GGUF 변환 시작: {adapter_path}")
    
    # 1. 모델 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    # 2. GGUF 저장 (q4_k_m 권장)
    save_path = f"{persona}_model_gguf"
    model.save_pretrained_gguf(
        save_path, 
        tokenizer, 
        quantization_method = "q4_k_m"
    )
    print(f">>> [{persona}] 변환 완료. 저장 경로: {save_path}")

if __name__ == "__main__":
    for p in PERSONAS:
        convert_to_gguf(p)
    print("\n모든 페르소나 변환이 완료되었습니다.")