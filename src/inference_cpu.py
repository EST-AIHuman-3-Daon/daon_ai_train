# inference_cpu.py
"""
CPU 추론 (GPU 메모리 제약 없음)
느림: ~3-5초/문장 but 안정적
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_VERSION = "Qwen2.5-7B-Instruct"
BASE_DIR = Path(__file__).resolve().parent.parent
ADAPTER_DIR = BASE_DIR / "adapters" / MODEL_VERSION

_model_cache = {}

def load_model(persona: str):
    """CPU 로드 (메모리 제약 없음)"""
    
    if persona in _model_cache:
        return _model_cache[persona]
    
    print(f"\n🔄 [{persona}] CPU에서 로딩 중... (1분 소요)")
    
    adapter_path = str(ADAPTER_DIR / f"{persona}_lora")
    
    try:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_path,
            torch_dtype=torch.float32,  # CPU는 float32
            device_map="cpu",  # 명시적으로 CPU
        )
        model = model.merge_and_unload()
    except:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype=torch.float32,
            device_map="cpu",
        )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # CPU에서 실행
    model.eval()
    
    _model_cache[persona] = (model, tokenizer)
    print(f"✅ [{persona}] CPU 로드 완료 (메모리 제약 없음)")
    
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_tokens: int = 100) -> str:
    """CPU 추론 (느림)"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    print("⏳ 생성 중 (3-5초 대기)...", end="", flush=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_beams=1,  # CPU는 beam search 비활성화
        )

    print("\r                    \r", end="")  # 진행중 메시지 제거
    
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()


def run_interview(persona: str, job_role: str):
    model, tokenizer = load_model(persona)
    history = []
    turn = 0

    print(f"\n{'='*50}")
    print(f"면접 시뮬레이터 (CPU 모드 - 느림)")
    print(f"페르소나: {persona} | 직무: {job_role}")
    print('='*50)
    print("⚠️  CPU 추론이므로 각 문장에 3-5초 소요됩니다.\n")

    while True:
        user_input = input("지원자: ").strip()
        
        if not user_input:
            continue
        
        if any(t in user_input for t in ["면접 종료", "종료", "끝"]):
            print(f"\n✅ 종료 ({turn}턴)\n")
            break
        
        if any(t in user_input for t in ["면접 시작", "시작"]):
            turn = 0
            history = []
            prompt = f"{job_role} 직무 첫 질문을 생성하세요."
            
            print("\n🎤 [면접관]: ", end="", flush=True)
            response = generate(model, tokenizer, prompt)
            print(response)
            history.append({"question": response, "answer": None})
            continue
        
        if history and history[-1]["answer"] is None:
            history[-1]["answer"] = user_input
            turn += 1
            
            if turn >= 10:
                print("\n🎤 [면접관]: 수고하셨습니다.")
                break
            
            recent = "\n".join([
                f"Q: {h['question']}\nA: {h['answer']}"
                for h in history[-2:] if h['answer']
            ])
            
            prompt = f"{recent}\n새로운 질문:"
            print("\n🎤 [면접관]: ", end="", flush=True)
            response = generate(model, tokenizer, prompt)
            print(response)
            history.append({"question": response, "answer": None})


if __name__ == "__main__":
    run_interview("pressure", "ICT")