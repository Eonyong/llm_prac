"""
SimpleLLM 인터랙티브 실행

사용자 입력을 받아 모델의 다음 토큰 예측을 보여줍니다.
"""

import torch
import torch.nn as nn
import tiktoken
import sys
from layer_normalization import SimpleLLM


def generate_text(model, tokenizer, prompt, max_new_tokens=20, temperature=1.0):
    """
    텍스트 생성

    Args:
        model: SimpleLLM 모델
        tokenizer: 토크나이저
        prompt: 입력 프롬프트
        max_new_tokens: 생성할 최대 토큰 수
        temperature: 샘플링 온도 (높을수록 다양한 출력)

    Returns:
        generated_text: 생성된 텍스트
    """
    model.eval()

    # 프롬프트 토큰화
    token_ids = tokenizer.encode(prompt)
    token_tensor = torch.tensor(token_ids).unsqueeze(0)  # [1, seq_len]

    generated_tokens = token_ids.copy()

    print(f"\n생성 중", end="", flush=True)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 현재 시퀀스에 대한 예측
            logits = model(token_tensor)

            # 마지막 토큰의 로짓
            last_logits = logits[0, -1, :] / temperature

            # 확률 계산 및 샘플링
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # 생성된 토큰 추가
            generated_tokens.append(next_token)
            token_tensor = torch.tensor(generated_tokens).unsqueeze(0)

            # 진행 표시
            print(".", end="", flush=True)

            # 종료 토큰 체크 (optional)
            if next_token == tokenizer.eot_token:
                break

    print(" 완료!\n")

    # 디코딩
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


def analyze_next_token(model, tokenizer, text, top_k=10):
    """
    다음 토큰 예측 분석

    Args:
        model: SimpleLLM 모델
        tokenizer: 토크나이저
        text: 입력 텍스트
        top_k: 상위 k개 토큰 표시
    """
    model.eval()

    # 토큰화
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    token_tensor = torch.tensor(token_ids).unsqueeze(0)

    print(f"\n📝 입력: \"{text}\"")
    print(f"토큰 개수: {len(token_ids)}")
    print(f"토큰: {tokens}\n")

    # 순전파
    with torch.no_grad():
        logits = model(token_tensor)

    # 마지막 토큰의 예측
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)

    # Top-k 토큰
    topk_probs, topk_indices = torch.topk(probs, top_k)

    print(f"🎯 다음 토큰 예측 (Top {top_k}):")
    print("-" * 60)
    for i, (prob, idx) in enumerate(zip(topk_probs, topk_indices), 1):
        token = tokenizer.decode([idx.item()])
        bar_length = int(prob.item() * 100)
        bar = "█" * (bar_length // 2)
        print(f"{i:2d}. '{token:15s}' {prob.item()*100:6.2f}% {bar}")


def main():
    print("="*80)
    print("🤖 SimpleLLM - 간단한 언어 모델")
    print("="*80)
    print()

    # 모델 초기화
    print("모델을 초기화하는 중...")
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    torch.manual_seed(42)
    model = SimpleLLM(
        vocab_size=vocab_size,
        d_embed=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=512,
        dropout=0.1
    )

    print("💡 참고: 이 모델은 학습되지 않았습니다 (무작위 가중치).")
    print("   학습된 모델이라면 더 의미있는 예측을 할 수 있습니다.\n")

    # 명령줄 인자 확인
    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "generate":
            # 텍스트 생성 모드
            prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Hello"
            print(f"📝 프롬프트: \"{prompt}\"")
            generated = generate_text(model, tokenizer, prompt, max_new_tokens=20)
            print(f"✨ 생성된 텍스트:\n{generated}\n")

        elif mode == "analyze":
            # 다음 토큰 분석 모드
            text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Hello"
            analyze_next_token(model, tokenizer, text, top_k=10)

        else:
            print(f"❌ 알 수 없는 모드: {mode}")
            print("사용법: python run_simple_llm.py [generate|analyze] 'your text'")

    else:
        # 인터랙티브 모드
        print("💡 사용법:")
        print("  - 'analyze: your text' - 다음 토큰 예측 분석")
        print("  - 'generate: your text' - 텍스트 생성")
        print("  - 'quit' 또는 'exit' - 종료")
        print()

        while True:
            try:
                user_input = input("📝 입력: ").strip()

                if not user_input:
                    print("⚠️  텍스트를 입력해주세요.\n")
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 프로그램을 종료합니다.")
                    break

                # 명령 파싱
                if user_input.startswith("analyze:"):
                    text = user_input[8:].strip()
                    if text:
                        analyze_next_token(model, tokenizer, text, top_k=10)
                    else:
                        print("⚠️  분석할 텍스트를 입력해주세요.\n")

                elif user_input.startswith("generate:"):
                    text = user_input[9:].strip()
                    if text:
                        generated = generate_text(model, tokenizer, text, max_new_tokens=20)
                        print(f"✨ 생성된 텍스트:\n{generated}\n")
                    else:
                        print("⚠️  생성할 프롬프트를 입력해주세요.\n")

                else:
                    # 기본값: 분석
                    analyze_next_token(model, tokenizer, user_input, top_k=10)

            except KeyboardInterrupt:
                print("\n\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류: {e}\n")


if __name__ == "__main__":
    main()
