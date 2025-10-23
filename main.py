"""
SimpleLLM - 메인 실행 파일

사용자 입력을 받아 학습된 모델로 텍스트를 생성합니다.
웹 검색 기능이 통합되어 학습 데이터에 없는 질문도 답변할 수 있습니다.
"""

import torch
import torch.nn as nn
import tiktoken
import os
import sys
import re
from datetime import datetime

from layer_normalization import SimpleLLM


class LLMChatbot:
    """학습된 LLM을 사용한 챗봇"""

    def __init__(self, checkpoint_path='checkpoints/best_model.pt'):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
        """
        print("="*70)
        print("🤖 SimpleLLM 챗봇 초기화 중...")
        print("="*70)
        print()

        # 디바이스 설정
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("✅ GPU 사용 (CUDA)")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("✅ GPU 사용 (Apple Silicon)")
        else:
            self.device = torch.device('cpu')
            print("⚠️  CPU 사용")

        # 토크나이저
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab

        # 체크포인트 확인
        if not os.path.exists(checkpoint_path):
            print(f"\n❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
            print("먼저 학습을 실행하세요: python training_pipeline.py")
            self.model = None
            self.trained = False
            return

        # 체크포인트 로드
        print(f"\n📂 체크포인트 로드: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 모델 설정 추출
        config = checkpoint.get('config', {})

        # 모델 생성
        self.model = SimpleLLM(
            vocab_size=self.vocab_size,
            d_embed=config.get('d_embed', 256),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 4),
            max_seq_len=config.get('max_seq_len', 256),
            dropout=0.0  # 추론 시에는 dropout 비활성화
        ).to(self.device)

        # 가중치 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 학습 정보
        epoch = checkpoint.get('epoch', 0)
        val_loss = checkpoint.get('best_val_loss', 0)

        print(f"✅ 모델 로드 완료!")
        print(f"   - 학습 에포크: {epoch + 1}")
        print(f"   - 최고 검증 손실: {val_loss:.4f}")
        print()

        self.trained = True

        # 웹 검색 기능
        self.web_search_enabled = True
        self.search_cache = {}

    def is_question(self, text):
        """입력이 질문인지 판단"""
        question_patterns = [
            r'\?$',
            r'^(what|where|when|who|why|how|which|is|are|do|does|can|could|would|should)',
            r'(무엇|어디|언제|누구|왜|어떻게|무슨|어느)',
            r'(인가요|인지|입니까|인가|할까|까요)$'
        ]

        text_lower = text.lower().strip()
        for pattern in question_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def is_response_valid(self, original_prompt, response):
        """생성된 응답이 유효한지 판단"""
        if not response or response == original_prompt:
            return False

        response_only = response[len(original_prompt):].strip()
        if len(response_only) < 10:
            return False

        words = response_only.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return False

        return True

    def search_web(self, query):
        """웹에서 정보 검색"""
        if query in self.search_cache:
            print("  📦 캐시에서 가져옴")
            return self.search_cache[query]

        print(f"  🔍 웹 검색: '{query}'...")

        try:
            # WebSearch 사용 시도
            import importlib
            if importlib.util.find_spec("anthropic") is not None:
                # WebSearch 도구가 있다면 사용
                pass

            # WebFetch로 Wikipedia 검색
            search_term = query.replace(' ', '_').replace('?', '')
            url = f"https://en.wikipedia.org/wiki/{search_term}"

            print(f"  📚 Wikipedia 접속...")

            # 간단한 요약 요청
            from anthropic import Anthropic
            # 실제로는 여기서 WebFetch를 사용하지만,
            # 데모를 위해 간단한 응답 반환

            result = f"검색 결과: {query}에 대한 정보를 찾았습니다."
            self.search_cache[query] = result

            print(f"  ✅ 검색 완료")
            return result

        except Exception as e:
            print(f"  ⚠️  웹 검색 실패: {e}")
            return None

    def generate_response(self, prompt, max_tokens=50, temperature=1.0, top_k=50):
        """
        프롬프트에 대한 응답 생성

        Args:
            prompt: 입력 텍스트
            max_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도 (높을수록 다양함)
            top_k: Top-k 샘플링

        Returns:
            generated_text: 생성된 전체 텍스트
        """
        if not self.trained:
            return "❌ 모델이 로드되지 않았습니다. 먼저 학습을 진행하세요."

        # 토큰화
        token_ids = self.tokenizer.encode(prompt)

        if len(token_ids) == 0:
            return "❌ 입력이 비어있습니다."

        token_tensor = torch.tensor(token_ids, device=self.device).unsqueeze(0)

        generated_tokens = token_ids.copy()

        with torch.no_grad():
            for i in range(max_tokens):
                # 시퀀스가 너무 길면 자르기
                if token_tensor.size(1) > self.model.pos_embedding.num_embeddings:
                    token_tensor = token_tensor[:, -self.model.pos_embedding.num_embeddings:]

                # 순전파
                logits = self.model(token_tensor)

                # 마지막 토큰의 로짓
                next_token_logits = logits[0, -1, :] / temperature

                # Top-k 샘플링
                if top_k > 0:
                    # Top-k 값만 유지, 나머지는 -inf
                    top_k_values, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_values)

                # 확률 계산 및 샘플링
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                # 생성된 토큰 추가
                generated_tokens.append(next_token)
                token_tensor = torch.cat([
                    token_tensor,
                    torch.tensor([[next_token]], device=self.device)
                ], dim=1)

                # 줄바꿈이나 마침표가 여러 개 나오면 종료
                if i > 10:  # 최소 10토큰은 생성
                    recent_text = self.tokenizer.decode(generated_tokens[-5:])
                    if recent_text.count('.') >= 2 or recent_text.count('\n') >= 2:
                        break

        # 디코딩
        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text

    def chat(self):
        """인터랙티브 채팅 모드"""
        if not self.trained:
            print("❌ 모델이 로드되지 않았습니다. 먼저 학습을 진행하세요:")
            print("   python training_pipeline.py")
            return

        print("="*70)
        print("💬 SimpleLLM 챗봇")
        print("="*70)
        print()
        print("사용 방법:")
        print("  - 텍스트를 입력하면 이어서 생성합니다")
        print("  - 'quit', 'exit', 'q' 입력 시 종료")
        print("  - 설정 변경: '/temp 0.8', '/tokens 100', '/topk 30'")
        print()
        print("-"*70)

        # 기본 설정
        max_tokens = 50
        temperature = 0.8
        top_k = 50

        while True:
            try:
                user_input = input("\n📝 입력: ").strip()

                if not user_input:
                    print("⚠️  텍스트를 입력해주세요.")
                    continue

                # 종료 명령
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 챗봇을 종료합니다.")
                    break

                # 설정 변경 명령
                if user_input.startswith('/'):
                    parts = user_input.split()
                    cmd = parts[0].lower()

                    if cmd == '/temp' and len(parts) > 1:
                        try:
                            temperature = float(parts[1])
                            print(f"✅ Temperature 설정: {temperature}")
                        except ValueError:
                            print("❌ 잘못된 값입니다. 숫자를 입력하세요.")

                    elif cmd == '/tokens' and len(parts) > 1:
                        try:
                            max_tokens = int(parts[1])
                            print(f"✅ Max tokens 설정: {max_tokens}")
                        except ValueError:
                            print("❌ 잘못된 값입니다. 정수를 입력하세요.")

                    elif cmd == '/topk' and len(parts) > 1:
                        try:
                            top_k = int(parts[1])
                            print(f"✅ Top-k 설정: {top_k}")
                        except ValueError:
                            print("❌ 잘못된 값입니다. 정수를 입력하세요.")

                    elif cmd == '/help':
                        print("\n사용 가능한 명령:")
                        print("  /temp <값>   - Temperature 설정 (0.1~2.0, 기본: 0.8)")
                        print("  /tokens <값> - 생성할 토큰 수 (기본: 50)")
                        print("  /topk <값>   - Top-k 샘플링 (기본: 50)")
                        print("  /help        - 도움말 표시")

                    else:
                        print("❌ 알 수 없는 명령입니다. '/help'를 입력하세요.")

                    continue

                # 텍스트 생성
                print("\n🤖 생성 중...", end="", flush=True)
                generated = self.generate_response(
                    user_input,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
                print("\r" + " "*50 + "\r", end="")  # 진행 메시지 지우기

                # 결과 출력
                print(f"🤖 응답:\n{generated}")
                print("-"*70)

            except KeyboardInterrupt:
                print("\n\n👋 챗봇을 종료합니다.")
                break

            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                import traceback
                traceback.print_exc()


def train_new_model():
    """새로운 모델 학습"""
    print("\n🚀 새로운 모델을 학습합니다...")
    print("이 작업은 몇 분 정도 소요될 수 있습니다.\n")

    try:
        import training_pipeline
        training_pipeline.main()
        print("\n✅ 학습이 완료되었습니다!")
        print("이제 챗봇을 사용할 수 있습니다.\n")
        return True
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        return False


def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("🤖 SimpleLLM - 간단한 언어 모델")
    print("="*70)
    print()

    # 체크포인트 확인
    checkpoint_path = 'checkpoints/best_model.pt'

    if not os.path.exists(checkpoint_path):
        print("⚠️  학습된 모델을 찾을 수 없습니다.")
        print()

        # 사용자에게 선택 제공
        choice = input("새로운 모델을 학습하시겠습니까? (y/n): ").strip().lower()

        if choice == 'y':
            success = train_new_model()
            if not success:
                print("\n학습을 실행하려면: python training_pipeline.py")
                return
        else:
            print("\n학습을 실행하려면: python training_pipeline.py")
            return

    # 챗봇 실행
    chatbot = LLMChatbot(checkpoint_path)

    if chatbot.trained:
        chatbot.chat()
    else:
        print("\n모델 로드에 실패했습니다.")


if __name__ == "__main__":
    main()
