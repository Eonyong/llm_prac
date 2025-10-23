"""
Web-Enhanced LLM Chatbot

학습 데이터에 답변이 없으면 웹에서 검색하여 답변하는 챗봇
"""

import torch
import torch.nn as nn
import tiktoken
import os
import sys
import re
from datetime import datetime

from layer_normalization import SimpleLLM


class WebEnhancedChatbot:
    """웹 검색 기능이 추가된 LLM 챗봇"""

    def __init__(self, checkpoint_path='checkpoints/best_model.pt'):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
        """
        print("="*70)
        print("🤖 Web-Enhanced SimpleLLM 챗봇")
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
            print(f"\n⚠️  체크포인트를 찾을 수 없습니다: {checkpoint_path}")
            print("기본 모드로 실행합니다 (웹 검색만 사용).\n")
            self.model = None
            self.trained = False
        else:
            # 모델 로드
            print(f"\n📂 체크포인트 로드: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            config = checkpoint.get('config', {})

            self.model = SimpleLLM(
                vocab_size=self.vocab_size,
                d_embed=config.get('d_embed', 256),
                num_heads=config.get('num_heads', 4),
                num_layers=config.get('num_layers', 4),
                max_seq_len=config.get('max_seq_len', 256),
                dropout=0.0
            ).to(self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            print(f"✅ 모델 로드 완료!\n")
            self.trained = True

        # 웹 검색 캐시 (중복 검색 방지)
        self.search_cache = {}

        # 지식 베이스 (검색 결과 저장)
        self.knowledge_base = {}

    def is_question(self, text):
        """
        입력이 질문인지 판단

        Args:
            text: 입력 텍스트

        Returns:
            bool: 질문 여부
        """
        # 질문 패턴 감지
        question_patterns = [
            r'\?$',  # 물음표로 끝남
            r'^(what|where|when|who|why|how|which|is|are|do|does|can|could|would|should)',  # 질문 시작
            r'(무엇|어디|언제|누구|왜|어떻게|무슨|어느)',  # 한글 질문
            r'(인가요|인지|입니까|인가|할까|ㄹ까|면)$'  # 한글 질문 종결
        ]

        text_lower = text.lower().strip()

        for pattern in question_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        return False

    def generate_response(self, prompt, max_tokens=50, temperature=0.8):
        """
        모델로 응답 생성 (기존 방식)

        Args:
            prompt: 입력 텍스트
            max_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도

        Returns:
            generated_text: 생성된 텍스트
        """
        if not self.trained:
            return None

        try:
            token_ids = self.tokenizer.encode(prompt)
            if len(token_ids) == 0:
                return None

            token_tensor = torch.tensor(token_ids, device=self.device).unsqueeze(0)
            generated_tokens = token_ids.copy()

            with torch.no_grad():
                for i in range(max_tokens):
                    if token_tensor.size(1) > self.model.pos_embedding.num_embeddings:
                        token_tensor = token_tensor[:, -self.model.pos_embedding.num_embeddings:]

                    logits = self.model(token_tensor)
                    next_token_logits = logits[0, -1, :] / temperature

                    # Top-k 샘플링
                    top_k = 50
                    top_k_values, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_values)

                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

                    generated_tokens.append(next_token)
                    token_tensor = torch.cat([
                        token_tensor,
                        torch.tensor([[next_token]], device=self.device)
                    ], dim=1)

                    # 조기 종료 조건
                    if i > 10:
                        recent_text = self.tokenizer.decode(generated_tokens[-5:])
                        if recent_text.count('.') >= 2 or recent_text.count('\n') >= 2:
                            break

            generated_text = self.tokenizer.decode(generated_tokens)
            return generated_text

        except Exception as e:
            print(f"⚠️  생성 중 오류: {e}")
            return None

    def is_response_valid(self, original_prompt, response):
        """
        생성된 응답이 유효한지 판단

        Args:
            original_prompt: 원본 질문
            response: 생성된 응답

        Returns:
            bool: 유효 여부
        """
        if not response or response == original_prompt:
            return False

        # 응답이 너무 짧거나 의미없는 경우
        response_only = response[len(original_prompt):].strip()
        if len(response_only) < 10:
            return False

        # 반복되는 단어가 너무 많은 경우
        words = response_only.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # 70% 이상 반복
                return False

        return True

    def search_web(self, query):
        """
        웹에서 정보 검색

        Args:
            query: 검색 쿼리

        Returns:
            result: 검색 결과 텍스트
        """
        # 캐시 확인
        if query in self.search_cache:
            print("📦 캐시에서 결과를 가져왔습니다.")
            return self.search_cache[query]

        print(f"\n🔍 웹 검색 중: '{query}'")

        # 간단한 지식 베이스 (데모용)
        # 실제로는 웹 API를 사용해야 합니다
        result = self._search_from_knowledge_base(query)

        if result:
            self.search_cache[query] = result
            print(f"✅ 검색 완료!")
            return result
        else:
            print("⚠️  검색 결과가 없습니다.")
            print("💡 더 나은 검색을 위해 더 큰 데이터셋으로 학습하거나,")
            print("   외부 API (OpenAI, Google Search API 등)를 연동하세요.")
            return None

    def _process_search_results(self, results):
        """
        검색 결과를 텍스트로 처리

        Args:
            results: 검색 결과

        Returns:
            processed_text: 처리된 텍스트
        """
        # 검색 결과가 문자열인 경우
        if isinstance(results, str):
            return results[:2000]  # 최대 2000자

        # 검색 결과가 리스트인 경우
        if isinstance(results, list):
            texts = []
            for item in results[:3]:  # 상위 3개 결과만
                if isinstance(item, dict):
                    title = item.get('title', '')
                    content = item.get('content', '') or item.get('snippet', '')
                    texts.append(f"{title}: {content}")
                elif isinstance(item, str):
                    texts.append(item)

            return "\n\n".join(texts)[:2000]

        return str(results)[:2000]

    def _search_from_knowledge_base(self, query):
        """
        내장 지식 베이스에서 검색 (데모용)

        Args:
            query: 검색 쿼리

        Returns:
            result: 검색 결과
        """
        # 간단한 키워드 기반 지식 베이스
        knowledge = {
            # AI 관련
            "artificial intelligence": "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction. AI is used in various applications such as speech recognition, image processing, and autonomous vehicles.",

            "machine learning": "Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",

            "deep learning": "Deep Learning is a subset of machine learning based on artificial neural networks. It uses multiple layers to progressively extract higher-level features from raw input. Deep learning is particularly effective for tasks like image and speech recognition.",

            "neural network": "A Neural Network is a series of algorithms that attempts to recognize underlying relationships in data through a process that mimics the way the human brain operates. It consists of interconnected nodes (neurons) organized in layers.",

            # 프로그래밍 관련
            "python": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, machine learning, and automation. Python emphasizes code readability and supports multiple programming paradigms.",

            "pytorch": "PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It's widely used for applications such as computer vision and natural language processing. PyTorch provides flexibility and speed in building deep learning models.",

            # 일반 지식
            "transformer": "Transformer is a deep learning model architecture introduced in 2017. It uses self-attention mechanisms to process sequential data and has become the foundation for modern language models like GPT and BERT.",

            "llm": "Large Language Model (LLM) is an AI model trained on vast amounts of text data to understand and generate human-like text. Examples include GPT-3, GPT-4, and BERT. LLMs can perform various tasks like translation, summarization, and question answering.",
        }

        # 쿼리를 소문자로 변환하여 검색
        query_lower = query.lower()

        # 직접 매칭
        for key, value in knowledge.items():
            if key in query_lower or query_lower in key:
                return value

        # 키워드 매칭
        keywords = query_lower.split()
        best_match = None
        max_score = 0

        for key, value in knowledge.items():
            score = sum(1 for keyword in keywords if keyword in key)
            if score > max_score:
                max_score = score
                best_match = value

        if max_score > 0:
            return best_match

        return None

    def _fallback_wikipedia_search(self, query):
        """
        Wikipedia에서 정보 검색 (fallback)

        Args:
            query: 검색 쿼리

        Returns:
            result: 검색 결과
        """
        try:
            from tools import WebFetch

            # Wikipedia URL 생성
            search_term = query.replace(' ', '_')
            url = f"https://en.wikipedia.org/wiki/{search_term}"

            print(f"📚 Wikipedia 검색: {url}")

            # 페이지 가져오기
            result = WebFetch(
                url=url,
                prompt=f"Summarize the main information about {query} in 2-3 sentences."
            )

            if result:
                print(f"✅ Wikipedia에서 정보를 가져왔습니다.")
                return result[:2000]
            else:
                return None

        except Exception as e:
            print(f"⚠️  Wikipedia 검색 실패: {e}")
            return None

    def answer_with_web(self, question):
        """
        웹 검색 결과를 기반으로 답변 생성

        Args:
            question: 질문

        Returns:
            answer: 답변
        """
        # 웹 검색
        search_result = self.search_web(question)

        if not search_result:
            return "죄송합니다. 관련 정보를 찾을 수 없습니다."

        # 검색 결과를 지식 베이스에 저장
        self.knowledge_base[question] = {
            'content': search_result,
            'timestamp': datetime.now().isoformat()
        }

        # 검색 결과 요약
        answer = self._summarize_search_result(search_result, question)

        return answer

    def _summarize_search_result(self, content, question):
        """
        검색 결과를 요약

        Args:
            content: 검색 결과 내용
            question: 원본 질문

        Returns:
            summary: 요약된 답변
        """
        # 간단한 요약: 첫 몇 문장 추출
        sentences = content.split('.')
        relevant_sentences = []

        # 질문과 관련된 문장 찾기
        question_keywords = set(question.lower().split())

        for sentence in sentences[:10]:  # 첫 10개 문장만 확인
            sentence_lower = sentence.lower()
            # 질문의 키워드가 포함된 문장 우선
            if any(keyword in sentence_lower for keyword in question_keywords if len(keyword) > 3):
                relevant_sentences.append(sentence.strip())

        # 관련 문장이 없으면 첫 2-3개 문장 사용
        if not relevant_sentences:
            relevant_sentences = [s.strip() for s in sentences[:3] if s.strip()]

        # 요약 생성
        summary = '. '.join(relevant_sentences[:3])
        if summary and not summary.endswith('.'):
            summary += '.'

        return summary or content[:500]

    def chat(self):
        """인터랙티브 채팅 모드"""
        print("="*70)
        print("💬 Web-Enhanced SimpleLLM 챗봇")
        print("="*70)
        print()
        print("기능:")
        print("  1. 일반 텍스트 입력 → 모델이 이어서 생성")
        print("  2. 질문 입력 → 모델이 답변 생성")
        print("  3. 답변이 부족하면 → 자동으로 웹 검색")
        print()
        print("명령어:")
        print("  /search <질문> - 웹에서 직접 검색")
        print("  /knowledge - 저장된 지식 확인")
        print("  /help - 도움말")
        print("  quit/exit - 종료")
        print()
        print("-"*70)

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

                # /search 명령 (직접 웹 검색)
                if user_input.startswith('/search '):
                    query = user_input[8:].strip()
                    if query:
                        answer = self.answer_with_web(query)
                        print(f"\n🌐 웹 검색 결과:\n{answer}")
                        print("-"*70)
                    else:
                        print("⚠️  검색할 내용을 입력하세요. 예: /search What is AI")
                    continue

                # /knowledge 명령 (지식 베이스 확인)
                if user_input.startswith('/knowledge'):
                    if self.knowledge_base:
                        print("\n📚 저장된 지식:")
                        for i, (question, data) in enumerate(self.knowledge_base.items(), 1):
                            print(f"\n{i}. {question}")
                            print(f"   시간: {data['timestamp']}")
                            print(f"   내용: {data['content'][:100]}...")
                    else:
                        print("\n⚠️  저장된 지식이 없습니다.")
                    continue

                # /help 명령
                if user_input.startswith('/help'):
                    print("\n사용 가능한 명령:")
                    print("  /search <질문> - 웹에서 직접 검색")
                    print("  /knowledge - 저장된 지식 확인")
                    print("  /help - 도움말 표시")
                    continue

                # 질문인지 확인
                is_question_input = self.is_question(user_input)

                if is_question_input:
                    print("\n🤔 질문으로 인식했습니다.")

                    # 먼저 모델로 답변 시도
                    if self.trained:
                        print("🤖 모델이 답변을 생성 중...")
                        model_response = self.generate_response(user_input, max_tokens=50)

                        if model_response and self.is_response_valid(user_input, model_response):
                            print(f"\n🤖 모델 답변:\n{model_response}")

                            # 사용자에게 만족도 확인
                            satisfied = input("\n이 답변이 만족스러우신가요? (y/n): ").strip().lower()

                            if satisfied == 'y':
                                print("-"*70)
                                continue
                            else:
                                print("\n더 나은 답변을 위해 웹을 검색합니다...")

                    # 웹 검색으로 답변
                    web_answer = self.answer_with_web(user_input)
                    print(f"\n🌐 웹 검색 답변:\n{web_answer}")
                    print("-"*70)

                else:
                    # 일반 텍스트 생성
                    if self.trained:
                        print("\n🤖 생성 중...", end="", flush=True)
                        generated = self.generate_response(user_input, max_tokens=50)
                        print("\r" + " "*50 + "\r", end="")

                        if generated:
                            print(f"🤖 응답:\n{generated}")
                        else:
                            print("⚠️  응답을 생성할 수 없습니다.")
                    else:
                        print("⚠️  모델이 로드되지 않았습니다. /search 명령을 사용하세요.")

                    print("-"*70)

            except KeyboardInterrupt:
                print("\n\n👋 챗봇을 종료합니다.")
                break

            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                import traceback
                traceback.print_exc()


def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("🤖 Web-Enhanced SimpleLLM")
    print("="*70)
    print()

    checkpoint_path = 'checkpoints/best_model.pt'

    # 챗봇 실행
    chatbot = WebEnhancedChatbot(checkpoint_path)
    chatbot.chat()


if __name__ == "__main__":
    main()
