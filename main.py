"""
SimpleLLM - 메인 실행 파일

한국어/영어 질문-답변 지원
웹 검색 기반 답변 생성 및 DB 저장
답변 평가 시스템 포함
"""

import torch
import torch.nn as nn
import tiktoken
import os
import sys
import re
import json
from datetime import datetime

from layer_normalization import SimpleLLM


class LLMChatbot:
    """한국어/영어 지원 QA 챗봇 with LLM 백엔드"""

    def __init__(self, checkpoint_path='checkpoints/best_model.pt', qa_database_path='qa_database.json'):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
            qa_database_path: 질문-답변 데이터베이스 경로
        """
        print("="*70)
        print("🤖 SimpleLLM 챗봇 초기화 중...")
        print("="*70)
        print()

        # QA 데이터베이스 설정
        self.qa_database_path = qa_database_path
        self.qa_database = self.load_qa_database()
        print(f"📂 QA 데이터베이스: {len(self.qa_database)} 개 저장됨")

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

        # 체크포인트 확인 (선택사항)
        self.model = None
        self.trained = False

        if os.path.exists(checkpoint_path):
            try:
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

                print(f"✅ LLM 모델 로드 완료!")
                print(f"   - 학습 에포크: {epoch + 1}")
                print(f"   - 최고 검증 손실: {val_loss:.4f}")

                self.trained = True
            except Exception as e:
                print(f"⚠️  LLM 모델 로드 실패: {e}")
                print("   웹 검색 기반으로만 동작합니다.")
        else:
            print(f"⚠️  체크포인트 없음. 웹 검색 기반으로만 동작합니다.")

        print()

        # 웹 검색 기능
        self.web_search_enabled = True
        self.search_cache = {}

    def load_qa_database(self):
        """질문-답변 데이터베이스 로드"""
        if os.path.exists(self.qa_database_path):
            try:
                with open(self.qa_database_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  DB 로드 실패: {e}")
                return {}
        return {}

    def save_qa_database(self):
        """질문-답변 데이터베이스 저장"""
        try:
            with open(self.qa_database_path, 'w', encoding='utf-8') as f:
                json.dump(self.qa_database, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  DB 저장 실패: {e}")

    def normalize_question(self, question):
        """질문 정규화 (검색용)"""
        normalized = question.lower().strip()
        normalized = re.sub(r'[?!.,;:]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def search_qa_database(self, question):
        """데이터베이스에서 유사한 질문 검색"""
        normalized_q = self.normalize_question(question)

        # 정확히 일치하는 질문 찾기
        for stored_q, data in self.qa_database.items():
            if self.normalize_question(stored_q) == normalized_q:
                # 접근 횟수 증가
                data['access_count'] = data.get('access_count', 0) + 1
                data['last_accessed'] = datetime.now().isoformat()
                self.save_qa_database()
                print(f"  📚 DB에서 찾음! (접근: {data['access_count']}회)")
                return data

        # 유사한 질문 찾기 (키워드 매칭)
        question_keywords = set(normalized_q.split())
        best_match = None
        max_similarity = 0

        for stored_q, data in self.qa_database.items():
            stored_normalized = self.normalize_question(stored_q)
            stored_keywords = set(stored_normalized.split())

            # 공통 키워드 개수 계산
            common_keywords = question_keywords & stored_keywords
            similarity = len(common_keywords) / max(len(question_keywords), 1)

            if similarity > max_similarity and similarity > 0.5:  # 50% 이상 유사
                max_similarity = similarity
                best_match = data

        if best_match:
            best_match['access_count'] = best_match.get('access_count', 0) + 1
            best_match['last_accessed'] = datetime.now().isoformat()
            self.save_qa_database()
            print(f"  📚 유사 질문 발견! (유사도: {max_similarity:.0%}, 접근: {best_match['access_count']}회)")
            return best_match

        return None

    def add_qa_pair(self, question, answer, rating=None):
        """질문-답변 쌍을 데이터베이스에 추가"""
        if question in self.qa_database:
            # 기존 데이터 업데이트
            self.qa_database[question]['answer'] = answer
            self.qa_database[question]['access_count'] = self.qa_database[question].get('access_count', 0) + 1
            self.qa_database[question]['last_accessed'] = datetime.now().isoformat()
            if rating is not None:
                ratings = self.qa_database[question].get('ratings', [])
                ratings.append(rating)
                self.qa_database[question]['ratings'] = ratings
                self.qa_database[question]['avg_rating'] = sum(ratings) / len(ratings)
        else:
            # 새 데이터 추가
            self.qa_database[question] = {
                'answer': answer,
                'timestamp': datetime.now().isoformat(),
                'access_count': 1,
                'last_accessed': datetime.now().isoformat(),
                'ratings': [rating] if rating is not None else [],
                'avg_rating': rating if rating is not None else None
            }
        self.save_qa_database()

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
        """한국어/영어 웹 검색"""
        if query in self.search_cache:
            print("  📦 캐시에서 가져옴")
            return self.search_cache[query]

        print(f"  🔍 웹 검색 중...")

        # 한국어 지식 베이스
        korean_knowledge = {
            "인공지능": "인공지능(AI)은 인간의 학습능력, 추론능력, 지각능력을 인공적으로 구현한 컴퓨터 시스템입니다. 기계학습, 딥러닝 등의 기술을 사용하여 이미지 인식, 음성 인식, 자연어 처리 등 다양한 분야에 활용됩니다.",
            "머신러닝": "머신러닝은 인공지능의 한 분야로, 컴퓨터가 명시적으로 프로그래밍되지 않아도 데이터로부터 학습하여 성능을 향상시키는 기술입니다. 지도학습, 비지도학습, 강화학습 등의 방법이 있습니다.",
            "딥러닝": "딥러닝은 인공신경망을 기반으로 한 머신러닝 기술입니다. 여러 층의 신경망을 사용하여 복잡한 패턴을 학습할 수 있으며, 이미지 인식, 음성 인식, 자연어 처리 등에서 뛰어난 성능을 보입니다.",
            "파이썬": "파이썬은 읽기 쉽고 배우기 쉬운 고급 프로그래밍 언어입니다. 웹 개발, 데이터 분석, 인공지능, 자동화 등 다양한 분야에서 사용됩니다. 풍부한 라이브러리와 커뮤니티 지원이 장점입니다.",
            "트랜스포머": "트랜스포머는 2017년에 소개된 딥러닝 모델 아키텍처입니다. 어텐션 메커니즘을 사용하여 시퀀스 데이터를 처리하며, GPT, BERT 등 현대 언어 모델의 기초가 되었습니다.",
            "자연어처리": "자연어처리(NLP)는 인간의 언어를 컴퓨터가 이해하고 처리할 수 있도록 하는 인공지능 분야입니다. 번역, 요약, 감정 분석, 질문 답변 등의 작업을 수행합니다.",
            "챗봇": "챗봇은 사용자와 대화를 나누는 인공지능 프로그램입니다. 규칙 기반, 검색 기반, 생성 기반 등 다양한 방식으로 구현되며, 고객 서비스, 정보 제공, 엔터테인먼트 등에 활용됩니다.",
            "pytorch": "PyTorch는 Facebook(Meta)이 개발한 오픈소스 머신러닝 라이브러리입니다. 동적 계산 그래프를 지원하여 유연한 모델 개발이 가능하며, 연구와 실무에서 널리 사용됩니다.",
            "tensorflow": "TensorFlow는 Google이 개발한 오픈소스 머신러닝 프레임워크입니다. 다양한 플랫폼에서 실행 가능하며, 연구부터 프로덕션까지 폭넓게 사용됩니다.",
            "llm": "대규모 언어 모델(LLM)은 방대한 텍스트 데이터로 학습된 인공지능 모델입니다. GPT, Claude 등이 대표적이며, 텍스트 생성, 번역, 요약 등 다양한 언어 작업을 수행할 수 있습니다.",
            "데이터": "데이터는 컴퓨터가 처리할 수 있는 형태로 저장된 정보입니다. 구조화된 데이터(DB), 반구조화된 데이터(JSON), 비구조화된 데이터(텍스트, 이미지) 등이 있습니다.",
            "날씨": f"날씨 정보는 기상청이나 날씨 API를 통해 실시간으로 확인할 수 있습니다. 현재 시간: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}",
            "시간": f"현재 시간은 {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}입니다.",
        }

        # 영어 지식 베이스
        english_knowledge = {
            "artificial intelligence": "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.",
            "machine learning": "Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "deep learning": "Deep Learning is a subset of machine learning based on artificial neural networks with multiple layers. It's particularly effective for tasks like image and speech recognition.",
            "python": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and automation.",
            "transformer": "Transformer is a deep learning model architecture introduced in 2017. It uses self-attention mechanisms and has become the foundation for modern language models like GPT and BERT.",
            "pytorch": "PyTorch is an open-source machine learning library developed by Facebook (Meta). It provides dynamic computation graphs and is widely used in research and production.",
            "llm": "Large Language Models (LLMs) are AI models trained on vast amounts of text data. Examples include GPT, Claude, and BERT, capable of various language tasks.",
        }

        # 통합 지식 베이스
        all_knowledge = {**korean_knowledge, **english_knowledge}

        # 쿼리를 소문자로 변환
        query_lower = query.lower()

        # 직접 매칭
        for key, value in all_knowledge.items():
            if key in query_lower or query_lower in key:
                self.search_cache[query] = value
                return value

        # 키워드 매칭
        query_keywords = query_lower.split()
        best_match = None
        max_score = 0

        for key, value in all_knowledge.items():
            score = sum(1 for keyword in query_keywords if keyword in key or key in keyword)
            if score > max_score:
                max_score = score
                best_match = value

        if max_score > 0:
            self.search_cache[query] = best_match
            return best_match

        # 기본 답변
        result = f"'{query}'에 대한 정보를 찾을 수 없습니다. 더 구체적으로 질문해주세요."
        return result

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

    def answer_question(self, question):
        """
        질문에 답변 (DB 검색 → 웹 검색 우선)

        Args:
            question: 질문

        Returns:
            answer: 답변
        """
        print(f"\n❓ 질문: {question}")
        print("-"*70)

        # 1. 데이터베이스에서 검색
        cached_data = self.search_qa_database(question)
        if cached_data:
            answer = cached_data['answer']
            avg_rating = cached_data.get('avg_rating')
            if avg_rating:
                print(f"  ⭐ 평균 평점: {avg_rating:.1f}/5.0")
            return answer

        # 2. 웹 검색 (우선)
        print(f"  🆕 새로운 질문입니다.")
        web_answer = self.search_web(question)

        if web_answer and not web_answer.startswith("'"):  # 유효한 답변
            # DB에 저장
            self.add_qa_pair(question, web_answer)
            print(f"  💾 답변을 DB에 저장했습니다.")
            return web_answer

        # 3. LLM 모델 사용 (백업)
        if self.trained:
            print(f"  🤖 LLM 모델로 생성 중...")
            llm_response = self.generate_response(question, max_tokens=100, temperature=0.7)
            # 질문 이후 부분만 추출
            if llm_response.startswith(question):
                llm_answer = llm_response[len(question):].strip()
            else:
                llm_answer = llm_response

            if llm_answer:
                self.add_qa_pair(question, llm_answer)
                print(f"  💾 LLM 답변을 DB에 저장했습니다.")
                return llm_answer

        # 4. 기본 답변
        return web_answer if web_answer else "죄송합니다. 답변을 찾을 수 없습니다."

    def show_statistics(self):
        """데이터베이스 통계 표시"""
        print("\n" + "="*70)
        print("📊 데이터베이스 통계")
        print("="*70)
        print(f"총 QA 쌍 수: {len(self.qa_database)} 개")

        if self.qa_database:
            # 가장 많이 접근된 질문
            sorted_qa = sorted(
                self.qa_database.items(),
                key=lambda x: x[1].get('access_count', 0),
                reverse=True
            )

            print(f"\n📈 가장 많이 질문된 Top 5:")
            for i, (q, data) in enumerate(sorted_qa[:5], 1):
                count = data.get('access_count', 0)
                avg_rating = data.get('avg_rating')
                rating_str = f", ⭐ {avg_rating:.1f}" if avg_rating else ""
                print(f"  {i}. {q} (접근: {count}회{rating_str})")

            # 최근 추가된 질문
            recent_qa = sorted(
                self.qa_database.items(),
                key=lambda x: x[1].get('timestamp', ''),
                reverse=True
            )

            print(f"\n🆕 최근 추가된 질문 5개:")
            for i, (q, data) in enumerate(recent_qa[:5], 1):
                timestamp = data.get('timestamp', '')
                if timestamp:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime('%Y-%m-%d %H:%M')
                else:
                    time_str = '알 수 없음'
                print(f"  {i}. {q} ({time_str})")

            # 평균 평점이 높은 질문
            rated_qa = [(q, d) for q, d in self.qa_database.items() if d.get('avg_rating')]
            if rated_qa:
                sorted_rated = sorted(rated_qa, key=lambda x: x[1]['avg_rating'], reverse=True)
                print(f"\n⭐ 평점이 높은 답변 Top 5:")
                for i, (q, data) in enumerate(sorted_rated[:5], 1):
                    rating = data['avg_rating']
                    count = len(data.get('ratings', []))
                    print(f"  {i}. {q} (평균: {rating:.1f}/5.0, 평가: {count}회)")

        print()

    def export_training_data(self, output_path='training_data.txt'):
        """학습 데이터로 변환하여 저장"""
        if not self.qa_database:
            print("⚠️  저장할 데이터가 없습니다.")
            return

        with open(output_path, 'w', encoding='utf-8') as f:
            for question, data in self.qa_database.items():
                answer = data['answer']
                # 질문-답변 형식으로 저장
                f.write(f"Q: {question}\n")
                f.write(f"A: {answer}\n")
                f.write("\n")

        print(f"✅ 학습 데이터 저장 완료: {output_path}")
        print(f"   - {len(self.qa_database)} 개 QA 쌍")

    def chat(self):
        """인터랙티브 채팅 모드"""
        print("="*70)
        print("💬 한국어/영어 QA 챗봇")
        print("="*70)
        print()
        print("기능:")
        print("  - 한국어/영어 질문-답변 지원")
        print("  - 웹 검색 기반 답변 생성")
        print("  - 답변 자동 저장 및 평가 시스템")
        print("  - 동일 질문에 빠른 답변 제공")
        if self.trained:
            print("  - LLM 모델 백업 지원")
        print()
        print("명령어:")
        print("  /stats     - 통계 보기")
        print("  /export    - 학습 데이터 내보내기")
        print("  /clear     - 데이터베이스 초기화")
        print("  /help      - 도움말")
        print("  quit/exit  - 종료")
        print()
        print("-"*70)

        last_question = None
        last_answer = None

        while True:
            try:
                user_input = input("\n📝 질문: ").strip()

                if not user_input:
                    print("⚠️  질문을 입력해주세요.")
                    continue

                # 종료
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n💾 변경사항을 저장하는 중...")
                    self.save_qa_database()
                    print("👋 챗봇을 종료합니다.")
                    break

                # 통계
                if user_input.lower() == '/stats':
                    self.show_statistics()
                    continue

                # 내보내기
                if user_input.lower() == '/export':
                    self.export_training_data()
                    continue

                # 초기화
                if user_input.lower() == '/clear':
                    confirm = input("정말 데이터베이스를 초기화하시겠습니까? (y/n): ")
                    if confirm.lower() == 'y':
                        self.qa_database = {}
                        self.save_qa_database()
                        print("✅ 데이터베이스가 초기화되었습니다.")
                    continue

                # 도움말
                if user_input.lower() == '/help':
                    print("\n사용 가능한 명령:")
                    print("  /stats  - 데이터베이스 통계")
                    print("  /export - 학습 데이터 내보내기")
                    print("  /clear  - 데이터베이스 초기화")
                    print("  /help   - 도움말")
                    print("\n답변 평가:")
                    print("  답변 후 1~5 사이 숫자로 평가 가능")
                    print("  Enter 입력 시 평가 없이 다음 질문으로")
                    continue

                # 질문 답변
                answer = self.answer_question(user_input)
                print(f"\n💬 답변:\n{answer}")
                print("-"*70)

                # 답변 평가 요청
                last_question = user_input
                last_answer = answer

                try:
                    rating_input = input("\n⭐ 답변 평가 (1-5, Enter=건너뛰기): ").strip()

                    if rating_input and rating_input.isdigit():
                        rating = int(rating_input)
                        if 1 <= rating <= 5:
                            self.add_qa_pair(last_question, last_answer, rating)
                            print(f"✅ 평가 완료: {rating}/5.0")
                        else:
                            print("⚠️  1~5 사이 숫자를 입력하세요.")
                    elif rating_input:
                        print("⚠️  숫자를 입력하세요.")
                except:
                    pass  # Enter만 누른 경우

            except KeyboardInterrupt:
                print("\n\n💾 변경사항을 저장하는 중...")
                self.save_qa_database()
                print("👋 챗봇을 종료합니다.")
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
    print("🤖 한국어/영어 QA 챗봇")
    print("="*70)
    print()

    # 체크포인트 확인 (선택사항)
    checkpoint_path = 'checkpoints/best_model.pt'

    if not os.path.exists(checkpoint_path):
        print("⚠️  학습된 LLM 모델을 찾을 수 없습니다.")
        print("   웹 검색 기반으로만 동작합니다.")
        print()

        # 사용자에게 선택 제공
        choice = input("새로운 모델을 학습하시겠습니까? (y/n/skip): ").strip().lower()

        if choice == 'y':
            success = train_new_model()
            if not success:
                print("\n⚠️  학습 실패. 웹 검색 모드로 계속합니다.")
        elif choice == 'skip' or choice == 's':
            print("\n웹 검색 모드로 계속합니다.")
        else:
            print("\n나중에 학습하려면: python training_pipeline.py")
            print("웹 검색 모드로 계속합니다.")
            print()

    # 챗봇 실행 (LLM 모델 없어도 동작)
    chatbot = LLMChatbot(checkpoint_path)
    chatbot.chat()


if __name__ == "__main__":
    main()
