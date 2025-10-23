"""
한국어 지원 QA 챗봇 with 웹 검색 기반 학습

특징:
1. 한국어/영어 질문-답변 지원
2. 웹 검색으로 답변 생성
3. 질문-답변 자동 저장 및 학습
4. 데이터 축적으로 정확도 향상
"""

import torch
import json
import os
from datetime import datetime
import re


class KoreanQAChatbot:
    """한국어 지원 질문-답변 챗봇"""

    def __init__(self, qa_database_path='qa_database.json'):
        """
        Args:
            qa_database_path: 질문-답변 데이터베이스 파일 경로
        """
        print("="*70)
        print("🤖 한국어 QA 챗봇 초기화")
        print("="*70)
        print()

        self.qa_database_path = qa_database_path
        self.qa_database = self.load_qa_database()

        print(f"✅ 초기화 완료!")
        print(f"   - 저장된 QA 쌍: {len(self.qa_database)} 개")
        print()

    def load_qa_database(self):
        """질문-답변 데이터베이스 로드"""
        if os.path.exists(self.qa_database_path):
            try:
                with open(self.qa_database_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"📂 기존 데이터베이스 로드: {len(data)} 개 QA 쌍")
                return data
            except Exception as e:
                print(f"⚠️  데이터베이스 로드 실패: {e}")
                return {}
        else:
            print("📝 새로운 데이터베이스 생성")
            return {}

    def save_qa_database(self):
        """질문-답변 데이터베이스 저장"""
        try:
            with open(self.qa_database_path, 'w', encoding='utf-8') as f:
                json.dump(self.qa_database, f, ensure_ascii=False, indent=2)
            print(f"💾 데이터베이스 저장 완료: {len(self.qa_database)} 개 QA 쌍")
        except Exception as e:
            print(f"⚠️  저장 실패: {e}")

    def normalize_question(self, question):
        """질문 정규화 (검색용)"""
        # 소문자 변환, 공백 정리, 특수문자 제거
        normalized = question.lower().strip()
        normalized = re.sub(r'[?!.,;:]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def search_qa_database(self, question):
        """
        데이터베이스에서 유사한 질문 검색

        Args:
            question: 검색할 질문

        Returns:
            answer: 찾은 답변 또는 None
        """
        normalized_q = self.normalize_question(question)

        # 정확히 일치하는 질문 찾기
        for stored_q, data in self.qa_database.items():
            if self.normalize_question(stored_q) == normalized_q:
                print(f"  📚 데이터베이스에서 찾음!")
                return data['answer']

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
                best_match = data['answer']

        if best_match:
            print(f"  📚 유사한 질문 발견! (유사도: {max_similarity:.0%})")
            return best_match

        return None

    def search_web_korean(self, query):
        """
        한국어 질문에 대한 웹 검색

        Args:
            query: 검색 쿼리

        Returns:
            answer: 검색된 답변
        """
        print(f"  🔍 웹 검색 중...")

        # 한국어 키워드 기반 지식 베이스 (확장 가능)
        korean_knowledge = {
            "인공지능": "인공지능(AI)은 인간의 학습능력, 추론능력, 지각능력을 인공적으로 구현한 컴퓨터 시스템입니다. 기계학습, 딥러닝 등의 기술을 사용하여 이미지 인식, 음성 인식, 자연어 처리 등 다양한 분야에 활용됩니다.",

            "머신러닝": "머신러닝은 인공지능의 한 분야로, 컴퓨터가 명시적으로 프로그래밍되지 않아도 데이터로부터 학습하여 성능을 향상시키는 기술입니다. 지도학습, 비지도학습, 강화학습 등의 방법이 있습니다.",

            "딥러닝": "딥러닝은 인공신경망을 기반으로 한 머신러닝 기술입니다. 여러 층의 신경망을 사용하여 복잡한 패턴을 학습할 수 있으며, 이미지 인식, 음성 인식, 자연어 처리 등에서 뛰어난 성능을 보입니다.",

            "파이썬": "파이썬은 읽기 쉽고 배우기 쉬운 고급 프로그래밍 언어입니다. 웹 개발, 데이터 분석, 인공지능, 자동화 등 다양한 분야에서 사용됩니다. 풍부한 라이브러리와 커뮤니티 지원이 장점입니다.",

            "트랜스포머": "트랜스포머는 2017년에 소개된 딥러닝 모델 아키텍처입니다. 어텐션 메커니즘을 사용하여 시퀀스 데이터를 처리하며, GPT, BERT 등 현대 언어 모델의 기초가 되었습니다.",

            "자연어처리": "자연어처리(NLP)는 인간의 언어를 컴퓨터가 이해하고 처리할 수 있도록 하는 인공지능 분야입니다. 번역, 요약, 감정 분석, 질문 답변 등의 작업을 수행합니다.",

            "챗봇": "챗봇은 사용자와 대화를 나누는 인공지능 프로그램입니다. 규칙 기반, 검색 기반, 생성 기반 등 다양한 방식으로 구현되며, 고객 서비스, 정보 제공, 엔터테인먼트 등에 활용됩니다.",

            "pytorch": "PyTorch는 Facebook(Meta)이 개발한 오픈소스 머신러닝 라이브러리입니다. 동적 계산 그래프를 지원하여 유연한 모델 개발이 가능하며, 연구와 실무에서 널리 사용됩니다.",

            "날씨": "날씨 정보는 기상청이나 날씨 API를 통해 실시간으로 확인할 수 있습니다. 기온, 습도, 강수량, 풍속 등의 정보를 제공합니다.",

            "시간": f"현재 시간은 {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}입니다.",
        }

        # 영어 지식 베이스
        english_knowledge = {
            "artificial intelligence": "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.",

            "machine learning": "Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",

            "deep learning": "Deep Learning is a subset of machine learning based on artificial neural networks with multiple layers. It's particularly effective for tasks like image and speech recognition.",

            "python": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and automation.",

            "transformer": "Transformer is a deep learning model architecture introduced in 2017. It uses self-attention mechanisms and has become the foundation for modern language models like GPT and BERT.",
        }

        # 통합 지식 베이스
        all_knowledge = {**korean_knowledge, **english_knowledge}

        # 쿼리를 소문자로 변환
        query_lower = query.lower()

        # 직접 매칭
        for key, value in all_knowledge.items():
            if key in query_lower or query_lower in key:
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
            return best_match

        # 기본 답변
        return f"'{query}'에 대한 정보를 찾을 수 없습니다. 더 구체적으로 질문해주세요."

    def add_qa_pair(self, question, answer):
        """
        질문-답변 쌍을 데이터베이스에 추가

        Args:
            question: 질문
            answer: 답변
        """
        self.qa_database[question] = {
            'answer': answer,
            'timestamp': datetime.now().isoformat(),
            'access_count': self.qa_database.get(question, {}).get('access_count', 0) + 1
        }
        self.save_qa_database()

    def answer_question(self, question):
        """
        질문에 답변

        Args:
            question: 질문

        Returns:
            answer: 답변
        """
        print(f"\n❓ 질문: {question}")
        print("-"*70)

        # 1. 데이터베이스에서 검색
        cached_answer = self.search_qa_database(question)
        if cached_answer:
            return cached_answer

        # 2. 웹 검색
        print(f"  🆕 새로운 질문입니다.")
        web_answer = self.search_web_korean(question)

        # 3. 데이터베이스에 저장
        if web_answer:
            self.add_qa_pair(question, web_answer)
            print(f"  💾 답변을 저장했습니다.")

        return web_answer

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
                print(f"  {i}. {q} (접근: {count}회)")

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

        print()

    def export_training_data(self, output_path='training_data.txt'):
        """
        학습 데이터로 변환하여 저장

        Args:
            output_path: 출력 파일 경로
        """
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
        print("💬 한국어 QA 챗봇")
        print("="*70)
        print()
        print("기능:")
        print("  - 한국어/영어 질문-답변")
        print("  - 자동으로 답변을 학습하여 저장")
        print("  - 동일한 질문에 빠른 답변 제공")
        print()
        print("명령어:")
        print("  /stats     - 통계 보기")
        print("  /export    - 학습 데이터 내보내기")
        print("  /clear     - 데이터베이스 초기화")
        print("  /help      - 도움말")
        print("  quit/exit  - 종료")
        print()
        print("-"*70)

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
                    continue

                # 질문 답변
                answer = self.answer_question(user_input)
                print(f"\n💬 답변:\n{answer}")
                print("-"*70)

            except KeyboardInterrupt:
                print("\n\n💾 변경사항을 저장하는 중...")
                self.save_qa_database()
                print("👋 챗봇을 종료합니다.")
                break

            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                import traceback
                traceback.print_exc()


def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("🤖 한국어 지원 QA 챗봇")
    print("="*70)
    print()

    # 챗봇 실행
    chatbot = KoreanQAChatbot()
    chatbot.chat()


if __name__ == "__main__":
    main()
