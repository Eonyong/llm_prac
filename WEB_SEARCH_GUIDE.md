# 웹 검색 기능 가이드

## 🌐 개요

SimpleLLM에 웹 검색 기능이 추가되어, 학습 데이터에 없는 질문도 답변할 수 있습니다.

---

## 🚀 사용 방법

### 방법 1: Web-Enhanced 챗봇 (권장)

```bash
python web_enhanced_chatbot.py
```

이 챗봇은 자동으로:
1. ✅ 질문 감지
2. ✅ 모델로 먼저 답변 시도
3. ✅ 답변이 부족하면 웹 검색
4. ✅ 검색 결과 요약 및 저장

### 방법 2: 기본 main.py (웹 검색 통합 중)

```bash
python main.py
```

---

## 💬 사용 예시

### 일반적인 질문

```
📝 입력: What is artificial intelligence?
🤔 질문으로 인식했습니다.
🤖 모델이 답변을 생성 중...

🤖 모델 답변:
[모델의 답변이 만족스럽지 않으면...]

이 답변이 만족스러우신가요? (y/n): n

더 나은 답변을 위해 웹을 검색합니다...
🔍 웹 검색 중: 'What is artificial intelligence?'
📚 Wikipedia 검색...
✅ 검색 완료!

🌐 웹 검색 답변:
Artificial intelligence (AI) is the simulation of human intelligence
processes by machines, especially computer systems...
```

### 직접 웹 검색

```
📝 입력: /search What is machine learning?
🔍 웹 검색 중: 'What is machine learning?'
✅ 검색 완료! (1250 문자)

🌐 웹 검색 결과:
Machine learning is a subset of artificial intelligence that enables
systems to learn and improve from experience...
```

### 저장된 지식 확인

```
📝 입력: /knowledge

📚 저장된 지식:
1. What is artificial intelligence?
   시간: 2025-10-23T22:00:00
   내용: Artificial intelligence (AI) is the simulation...

2. What is machine learning?
   시간: 2025-10-23T22:01:00
   내용: Machine learning is a subset of AI...
```

---

## 🎯 작동 원리

### 1. 질문 감지

다음 패턴을 감지합니다:
- 물음표로 끝나는 문장 (`?`)
- 의문사로 시작 (`what, where, when, who, why, how`)
- 한글 질문 (`무엇, 어디, 언제, 인가요`)

### 2. 응답 유효성 검사

생성된 응답이 다음 조건을 만족하는지 확인:
- ❌ 너무 짧음 (10자 미만)
- ❌ 반복이 너무 많음 (70% 이상 반복)
- ❌ 원본과 동일

### 3. 웹 검색 (필요시)

```
모델 답변 생성
    ↓
답변 유효성 검사
    ↓
    ├─ 유효함 → 사용자에게 확인
    │              ↓
    │          만족 → 완료
    │              ↓
    │       불만족 → 웹 검색
    │
    └─ 무효함 → 웹 검색
                  ↓
            결과 요약 및 저장
```

### 4. 검색 결과 처리

- Wikipedia에서 정보 가져오기
- 관련 문장 추출
- 요약 생성 (2-3문장)
- 캐시에 저장 (중복 검색 방지)

---

## 🛠️ 주요 기능

### ✅ 자동 질문 감지
- 질문 패턴 인식
- 영어/한글 지원

### ✅ 하이브리드 응답
- 먼저 학습된 모델 시도
- 필요시 웹 검색으로 보강

### ✅ 검색 결과 캐싱
- 동일 질문 재검색 방지
- 빠른 응답

### ✅ 지식 베이스 구축
- 검색 결과 자동 저장
- 시간 기록

---

## 📝 명령어

### 웹 검색 관련
```bash
/search <질문>    # 웹에서 직접 검색
/knowledge        # 저장된 지식 확인
/help             # 도움말 표시
```

### 일반 명령어
```bash
/temp 0.8         # Temperature 설정
/tokens 100       # 생성 토큰 수
/topk 50          # Top-k 샘플링
quit/exit         # 종료
```

---

## 🔍 예제 시나리오

### 시나리오 1: 학습 데이터에 없는 최신 정보

```
📝 입력: What happened in 2024?

🤖 모델 답변: [학습 데이터가 2024년 이전이므로 부정확]

이 답변이 만족스러우신가요? (y/n): n

🔍 웹 검색 중...
🌐 웹 검색 답변: [2024년 최신 정보]
```

### 시나리오 2: 전문 지식

```
📝 입력: What is quantum computing?

🤖 모델 답변: [일반적인 답변]

이 답변이 만족스러우신가요? (y/n): n

🔍 웹 검색 중...
🌐 웹 검색 답변: [Wikipedia의 상세한 설명]
```

### 시나리오 3: 빠른 검색

```
📝 입력: /search What is the capital of France?

🔍 웹 검색 중...
✅ 검색 완료!

🌐 웹 검색 결과:
Paris is the capital and most populous city of France...
```

---

## ⚙️ 설정

### `web_enhanced_chatbot.py` 설정

```python
# 검색 결과 최대 길이
result_text[:2000]  # 2000자로 제한

# 요약 문장 수
relevant_sentences[:3]  # 최대 3문장

# 검색할 문장 수
for sentence in sentences[:10]  # 첫 10개 문장 검색
```

---

## 🚧 제한사항

### 현재 버전
- Wikipedia 검색만 지원
- 영어 콘텐츠 위주
- 간단한 요약만 제공

### 향후 계획
- [ ] 다양한 검색 엔진 지원
- [ ] 더 정교한 요약 알고리즘
- [ ] 다국어 지원 강화
- [ ] 검색 결과 검증
- [ ] 소스 표시

---

## 💡 팁

### 1. 명확한 질문 작성
**❌ 나쁜 예:**
```
tell me about AI
```

**✅ 좋은 예:**
```
What is artificial intelligence and how does it work?
```

### 2. 답변 만족도 확인
- 모델 답변이 정확하면 `y` 입력
- 더 나은 답변 원하면 `n` 입력 → 웹 검색

### 3. 직접 검색 활용
```
/search [구체적인 질문]
```

### 4. 지식 베이스 활용
```
/knowledge  # 이전 검색 결과 확인
```

---

## 🔧 문제 해결

### Q: 웹 검색이 작동하지 않아요
```
A: 인터넷 연결을 확인하세요.
   Wikipedia 접근이 차단되어 있지 않은지 확인하세요.
```

### Q: 검색 결과가 이상해요
```
A: 질문을 더 명확하게 작성하세요.
   영어로 질문하면 더 정확한 결과를 얻을 수 있습니다.
```

### Q: 같은 질문을 다시 하면 어떻게 되나요?
```
A: 캐시에서 이전 결과를 가져옵니다 (빠름).
```

---

## 📚 기술 스택

- **모델**: SimpleLLM (GPT 스타일)
- **검색**: Wikipedia API
- **토크나이저**: tiktoken (GPT-2)
- **프레임워크**: PyTorch

---

**Happy Learning with Web-Enhanced AI! 🌐🤖**
