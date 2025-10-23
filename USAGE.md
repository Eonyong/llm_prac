# SimpleLLM 사용 가이드

## 🚀 시작하기

### Step 1: 프로그램 실행

```bash
python main.py
```

### Step 2: 자동 처리

프로그램이 자동으로 처리합니다:

**시나리오 1: 처음 실행 (학습된 모델 없음)**
```
⚠️  학습된 모델을 찾을 수 없습니다.

새로운 모델을 학습하시겠습니까? (y/n): y

🚀 새로운 모델을 학습합니다...
[학습 진행...]
✅ 학습이 완료되었습니다!
```

**시나리오 2: 학습된 모델 있음**
```
✅ 모델 로드 완료!
💬 SimpleLLM 챗봇

📝 입력:
```

---

## 💬 챗봇 사용법

### 기본 사용

1. **텍스트 입력** - 모델이 이어서 생성합니다
```
📝 입력: Once upon a time
🤖 응답: Once upon a time, there was a small village...
```

2. **다양한 프롬프트 시도**
```
📝 입력: The meaning of life is
🤖 응답: The meaning of life is to find happiness and purpose...

📝 입력: In the year 2050,
🤖 응답: In the year 2050, technology will have advanced...
```

### 설정 명령어

#### 🌡️ Temperature (다양성 조절)

```bash
/temp <값>
```

- **낮음 (0.3~0.5)**: 보수적, 예측 가능한 출력
- **중간 (0.7~0.9)**: 균형잡힌 출력 (기본: 0.8)
- **높음 (1.0~2.0)**: 창의적, 다양한 출력

**예시:**
```
📝 입력: /temp 0.3
✅ Temperature 설정: 0.3

📝 입력: Hello
🤖 응답: Hello, how are you today? [보수적]

📝 입력: /temp 1.5
✅ Temperature 설정: 1.5

📝 입력: Hello
🤖 응답: Hello there! What an amazing day... [창의적]
```

#### 📝 Max Tokens (생성 길이)

```bash
/tokens <값>
```

- **짧게 (10~30)**: 짧은 응답
- **중간 (50~100)**: 일반적인 응답 (기본: 50)
- **길게 (100~200)**: 긴 응답

**예시:**
```
📝 입력: /tokens 10
✅ Max tokens 설정: 10

📝 입력: Tell me a story
🤖 응답: Once upon a time there... [짧음]

📝 입력: /tokens 150
✅ Max tokens 설정: 150

📝 입력: Tell me a story
🤖 응답: Once upon a time there was a brave knight who... [길게 생성]
```

#### 🎯 Top-k (샘플링 범위)

```bash
/topk <값>
```

- **작음 (10~30)**: 더 집중된 응답
- **중간 (40~60)**: 균형 (기본: 50)
- **큼 (70~100)**: 더 다양한 응답

**예시:**
```
📝 입력: /topk 10
✅ Top-k 설정: 10

📝 입력: /topk 100
✅ Top-k 설정: 100
```

#### ❓ 도움말

```bash
/help
```

사용 가능한 모든 명령어를 표시합니다.

---

## 🎯 실전 예제

### 예제 1: 이야기 생성

```
📝 입력: /tokens 100
✅ Max tokens 설정: 100

📝 입력: /temp 1.0
✅ Temperature 설정: 1.0

📝 입력: Once upon a time in a magical forest,
🤖 응답: Once upon a time in a magical forest, there lived a wise old owl...
```

### 예제 2: 짧고 정확한 응답

```
📝 입력: /tokens 20
✅ Max tokens 설정: 20

📝 입력: /temp 0.5
✅ Temperature 설정: 0.5

📝 입력: The capital of France is
🤖 응답: The capital of France is Paris.
```

### 예제 3: 창의적인 글쓰기

```
📝 입력: /temp 1.3
✅ Temperature 설정: 1.3

📝 입력: /tokens 150
✅ Max tokens 설정: 150

📝 입력: In a world where robots and humans live together,
🤖 응답: In a world where robots and humans live together, society has evolved...
```

---

## ⚙️ 고급 설정 조합

### 1. 보수적 모드 (정확성 우선)
```
/temp 0.3
/tokens 30
/topk 20
```
→ 짧고 예측 가능한 응답

### 2. 균형 모드 (기본)
```
/temp 0.8
/tokens 50
/topk 50
```
→ 적당히 창의적이고 적당한 길이

### 3. 창의적 모드 (다양성 우선)
```
/temp 1.5
/tokens 100
/topk 80
```
→ 길고 창의적인 응답

### 4. 짧은 대화 모드
```
/temp 0.7
/tokens 20
/topk 40
```
→ 대화형 짧은 응답

---

## 📊 출력 품질 향상 팁

### 1. 더 나은 프롬프트 작성

**❌ 나쁜 예:**
```
📝 입력: tell me
```

**✅ 좋은 예:**
```
📝 입력: Tell me a story about a brave knight who
```

### 2. 문맥 제공

**❌ 나쁜 예:**
```
📝 입력: It was
```

**✅ 좋은 예:**
```
📝 입력: The ancient castle stood tall on the hill. It was
```

### 3. 명확한 시작

**❌ 나쁜 예:**
```
📝 입력: weather
```

**✅ 좋은 예:**
```
📝 입력: The weather today is
```

---

## 🔧 문제 해결

### Q: 응답이 이상해요 (무작위 단어들)

**원인**: 모델이 충분히 학습되지 않았거나 데이터가 부족합니다.

**해결책:**
1. 더 많은 데이터로 다시 학습
2. Temperature를 낮추세요 (`/temp 0.5`)
3. Top-k를 줄이세요 (`/topk 20`)

### Q: 응답이 너무 짧아요

**해결책:**
```
/tokens 100
/temp 0.9
```

### Q: 같은 말만 반복해요

**해결책:**
1. Temperature를 높이세요 (`/temp 1.2`)
2. Top-k를 높이세요 (`/topk 80`)
3. 더 다양한 데이터로 재학습

### Q: 모델이 로드되지 않아요

**해결책:**
```bash
# 먼저 학습 실행
python training_pipeline.py

# 그 다음 main.py 실행
python main.py
```

---

## 🎓 학습 향상을 위한 팁

### 더 나은 모델을 위해:

1. **더 많은 데이터**
   - 현재: 20KB (작은 데이터)
   - 권장: 1MB 이상
   - 이상적: 10MB+

2. **더 긴 학습**
   - `training_pipeline.py`에서 `num_epochs` 증가
   ```python
   config['num_epochs'] = 20  # 기본: 10
   ```

3. **더 큰 모델**
   ```python
   config['d_embed'] = 512     # 기본: 256
   config['num_layers'] = 6    # 기본: 4
   ```

---

## 📝 사용 예시 모음

### 이야기 생성
```
Once upon a time in a far away land,
There was a young girl who dreamed of
In a small village by the sea,
```

### 문장 완성
```
The most important thing in life is
If I could travel anywhere, I would
The future of technology will be
```

### 설명 생성
```
Machine learning is a technology that
The reason why the sky is blue is because
Climate change is important because
```

---

## 🚪 종료

프로그램을 종료하려면:
```
quit
exit
q
```

또는 `Ctrl+C`를 누르세요.

---

**즐거운 AI 텍스트 생성 되세요! 🎉**
