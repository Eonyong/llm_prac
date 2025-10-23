# SimpleLLM - 간단한 언어 모델 프로젝트

PyTorch를 사용한 GPT 스타일의 언어 모델 구현 및 학습

---

## 📁 프로젝트 구조

```
LLM/
├── layer_normalization.py      # 핵심 모델 코드 (SimpleLLM, Transformer 블록)
├── training_pipeline.py         # 학습 파이프라인 (데이터로더, 학습 루프, 옵티마이저)
├── run_simple_llm.py           # 텍스트 생성 및 분석 인터페이스
├── training_guide.md           # 학습 가이드 문서
├── the-verdict.txt             # 학습 데이터
├── checkpoints/                # 학습된 모델 저장 위치
└── examples/                   # 예제 및 데모 코드
    ├── chapter3.py
    ├── main.py
    ├── simple_attention.py
    ├── multihead_attention.py
    ├── interactive_attention.py
    ├── run_attention.py
    └── ...
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 활성화
source .venv/bin/activate

# 필요한 패키지가 설치되어 있는지 확인
pip list | grep -E "torch|tiktoken|tqdm"
```

### 2. 메인 프로그램 실행 (추천)

```bash
# main.py 실행 - 자동으로 학습 또는 챗봇 실행
python main.py
```

프로그램이 자동으로:
- 학습된 모델이 없으면 → 학습 진행
- 학습된 모델이 있으면 → 챗봇 시작

### 3. 또는 수동 학습

```bash
# 기본 설정으로 학습 시작
python training_pipeline.py
```

**학습 설정 수정**: `training_pipeline.py`의 `config` 딕셔너리 수정

```python
config = {
    'd_embed': 256,        # 임베딩 차원
    'num_heads': 4,        # 헤드 개수
    'num_layers': 4,       # 레이어 개수
    'batch_size': 8,       # 배치 크기
    'num_epochs': 10,      # 에포크 수
    'learning_rate': 3e-4, # 학습률
}
```

### 4. 챗봇 사용 (main.py)

```bash
# 인터랙티브 챗봇 시작
python main.py
```

**챗봇 명령어:**
- 일반 텍스트 입력 → 이어서 생성
- `/temp 0.8` → Temperature 설정 (다양성 조절)
- `/tokens 100` → 생성할 토큰 수 설정
- `/topk 30` → Top-k 샘플링 설정
- `/help` → 도움말 표시
- `quit` 또는 `exit` → 종료

**예시:**
```
📝 입력: Once upon a time
🤖 응답: Once upon a time, there was a small village...

📝 입력: /temp 1.2
✅ Temperature 설정: 1.2

📝 입력: The weather is
🤖 응답: The weather is beautiful today, with clear skies...
```

### 5. 추가 도구 (run_simple_llm.py)

```bash
# 다음 토큰 예측 분석
python run_simple_llm.py analyze "Hello, how are you"

# 텍스트 생성
python run_simple_llm.py generate "Once upon a time"

# 인터랙티브 모드
python run_simple_llm.py
```

---

## 📚 핵심 파일 설명

### `layer_normalization.py`
**모델 아키텍처 코드**

포함된 클래스:
- `LayerNorm`: 층 정규화
- `FeedForward`: 피드포워드 네트워크
- `MultiHeadAttention`: 멀티-헤드 어텐션
- `TransformerBlock`: Transformer 블록 (Pre-LN 스타일)
- `SimpleLLM`: 완전한 언어 모델 (GPT 스타일)

### `training_pipeline.py`
**학습 파이프라인**

포함된 클래스:
- `TextDataset`: 학습 데이터셋
- `WarmupCosineScheduler`: 학습률 스케줄러
- `Trainer`: 학습 루프 및 검증

주요 기능:
- Cross Entropy Loss
- AdamW 옵티마이저
- Gradient Clipping
- 체크포인트 저장/로드
- Perplexity & Accuracy 계산

### `run_simple_llm.py`
**텍스트 생성 및 분석**

주요 기능:
- 다음 토큰 예측 (Top-k)
- 텍스트 자동 생성
- 인터랙티브 모드

---

## 🎯 모델 구조

```
SimpleLLM Architecture:

Input Tokens
    ↓
Token Embedding + Positional Embedding
    ↓
Dropout
    ↓
┌─────────────────────────────────┐
│  Transformer Block × N          │
│  ┌───────────────────────────┐  │
│  │ LayerNorm                 │  │
│  │    ↓                      │  │
│  │ Multi-Head Attention      │  │
│  │    ↓                      │  │
│  │ Residual Connection       │  │
│  │    ↓                      │  │
│  │ LayerNorm                 │  │
│  │    ↓                      │  │
│  │ Feed Forward Network      │  │
│  │    ↓                      │  │
│  │ Residual Connection       │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    ↓
Final LayerNorm
    ↓
Output Projection (Logits)
```

---

## 📊 학습 예제

### 기본 학습 결과 (the-verdict.txt)

```
데이터셋: 20,479 문자, 5,145 토큰
학습 시간: ~30초 (Apple Silicon GPU)

Epoch 1:  PPL: 41,729 → Acc: 0.59%
Epoch 5:  PPL: 5,740  → Acc: 3.61%
Epoch 10: PPL: 2,874  → Acc: 3.61%
```

---

## 🔧 학습 팁

### 1. GPU 메모리 부족 시
```python
config = {
    'batch_size': 4,      # 배치 크기 줄이기
    'max_seq_len': 128,   # 시퀀스 길이 줄이기
    'd_embed': 128,       # 모델 크기 줄이기
}
```

### 2. 학습 속도 향상
```python
# 더 큰 배치 크기 사용 (메모리 허용 시)
config['batch_size'] = 16

# 더 짧은 시퀀스 길이
config['max_seq_len'] = 128
```

### 3. 성능 향상
```python
# 더 큰 모델
config = {
    'd_embed': 512,
    'num_layers': 6,
    'num_heads': 8,
}

# 더 많은 에포크
config['num_epochs'] = 20
```

---

## 📖 추가 자료

- `training_guide.md`: 상세한 학습 가이드
- `examples/`: 다양한 예제 코드
  - `simple_attention.py`: 기본 어텐션 메커니즘
  - `multihead_attention.py`: 멀티-헤드 어텐션 데모
  - `interactive_attention.py`: 인터랙티브 어텐션 시각화

---

## 🎓 학습 내용

이 프로젝트를 통해 학습할 수 있는 것들:

1. **Transformer 아키텍처**
   - Multi-Head Attention
   - Layer Normalization
   - Residual Connections
   - Feed Forward Networks

2. **학습 기법**
   - AdamW 옵티마이저
   - Warmup + Cosine Decay 스케줄러
   - Gradient Clipping
   - Dropout (정규화)

3. **LLM 기초**
   - Token Embedding
   - Positional Embedding
   - Causal Masking
   - Next Token Prediction

---

## 🔍 문제 해결

### Q: 학습이 너무 느려요
```
A: GPU를 사용하고 있는지 확인하세요.
   출력에 "✅ GPU 사용 (MPS)" 또는 "✅ GPU 사용 (CUDA)"가 표시되어야 합니다.
```

### Q: 손실이 감소하지 않아요
```
A: 1. 학습률을 낮추세요 (3e-4 → 1e-4)
   2. 데이터가 충분한지 확인하세요
   3. 그래디언트 클리핑 값 확인 (grad_clip = 1.0)
```

### Q: 모델이 같은 단어만 반복해요
```
A: 1. 더 많은 데이터로 더 오래 학습하세요
   2. Temperature를 조정하세요 (텍스트 생성 시)
   3. Top-k 또는 Top-p 샘플링 사용
```

---

## 📝 TODO (향후 개선 사항)

- [ ] Gradient Accumulation 구현
- [ ] Mixed Precision Training 추가
- [ ] TensorBoard 로깅
- [ ] Early Stopping
- [ ] Top-k, Top-p 샘플링
- [ ] Beam Search
- [ ] Model Export (ONNX)
- [ ] 더 큰 데이터셋 예제

---

## 📜 라이센스

MIT License

---

## 🤝 기여

이슈와 Pull Request를 환영합니다!

---

## 📧 연락처

프로젝트 문의: [이메일 주소]

---

**Happy Learning! 🚀**
