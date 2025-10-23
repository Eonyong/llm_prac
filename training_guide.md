# LLM 학습 가이드

## 📋 학습에 필요한 구성 요소

### 1️⃣ **손실 함수 (Loss Function)**
```python
loss = CrossEntropyLoss(logits, targets)
```
- **역할**: 모델 예측과 정답 사이의 차이 측정
- **Cross Entropy Loss**: 다중 클래스 분류에 사용
- **낮을수록 좋음**: 모델이 정답을 잘 예측함

### 2️⃣ **옵티마이저 (Optimizer)**
```python
optimizer = AdamW(
    model.parameters(),
    lr=3e-4,              # 학습률
    weight_decay=0.1,     # L2 정규화
    betas=(0.9, 0.95)     # 모멘텀 파라미터
)
```
- **역할**: 손실을 줄이는 방향으로 가중치 업데이트
- **AdamW**: Adam + Weight Decay (LLM에서 표준)
- **학습률 (Learning Rate)**: 가장 중요한 하이퍼파라미터

### 3️⃣ **학습률 스케줄러 (Learning Rate Scheduler)**
```
Warmup (0 → max_lr) → Cosine Decay (max_lr → min_lr)
```
- **Warmup**: 초반에 학습률을 천천히 증가 → 학습 안정화
- **Cosine Decay**: 학습률을 코사인 함수로 감소 → 부드러운 수렴

### 4️⃣ **데이터로더 (DataLoader)**
```python
DataLoader(dataset, batch_size=8, shuffle=True)
```
- **역할**: 배치 단위로 데이터 제공
- **Shuffle**: 학습 데이터 순서 섞음 → 일반화 성능 향상
- **Batch Size**: 한 번에 처리하는 샘플 수

### 5️⃣ **학습 루프 (Training Loop)**
```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # 1. 순전파
        logits = model(inputs)

        # 2. 손실 계산
        loss = compute_loss(logits, targets)

        # 3. 역전파
        loss.backward()

        # 4. 가중치 업데이트
        optimizer.step()
        optimizer.zero_grad()
```

### 6️⃣ **검증 (Validation)**
```python
model.eval()
with torch.no_grad():
    val_loss = evaluate(model, val_loader)
```
- **역할**: 과적합 확인
- **검증 손실이 증가**하면 → 과적합 발생

### 7️⃣ **체크포인트 (Checkpointing)**
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'best_val_loss': best_val_loss
}, 'checkpoint.pt')
```
- **역할**: 학습 중간 결과 저장
- **Best Model**: 검증 손실이 가장 낮은 모델 저장

### 8️⃣ **평가 지표 (Metrics)**

#### **Perplexity (혼란도)**
```python
perplexity = exp(loss)
```
- **의미**: 모델이 다음 토큰을 예측할 때의 불확실성
- **낮을수록 좋음**
- **예시**:
  - Perplexity = 10 → 평균 10개 후보 중 선택
  - Perplexity = 100 → 평균 100개 후보 중 선택

#### **Accuracy (정확도)**
```python
accuracy = (predictions == targets).mean()
```
- **의미**: 다음 토큰을 정확히 맞춘 비율
- **높을수록 좋음** (0~1)

---

## 🔧 학습 실행 방법

### **기본 학습**
```bash
python training_pipeline.py
```

### **학습 설정 수정**
`training_pipeline.py`의 `config` 딕셔너리 수정:

```python
config = {
    # 모델 크기
    'd_embed': 256,        # 임베딩 차원 (클수록 성능↑, 느림↑)
    'num_heads': 4,        # 헤드 개수
    'num_layers': 4,       # 레이어 개수 (깊을수록 성능↑)

    # 학습 파라미터
    'batch_size': 8,       # 배치 크기 (클수록 안정적, 메모리↑)
    'num_epochs': 10,      # 에포크 수
    'learning_rate': 3e-4, # 학습률 (가장 중요!)

    # 정규화
    'dropout': 0.1,        # 드롭아웃 (과적합 방지)
    'weight_decay': 0.1,   # L2 정규화
}
```

---

## 📊 학습 과정 모니터링

### **학습 중 출력 예시**
```
Epoch 1/10: 100%|██████████| 150/150 [02:30<00:00,  1.00it/s, loss=5.2341, acc=0.1523, lr=1.5e-04]

Epoch 1/10
  Train Loss: 5.2341, PPL: 187.23, Acc: 0.1523
  Val   Loss: 5.1234, PPL: 168.45, Acc: 0.1678
  LR: 1.5e-04
  ✅ 최고 모델 저장! (Val Loss: 5.1234)
```

### **주요 지표 해석**

| 지표 | 의미 | 목표 |
|------|------|------|
| **Train Loss** | 학습 손실 | 감소 |
| **Val Loss** | 검증 손실 | 감소 (Train과 비슷) |
| **PPL (Perplexity)** | 혼란도 | 낮을수록 좋음 |
| **Acc (Accuracy)** | 정확도 | 높을수록 좋음 |
| **LR** | 현재 학습률 | Warmup 후 감소 |

### **과적합 징후**
```
Epoch 5:
  Train Loss: 2.1  ← 계속 감소
  Val Loss: 3.5    ← 증가하기 시작!
```
→ **해결책**: Dropout 증가, Weight Decay 증가, 조기 종료

---

## 🎯 하이퍼파라미터 튜닝 가이드

### **1. 학습률 (Learning Rate)** ⭐⭐⭐⭐⭐
- **가장 중요!**
- **너무 크면**: 발산 (loss = NaN)
- **너무 작으면**: 학습이 느림
- **권장 범위**: 1e-4 ~ 1e-3
- **찾는 방법**: Learning Rate Finder 사용

### **2. 배치 크기 (Batch Size)** ⭐⭐⭐⭐
- **큰 배치**: 안정적, 빠름, 메모리 많이 사용
- **작은 배치**: 불안정, 느림, 메모리 적게 사용
- **권장**: GPU 메모리가 허용하는 최대 크기
- **일반적**: 8, 16, 32, 64

### **3. 모델 크기** ⭐⭐⭐⭐
- **d_embed**: 128(작음) → 256(중간) → 512(큼) → 768(매우 큼)
- **num_layers**: 4(작음) → 6(중간) → 12(큼)
- **원칙**: 데이터가 많을수록 큰 모델

### **4. Dropout** ⭐⭐⭐
- **과적합 방지**
- **일반적**: 0.1
- **데이터 적을 때**: 0.2~0.3
- **데이터 많을 때**: 0.0~0.1

### **5. Weight Decay** ⭐⭐⭐
- **L2 정규화**
- **일반적**: 0.1
- **과적합 심할 때**: 0.2~0.5

---

## 💡 학습 팁

### **1. 작게 시작하기**
```python
# 빠른 실험용 설정
config = {
    'd_embed': 128,
    'num_layers': 2,
    'batch_size': 4,
    'num_epochs': 2
}
```
→ 코드가 작동하는지 빠르게 확인

### **2. 학습률 찾기**
```python
# 여러 학습률 시도
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
for lr in learning_rates:
    # 몇 에포크만 학습
    # 손실이 안정적으로 감소하는 lr 선택
```

### **3. 과적합 모니터링**
```python
if val_loss > best_val_loss + 0.5:
    print("⚠️  과적합 발생!")
    # Dropout 증가 또는 조기 종료
```

### **4. 그래디언트 클리핑**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
→ 그래디언트 폭발 방지 (필수!)

### **5. Mixed Precision Training**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    logits = model(inputs)
    loss = compute_loss(logits, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
→ GPU 메모리 절약 + 속도 향상

---

## 📈 학습 예상 결과

### **작은 데이터셋 (20KB, the-verdict.txt)**
```
초기 (Epoch 1):
  PPL: ~180, Acc: ~15%

중간 (Epoch 5):
  PPL: ~80, Acc: ~30%

최종 (Epoch 10):
  PPL: ~50, Acc: ~40%
```

### **큰 데이터셋 (수 GB)**
```
초기: PPL: ~100
중간: PPL: ~30
최종: PPL: ~10-20 (GPT-2 수준)
```

---

## 🚀 다음 단계

학습 후 할 수 있는 것들:

1. **텍스트 생성**
```python
from training_pipeline import Trainer
trainer.load_checkpoint('checkpoints/best_model.pt')
generated = generate_text(model, "Once upon a time", max_tokens=100)
```

2. **Fine-tuning**
- 특정 도메인 데이터로 추가 학습
- 작은 학습률 (1e-5) 사용

3. **모델 평가**
- 다양한 프롬프트로 생성 품질 테스트
- Perplexity로 정량 평가

4. **배포**
- ONNX 변환
- 양자화 (Quantization)
- API 서버 구축

---

## 📚 참고 자료

- **논문**: "Attention Is All You Need" (Transformer 원본)
- **GPT-2**: "Language Models are Unsupervised Multitask Learners"
- **GPT-3**: "Language Models are Few-Shot Learners"
- **최적화**: "Decoupled Weight Decay Regularization" (AdamW)

---

## ❓ 자주 묻는 질문

### Q: GPU 메모리 부족 에러가 나요
```
A: batch_size를 줄이거나 max_seq_len을 줄이세요
   또는 gradient_accumulation 사용
```

### Q: 손실이 감소하지 않아요
```
A: 1. 학습률을 낮추세요 (1e-4 → 1e-5)
   2. 그래디언트 클리핑 확인
   3. 데이터 품질 확인
```

### Q: 얼마나 학습해야 하나요?
```
A: 검증 손실이 더 이상 감소하지 않을 때까지
   보통 10~100 에포크
```

### Q: 모델이 같은 단어만 반복해요
```
A: Temperature를 높이거나 Top-k/Top-p 샘플링 사용
   또는 더 많은 데이터로 더 오래 학습
```
