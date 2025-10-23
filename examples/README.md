# Examples - 예제 코드 모음

이 폴더에는 학습 과정에서 만든 예제 및 데모 파일들이 있습니다.

## 📚 파일 설명

### 기초 예제
- `chapter3.py` - Chapter 3 학습 코드 (초기 어텐션 구현)
- `main.py` - 초기 테스트 코드
- `tokenizer.py` - 커스텀 토크나이저 구현
- `pytorch_example.py` - PyTorch 기본 예제

### 어텐션 메커니즘 예제
- `simple_attention.py` - 기본 Self-Attention 구현
- `multihead_attention.py` - Multi-Head Attention 구현 및 데모
- `interactive_attention.py` - 어텐션 시각화 (인터랙티브)
- `run_attention.py` - 어텐션 실행 스크립트

## 🚀 실행 방법

각 파일은 독립적으로 실행 가능합니다:

```bash
# 가상환경 활성화
source ../.venv/bin/activate

# 예제 실행
python simple_attention.py
python multihead_attention.py
python run_attention.py "your text here"
```

## 📝 참고

이 파일들은 학습용 예제이며, 실제 LLM 학습은 메인 디렉토리의 파일들을 사용하세요:
- `layer_normalization.py` - 모델 정의
- `training_pipeline.py` - 학습 실행
- `run_simple_llm.py` - 텍스트 생성
