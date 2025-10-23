import torch
import torch.nn as nn
import tiktoken


class TokenEmbedding(nn.Module):
    """토큰 ID를 임베딩 벡터로 변환"""

    def __init__(self, vocab_size, d_embed):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)

    def forward(self, token_ids):
        return self.embedding(token_ids)


class CausalAttention(nn.Module):
    """인과적 Self-Attention (GPT 스타일)"""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x, verbose=False):
        batch_size, seq_len, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Attention scores 계산
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores_scaled = attn_scores / (self.d_out ** 0.5)

        # Causal mask 적용
        mask = torch.tril(torch.ones(seq_len, seq_len))
        attn_scores_masked = attn_scores_scaled.masked_fill(mask == 0, float('-inf'))

        # Softmax로 가중치 계산
        attn_weights = torch.softmax(attn_scores_masked, dim=-1)

        if verbose:
            print(f"\n[ Attention Weights ]")
            print("각 토큰이 이전 토큰들에 얼마나 집중하는지:")
            print(attn_weights[0].detach().numpy())

        # Context vector 계산
        context = attn_weights @ values

        return context, attn_weights


class InteractiveAttentionModel:
    """
    사용자 입력을 받아 어텐션을 적용하는 인터랙티브 모델
    """

    def __init__(self, d_embed=128, d_out=128):
        """
        Args:
            d_embed: 임베딩 차원
            d_out: 어텐션 출력 차원
        """
        # 토크나이저 초기화
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab

        # 임베딩 레이어
        self.embedding = TokenEmbedding(self.vocab_size, d_embed)

        # 어텐션 레이어
        self.attention = CausalAttention(d_embed, d_out)

        print(f"✅ 모델 초기화 완료")
        print(f"   - 어휘 크기: {self.vocab_size}")
        print(f"   - 임베딩 차원: {d_embed}")
        print(f"   - 어텐션 출력 차원: {d_out}\n")

    def process_text(self, text, verbose=True):
        """
        입력 텍스트를 처리하여 어텐션 적용

        Args:
            text: 입력 텍스트
            verbose: 상세 정보 출력 여부

        Returns:
            output: 어텐션이 적용된 출력 벡터
            tokens: 토큰 리스트
            attn_weights: 어텐션 가중치
        """
        # 1. 토큰화
        token_ids = self.tokenizer.encode(text)
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        if verbose:
            print(f"📝 입력 텍스트: \"{text}\"")
            print(f"\n[ Step 1: 토큰화 ]")
            print(f"토큰 개수: {len(token_ids)}")
            print(f"토큰 ID: {token_ids}")
            print(f"토큰: {tokens}")

        # 2. 텐서로 변환 (배치 차원 추가)
        token_tensor = torch.tensor(token_ids).unsqueeze(0)  # [1, seq_len]

        # 3. 임베딩
        embeddings = self.embedding(token_tensor)  # [1, seq_len, d_embed]

        if verbose:
            print(f"\n[ Step 2: 임베딩 ]")
            print(f"임베딩 크기: {embeddings.shape}")
            print(f"첫 번째 토큰의 임베딩 벡터 (처음 5개 차원):")
            print(f"{embeddings[0, 0, :5].detach().numpy()}")

        # 4. 어텐션 적용
        output, attn_weights = self.attention(embeddings, verbose=verbose)

        if verbose:
            print(f"\n[ Step 3: 어텐션 적용 ]")
            print(f"출력 크기: {output.shape}")
            print(f"첫 번째 토큰의 출력 벡터 (처음 5개 차원):")
            print(f"{output[0, 0, :5].detach().numpy()}")

        return output, tokens, attn_weights

    def visualize_attention(self, tokens, attn_weights):
        """
        어텐션 가중치를 시각화

        Args:
            tokens: 토큰 리스트
            attn_weights: 어텐션 가중치 [1, seq_len, seq_len]
        """
        print(f"\n{'='*70}")
        print("🔍 어텐션 가중치 시각화")
        print(f"{'='*70}")
        print("각 토큰(행)이 이전 토큰들(열)에 얼마나 집중하는지 보여줍니다.")
        print("숫자가 클수록 더 많이 집중하는 것입니다.\n")

        attn_matrix = attn_weights[0].detach().numpy()

        # 헤더 출력
        max_token_len = max(len(t) for t in tokens)
        header = " " * (max_token_len + 2)
        for token in tokens:
            header += f"{token[:6]:>8}"
        print(header)
        print("-" * len(header))

        # 각 행 출력
        for i, token in enumerate(tokens):
            row = f"{token[:max_token_len]:<{max_token_len+2}}"
            for j in range(len(tokens)):
                if j <= i:  # Causal mask: 현재와 이전 토큰만
                    row += f"{attn_matrix[i, j]:>8.3f}"
                else:
                    row += f"{'---':>8}"
            print(row)

    def compare_tokens(self, tokens, output):
        """
        각 토큰의 출력 벡터 비교

        Args:
            tokens: 토큰 리스트
            output: 어텐션 출력 [1, seq_len, d_out]
        """
        print(f"\n{'='*70}")
        print("📊 각 토큰의 출력 벡터 비교 (처음 10개 차원)")
        print(f"{'='*70}\n")

        output_matrix = output[0].detach().numpy()

        for i, token in enumerate(tokens):
            print(f"토큰 {i+1}: '{token}'")
            print(f"  출력: {output_matrix[i, :10]}")
            print()


def interactive_mode():
    """인터랙티브 모드로 실행"""
    print("="*70)
    print("🤖 인터랙티브 어텐션 모델")
    print("="*70)
    print()

    # 모델 초기화
    torch.manual_seed(42)
    model = InteractiveAttentionModel(d_embed=128, d_out=128)

    while True:
        print("\n" + "="*70)
        user_input = input("📝 텍스트를 입력하세요 (종료: 'quit' 또는 'exit'): ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 프로그램을 종료합니다.")
            break

        if not user_input:
            print("⚠️  텍스트를 입력해주세요.")
            continue

        try:
            # 텍스트 처리
            output, tokens, attn_weights = model.process_text(user_input, verbose=True)

            # 어텐션 가중치 시각화
            model.visualize_attention(tokens, attn_weights)

            # 토큰별 출력 비교
            model.compare_tokens(tokens, output)

        except Exception as e:
            print(f"❌ 오류 발생: {e}")


def example_mode():
    """예제 모드로 실행"""
    print("="*70)
    print("📚 예제 모드: 미리 정의된 텍스트로 실행")
    print("="*70)
    print()

    # 모델 초기화
    torch.manual_seed(42)
    model = InteractiveAttentionModel(d_embed=128, d_out=128)

    # 예제 텍스트들
    examples = [
        "Hello, how are you?",
        "I love machine learning",
        "The quick brown fox jumps over the lazy dog",
        "Attention is all you need"
    ]

    for idx, text in enumerate(examples, 1):
        print(f"\n{'='*70}")
        print(f"예제 {idx}/{len(examples)}")
        print(f"{'='*70}")

        # 텍스트 처리
        output, tokens, attn_weights = model.process_text(text, verbose=True)

        # 어텐션 가중치 시각화
        model.visualize_attention(tokens, attn_weights)

        if idx < len(examples):
            input("\n⏸️  다음 예제를 보려면 Enter를 누르세요...")


if __name__ == "__main__":
    print("\n실행 모드를 선택하세요:")
    print("1. 인터랙티브 모드 (직접 입력)")
    print("2. 예제 모드 (미리 정의된 텍스트)")
    print()

    choice = input("선택 (1 또는 2): ").strip()

    if choice == "1":
        interactive_mode()
    elif choice == "2":
        example_mode()
    else:
        print("❌ 잘못된 선택입니다. 1 또는 2를 입력해주세요.")
