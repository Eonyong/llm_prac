import torch
import torch.nn as nn
import tiktoken
import sys


class TokenEmbedding(nn.Module):
    """토큰 ID를 임베딩 벡터로 변환"""

    def __init__(self, vocab_size, d_embed):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)

    def forward(self, token_ids):
        return self.embedding(token_ids)


class MultiHeadAttention(nn.Module):
    """
    멀티-헤드 어텐션 (Multi-Head Attention)

    여러 개의 어텐션 헤드를 병렬로 실행하여 다양한 관점에서 정보를 학습

    작동 원리:
    1. 입력을 num_heads개로 분할
    2. 각 헤드에서 독립적으로 어텐션 계산
    3. 모든 헤드의 결과를 합침(concatenate)
    4. 최종 선형 변환 적용
    """

    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        """
        Args:
            d_in: 입력 차원
            d_out: 출력 차원
            num_heads: 헤드 개수
            dropout: 드롭아웃 비율
            qkv_bias: Query, Key, Value 레이어에 bias 사용 여부
        """
        super().__init__()

        assert d_out % num_heads == 0, "d_out은 num_heads로 나누어떨어져야 합니다"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 각 헤드의 차원

        # Query, Key, Value 변환 레이어
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 최종 출력 변환 레이어
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # 어텐션 가중치 저장 (시각화용)
        self.attn_weights = None

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_in]

        Returns:
            output: [batch_size, seq_len, d_out]
        """
        batch_size, seq_len, d_in = x.shape

        # Step 1: Query, Key, Value 계산
        queries = self.W_query(x)  # [batch_size, seq_len, d_out]
        keys = self.W_key(x)       # [batch_size, seq_len, d_out]
        values = self.W_value(x)   # [batch_size, seq_len, d_out]

        # Step 2: 멀티-헤드로 분할
        # [batch_size, seq_len, d_out] → [batch_size, seq_len, num_heads, head_dim]
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 차원 재배치: [batch_size, num_heads, seq_len, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Step 3: 각 헤드별로 Scaled Dot-Product Attention 계산
        # [batch_size, num_heads, seq_len, seq_len]
        attn_scores = queries @ keys.transpose(2, 3)

        # Scaling
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        # Causal mask 적용 (현재 위치보다 뒤의 토큰은 보지 못하게)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 시각화를 위해 저장
        self.attn_weights = attn_weights.detach()

        # Step 4: Value에 어텐션 가중치 적용
        # [batch_size, num_heads, seq_len, head_dim]
        context = attn_weights @ values

        # Step 5: 헤드들을 다시 합침 (concatenate)
        # [batch_size, seq_len, num_heads, head_dim]
        context = context.transpose(1, 2)

        # [batch_size, seq_len, d_out]
        context = context.contiguous().view(batch_size, seq_len, self.d_out)

        # Step 6: 최종 선형 변환
        output = self.out_proj(context)

        return output


class MultiHeadAttentionModel:
    """멀티-헤드 어텐션 모델"""

    def __init__(self, d_embed=128, d_out=128, num_heads=24, dropout=0.1):
        """
        Args:
            d_embed: 임베딩 차원
            d_out: 출력 차원
            num_heads: 헤드 개수 (d_out을 num_heads로 나눈 값이 각 헤드의 차원)
            dropout: 드롭아웃 비율
        """
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab
        self.num_heads = num_heads

        self.embedding = TokenEmbedding(self.vocab_size, d_embed)
        self.attention = MultiHeadAttention(
            d_in=d_embed,
            d_out=d_out,
            num_heads=num_heads,
            dropout=dropout
        )

        print(f"✅ 멀티-헤드 어텐션 모델 초기화 완료")
        print(f"   - 어휘 크기: {self.vocab_size}")
        print(f"   - 임베딩 차원: {d_embed}")
        print(f"   - 출력 차원: {d_out}")
        print(f"   - 헤드 개수: {num_heads}")
        print(f"   - 각 헤드 차원: {d_out // num_heads}")
        print(f"   - 드롭아웃: {dropout}\n")

    def process(self, text):
        """텍스트 처리"""
        # 토큰화
        token_ids = self.tokenizer.encode(text)
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        # 텐서로 변환
        token_tensor = torch.tensor(token_ids).unsqueeze(0)

        # 임베딩 + 멀티-헤드 어텐션
        embeddings = self.embedding(token_tensor)
        output = self.attention(embeddings)

        # 어텐션 가중치 가져오기
        attn_weights = self.attention.attn_weights

        return output, tokens, attn_weights

    def visualize_all_heads(self, tokens, attn_weights):
        """모든 헤드의 어텐션 가중치 시각화"""
        print(f"\n{'='*80}")
        print(f"🔍 멀티-헤드 어텐션 가중치 시각화 (총 {self.num_heads}개 헤드)")
        print(f"{'='*80}\n")

        # attn_weights shape: [batch_size, num_heads, seq_len, seq_len]
        attn_matrix = attn_weights[0]  # 첫 번째 배치만

        for head_idx in range(self.num_heads):
            print(f"\n[ Head {head_idx + 1}/{self.num_heads} ]")
            print("-" * 80)
            self._visualize_single_head(tokens, attn_matrix[head_idx])

    def _visualize_single_head(self, tokens, attn_matrix):
        """단일 헤드의 어텐션 가중치 시각화"""
        attn_np = attn_matrix.numpy()

        # 헤더
        max_len = max(len(t) for t in tokens)
        header = " " * (max_len + 2)
        for token in tokens:
            header += f"{token[:8]:>10}"
        print(header)
        print("-" * len(header))

        # 각 행
        for i, token in enumerate(tokens):
            row = f"{token[:max_len]:<{max_len+2}}"
            for j in range(len(tokens)):
                if j <= i:
                    row += f"{attn_np[i, j]:>10.3f}"
                else:
                    row += f"{'---':>10}"
            print(row)

    def visualize_average_attention(self, tokens, attn_weights):
        """모든 헤드의 평균 어텐션 가중치 시각화"""
        print(f"\n{'='*80}")
        print("📊 평균 어텐션 가중치 (모든 헤드의 평균)")
        print(f"{'='*80}")

        # 모든 헤드의 평균 계산
        attn_matrix = attn_weights[0].mean(dim=0)  # [seq_len, seq_len]

        attn_np = attn_matrix.numpy()

        # 헤더
        max_len = max(len(t) for t in tokens)
        header = " " * (max_len + 2)
        for token in tokens:
            header += f"{token[:8]:>10}"
        print(header)
        print("-" * len(header))

        # 각 행
        for i, token in enumerate(tokens):
            row = f"{token[:max_len]:<{max_len+2}}"
            for j in range(len(tokens)):
                if j <= i:
                    row += f"{attn_np[i, j]:>10.3f}"
                else:
                    row += f"{'---':>10}"
            print(row)
        print()

    def compare_heads(self, tokens, attn_weights):
        """각 헤드가 특정 토큰에 집중하는 정도 비교"""
        print(f"\n{'='*80}")
        print("🎯 각 헤드가 각 토큰에 집중하는 정도 (마지막 토큰 기준)")
        print(f"{'='*80}\n")

        # 마지막 토큰의 어텐션 가중치 [num_heads, seq_len]
        last_token_attn = attn_weights[0, :, -1, :]

        print(f"{'헤드':<8}", end="")
        for token in tokens:
            print(f"{token[:8]:>10}", end="")
        print("\n" + "-" * (8 + 10 * len(tokens)))

        for head_idx in range(self.num_heads):
            print(f"Head {head_idx+1:<3}", end="")
            for token_idx in range(len(tokens)):
                print(f"{last_token_attn[head_idx, token_idx]:>10.3f}", end="")
            print()
        print()


def main():
    """메인 함수"""
    print("="*80)
    print("🤖 멀티-헤드 어텐션 모델 - 텍스트 입력 받기")
    print("="*80)
    print()

    # 모델 초기화
    torch.manual_seed(42)
    model = MultiHeadAttentionModel(
        d_embed=512,
        d_out=128,
        num_heads=8,  # 4개의 헤드 사용
        dropout=0.1
    )

    # 커맨드 라인 인자로 입력 받기
    if len(sys.argv) > 1:
        # 명령어로 실행: python multihead_attention.py "your text here"
        text = " ".join(sys.argv[1:])
        print(f"📝 입력: \"{text}\"\n")

        # 처리
        output, tokens, attn_weights = model.process(text)

        print(f"토큰 개수: {len(tokens)}")
        print(f"토큰: {tokens}")
        print(f"출력 크기: {output.shape}")

        # 시각화
        model.visualize_all_heads(tokens, attn_weights)
        model.visualize_average_attention(tokens, attn_weights)
        model.compare_heads(tokens, attn_weights)

    else:
        # 인터랙티브 모드
        print("💡 사용법: python multihead_attention.py \"your text here\"")
        print("또는 아래에 직접 입력하세요.\n")

        while True:
            try:
                text = input("📝 텍스트 입력 (종료: Ctrl+C): ").strip()

                if not text:
                    print("⚠️  텍스트를 입력해주세요.\n")
                    continue

                # 처리
                output, tokens, attn_weights = model.process(text)

                print(f"\n토큰 개수: {len(tokens)}")
                print(f"토큰: {tokens}")
                print(f"출력 크기: {output.shape}")

                # 시각화
                model.visualize_all_heads(tokens, attn_weights)
                model.visualize_average_attention(tokens, attn_weights)
                model.compare_heads(tokens, attn_weights)

            except KeyboardInterrupt:
                print("\n\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류: {e}\n")


if __name__ == "__main__":
    main()
