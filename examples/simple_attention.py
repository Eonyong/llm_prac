import torch
import torch.nn as nn

class SimpleAttention(nn.Module):
    """
    간단한 Self-Attention 메커니즘

    작동 방식:
    1. 입력을 Query, Key, Value로 변환
    2. Query와 Key의 유사도를 계산 (어떤 단어가 중요한가?)
    3. Softmax로 확률(가중치)로 변환
    4. Value에 가중치를 적용하여 최종 출력 생성
    """

    def __init__(self, d_in, d_out):
        """
        Args:
            d_in: 입력 임베딩 차원 (예: 단어 벡터의 크기)
            d_out: 출력 차원 (Query, Key, Value의 차원)
        """
        super().__init__()
        self.d_out = d_out

        # 입력을 Query, Key, Value로 변환하는 가중치 행렬
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        """
        Args:
            x: 입력 텐서 [batch_size, seq_len, d_in]
               예: [2, 4, 3] → 2개 문장, 각 4개 단어, 각 단어는 3차원 벡터

        Returns:
            context: 어텐션이 적용된 출력 [batch_size, seq_len, d_out]
        """
        # Step 1: 입력을 Query, Key, Value로 변환
        queries = self.W_query(x)  # [batch, seq_len, d_out]
        keys = self.W_key(x)       # [batch, seq_len, d_out]
        values = self.W_value(x)   # [batch, seq_len, d_out]

        print(f"입력 x의 크기: {x.shape}")
        print(f"Queries 크기: {queries.shape}")
        print(f"Keys 크기: {keys.shape}")
        print(f"Values 크기: {values.shape}\n")

        # Step 2: Query와 Key의 유사도 계산 (Attention Scores)
        # queries @ keys^T → 각 단어가 다른 단어들과 얼마나 관련있는지
        attn_scores = queries @ keys.transpose(1, 2)  # [batch, seq_len, seq_len]

        print(f"Attention Scores 크기: {attn_scores.shape}")
        print(f"Attention Scores (첫 번째 배치):\n{attn_scores[0]}\n")

        # Step 3: Scaling (큰 값으로 인한 gradient 문제 방지)
        attn_scores_scaled = attn_scores / (self.d_out ** 0.5)

        # Step 4: Softmax로 확률(가중치)로 변환
        attn_weights = torch.softmax(attn_scores_scaled, dim=-1)

        print(f"Attention Weights (첫 번째 배치):")
        print(attn_weights[0])
        print(f"각 행의 합 (1이어야 함): {attn_weights[0].sum(dim=-1)}\n")

        # Step 5: Value에 가중치를 적용하여 최종 출력 생성
        context = attn_weights @ values  # [batch, seq_len, d_out]

        print(f"최종 Context Vector 크기: {context.shape}")

        return context


class CausalAttention(nn.Module):
    """
    인과적(Causal) Self-Attention

    GPT처럼 미래 단어를 보지 못하게 마스킹
    (다음 단어 예측 시 뒤의 단어를 미리 보면 안 됨)
    """

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        batch_size, seq_len, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Attention Scores 계산
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores_scaled = attn_scores / (self.d_out ** 0.5)

        # 인과적 마스크 생성 (하삼각 행렬)
        # 현재 위치 이후의 단어들은 -inf로 설정 → softmax 후 0이 됨
        mask = torch.tril(torch.ones(seq_len, seq_len))
        attn_scores_masked = attn_scores_scaled.masked_fill(mask == 0, float('-inf'))

        print(f"Causal Mask:\n{mask}\n")
        print(f"Masked Attention Scores (첫 번째 배치):\n{attn_scores_masked[0]}\n")

        # Softmax 적용
        attn_weights = torch.softmax(attn_scores_masked, dim=-1)

        print(f"Masked Attention Weights (첫 번째 배치):")
        print(attn_weights[0])
        print("→ 위쪽 삼각형 부분이 0인 것을 확인 (미래 단어를 보지 않음)\n")

        # Context Vector 계산
        context = attn_weights @ values

        return context


# 실행 예제
if __name__ == "__main__":
    print("=" * 60)
    print("1. 기본 Self-Attention 예제")
    print("=" * 60 + "\n")

    # 입력 데이터 생성
    # 예: 2개의 문장, 각 문장은 4개 단어, 각 단어는 3차원 임베딩
    torch.manual_seed(123)
    batch_size = 2
    seq_len = 4
    d_in = 3
    d_out = 2

    x = torch.randn(batch_size, seq_len, d_in)
    print(f"입력 예시 (첫 번째 문장):\n{x[0]}\n")

    # Simple Attention 적용
    simple_attn = SimpleAttention(d_in, d_out)
    output = simple_attn(x)

    print(f"출력 (첫 번째 문장):\n{output[0]}\n")

    print("\n" + "=" * 60)
    print("2. Causal Attention 예제 (GPT 스타일)")
    print("=" * 60 + "\n")

    # Causal Attention 적용
    causal_attn = CausalAttention(d_in, d_out)
    output_causal = causal_attn(x)

    print(f"출력 (첫 번째 문장):\n{output_causal[0]}\n")
