"""
층 정규화 (Layer Normalization)

LLM의 핵심 구성 요소:
1. Layer Normalization: 각 층의 출력을 정규화하여 학습 안정화
2. Residual Connection (Skip Connection): 기울기 소실 문제 해결
3. Feed Forward Network: 추가적인 비선형 변환

Transformer 블록 구조:
Input → LayerNorm → Multi-Head Attention → Add & Norm → FFN → Add & Norm → Output
"""

import torch
import torch.nn as nn
import tiktoken
import sys


class LayerNorm(nn.Module):
    """
    층 정규화 (Layer Normalization)

    배치 정규화(Batch Norm)와 달리 각 샘플을 독립적으로 정규화
    → 배치 크기에 영향을 받지 않아 NLP에서 더 효과적

    작동 원리:
    1. 각 샘플의 평균(mean)과 분산(variance) 계산
    2. 정규화: (x - mean) / sqrt(variance + eps)
    3. 스케일(γ)과 시프트(β) 학습 파라미터 적용
    """

    def __init__(self, d_embed, eps=1e-5):
        """
        Args:
            d_embed: 임베딩 차원
            eps: 수치 안정성을 위한 작은 값 (0으로 나누는 것을 방지)
        """
        super().__init__()
        self.eps = eps

        # 학습 가능한 파라미터
        self.scale = nn.Parameter(torch.ones(d_embed))   # γ (감마)
        self.shift = nn.Parameter(torch.zeros(d_embed))  # β (베타)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_embed]

        Returns:
            normalized: [batch_size, seq_len, d_embed]
        """
        # 마지막 차원(d_embed)에 대해 평균과 분산 계산
        mean = x.mean(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [batch_size, seq_len, 1]

        # 정규화
        normalized = (x - mean) / torch.sqrt(var + self.eps)

        # 스케일과 시프트 적용
        output = self.scale * normalized + self.shift

        return output


class FeedForward(nn.Module):
    """
    피드포워드 네트워크 (Feed Forward Network)

    구조: Linear → GELU → Linear
    - 중간 차원을 4배로 확장했다가 다시 축소
    - GELU: 부드러운 비선형 활성화 함수 (ReLU보다 성능 좋음)
    """

    def __init__(self, d_embed, d_ff=None, dropout=0.1):
        """
        Args:
            d_embed: 임베딩 차원
            d_ff: 피드포워드 중간 차원 (일반적으로 d_embed의 4배)
            dropout: 드롭아웃 비율
        """
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_embed  # 기본값: 4배 확장

        self.fc1 = nn.Linear(d_embed, d_ff)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_embed]

        Returns:
            output: [batch_size, seq_len, d_embed]
        """
        x = self.fc1(x)      # [batch_size, seq_len, d_ff]
        x = self.gelu(x)     # 비선형 활성화
        x = self.fc2(x)      # [batch_size, seq_len, d_embed]
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    """멀티-헤드 어텐션"""

    def __init__(self, d_embed, num_heads, dropout=0.1):
        super().__init__()
        assert d_embed % num_heads == 0

        self.d_embed = d_embed
        self.num_heads = num_heads
        self.head_dim = d_embed // num_heads

        self.W_query = nn.Linear(d_embed, d_embed)
        self.W_key = nn.Linear(d_embed, d_embed)
        self.W_value = nn.Linear(d_embed, d_embed)
        self.out_proj = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, d_embed = x.shape

        # Query, Key, Value 계산
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # 멀티-헤드로 분할
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = queries @ keys.transpose(2, 3) / (self.head_dim ** 0.5)

        # Causal mask (디바이스 일치)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context 계산
        context = attn_weights @ values
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_embed)

        # 출력 투영
        output = self.out_proj(context)
        return output


class TransformerBlock(nn.Module):
    """
    Transformer 블록 (GPT 스타일)

    구조:
    1. LayerNorm → Multi-Head Attention → Residual Connection
    2. LayerNorm → Feed Forward Network → Residual Connection

    Pre-LN (Pre-Layer Normalization):
    - LayerNorm을 서브레이어 앞에 배치
    - 학습 안정성이 더 좋음 (GPT-3, GPT-4에서 사용)
    """

    def __init__(self, d_embed, num_heads, dropout=0.1, d_ff=None):
        """
        Args:
            d_embed: 임베딩 차원
            num_heads: 헤드 개수
            dropout: 드롭아웃 비율
            d_ff: 피드포워드 중간 차원
        """
        super().__init__()

        # Layer Normalization
        self.ln1 = LayerNorm(d_embed)
        self.ln2 = LayerNorm(d_embed)

        # Multi-Head Attention
        self.attn = MultiHeadAttention(d_embed, num_heads, dropout)

        # Feed Forward Network
        self.ffn = FeedForward(d_embed, d_ff, dropout)

        # Dropout (Residual Connection에 적용)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_embed]

        Returns:
            output: [batch_size, seq_len, d_embed]
        """
        # 1. Multi-Head Attention with Residual Connection
        # Pre-LN: LayerNorm → Attention
        attn_output = self.attn(self.ln1(x))
        x = x + self.dropout(attn_output)  # Residual Connection

        # 2. Feed Forward Network with Residual Connection
        # Pre-LN: LayerNorm → FFN
        ffn_output = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_output)  # Residual Connection

        return x


class SimpleLLM(nn.Module):
    """
    간단한 LLM 모델

    구조:
    Token Embedding → Positional Embedding → N × Transformer Blocks → LayerNorm → Output
    """

    def __init__(self, vocab_size, d_embed=256, num_heads=4, num_layers=4,
                 max_seq_len=512, dropout=0.1):
        """
        Args:
            vocab_size: 어휘 크기
            d_embed: 임베딩 차원
            num_heads: 헤드 개수
            num_layers: Transformer 블록 개수
            max_seq_len: 최대 시퀀스 길이
            dropout: 드롭아웃 비율
        """
        super().__init__()

        self.d_embed = d_embed

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_embed)

        # Positional Embedding (학습 가능)
        self.pos_embedding = nn.Embedding(max_seq_len, d_embed)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_embed, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 최종 Layer Normalization
        self.ln_final = LayerNorm(d_embed)

        # Output projection (토큰 예측)
        self.lm_head = nn.Linear(d_embed, vocab_size, bias=False)

        print(f"✅ SimpleLLM 초기화 완료")
        print(f"   - 어휘 크기: {vocab_size:,}")
        print(f"   - 임베딩 차원: {d_embed}")
        print(f"   - 헤드 개수: {num_heads}")
        print(f"   - 레이어 개수: {num_layers}")
        print(f"   - 최대 시퀀스 길이: {max_seq_len}")
        print(f"   - 드롭아웃: {dropout}")

        # 파라미터 개수 계산
        num_params = sum(p.numel() for p in self.parameters())
        print(f"   - 총 파라미터 수: {num_params:,}\n")

    def forward(self, token_ids):
        """
        Args:
            token_ids: [batch_size, seq_len]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = token_ids.shape

        # Token Embedding
        token_embeds = self.token_embedding(token_ids)  # [batch_size, seq_len, d_embed]

        # Positional Embedding
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)  # [1, seq_len]
        pos_embeds = self.pos_embedding(positions)  # [1, seq_len, d_embed]

        # 임베딩 결합
        x = token_embeds + pos_embeds
        x = self.dropout(x)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # 최종 LayerNorm
        x = self.ln_final(x)

        # 출력 투영 (로짓)
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]

        return logits


def test_layer_normalization():
    """층 정규화 테스트"""
    print("="*80)
    print("🧪 층 정규화 (Layer Normalization) 테스트")
    print("="*80)
    print()

    # 테스트 데이터
    batch_size = 2
    seq_len = 4
    d_embed = 8

    x = torch.randn(batch_size, seq_len, d_embed)
    print(f"입력 크기: {x.shape}")
    print(f"입력 데이터 (첫 번째 샘플, 첫 번째 토큰):\n{x[0, 0]}\n")

    # Layer Normalization 적용
    ln = LayerNorm(d_embed)
    normalized = ln(x)

    print(f"정규화 후 크기: {normalized.shape}")
    print(f"정규화 후 데이터 (첫 번째 샘플, 첫 번째 토큰):\n{normalized[0, 0]}\n")

    # 통계 확인
    mean = normalized[0, 0].mean()
    std = normalized[0, 0].std(unbiased=False)
    print(f"정규화 후 평균: {mean:.6f} (0에 가까워야 함)")
    print(f"정규화 후 표준편차: {std:.6f} (1에 가까워야 함)")
    print()


def test_transformer_block():
    """Transformer 블록 테스트"""
    print("="*80)
    print("🧪 Transformer 블록 테스트")
    print("="*80)
    print()

    batch_size = 2
    seq_len = 6
    d_embed = 128
    num_heads = 4

    x = torch.randn(batch_size, seq_len, d_embed)
    print(f"입력 크기: {x.shape}")

    # Transformer Block
    block = TransformerBlock(d_embed, num_heads, dropout=0.1)
    output = block(x)

    print(f"출력 크기: {output.shape}")
    print(f"✅ Transformer 블록 작동 확인\n")

    # 파라미터 개수
    num_params = sum(p.numel() for p in block.parameters())
    print(f"Transformer 블록 파라미터 수: {num_params:,}\n")


def test_simple_llm():
    """SimpleLLM 테스트"""
    print("="*80)
    print("🤖 SimpleLLM 모델 테스트")
    print("="*80)
    print()

    # 토크나이저
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    # 모델 생성
    torch.manual_seed(42)
    model = SimpleLLM(
        vocab_size=vocab_size,
        d_embed=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=512,
        dropout=0.1
    )

    # 테스트 문장
    text = "Hello, how are you today?"
    token_ids = tokenizer.encode(text)
    token_tensor = torch.tensor(token_ids).unsqueeze(0)  # [1, seq_len]

    print(f"📝 입력 문장: \"{text}\"")
    print(f"토큰 ID: {token_ids}")
    print(f"입력 크기: {token_tensor.shape}\n")

    # 순전파
    model.eval()
    with torch.no_grad():
        logits = model(token_tensor)

    print(f"출력 로짓 크기: {logits.shape}")
    print(f"  - Batch size: {logits.shape[0]}")
    print(f"  - Sequence length: {logits.shape[1]}")
    print(f"  - Vocab size: {logits.shape[2]}\n")

    # 다음 토큰 예측
    last_token_logits = logits[0, -1, :]  # 마지막 토큰의 로짓
    probs = torch.softmax(last_token_logits, dim=-1)
    top5_probs, top5_indices = torch.topk(probs, 5)

    print("🎯 다음 토큰 예측 (Top 5):")
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices), 1):
        token = tokenizer.decode([idx.item()])
        print(f"  {i}. '{token}' - 확률: {prob.item():.4f} ({prob.item()*100:.2f}%)")


def compare_normalizations():
    """Batch Norm vs Layer Norm 비교"""
    print("\n" + "="*80)
    print("📊 Batch Normalization vs Layer Normalization 비교")
    print("="*80)
    print()

    batch_size = 3
    seq_len = 4
    d_embed = 6

    x = torch.randn(batch_size, seq_len, d_embed)

    print(f"입력 크기: {x.shape}")
    print(f"입력 데이터:\n{x[0, 0]}\n")

    # Layer Normalization
    ln = LayerNorm(d_embed)
    ln_output = ln(x)

    print("Layer Normalization:")
    print("  - 각 샘플을 독립적으로 정규화")
    print("  - 배치 크기에 영향 없음")
    print(f"  - 출력: {ln_output[0, 0]}")
    print(f"  - 평균: {ln_output[0, 0].mean():.6f}, 표준편차: {ln_output[0, 0].std(unbiased=False):.6f}\n")

    # PyTorch의 Layer Norm과 비교
    pytorch_ln = nn.LayerNorm(d_embed)
    pytorch_ln.weight.data = ln.scale.data
    pytorch_ln.bias.data = ln.shift.data
    pytorch_output = pytorch_ln(x)

    diff = (ln_output - pytorch_output).abs().max()
    print(f"✅ PyTorch LayerNorm과의 차이: {diff:.8f} (거의 동일)\n")


if __name__ == "__main__":
    # 1. Layer Normalization 테스트
    test_layer_normalization()

    # 2. Transformer Block 테스트
    test_transformer_block()

    # 3. SimpleLLM 전체 모델 테스트
    test_simple_llm()

    # 4. Normalization 비교
    compare_normalizations()

    print("="*80)
    print("✅ 모든 테스트 완료!")
    print("="*80)
