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


class CausalAttention(nn.Module):
    """인과적 Self-Attention (GPT 스타일)"""

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

        # Attention scores 계산
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores_scaled = attn_scores / (self.d_out ** 0.5)

        # Causal mask 적용
        mask = torch.tril(torch.ones(seq_len, seq_len))
        attn_scores_masked = attn_scores_scaled.masked_fill(mask == 0, float('-inf'))

        # Softmax로 가중치 계산
        attn_weights = torch.softmax(attn_scores_masked, dim=-1)

        # Context vector 계산
        context = attn_weights @ values

        return context, attn_weights


class AttentionModel:
    """어텐션 모델"""

    def __init__(self, d_embed=128, d_out=128):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab

        self.embedding = TokenEmbedding(self.vocab_size, d_embed)
        self.attention = CausalAttention(d_embed, d_out)

    def process(self, text):
        """텍스트 처리"""
        # 토큰화
        token_ids = self.tokenizer.encode(text)
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        # 텐서로 변환
        token_tensor = torch.tensor(token_ids).unsqueeze(0)

        # 임베딩 + 어텐션
        embeddings = self.embedding(token_tensor)
        output, attn_weights = self.attention(embeddings)

        return output, tokens, attn_weights

    def visualize(self, tokens, attn_weights):
        """어텐션 가중치 시각화"""
        print(f"\n{'='*80}")
        print("🔍 어텐션 가중치 시각화")
        print(f"{'='*80}")
        print("각 토큰(행)이 이전 토큰들(열)에 얼마나 집중하는지 보여줍니다.\n")

        attn_matrix = attn_weights[0].detach().numpy()

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
                    row += f"{attn_matrix[i, j]:>10.3f}"
                else:
                    row += f"{'---':>10}"
            print(row)
        print()


def main():
    """메인 함수"""
    print("="*80)
    print("🤖 어텐션 모델 - 텍스트 입력 받기")
    print("="*80)
    print()

    # 모델 초기화
    torch.manual_seed(42)
    model = AttentionModel(d_embed=128, d_out=128)
    print("✅ 모델 초기화 완료\n")

    # 커맨드 라인 인자로 입력 받기
    if len(sys.argv) > 1:
        # 명령어로 실행: python run_attention.py "your text here"
        text = " ".join(sys.argv[1:])
        print(f"📝 입력: \"{text}\"\n")

        # 처리
        output, tokens, attn_weights = model.process(text)

        print(f"토큰 개수: {len(tokens)}")
        print(f"토큰: {tokens}")
        print(f"출력 크기: {output.shape}")

        # 시각화
        model.visualize(tokens, attn_weights)

    else:
        # 인터랙티브 모드
        print("💡 사용법: python run_attention.py \"your text here\"")
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
                model.visualize(tokens, attn_weights)

            except KeyboardInterrupt:
                print("\n\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류: {e}\n")


if __name__ == "__main__":
    main()
