"""
LLM 학습 파이프라인

학습에 필요한 구성 요소:
1. 손실 함수 (Loss Function): Cross Entropy Loss
2. 옵티마이저 (Optimizer): AdamW
3. 학습률 스케줄러 (Learning Rate Scheduler): Warmup + Cosine Decay
4. 데이터로더 (DataLoader): 배치 단위 데이터 제공
5. 학습 루프 (Training Loop): 실제 학습 진행
6. 검증 (Validation): 과적합 방지
7. 체크포인트 저장 (Checkpointing): 모델 저장/로드
8. 평가 지표 (Metrics): Perplexity, Accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import tiktoken
import math
import os
from tqdm import tqdm
import time

from layer_normalization import SimpleLLM


# ===================================================================
# 1. 데이터셋 클래스
# ===================================================================

class TextDataset(Dataset):
    """
    언어 모델 학습용 데이터셋

    입력: [토큰1, 토큰2, 토큰3, 토큰4]
    타겟: [토큰2, 토큰3, 토큰4, 토큰5]
    → 다음 토큰 예측 학습
    """

    def __init__(self, text_path, tokenizer, max_length=256, stride=None):
        """
        Args:
            text_path: 텍스트 파일 경로
            tokenizer: 토크나이저
            max_length: 시퀀스 최대 길이
            stride: 슬라이딩 윈도우 간격 (None이면 max_length와 동일)
        """
        # 텍스트 로드
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 토큰화
        self.token_ids = tokenizer.encode(text)
        self.max_length = max_length
        self.stride = stride if stride is not None else max_length

        print(f"✅ 데이터셋 로드 완료")
        print(f"   - 총 문자 수: {len(text):,}")
        print(f"   - 총 토큰 수: {len(self.token_ids):,}")
        print(f"   - 시퀀스 길이: {max_length}")
        print(f"   - Stride: {self.stride}")

    def __len__(self):
        # 생성 가능한 샘플 수
        return (len(self.token_ids) - self.max_length - 1) // self.stride

    def __getitem__(self, idx):
        # 시작 위치 계산
        start_idx = idx * self.stride

        # 입력과 타겟 추출
        input_ids = self.token_ids[start_idx:start_idx + self.max_length]
        target_ids = self.token_ids[start_idx + 1:start_idx + self.max_length + 1]

        return torch.tensor(input_ids), torch.tensor(target_ids)


# ===================================================================
# 2. 손실 함수 및 평가 지표
# ===================================================================

def compute_loss(logits, targets):
    """
    Cross Entropy Loss 계산

    Args:
        logits: [batch_size, seq_len, vocab_size]
        targets: [batch_size, seq_len]

    Returns:
        loss: 스칼라 손실 값
    """
    # logits를 2D로, targets를 1D로 변환
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
    targets_flat = targets.view(-1)  # [batch_size * seq_len]

    # Cross Entropy Loss
    loss = F.cross_entropy(logits_flat, targets_flat)

    return loss


def compute_perplexity(loss):
    """
    Perplexity 계산

    Perplexity = exp(loss)
    낮을수록 좋음 (모델이 다음 토큰을 잘 예측함)
    """
    return torch.exp(loss)


def compute_accuracy(logits, targets):
    """
    정확도 계산

    Args:
        logits: [batch_size, seq_len, vocab_size]
        targets: [batch_size, seq_len]

    Returns:
        accuracy: 정확도 (0~1)
    """
    predictions = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
    correct = (predictions == targets).float()
    accuracy = correct.mean()
    return accuracy


# ===================================================================
# 3. 학습률 스케줄러 (Warmup + Cosine Decay)
# ===================================================================

class WarmupCosineScheduler:
    """
    Warmup + Cosine Decay 학습률 스케줄러

    1. Warmup 단계: 학습률을 0에서 max_lr까지 선형 증가
    2. Cosine Decay: 학습률을 코사인 함수로 감소
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        """
        Args:
            optimizer: 옵티마이저
            warmup_steps: Warmup 스텝 수
            total_steps: 전체 학습 스텝 수
            min_lr: 최소 학습률
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        """학습률 업데이트"""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Warmup 단계: 선형 증가
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine Decay 단계
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        # 옵티마이저에 적용
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


# ===================================================================
# 4. 학습 클래스
# ===================================================================

class Trainer:
    """
    LLM 학습 담당 클래스
    """

    def __init__(self, model, train_loader, val_loader, device, config):
        """
        Args:
            model: SimpleLLM 모델
            train_loader: 학습 데이터로더
            val_loader: 검증 데이터로더
            device: 디바이스 (cuda/mps/cpu)
            config: 학습 설정 딕셔너리
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # 옵티마이저 (AdamW)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.95)  # GPT-3 설정
        )

        # 학습률 스케줄러
        total_steps = len(train_loader) * config['num_epochs']
        warmup_steps = int(total_steps * config['warmup_ratio'])
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=config['min_lr']
        )

        # 학습 기록
        self.history = {
            'train_loss': [],
            'train_perplexity': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_perplexity': [],
            'val_accuracy': [],
            'lr': []
        }

        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        """1 에포크 학습"""
        self.model.train()

        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # 순전파
            logits = self.model(inputs)

            # 손실 계산
            loss = compute_loss(logits, targets)

            # 역전파
            self.optimizer.zero_grad()
            loss.backward()

            # 그래디언트 클리핑 (기울기 폭발 방지)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )

            # 옵티마이저 스텝
            self.optimizer.step()

            # 학습률 업데이트
            current_lr = self.scheduler.step()

            # 정확도 계산
            accuracy = compute_accuracy(logits, targets)

            # 기록
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1

            # 진행 표시
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })

        # 평균 계산
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_perplexity = math.exp(avg_loss)

        return avg_loss, avg_perplexity, avg_accuracy, current_lr

    def validate(self):
        """검증"""
        self.model.eval()

        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # 순전파
                logits = self.model(inputs)

                # 손실 계산
                loss = compute_loss(logits, targets)
                accuracy = compute_accuracy(logits, targets)

                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_perplexity = math.exp(avg_loss)

        return avg_loss, avg_perplexity, avg_accuracy

    def train(self):
        """전체 학습 실행"""
        print("\n" + "="*80)
        print("🚀 학습 시작")
        print("="*80)

        start_time = time.time()

        for epoch in range(self.config['num_epochs']):
            # 학습
            train_loss, train_ppl, train_acc, lr = self.train_epoch(epoch)

            # 검증
            val_loss, val_ppl, val_acc = self.validate()

            # 기록
            self.history['train_loss'].append(train_loss)
            self.history['train_perplexity'].append(train_ppl)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_perplexity'].append(val_ppl)
            self.history['val_accuracy'].append(val_acc)
            self.history['lr'].append(lr)

            # 출력
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f}, PPL: {train_ppl:.2f}, Acc: {train_acc:.4f}")
            print(f"  Val   Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}, Acc: {val_acc:.4f}")
            print(f"  LR: {lr:.2e}")

            # 체크포인트 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✅ 최고 모델 저장! (Val Loss: {val_loss:.4f})")

            # 일반 체크포인트
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(epoch, is_best=False)

        elapsed_time = time.time() - start_time
        print(f"\n✅ 학습 완료! 소요 시간: {elapsed_time/60:.2f}분")

    def save_checkpoint(self, epoch, is_best=False):
        """체크포인트 저장"""
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }

        if is_best:
            path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
        else:
            path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pt')

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        print(f"✅ 체크포인트 로드 완료: {path}")
        return checkpoint['epoch']


# ===================================================================
# 5. 메인 실행
# ===================================================================

def main():
    print("="*80)
    print("🤖 SimpleLLM 학습 파이프라인")
    print("="*80)
    print()

    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("✅ GPU 사용 (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ GPU 사용 (MPS - Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("⚠️  CPU 사용 (학습이 느릴 수 있습니다)")
    print()

    # 학습 설정
    config = {
        # 모델 설정
        'd_embed': 256,
        'num_heads': 4,
        'num_layers': 4,
        'max_seq_len': 256,
        'dropout': 0.1,

        # 학습 설정
        'batch_size': 8,
        'num_epochs': 10,
        'learning_rate': 3e-4,
        'min_lr': 3e-5,
        'weight_decay': 0.1,
        'warmup_ratio': 0.1,
        'grad_clip': 1.0,

        # 데이터 설정
        'train_stride': 128,  # 오버랩 허용
        'val_stride': 256,    # 검증은 오버랩 없이

        # 체크포인트 설정
        'checkpoint_dir': './checkpoints',
        'save_every': 2  # 2 에포크마다 저장
    }

    # 토크나이저
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    # 데이터셋 생성
    print("📚 데이터셋 로드 중...")
    try:
        train_dataset = TextDataset(
            'the-verdict.txt',
            tokenizer,
            max_length=config['max_seq_len'],
            stride=config['train_stride']
        )

        # 검증 데이터셋 (마지막 10% 사용)
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        print(f"   - 학습 샘플 수: {len(train_dataset):,}")
        print(f"   - 검증 샘플 수: {len(val_dataset):,}\n")

    except FileNotFoundError:
        print("❌ 'the-verdict.txt' 파일을 찾을 수 없습니다.")
        print("   학습 데이터를 준비해주세요.\n")
        return

    # 데이터로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Windows에서는 0으로 설정
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 모델 생성
    print("🏗️  모델 생성 중...")
    model = SimpleLLM(
        vocab_size=vocab_size,
        d_embed=config['d_embed'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )

    # 학습
    trainer = Trainer(model, train_loader, val_loader, device, config)
    trainer.train()

    print("\n" + "="*80)
    print("🎉 학습 완료!")
    print("="*80)
    print(f"최고 검증 손실: {trainer.best_val_loss:.4f}")
    print(f"체크포인트 저장 위치: {config['checkpoint_dir']}")
    print()


if __name__ == "__main__":
    main()
