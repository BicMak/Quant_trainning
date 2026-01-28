"""
모델 평가 유틸리티
- Precision, Recall, F1 Score 계산
- Top-k Accuracy
- Classification Report
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import numpy as np


def compute_precision_recall_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Precision, Recall, F1 Score 계산

    Args:
        predictions: 예측 클래스 인덱스 (N,)
        targets: 실제 클래스 인덱스 (N,)
        num_classes: 클래스 수
        average: 'macro', 'micro', 'weighted' 중 선택

    Returns:
        Dictionary with precision, recall, f1 scores
    """
    predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
    targets = targets.cpu().numpy() if torch.is_tensor(targets) else targets

    # Per-class TP, FP, FN 계산
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    support = np.zeros(num_classes)

    for c in range(num_classes):
        tp[c] = np.sum((predictions == c) & (targets == c))
        fp[c] = np.sum((predictions == c) & (targets != c))
        fn[c] = np.sum((predictions != c) & (targets == c))
        support[c] = np.sum(targets == c)

    # Per-class precision, recall, f1
    with np.errstate(divide='ignore', invalid='ignore'):
        precision_per_class = tp / (tp + fp)
        recall_per_class = tp / (tp + fn)
        f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class)

    # NaN을 0으로 대체
    precision_per_class = np.nan_to_num(precision_per_class)
    recall_per_class = np.nan_to_num(recall_per_class)
    f1_per_class = np.nan_to_num(f1_per_class)

    if average == 'micro':
        # Global TP, FP, FN
        total_tp = np.sum(tp)
        total_fp = np.sum(fp)
        total_fn = np.sum(fn)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    elif average == 'macro':
        # 클래스별 평균 (support 무시)
        active_classes = support > 0
        precision = np.mean(precision_per_class[active_classes]) if np.any(active_classes) else 0.0
        recall = np.mean(recall_per_class[active_classes]) if np.any(active_classes) else 0.0
        f1 = np.mean(f1_per_class[active_classes]) if np.any(active_classes) else 0.0

    elif average == 'weighted':
        # Support로 가중 평균
        total_support = np.sum(support)
        if total_support > 0:
            precision = np.sum(precision_per_class * support) / total_support
            recall = np.sum(recall_per_class * support) / total_support
            f1 = np.sum(f1_per_class * support) / total_support
        else:
            precision = recall = f1 = 0.0
    else:
        raise ValueError(f"Unknown average: {average}")

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def compute_topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5)
) -> Dict[str, float]:
    """
    Top-k Accuracy 계산

    Args:
        logits: 모델 출력 로짓 (N, C)
        targets: 실제 클래스 인덱스 (N,)
        topk: k 값들의 튜플

    Returns:
        Dictionary with top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        results = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / batch_size).item()
            results[f'top{k}_acc'] = acc

        return results


class ModelEvaluator:
    """
    모델 평가 클래스
    - 배치 단위로 온라인 메트릭 계산 (메모리 효율적)
    - 최종 메트릭 집계
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """평가 상태 초기화"""
        # 온라인 계산용 - logits 저장하지 않음
        self.tp = np.zeros(self.num_classes)
        self.fp = np.zeros(self.num_classes)
        self.fn = np.zeros(self.num_classes)
        self.support = np.zeros(self.num_classes)

        self.correct_top1 = 0
        self.correct_top5 = 0
        self.total_samples = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        배치 결과 추가 (온라인 계산, 벡터화)

        메모리 누수 방지: detach().cpu()로 computational graph에서 분리

        Args:
            logits: 모델 출력 로짓 (N, C)
            targets: 실제 클래스 인덱스 (N,)
        """
        with torch.no_grad():
            # GPU 메모리 점유 방지: 즉시 detach하여 graph에서 분리 후 CPU로 이동
            logits_cpu = logits.detach().cpu()
            targets_cpu = targets.detach().cpu()

            pred_np = logits_cpu.argmax(dim=1).numpy()
            target_np = targets_cpu.numpy()

            # Top-5 계산 (CPU에서)
            _, top5_idx = logits_cpu.topk(5, dim=1)
            top5_np = top5_idx.numpy()

            # Top-1
            self.correct_top1 += (pred_np == target_np).sum()

            # Top-5 (벡터화)
            self.correct_top5 += np.any(top5_np == target_np[:, None], axis=1).sum()

            # Per-class TP, FP, FN 업데이트
            # 실제 등장한 클래스만 처리
            unique_classes = np.unique(np.concatenate([pred_np, target_np]))
            for c in unique_classes:
                pred_c = pred_np == c
                target_c = target_np == c
                self.tp[c] += np.sum(pred_c & target_c)
                self.fp[c] += np.sum(pred_c & ~target_c)
                self.fn[c] += np.sum(~pred_c & target_c)
                self.support[c] += np.sum(target_c)

            self.total_samples += target_np.shape[0]

    def compute_metrics(
        self,
        average: str = 'macro'
    ) -> Dict[str, float]:
        """
        전체 메트릭 계산 (온라인 집계 결과 사용)

        Args:
            average: precision/recall/f1 평균 방식

        Returns:
            모든 메트릭이 담긴 dictionary
        """
        if self.total_samples == 0:
            return {}

        # Per-class precision, recall, f1
        with np.errstate(divide='ignore', invalid='ignore'):
            precision_per_class = self.tp / (self.tp + self.fp)
            recall_per_class = self.tp / (self.tp + self.fn)
            f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class)

        precision_per_class = np.nan_to_num(precision_per_class)
        recall_per_class = np.nan_to_num(recall_per_class)
        f1_per_class = np.nan_to_num(f1_per_class)

        if average == 'micro':
            total_tp = np.sum(self.tp)
            total_fp = np.sum(self.fp)
            total_fn = np.sum(self.fn)
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        elif average == 'macro':
            active_classes = self.support > 0
            precision = np.mean(precision_per_class[active_classes]) if np.any(active_classes) else 0.0
            recall = np.mean(recall_per_class[active_classes]) if np.any(active_classes) else 0.0
            f1 = np.mean(f1_per_class[active_classes]) if np.any(active_classes) else 0.0

        elif average == 'weighted':
            total_support = np.sum(self.support)
            if total_support > 0:
                precision = np.sum(precision_per_class * self.support) / total_support
                recall = np.sum(recall_per_class * self.support) / total_support
                f1 = np.sum(f1_per_class * self.support) / total_support
            else:
                precision = recall = f1 = 0.0
        else:
            raise ValueError(f"Unknown average: {average}")

        # Top-k accuracy
        top1_acc = self.correct_top1 / self.total_samples * 100
        top5_acc = self.correct_top5 / self.total_samples * 100

        return {
            'accuracy': top1_acc,
            'top1_acc': top1_acc,
            'top5_acc': top5_acc,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'total_samples': self.total_samples
        }

    def get_per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """
        클래스별 메트릭 계산

        Returns:
            클래스 인덱스를 키로 하는 메트릭 dictionary
        """
        if self.total_samples == 0:
            return {}

        per_class = {}
        for c in range(self.num_classes):
            if self.support[c] == 0:
                continue

            precision = self.tp[c] / (self.tp[c] + self.fp[c]) if (self.tp[c] + self.fp[c]) > 0 else 0.0
            recall = self.tp[c] / (self.tp[c] + self.fn[c]) if (self.tp[c] + self.fn[c]) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class[c] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(self.support[c])
            }

        return per_class


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    device: str = 'cuda',
    num_classes: int = 1000,
    verbose: bool = True
) -> Dict[str, float]:
    """
    모델 전체 평가 함수

    Args:
        model: 평가할 모델
        dataloader: 평가 데이터로더
        device: 디바이스
        num_classes: 클래스 수
        verbose: 진행 상황 출력 여부

    Returns:
        평가 메트릭 dictionary
    """
    model.eval()
    evaluator = ModelEvaluator(num_classes)

    total_batches = len(dataloader)

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        evaluator.update(logits, targets)

        # 메모리 해제: 명시적 삭제
        del logits, images, targets

        # 주기적으로 CUDA 캐시 비우기
        if torch.cuda.is_available() and (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

        if verbose and (i + 1) % 10 == 0:
            print(f"  Evaluating: {i+1}/{total_batches} batches")

    metrics = evaluator.compute_metrics()

    if verbose:
        print(f"\n  Evaluation Results ({metrics['total_samples']} samples):")
        print(f"    Accuracy:  {metrics['accuracy']:.2f}%")
        print(f"    Top-5 Acc: {metrics['top5_acc']:.2f}%")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1 Score:  {metrics['f1']:.4f}")

    return metrics


def compare_models(
    metrics_fp32: Dict[str, float],
    metrics_quant: Dict[str, float],
    model_names: Tuple[str, str] = ('FP32', 'Quantized')
) -> Dict[str, float]:
    """
    두 모델의 메트릭 비교

    Args:
        metrics_fp32: FP32 모델 메트릭
        metrics_quant: 양자화 모델 메트릭
        model_names: 모델 이름 튜플

    Returns:
        차이 메트릭
    """
    diff = {}

    for key in ['accuracy', 'precision', 'recall', 'f1', 'top1_acc', 'top5_acc']:
        if key in metrics_fp32 and key in metrics_quant:
            diff[f'{key}_diff'] = metrics_quant[key] - metrics_fp32[key]

    print(f"\n  Model Comparison: {model_names[0]} vs {model_names[1]}")
    print("-" * 50)
    print(f"  {'Metric':<15} {model_names[0]:>10} {model_names[1]:>10} {'Diff':>10}")
    print("-" * 50)

    for key in ['accuracy', 'top5_acc', 'precision', 'recall', 'f1']:
        if key in metrics_fp32:
            fp32_val = metrics_fp32[key]
            quant_val = metrics_quant.get(key, 0)
            diff_val = quant_val - fp32_val
            sign = '+' if diff_val >= 0 else ''
            print(f"  {key:<15} {fp32_val:>10.4f} {quant_val:>10.4f} {sign}{diff_val:>9.4f}")

    print("-" * 50)

    return diff