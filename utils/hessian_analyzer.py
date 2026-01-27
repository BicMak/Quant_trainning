"""
Hessian Analyzer for Block-wise Sensitivity Analysis

Fisher Information 기반 Hessian 분석:
- 각 블록 출력에 대한 민감도 계산
- Mixed-precision 결정에 활용
- 양자화 에러 민감도 측정

Methods:
1. Fisher Information: E[∂L/∂θ · (∂L/∂θ)^T] - gradient의 outer product 기대값
2. Hessian Trace: tr(H) = Σ ∂²L/∂θᵢ² - 대각 성분의 합
3. Hutchinson's Estimator: 확률적 trace 추정
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np


class HessianAnalyzer:
    """
    Block-wise Hessian/Fisher Information Analyzer.

    각 블록의 출력에 대한 민감도를 계산하여
    양자화에 민감한 블록을 식별합니다.
    """

    def __init__(self, model: nn.Module, criterion: Callable = None):
        """
        Args:
            model: QVit 또는 블록 리스트를 포함한 모델
            criterion: Loss function (default: CrossEntropyLoss)
        """
        self.model = model
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = next(model.parameters()).device

    def compute_block_fisher(self,
                             data_loader,
                             num_samples: int = 100,
                             use_labels: bool = True) -> Dict[int, float]:
        """
        각 블록 출력에 대한 Fisher Information을 계산합니다.

        Fisher Information = E[||∂L/∂h||²]
        where h is the block output

        Args:
            data_loader: Calibration data loader
            num_samples: Number of samples for estimation
            use_labels: Whether to use real labels (True) or model predictions (False)

        Returns:
            Dict mapping block index to Fisher trace value
        """
        self.model.eval()
        fisher_dict = {i: 0.0 for i in range(len(self.model.blocks))}
        sample_count = 0

        for batch_idx, (images, labels) in enumerate(data_loader):
            if sample_count >= num_samples:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            # 각 블록별로 Fisher 계산
            for block_idx in range(len(self.model.blocks)):
                fisher_val = self._compute_single_block_fisher(
                    images, labels, block_idx, use_labels
                )
                fisher_dict[block_idx] += fisher_val * batch_size

            sample_count += batch_size

            if (batch_idx + 1) % 10 == 0:
                print(f"  Fisher computation: {sample_count}/{num_samples} samples")

        # Average over samples
        for block_idx in fisher_dict:
            fisher_dict[block_idx] /= sample_count

        return fisher_dict

    def _compute_single_block_fisher(self,
                                     x: torch.Tensor,
                                     labels: torch.Tensor,
                                     block_idx: int,
                                     use_labels: bool = True) -> float:
        """
        단일 블록에 대한 Fisher Information 계산.

        블록 출력 h에 대해:
        Fisher = E[||∂L/∂h||²] = trace(∂L/∂h · (∂L/∂h)^T)
        """
        self.model.zero_grad()

        # Forward pass with gradient tracking for block output
        # 1. Embedding (no grad needed here)
        if self.model.embedding is not None:
            self.model.embedding.set_mode('fp32')
            with torch.no_grad():
                h = self.model.embedding.forward(x)
        else:
            with torch.no_grad():
                h = self.model.patch_embed(x)
                cls_token = self.model.cls_token.expand(h.shape[0], -1, -1)
                h = torch.cat((cls_token, h), dim=1)
                h = h + self.model.pos_embed

        # 2. Pass through blocks until target block
        for i in range(block_idx):
            self.model.blocks[i].set_mode('fp32')
            with torch.no_grad():
                h = self.model.blocks[i].forward(h)

        # 3. Target block - need gradients here
        h = h.detach().requires_grad_(True)
        self.model.blocks[block_idx].set_mode('fp32')
        h_out = self.model.blocks[block_idx].forward(h)

        # 4. Continue through remaining blocks
        h_continue = h_out
        for i in range(block_idx + 1, len(self.model.blocks)):
            self.model.blocks[i].set_mode('fp32')
            h_continue = self.model.blocks[i].forward(h_continue)

        # 5. FP32 blocks (if any)
        for block in self.model.fp32_blocks:
            h_continue = block(h_continue)

        # 6. Final norm and head
        if hasattr(self.model, 'norm'):
            if hasattr(self.model.norm, 'forward'):
                h_continue = self.model.norm.forward(h_continue)
            else:
                h_continue = self.model.norm(h_continue)

        h_continue = h_continue[:, 0]  # CLS token

        if hasattr(self.model, 'head'):
            if hasattr(self.model.head, 'forward'):
                logits = self.model.head.forward(h_continue)
            else:
                logits = self.model.head(h_continue)

        # Compute loss
        if use_labels:
            loss = self.criterion(logits, labels)
        else:
            # Use model's own predictions as pseudo-labels
            pseudo_labels = logits.argmax(dim=1)
            loss = self.criterion(logits, pseudo_labels)

        # Compute gradient w.r.t. block output
        grad_h = torch.autograd.grad(loss, h, create_graph=False)[0]

        # Fisher = E[||∂L/∂h||²] = mean(sum(grad²))
        fisher_trace = (grad_h ** 2).sum(dim=[1, 2]).mean().item()

        return fisher_trace

    def compute_block_hessian_trace(self,
                                    data_loader,
                                    num_samples: int = 50,
                                    num_hutchinson: int = 10) -> Dict[int, float]:
        """
        Hutchinson's method를 사용한 블록별 Hessian trace 추정.

        tr(H) ≈ E[v^T H v] where v ~ Rademacher distribution

        Args:
            data_loader: Calibration data loader
            num_samples: Number of data samples
            num_hutchinson: Number of Hutchinson vectors per sample

        Returns:
            Dict mapping block index to Hessian trace estimate
        """
        self.model.eval()
        hessian_dict = {i: 0.0 for i in range(len(self.model.blocks))}
        sample_count = 0

        for batch_idx, (images, labels) in enumerate(data_loader):
            if sample_count >= num_samples:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            for block_idx in range(len(self.model.blocks)):
                hessian_val = self._compute_hutchinson_trace(
                    images, labels, block_idx, num_hutchinson
                )
                hessian_dict[block_idx] += hessian_val * batch_size

            sample_count += batch_size

            if (batch_idx + 1) % 5 == 0:
                print(f"  Hessian computation: {sample_count}/{num_samples} samples")

        for block_idx in hessian_dict:
            hessian_dict[block_idx] /= sample_count

        return hessian_dict

    def _compute_hutchinson_trace(self,
                                  x: torch.Tensor,
                                  labels: torch.Tensor,
                                  block_idx: int,
                                  num_vectors: int = 10) -> float:
        """
        Hutchinson's estimator for Hessian trace.

        tr(H) ≈ (1/m) Σᵢ vᵢ^T H vᵢ
        where vᵢ ~ Rademacher({-1, +1})
        """
        traces = []

        for _ in range(num_vectors):
            self.model.zero_grad()

            # Forward to get block output
            if self.model.embedding is not None:
                self.model.embedding.set_mode('fp32')
                with torch.no_grad():
                    h = self.model.embedding.forward(x)
            else:
                with torch.no_grad():
                    h = self.model.patch_embed(x)
                    cls_token = self.model.cls_token.expand(h.shape[0], -1, -1)
                    h = torch.cat((cls_token, h), dim=1)
                    h = h + self.model.pos_embed

            for i in range(block_idx):
                self.model.blocks[i].set_mode('fp32')
                with torch.no_grad():
                    h = self.model.blocks[i].forward(h)

            h = h.detach().requires_grad_(True)
            self.model.blocks[block_idx].set_mode('fp32')
            h_out = self.model.blocks[block_idx].forward(h)

            # Continue forward
            h_continue = h_out
            for i in range(block_idx + 1, len(self.model.blocks)):
                self.model.blocks[i].set_mode('fp32')
                h_continue = self.model.blocks[i].forward(h_continue)

            for block in self.model.fp32_blocks:
                h_continue = block(h_continue)

            if hasattr(self.model, 'norm'):
                if hasattr(self.model.norm, 'forward'):
                    h_continue = self.model.norm.forward(h_continue)
                else:
                    h_continue = self.model.norm(h_continue)

            h_continue = h_continue[:, 0]

            if hasattr(self.model, 'head'):
                if hasattr(self.model.head, 'forward'):
                    logits = self.model.head.forward(h_continue)
                else:
                    logits = self.model.head(h_continue)

            loss = self.criterion(logits, labels)

            # First gradient
            grad_h = torch.autograd.grad(loss, h, create_graph=True)[0]

            # Rademacher vector
            v = torch.randint(0, 2, h.shape, device=self.device).float() * 2 - 1

            # Hessian-vector product: H @ v = ∂(∂L/∂h · v)/∂h
            grad_v = (grad_h * v).sum()
            hv = torch.autograd.grad(grad_v, h, retain_graph=False)[0]

            # v^T H v
            trace_est = (v * hv).sum().item()
            traces.append(trace_est)

        return np.mean(traces)

    def compute_output_sensitivity(self,
                                   data_loader,
                                   num_samples: int = 100) -> Dict[int, Dict[str, float]]:
        """
        블록 출력 perturbation에 대한 최종 출력 민감도.

        작은 noise δ를 블록 출력에 더했을 때 최종 출력의 변화량 측정.

        Args:
            data_loader: Data loader
            num_samples: Number of samples

        Returns:
            Dict with 'l2_sensitivity' and 'cosine_sensitivity' per block
        """
        self.model.eval()
        sensitivity_dict = {i: {'l2': 0.0, 'cosine': 0.0} for i in range(len(self.model.blocks))}
        sample_count = 0

        noise_scale = 0.01  # 1% noise

        for batch_idx, (images, _) in enumerate(data_loader):
            if sample_count >= num_samples:
                break

            images = images.to(self.device)
            batch_size = images.size(0)

            for block_idx in range(len(self.model.blocks)):
                l2_sens, cos_sens = self._compute_perturbation_sensitivity(
                    images, block_idx, noise_scale
                )
                sensitivity_dict[block_idx]['l2'] += l2_sens * batch_size
                sensitivity_dict[block_idx]['cosine'] += cos_sens * batch_size

            sample_count += batch_size

        for block_idx in sensitivity_dict:
            sensitivity_dict[block_idx]['l2'] /= sample_count
            sensitivity_dict[block_idx]['cosine'] /= sample_count

        return sensitivity_dict

    def _compute_perturbation_sensitivity(self,
                                          x: torch.Tensor,
                                          block_idx: int,
                                          noise_scale: float) -> Tuple[float, float]:
        """단일 블록에 대한 perturbation sensitivity 계산."""
        with torch.no_grad():
            # Clean forward
            if self.model.embedding is not None:
                self.model.embedding.set_mode('fp32')
                h = self.model.embedding.forward(x)
            else:
                h = self.model.patch_embed(x)
                cls_token = self.model.cls_token.expand(h.shape[0], -1, -1)
                h = torch.cat((cls_token, h), dim=1)
                h = h + self.model.pos_embed

            # Forward to target block
            for i in range(block_idx):
                self.model.blocks[i].set_mode('fp32')
                h = self.model.blocks[i].forward(h)

            # Save clean block output
            self.model.blocks[block_idx].set_mode('fp32')
            h_clean = self.model.blocks[block_idx].forward(h)

            # Add noise
            noise = torch.randn_like(h_clean) * noise_scale * h_clean.std()
            h_noisy = h_clean + noise

            # Continue with clean path
            h_continue_clean = h_clean.clone()
            h_continue_noisy = h_noisy.clone()

            for i in range(block_idx + 1, len(self.model.blocks)):
                self.model.blocks[i].set_mode('fp32')
                h_continue_clean = self.model.blocks[i].forward(h_continue_clean)
                h_continue_noisy = self.model.blocks[i].forward(h_continue_noisy)

            for block in self.model.fp32_blocks:
                h_continue_clean = block(h_continue_clean)
                h_continue_noisy = block(h_continue_noisy)

            if hasattr(self.model, 'norm'):
                if hasattr(self.model.norm, 'forward'):
                    h_continue_clean = self.model.norm.forward(h_continue_clean)
                    h_continue_noisy = self.model.norm.forward(h_continue_noisy)
                else:
                    h_continue_clean = self.model.norm(h_continue_clean)
                    h_continue_noisy = self.model.norm(h_continue_noisy)

            out_clean = h_continue_clean[:, 0]
            out_noisy = h_continue_noisy[:, 0]

            if hasattr(self.model, 'head'):
                if hasattr(self.model.head, 'forward'):
                    logits_clean = self.model.head.forward(out_clean)
                    logits_noisy = self.model.head.forward(out_noisy)
                else:
                    logits_clean = self.model.head(out_clean)
                    logits_noisy = self.model.head(out_noisy)

            # L2 sensitivity: ||output_noisy - output_clean|| / ||noise||
            output_diff = (logits_noisy - logits_clean).norm(dim=1).mean()
            noise_norm = noise.norm() / np.sqrt(noise.numel())
            l2_sens = (output_diff / (noise_norm + 1e-10)).item()

            # Cosine sensitivity: 1 - cos_sim(output_noisy, output_clean)
            cos_sim = F.cosine_similarity(
                logits_clean.flatten(),
                logits_noisy.flatten(),
                dim=0
            ).item()
            cos_sens = 1 - cos_sim

            return l2_sens, cos_sens

    def get_sensitivity_summary(self,
                                data_loader,
                                num_samples: int = 100,
                                method: str = 'fisher') -> Dict[int, float]:
        """
        블록별 민감도 요약.

        Args:
            data_loader: Data loader
            num_samples: Number of samples
            method: 'fisher', 'hessian', or 'perturbation'

        Returns:
            Dict mapping block index to sensitivity score
        """
        print(f"\n[Sensitivity Analysis] Method: {method}")
        print("-" * 50)

        if method == 'fisher':
            result = self.compute_block_fisher(data_loader, num_samples)
        elif method == 'hessian':
            result = self.compute_block_hessian_trace(data_loader, num_samples)
        elif method == 'perturbation':
            pert_result = self.compute_output_sensitivity(data_loader, num_samples)
            result = {i: v['l2'] for i, v in pert_result.items()}
        else:
            raise ValueError(f"Unknown method: {method}")

        return result


def analyze_block_sensitivity(qvit, data_loader, num_samples: int = 100):
    """
    QVit 블록별 민감도 분석 utility 함수.

    Args:
        qvit: QVit model
        data_loader: Calibration data loader
        num_samples: Number of samples for analysis

    Returns:
        Dict with Fisher, perturbation sensitivity results
    """
    analyzer = HessianAnalyzer(qvit)

    results = {}

    # 1. Fisher Information
    print("\n[1] Computing Fisher Information...")
    results['fisher'] = analyzer.compute_block_fisher(data_loader, num_samples, use_labels=False)

    # 2. Perturbation Sensitivity
    print("\n[2] Computing Perturbation Sensitivity...")
    results['perturbation'] = analyzer.compute_output_sensitivity(data_loader, num_samples)

    # 3. Summary
    print("\n" + "=" * 70)
    print("Block Sensitivity Summary")
    print("=" * 70)
    print(f"{'Block':<8} {'Fisher':>15} {'L2 Sens':>15} {'Cos Sens':>15}")
    print("-" * 70)

    for i in range(len(qvit.blocks)):
        fisher = results['fisher'][i]
        l2_sens = results['perturbation'][i]['l2']
        cos_sens = results['perturbation'][i]['cosine']
        print(f"Block {i:<3} {fisher:>15.6f} {l2_sens:>15.6f} {cos_sens:>15.8f}")

    print("=" * 70)

    # Rank blocks by sensitivity
    fisher_sorted = sorted(results['fisher'].items(), key=lambda x: x[1], reverse=True)
    print("\n[Most Sensitive Blocks (Fisher)]")
    for rank, (block_idx, score) in enumerate(fisher_sorted[:3], 1):
        print(f"  {rank}. Block {block_idx}: {score:.6f}")

    return results
