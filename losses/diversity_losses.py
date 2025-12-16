"""
MBEE 다양성 손실 함수

1. NCL (Negative Correlation Learning): Branch 예측 간 상관 최소화
2. OR (Orthogonality Regularization): Classifier 가중치 직교성
3. FDL (Feature Decorrelation Loss): Branch features 간 상관 최소화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class NegativeCorrelationLoss(nn.Module):
    """
    Negative Correlation Learning Loss
    
    Branch 간 예측(softmax outputs)의 상관을 최소화
    L_NCL = sum_{i≠j} corr(p_i, p_j)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, branch_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            branch_probs: [B, K, C] 각 branch의 softmax 출력
        
        Returns:
            NCL loss (scalar)
        """
        B, K, C = branch_probs.shape
        
        # Flatten to [B, K, C] -> [K, B*C]
        probs_flat = branch_probs.permute(1, 0, 2).reshape(K, -1)  # [K, B*C]
        
        # 각 branch의 mean 제거 (correlation 계산 위해)
        probs_centered = probs_flat - probs_flat.mean(dim=1, keepdim=True)
        
        # Correlation matrix [K, K]
        # corr(i,j) = cov(i,j) / (std(i) * std(j))
        cov_matrix = torch.mm(probs_centered, probs_centered.t()) / (B * C - 1)
        std = probs_centered.std(dim=1, keepdim=True) + 1e-8
        corr_matrix = cov_matrix / (std @ std.t())
        
        # 대각선 제외한 상관의 합 (off-diagonal)
        mask = 1 - torch.eye(K, device=branch_probs.device)
        ncl_loss = (corr_matrix.abs() * mask).sum() / (K * (K - 1))
        
        return ncl_loss


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality Regularization Loss
    
    각 branch의 classifier weights가 직교하도록 유도
    L_OR = sum_{i≠j} ||W_i^T W_j||_F^2
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, classifier_weights: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            classifier_weights: List of [C, D] weight matrices for each branch
        
        Returns:
            OR loss (scalar)
        """
        K = len(classifier_weights)
        
        total_loss = 0.0
        count = 0
        
        for i in range(K):
            for j in range(i + 1, K):
                # W_i^T @ W_j
                Wi = classifier_weights[i]  # [C, D]
                Wj = classifier_weights[j]  # [C, D]
                
                # Frobenius norm of W_i^T @ W_j
                cross = torch.mm(Wi, Wj.t())  # [C, C]
                loss = (cross ** 2).sum()
                
                total_loss += loss
                count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        return total_loss


class FeatureDecorrelationLoss(nn.Module):
    """
    Feature Decorrelation Loss
    
    Branch별 feature embeddings 간 상관 최소화
    L_FDL = sum_{i≠j} ||Cov(F_i, F_j)||_F^2
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, branch_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            branch_features: [B, K, D] 각 branch의 feature
        
        Returns:
            FDL loss (scalar)
        """
        B, K, D = branch_features.shape
        
        # Center features
        features_centered = branch_features - branch_features.mean(dim=0, keepdim=True)
        
        total_loss = 0.0
        count = 0
        
        for i in range(K):
            for j in range(i + 1, K):
                Fi = features_centered[:, i, :]  # [B, D]
                Fj = features_centered[:, j, :]  # [B, D]
                
                # Cross-covariance matrix
                cov = torch.mm(Fi.t(), Fj) / (B - 1)  # [D, D]
                
                # Frobenius norm
                loss = (cov ** 2).sum()
                
                total_loss += loss
                count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        # Normalize by dimension
        total_loss = total_loss / (D * D)
        
        return total_loss


class DiversityLoss(nn.Module):
    """
    통합 다양성 손실
    
    L_div = λ_ncl * L_NCL + λ_or * L_OR + λ_fdl * L_FDL
    """
    def __init__(
        self,
        lambda_ncl: float = 0.1,
        lambda_or: float = 0.01,
        lambda_fdl: float = 0.05
    ):
        super().__init__()
        self.lambda_ncl = lambda_ncl
        self.lambda_or = lambda_or
        self.lambda_fdl = lambda_fdl
        
        self.ncl_loss = NegativeCorrelationLoss()
        self.or_loss = OrthogonalityLoss()
        self.fdl_loss = FeatureDecorrelationLoss()
    
    def forward(
        self,
        branch_probs: torch.Tensor,
        classifier_weights: List[torch.Tensor],
        branch_features: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            branch_probs: [B, K, C] softmax outputs
            classifier_weights: List of [C, D] weight matrices
            branch_features: [B, K, D] branch features (optional)
        
        Returns:
            Dict with 'total', 'ncl', 'or', 'fdl' losses
        """
        losses = {}
        total = 0.0
        
        # NCL Loss
        if self.lambda_ncl > 0:
            ncl = self.ncl_loss(branch_probs)
            losses['ncl'] = ncl
            total += self.lambda_ncl * ncl
        else:
            losses['ncl'] = torch.tensor(0.0)
        
        # OR Loss
        if self.lambda_or > 0:
            or_loss = self.or_loss(classifier_weights)
            losses['or'] = or_loss
            total += self.lambda_or * or_loss
        else:
            losses['or'] = torch.tensor(0.0)
        
        # FDL Loss
        if self.lambda_fdl > 0 and branch_features is not None:
            fdl = self.fdl_loss(branch_features)
            losses['fdl'] = fdl
            total += self.lambda_fdl * fdl
        else:
            losses['fdl'] = torch.tensor(0.0)
        
        losses['total'] = total
        
        return losses


class MBEELoss(nn.Module):
    """
    MBEE 전체 손실 함수
    
    L = L_cls + L_div
    
    - L_cls: 각 branch의 분류 손실 평균
    - L_div: 다양성 손실
    """
    def __init__(
        self,
        lambda_ncl: float = 0.1,
        lambda_or: float = 0.01,
        lambda_fdl: float = 0.05
    ):
        super().__init__()
        self.diversity_loss = DiversityLoss(lambda_ncl, lambda_or, lambda_fdl)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        classifier_weights: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: MBEE model outputs
            targets: [B] ground truth labels
            classifier_weights: List of classifier weight matrices
        
        Returns:
            Dict with all losses
        """
        branch_logits = outputs['logits']  # [B, K, C]
        branch_probs = outputs['probs']  # [B, K, C]
        B, K, C = branch_logits.shape
        
        # 1. Classification Loss (각 branch 평균)
        cls_loss = 0.0
        for k in range(K):
            cls_loss += self.ce_loss(branch_logits[:, k, :], targets)
        cls_loss = cls_loss / K
        
        # 2. Diversity Loss
        branch_features = outputs.get('branch_features', None)
        div_losses = self.diversity_loss(
            branch_probs,
            classifier_weights,
            branch_features
        )
        
        # 3. Total Loss
        total_loss = cls_loss + div_losses['total']
        
        return {
            'total': total_loss,
            'cls': cls_loss,
            'div_total': div_losses['total'],
            'ncl': div_losses['ncl'],
            'or': div_losses['or'],
            'fdl': div_losses['fdl']
        }


if __name__ == "__main__":
    # 손실 함수 테스트
    print("Testing diversity losses...")
    
    B, K, C, D = 8, 4, 10, 512
    
    # Dummy data
    branch_probs = F.softmax(torch.randn(B, K, C), dim=-1)
    classifier_weights = [torch.randn(C, D) for _ in range(K)]
    branch_features = torch.randn(B, K, D)
    
    # NCL Loss
    ncl = NegativeCorrelationLoss()
    ncl_val = ncl(branch_probs)
    print(f"NCL Loss: {ncl_val.item():.4f}")
    
    # OR Loss
    or_loss = OrthogonalityLoss()
    or_val = or_loss(classifier_weights)
    print(f"OR Loss: {or_val.item():.4f}")
    
    # FDL Loss
    fdl = FeatureDecorrelationLoss()
    fdl_val = fdl(branch_features)
    print(f"FDL Loss: {fdl_val.item():.4f}")
    
    # Combined Diversity Loss
    div_loss = DiversityLoss(lambda_ncl=0.1, lambda_or=0.01, lambda_fdl=0.05)
    div_losses = div_loss(branch_probs, classifier_weights, branch_features)
    print(f"\nCombined Diversity Loss: {div_losses['total'].item():.4f}")
    
    # MBEE Loss
    print("\nTesting MBEE Loss...")
    mbee_loss = MBEELoss(lambda_ncl=0.1, lambda_or=0.01, lambda_fdl=0.05)
    
    outputs = {
        'logits': torch.randn(B, K, C),
        'probs': branch_probs,
        'branch_features': branch_features
    }
    targets = torch.randint(0, C, (B,))
    
    losses = mbee_loss(outputs, targets, classifier_weights)
    print(f"Total: {losses['total'].item():.4f}")
    print(f"Classification: {losses['cls'].item():.4f}")
    print(f"Diversity: {losses['div_total'].item():.4f}")
