"""
EDL: Evidential Deep Learning

Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018

핵심:
- Softmax 대신 Dirichlet distribution 파라미터 (evidence) 예측
- Epistemic uncertainty = K / sum(alpha) where alpha = evidence + 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple


class EDL(nn.Module):
    """
    Evidential Deep Learning for Classification
    
    Evidence를 예측하고, Dirichlet distribution으로 uncertainty 추정
    """
    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 10,
        pretrained: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone
        self.backbone, self.feature_dim = self._create_backbone(backbone, pretrained)
        
        # Evidence 예측 head (softplus로 양수 보장)
        self.evidence_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.Softplus(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def _create_backbone(self, backbone: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """Backbone 생성"""
        if backbone == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif backbone == "resnet34":
            model = models.resnet34(pretrained=pretrained)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        return model, feature_dim
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] input images
        
        Returns:
            Dict with:
                - 'evidence': [B, num_classes] non-negative evidence
                - 'alpha': [B, num_classes] Dirichlet parameters (evidence + 1)
                - 'probs': [B, num_classes] expected probability (alpha / S)
                - 'epistemic': [B] epistemic uncertainty (K / S)
                - 'aleatoric': [B] aleatoric uncertainty (entropy of expected prob)
                - 'prediction': [B] predicted class
        """
        # Backbone features
        features = self.backbone(x)  # [B, feature_dim]
        
        # Evidence (non-negative)
        evidence = F.softplus(self.evidence_head(features))  # [B, K]
        
        # Dirichlet parameters
        alpha = evidence + 1  # [B, K]
        S = alpha.sum(dim=-1, keepdim=True)  # [B, 1]
        
        # Expected probability
        probs = alpha / S  # [B, K]
        
        # Epistemic uncertainty: K / S (낮을수록 confident)
        # 또는 1 - max(alpha)/S 로도 계산 가능
        epistemic = self.num_classes / S.squeeze(-1)  # [B]
        
        # Aleatoric uncertainty: entropy of expected probability
        aleatoric = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B]
        
        # Prediction
        prediction = probs.argmax(dim=-1)  # [B]
        
        return {
            'evidence': evidence,
            'alpha': alpha,
            'probs': probs,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'prediction': prediction,
            'S': S.squeeze(-1)  # Dirichlet strength
        }
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        예측 + 불확실성 반환
        
        Returns:
            predictions: [B] 예측 클래스
            confidence: [B] 예측 확률
            epistemic: [B] epistemic uncertainty
        """
        outputs = self.forward(x)
        predictions = outputs['prediction']
        confidence = outputs['probs'].max(dim=-1)[0]
        epistemic = outputs['epistemic']
        
        return predictions, confidence, epistemic


class EDLLoss(nn.Module):
    """
    EDL 손실 함수
    
    L = L_mse + λ * L_kl
    
    - L_mse: Mean squared error for correct class
    - L_kl: KL divergence regularization
    """
    def __init__(self, num_classes: int = 10, lambda_kl: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_kl = lambda_kl
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: torch.Tensor,
        epoch: int = 0,
        total_epochs: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: EDL model outputs
            targets: [B] ground truth labels
            epoch: current epoch (for annealing)
            total_epochs: total training epochs
        
        Returns:
            Dict with 'total', 'mse', 'kl' losses
        """
        evidence = outputs['evidence']
        alpha = outputs['alpha']
        S = outputs['S']
        
        # One-hot encoding
        one_hot = F.one_hot(targets, self.num_classes).float()
        
        # MSE Loss
        probs = alpha / S.unsqueeze(-1)
        mse_loss = ((one_hot - probs) ** 2).sum(dim=-1).mean()
        
        # KL Divergence regularization
        # KL(Dir(alpha) || Dir(1, ..., 1))
        # Annealing: gradually increase lambda
        annealing_coef = min(1.0, epoch / (total_epochs * 0.5)) # Annealing coefficient
        
        # KL divergence for all classes
        kl_alpha = (alpha - 1) * (1 - one_hot) + 1
        kl_div = self._kl_divergence(kl_alpha)
        
        total_loss = mse_loss + self.lambda_kl * annealing_coef * kl_div
        
        return {
            'total': total_loss,
            'mse': mse_loss,
            'kl': kl_div
        }
    
    def _kl_divergence(self, alpha: torch.Tensor) -> torch.Tensor:
        """KL divergence for incorrect classes."""
        beta = torch.ones((1, self.num_classes), dtype=torch.float, device=alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl.mean()


class EDLForCIFAR(EDL):
    """CIFAR-10/100용 EDL"""
    def __init__(
        self,
        num_classes: int = 10,
        **kwargs
    ):
        super().__init__(
            backbone="resnet18",
            num_classes=num_classes,
            **kwargs
        )
        
        # CIFAR용 ResNet 수정
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()


if __name__ == "__main__":
    # EDL 모델 테스트
    print("Testing EDL model...")
    
    model = EDLForCIFAR(num_classes=10)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass 테스트
    x = torch.randn(8, 3, 32, 32)
    outputs = model(x)
    
    print(f"\nOutput shapes:")
    print(f"  evidence: {outputs['evidence'].shape}")
    print(f"  alpha: {outputs['alpha'].shape}")
    print(f"  probs: {outputs['probs'].shape}")
    print(f"  epistemic: {outputs['epistemic'].shape}")
    print(f"  aleatoric: {outputs['aleatoric'].shape}")
    
    # 손실 함수 테스트
    criterion = EDLLoss(num_classes=10, lambda_kl=0.1)
    targets = torch.randint(0, 10, (8,))
    losses = criterion(outputs, targets, epoch=50, total_epochs=100)
    
    print(f"\nLosses:")
    print(f"  total: {losses['total'].item():.4f}")
    print(f"  mse: {losses['mse'].item():.4f}")
    print(f"  kl: {losses['kl'].item():.4f}")
