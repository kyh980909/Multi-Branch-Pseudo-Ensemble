"""
MBEE: Multi-Branch Epistemic Ensemble

핵심 구조:
1. Shared Backbone (ResNet)
2. Diversity Injection Module (Dropout, Perturbation, Projection)
3. K개의 병렬 Classification Branches
4. Aggregation Head (Variance 기반 Epistemic Uncertainty)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple, Optional, List


class DiversityInjectionModule(nn.Module):
    """
    다양성 주입 모듈
    - Branch별 Dropout (다른 seed)
    - Learnable Feature Perturbation
    - Low-rank Projection
    """
    def __init__(
        self,
        in_features: int,
        num_branches: int,
        dropout_rate: float = 0.1,
        use_perturbation: bool = True,
        perturbation_dim: int = 128,
        use_projection: bool = True,
        projection_rank: int = 64
    ):
        super().__init__()
        self.num_branches = num_branches
        self.in_features = in_features
        
        # Branch별 Dropout
        self.dropouts = nn.ModuleList([
            nn.Dropout(p=dropout_rate) for _ in range(num_branches)
        ])
        
        # Learnable Feature Perturbation (branch별 작은 MLP)
        self.use_perturbation = use_perturbation
        if use_perturbation:
            self.perturbation_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_features, perturbation_dim),
                    nn.ReLU(),
                    nn.Linear(perturbation_dim, in_features),
                    nn.Tanh()  # [-1, 1] 범위의 perturbation
                ) for _ in range(num_branches)
            ])
            self.perturbation_scale = nn.Parameter(torch.ones(num_branches) * 0.1)
        
        # Low-rank Projection
        self.use_projection = use_projection
        if use_projection:
            self.projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_features, projection_rank),
                    nn.ReLU(),
                    nn.Linear(projection_rank, in_features)
                ) for _ in range(num_branches)
            ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, in_features] shared backbone output
        
        Returns:
            List of [B, in_features] for each branch
        """
        branch_features = []
        
        for k in range(self.num_branches):
            feat = x.clone()
            
            # 1. Branch-specific Dropout
            feat = self.dropouts[k](feat)
            
            # 2. Learnable Perturbation
            if self.use_perturbation:
                perturbation = self.perturbation_nets[k](x)
                feat = feat + self.perturbation_scale[k] * perturbation
            
            # 3. Low-rank Projection
            if self.use_projection:
                proj = self.projections[k](x)
                feat = feat + proj
            
            branch_features.append(feat)
        
        return branch_features


class BranchHead(nn.Module):
    """개별 분류 Branch"""
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, num_classes]"""
        return self.classifier(x)


class MBEE(nn.Module):
    """
    Multi-Branch Epistemic Ensemble
    
    구조:
        Input -> Backbone -> Diversity Injection -> K Branches -> Aggregation
    """
    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 10,
        num_branches: int = 4,
        branch_hidden_dim: int = 256,
        dropout_rate: float = 0.1,
        use_perturbation: bool = True,
        use_projection: bool = True,
        pretrained: bool = False
    ):
        super().__init__()
        self.num_branches = num_branches
        self.num_classes = num_classes
        
        # 1. Shared Backbone
        self.backbone, self.feature_dim = self._create_backbone(backbone, pretrained)
        
        # 2. Diversity Injection Module
        self.diversity_module = DiversityInjectionModule(
            in_features=self.feature_dim,
            num_branches=num_branches,
            dropout_rate=dropout_rate,
            use_perturbation=use_perturbation,
            use_projection=use_projection
        )
        
        # 3. K Branch Heads
        self.branch_heads = nn.ModuleList([
            BranchHead(self.feature_dim, branch_hidden_dim, num_classes)
            for _ in range(num_branches)
        ])
    
    def _create_backbone(self, backbone: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """Backbone 생성 (마지막 FC layer 제거)"""
        if backbone == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif backbone == "resnet34":
            model = models.resnet34(pretrained=pretrained)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif backbone == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        return model, feature_dim
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] input images
            return_features: branch features도 반환할지
        
        Returns:
            Dict with:
                - 'logits': [B, K, num_classes] 각 branch의 logits
                - 'probs': [B, K, num_classes] 각 branch의 softmax
                - 'mean_probs': [B, num_classes] 평균 확률
                - 'variance': [B, num_classes] 분기 간 분산 (epistemic)
                - 'epistemic': [B] 전체 epistemic uncertainty (분산의 평균)
                - 'prediction': [B] 최종 예측 클래스
        """
        batch_size = x.size(0)
        
        # 1. Shared Backbone
        shared_features = self.backbone(x)  # [B, feature_dim]
        
        # 2. Diversity Injection
        branch_features = self.diversity_module(shared_features)  # List of [B, feature_dim]
        
        # 3. Branch별 Classification
        branch_logits = []
        for k in range(self.num_branches):
            logits = self.branch_heads[k](branch_features[k])  # [B, num_classes]
            branch_logits.append(logits)
        
        # Stack: [B, K, num_classes]
        branch_logits = torch.stack(branch_logits, dim=1)
        branch_probs = F.softmax(branch_logits, dim=-1)
        
        # 4. Aggregation
        mean_probs = branch_probs.mean(dim=1)  # [B, num_classes]
        variance = branch_probs.var(dim=1)  # [B, num_classes]
        
        # Epistemic uncertainty: 전체 분산의 평균
        epistemic = variance.mean(dim=-1)  # [B]
        
        # 최종 예측
        prediction = mean_probs.argmax(dim=-1)  # [B]
        
        outputs = {
            'logits': branch_logits,
            'probs': branch_probs,
            'mean_probs': mean_probs,
            'variance': variance,
            'epistemic': epistemic,
            'prediction': prediction
        }
        
        if return_features:
            outputs['branch_features'] = torch.stack(branch_features, dim=1)
            outputs['shared_features'] = shared_features
        
        return outputs
    
    def get_classifier_weights(self) -> List[torch.Tensor]:
        """각 branch의 classifier weights 반환 (Orthogonality Loss용)"""
        weights = []
        for branch in self.branch_heads:
            # classifier의 마지막 linear layer의 weight
            last_layer = branch.classifier[-1]
            weights.append(last_layer.weight)
        return weights
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        예측 + 불확실성 반환
        
        Returns:
            predictions: [B] 예측 클래스
            confidence: [B] 평균 확률의 최댓값
            epistemic: [B] epistemic uncertainty
        """
        outputs = self.forward(x)
        predictions = outputs['prediction']
        confidence = outputs['mean_probs'].max(dim=-1)[0]
        epistemic = outputs['epistemic']
        
        return predictions, confidence, epistemic


class MBEEForCIFAR(MBEE):
    """CIFAR-10/100용 MBEE (32x32 입력 최적화)"""
    def __init__(
        self,
        num_classes: int = 10,
        num_branches: int = 4,
        **kwargs
    ):
        # CIFAR용 작은 backbone
        super().__init__(
            backbone="resnet18",
            num_classes=num_classes,
            num_branches=num_branches,
            branch_hidden_dim=256,
            **kwargs
        )
        
        # ResNet 첫 번째 conv를 CIFAR용으로 수정
        # 7x7 -> 3x3, stride 2 -> 1, maxpool 제거
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()


if __name__ == "__main__":
    # 모델 테스트
    print("Testing MBEE model...")
    
    model = MBEEForCIFAR(num_classes=10, num_branches=4)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass 테스트
    x = torch.randn(8, 3, 32, 32)
    outputs = model(x, return_features=True)
    
    print(f"\nOutput shapes:")
    print(f"  logits: {outputs['logits'].shape}")
    print(f"  probs: {outputs['probs'].shape}")
    print(f"  mean_probs: {outputs['mean_probs'].shape}")
    print(f"  variance: {outputs['variance'].shape}")
    print(f"  epistemic: {outputs['epistemic'].shape}")
    print(f"  prediction: {outputs['prediction'].shape}")
    print(f"  branch_features: {outputs['branch_features'].shape}")
    
    # Uncertainty 확인
    preds, conf, epist = model.predict_with_uncertainty(x)
    print(f"\nPredictions: {preds}")
    print(f"Confidence: {conf}")
    print(f"Epistemic: {epist}")
    
    # Classifier weights
    weights = model.get_classifier_weights()
    print(f"\nClassifier weights shapes: {[w.shape for w in weights]}")
