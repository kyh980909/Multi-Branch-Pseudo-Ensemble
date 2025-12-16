"""
MBEE 실험 기본 설정
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """데이터 관련 설정"""
    dataset: str = "cifar10"  # cifar10, cifar100
    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    
    # OOD 데이터셋
    ood_datasets: List[str] = field(default_factory=lambda: ["svhn", "lsun", "tiny_imagenet"])


@dataclass
class ModelConfig:
    """모델 관련 설정"""
    # Backbone
    backbone: str = "resnet18"  # resnet18, resnet34, resnet50
    num_classes: int = 10
    pretrained: bool = False
    
    # MBEE 설정
    num_branches: int = 4  # K: branch 개수
    branch_hidden_dim: int = 256
    
    # Diversity Injection
    use_branch_dropout: bool = True
    branch_dropout_rate: float = 0.1
    use_feature_perturbation: bool = True
    perturbation_dim: int = 128
    use_lowrank_projection: bool = True
    projection_rank: int = 64


@dataclass
class LossConfig:
    """손실 함수 관련 설정"""
    # 기본 분류 손실
    classification_loss: str = "cross_entropy"
    
    # 다양성 손실 가중치
    lambda_ncl: float = 0.1      # Negative Correlation Learning
    lambda_or: float = 0.01      # Orthogonality Regularization
    lambda_fdl: float = 0.05     # Feature Decorrelation Loss
    
    # EDL 설정 (비교용)
    edl_lambda: float = 0.1      # EDL KL regularization weight


@dataclass
class TrainConfig:
    """학습 관련 설정"""
    epochs: int = 100
    optimizer: str = "adamw"  # adam, adamw, sgd
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # cosine, step, none
    warmup_epochs: int = 5
    
    # 기타
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "./checkpoints"
    log_interval: int = 100


@dataclass
class EvalConfig:
    """평가 관련 설정"""
    # Uncertainty threshold for OOD detection
    uncertainty_threshold: float = 0.5
    
    # Metrics to compute
    compute_auroc: bool = True
    compute_aupr: bool = True
    compute_fpr95: bool = True
    compute_ece: bool = True


@dataclass
class Config:
    """전체 설정"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # 실험 이름
    exp_name: str = "mbee_cifar10"
    
    def __post_init__(self):
        """CIFAR 데이터셋에 맞게 num_classes 자동 설정"""
        if self.data.dataset == "cifar10":
            self.model.num_classes = 10
        elif self.data.dataset == "cifar100":
            self.model.num_classes = 100


def get_config(exp_type: str = "mbee") -> Config:
    """실험 타입에 따른 설정 반환"""
    config = Config()
    
    if exp_type == "mbee":
        config.exp_name = "mbee_cifar10"
    elif exp_type == "edl":
        config.exp_name = "edl_cifar10"
        # EDL은 다양성 손실 사용 안함
        config.loss.lambda_ncl = 0.0
        config.loss.lambda_or = 0.0
        config.loss.lambda_fdl = 0.0
    elif exp_type == "mbee_no_diversity":
        config.exp_name = "mbee_no_diversity"
        config.loss.lambda_ncl = 0.0
        config.loss.lambda_or = 0.0
        config.loss.lambda_fdl = 0.0
    elif exp_type == "deep_ensemble":
        config.exp_name = "deep_ensemble"
        config.model.num_branches = 1  # 개별 모델 학습
    
    return config


if __name__ == "__main__":
    # 설정 테스트
    config = get_config("mbee")
    print(f"Experiment: {config.exp_name}")
    print(f"Backbone: {config.model.backbone}")
    print(f"Num branches: {config.model.num_branches}")
    print(f"Diversity losses: NCL={config.loss.lambda_ncl}, OR={config.loss.lambda_or}, FDL={config.loss.lambda_fdl}")
