"""
MBPE 학습 스크립트

사용법:
    python train.py --model mbpe --epochs 100
    python train.py --model edl --epochs 100
    python train.py --model mbpe --no_diversity  # 소거 실험

    # λ sensitivity 실험
    python train.py --model edl --edl_lambda 0.1
    python train.py --model mbpe --lambda_ncl 0.2 --lambda_or 0.1 --lambda_fdl 0.1

    # wandb는 설치되어 있으면 자동으로 사용됩니다
    # 프로젝트명/entity/run name 커스터마이징:
    python train.py --model mbpe --wandb_project my-project --wandb_entity my-team
"""

import os
import sys
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
try:
    import wandb
except ImportError:
    wandb = None

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import Config, get_config
from data.datasets import get_cifar10_loaders
from models.mbee import MBEEForCIFAR
from models.edl import EDLForCIFAR, EDLLoss
from losses.diversity_losses import MBEELoss


def set_seed(seed: int):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config: Config, model_type: str):
    """모델 생성"""
    if model_type == "mbpe":
        model = MBEEForCIFAR(
            num_classes=config.model.num_classes,
            num_branches=config.model.num_branches,
            dropout_rate=config.model.branch_dropout_rate,
            use_perturbation=config.model.use_feature_perturbation,
            use_projection=config.model.use_lowrank_projection
        )
    elif model_type == "edl":
        model = EDLForCIFAR(num_classes=config.model.num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_criterion(config: Config, model_type: str):
    """손실 함수 생성"""
    if model_type == "mbpe":
        criterion = MBEELoss(
            lambda_ncl=config.loss.lambda_ncl,
            lambda_or=config.loss.lambda_or,
            lambda_fdl=config.loss.lambda_fdl
        )
    elif model_type == "edl":
        criterion = EDLLoss(
            num_classes=config.model.num_classes,
            lambda_kl=config.loss.edl_lambda
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return criterion


def create_optimizer(model: nn.Module, config: Config):
    """옵티마이저 생성"""
    if config.train.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay
        )
    elif config.train.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay
        )
    elif config.train.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.train.learning_rate,
            momentum=0.9,
            weight_decay=config.train.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.train.optimizer}")
    
    return optimizer


def create_scheduler(optimizer, config: Config, steps_per_epoch: int):
    """Learning rate scheduler 생성"""
    total_steps = config.train.epochs * steps_per_epoch
    warmup_steps = config.train.warmup_epochs * steps_per_epoch
    
    if config.train.scheduler == "cosine":
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=1e-6
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=1e-6
            )
    else:
        scheduler = None
    
    return scheduler


def train_one_epoch_mbpe(
    model: nn.Module,
    train_loader,
    criterion: MBEELoss,
    optimizer,
    scheduler,
    device: str,
    epoch: int
):
    """MBPE 한 에폭 학습"""
    model.train()
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_div_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(images, return_features=True)
        classifier_weights = model.get_classifier_weights()
        
        # Loss
        losses = criterion(outputs, targets, classifier_weights)
        loss = losses['total']
        
        # Backward
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        total_cls_loss += losses['cls'].item()
        total_div_loss += losses['div_total'].item()
        
        pred = outputs['prediction']
        correct += (pred == targets).sum().item()
        total += targets.size(0)
        
        # Progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'cls_loss': total_cls_loss / len(train_loader),
        'div_loss': total_div_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }


def convert_to_serializable(obj):
    """JSON 직렬화 가능한 형식으로 변환"""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj


def init_wandb(args, config: Config):
    """Initialize Weights & Biases automatically if installed"""
    if wandb is None:
        print("wandb is not installed. Skipping logging to W&B.")
        return None

    # wandb가 설치되어 있으면 자동으로 사용
    run_name = args.wandb_run_name or config.exp_name
    project = args.wandb_project or f"mbpe-{args.model}"
    init_config = {
        'model': args.model,
        'epochs': config.train.epochs,
        'batch_size': config.data.batch_size,
        'learning_rate': config.train.learning_rate,
        'optimizer': config.train.optimizer,
        'scheduler': config.train.scheduler,
        'num_branches': getattr(config.model, "num_branches", None),
        'lambda_ncl': getattr(config.loss, "lambda_ncl", None),
        'lambda_or': getattr(config.loss, "lambda_or", None),
        'lambda_fdl': getattr(config.loss, "lambda_fdl", None),
        'edl_lambda': getattr(config.loss, "edl_lambda", None),
        'seed': config.train.seed
    }

    try:
        return wandb.init(
            project=project,
            entity=args.wandb_entity or None,
            name=run_name,
            config=convert_to_serializable(init_config)
        )
    except Exception as e:
        print(f"Failed to initialize wandb: {e}")
        print("Continuing without W&B logging...")
        return None


def train_one_epoch_edl(
    model: nn.Module,
    train_loader,
    criterion: EDLLoss,
    optimizer,
    scheduler,
    device: str,
    epoch: int,
    total_epochs: int
):
    """EDL 한 에폭 학습"""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(images)
        
        # Loss
        losses = criterion(outputs, targets, epoch=epoch, total_epochs=total_epochs)
        loss = losses['total']
        
        # Backward
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        
        pred = outputs['prediction']
        correct += (pred == targets).sum().item()
        total += targets.size(0)
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, device: str, model_type: str):
    """검증 평가"""
    model.eval()
    
    correct = 0
    total = 0
    all_uncertainties = []
    all_confidences = []
    
    for images, targets in val_loader:
        images, targets = images.to(device), targets.to(device)
        
        if model_type == "mbpe":
            outputs = model(images)
            pred = outputs['prediction']
            uncertainty = outputs['epistemic']
            confidence = outputs['mean_probs'].max(dim=-1)[0]
        else:  # edl
            outputs = model(images)
            pred = outputs['prediction']
            uncertainty = outputs['epistemic']
            confidence = outputs['probs'].max(dim=-1)[0]
        
        correct += (pred == targets).sum().item()
        total += targets.size(0)
        
        all_uncertainties.append(uncertainty.cpu())
        all_confidences.append(confidence.cpu())
    
    all_uncertainties = torch.cat(all_uncertainties).numpy()
    all_confidences = torch.cat(all_confidences).numpy()
    
    return {
        'accuracy': 100. * correct / total,
        'mean_uncertainty': all_uncertainties.mean(),
        'mean_confidence': all_confidences.mean()
    }


def train(args):
    """전체 학습 루프"""
    # Config
    if args.no_diversity:
        config = get_config("mbpe_no_diversity")
    else:
        config = get_config(args.model)
    
    config.train.epochs = args.epochs
    
    # λ 값 오버라이드 (커맨드라인에서 지정한 경우)
    if args.model == "edl" and args.edl_lambda is not None:
        config.loss.edl_lambda = args.edl_lambda
        lambda_str = f"_lambda{args.edl_lambda}"
    elif args.model == "mbpe":
        lambda_str = ""
        if args.lambda_ncl is not None:
            config.loss.lambda_ncl = args.lambda_ncl
            lambda_str += f"_ncl{args.lambda_ncl}"
        if args.lambda_or is not None:
            config.loss.lambda_or = args.lambda_or
            lambda_str += f"_or{args.lambda_or}"
        if args.lambda_fdl is not None:
            config.loss.lambda_fdl = args.lambda_fdl
            lambda_str += f"_fdl{args.lambda_fdl}"
    else:
        lambda_str = ""
    
    # 실험 이름에 λ 값 포함
    config.exp_name = f"{args.model}{lambda_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Seed
    set_seed(config.train.seed)
    
    # Device
    device = config.train.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 현재 λ 설정 출력
    print("\n" + "="*60)
    print("HYPERPARAMETER SETTINGS")
    print("="*60)
    if args.model == "edl":
        print(f"  EDL λ (KL weight): {config.loss.edl_lambda}")
    else:
        print(f"  λ_NCL: {config.loss.lambda_ncl}")
        print(f"  λ_OR:  {config.loss.lambda_or}")
        print(f"  λ_FDL: {config.loss.lambda_fdl}")
    print("="*60 + "\n")
    
    # Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_root=config.data.data_root,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    print(f"Creating {args.model} model...")
    model = create_model(config, args.model)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    wandb_run = init_wandb(args, config)
    if wandb_run is not None:
        wandb.watch(model, log="gradients", log_freq=100)
    
    # Loss
    criterion = create_criterion(config, args.model)
    
    # Optimizer
    optimizer = create_optimizer(model, config)
    
    # Scheduler
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    
    # Training
    print(f"\nStarting training for {config.train.epochs} epochs...")
    
    save_dir = os.path.join(config.train.save_dir, config.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config (λ 값 포함)
    config_to_save = {
        'model': args.model,
        'epochs': config.train.epochs,
        'lambda_ncl': config.loss.lambda_ncl,
        'lambda_or': config.loss.lambda_or,
        'lambda_fdl': config.loss.lambda_fdl,
        'edl_lambda': config.loss.edl_lambda,
        'num_branches': config.model.num_branches,
        'no_diversity': args.no_diversity
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_to_save, f, indent=2)
    
    best_acc = 0.0
    history = []
    
    for epoch in range(1, config.train.epochs + 1):
        # Train
        if args.model == "mbpe":
            train_metrics = train_one_epoch_mbpe(
                model, train_loader, criterion, optimizer, scheduler, device, epoch
            )
        else:
            train_metrics = train_one_epoch_edl(
                model, train_loader, criterion, optimizer, scheduler, device, epoch, config.train.epochs
            )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, args.model)
        
        # Log
        log = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy'],
            'val_uncertainty': val_metrics['mean_uncertainty'],
            'val_confidence': val_metrics['mean_confidence']
        }
        
        if args.model == "mbpe":
            log['div_loss'] = train_metrics['div_loss']
        
        history.append(log)

        if wandb_run is not None:
            lr = optimizer.param_groups[0]['lr']
            wandb.log({**log, 'learning_rate': lr}, step=epoch)
        
        print(f"Epoch {epoch}: Train Acc={train_metrics['accuracy']:.2f}%, "
              f"Val Acc={val_metrics['accuracy']:.2f}%, "
              f"Val Uncertainty={val_metrics['mean_uncertainty']:.4f}")
        
        # Save best
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config_to_save
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"  -> New best model saved! (Acc: {best_acc:.2f}%)")
    
    # Save final model and history
    torch.save({
        'epoch': config.train.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'config': config_to_save
    }, os.path.join(save_dir, "final_model.pth"))
    
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(convert_to_serializable(history), f, indent=2)
    
    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    print(f"Models saved to: {save_dir}")

    if wandb_run is not None:
        wandb.log({'best_val_acc': best_acc})
        wandb.finish()
    
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Train MBPE or EDL model")
    parser.add_argument("--model", type=str, default="mbpe", choices=["mbpe", "edl"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--no_diversity", action="store_true", 
                        help="Disable diversity losses (for ablation)")
    parser.add_argument("--seed", type=int, default=42)
    
    # λ sensitivity 실험용 하이퍼파라미터
    parser.add_argument("--edl_lambda", type=float, default=None,
                        help="EDL KL divergence weight (default: config value)")
    parser.add_argument("--lambda_ncl", type=float, default=None,
                        help="MBPE NCL loss weight (default: config value)")
    parser.add_argument("--lambda_or", type=float, default=None,
                        help="MBPE OR loss weight (default: config value)")
    parser.add_argument("--lambda_fdl", type=float, default=None,
                        help="MBPE FDL loss weight (default: config value)")
    
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (defaults to mbpe-<model>).")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (team or username).")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Custom W&B run name (defaults to experiment name with timestamp).")
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()