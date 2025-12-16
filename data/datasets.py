"""
데이터 로더: CIFAR-10 (ID), SVHN/LSUN (OOD)
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Dict, Optional
import numpy as np
import os


def get_cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """CIFAR-10 학습/테스트 변환"""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    return train_transform, test_transform


def get_svhn_transform() -> transforms.Compose:
    """SVHN 변환 (CIFAR-10 정규화 사용)"""
    return transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])


def get_cifar10_loaders(
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    CIFAR-10 데이터 로더 반환
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform, test_transform = get_cifar10_transforms()
    
    # 전체 학습 데이터
    full_train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # 검증 데이터용 (augmentation 없이)
    val_dataset_base = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=test_transform
    )
    
    # 학습/검증 분할
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split = int(np.floor(val_split * num_train))
    train_idx, val_idx = indices[split:], indices[:split]
    
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(val_dataset_base, val_idx)
    
    # 테스트 데이터
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 데이터 로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_svhn_loader(
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4
) -> DataLoader:
    """SVHN 데이터 로더 (OOD용)"""
    transform = get_svhn_transform()
    
    dataset = datasets.SVHN(
        root=os.path.join(data_root, "svhn"),
        split="test",
        download=True,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def get_ood_loaders(
    ood_datasets: list,
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    OOD 데이터 로더들 반환
    
    Args:
        ood_datasets: ["svhn", "lsun", "tiny_imagenet"] 등
    
    Returns:
        Dict[str, DataLoader]: 데이터셋 이름 -> 로더
    """
    ood_loaders = {}
    
    for ood_name in ood_datasets:
        if ood_name.lower() == "svhn":
            ood_loaders["svhn"] = get_svhn_loader(data_root, batch_size, num_workers)
        # 다른 OOD 데이터셋은 필요시 추가
    
    return ood_loaders


class OODDataset(torch.utils.data.Dataset):
    """
    ID + OOD 결합 데이터셋 (평가용)
    """
    def __init__(
        self, 
        id_dataset: torch.utils.data.Dataset,
        ood_dataset: torch.utils.data.Dataset,
        id_label: int = 0,  # In-distribution label
        ood_label: int = 1  # Out-of-distribution label
    ):
        self.id_dataset = id_dataset
        self.ood_dataset = ood_dataset
        self.id_label = id_label
        self.ood_label = ood_label
        
        self.id_len = len(id_dataset)
        self.ood_len = len(ood_dataset)
    
    def __len__(self):
        return self.id_len + self.ood_len
    
    def __getitem__(self, idx):
        if idx < self.id_len:
            img, _ = self.id_dataset[idx]
            return img, self.id_label
        else:
            img, _ = self.ood_dataset[idx - self.id_len]
            return img, self.ood_label


def get_id_ood_loader(
    id_loader: DataLoader,
    ood_loader: DataLoader,
    batch_size: int = 128,
    num_workers: int = 4
) -> DataLoader:
    """ID + OOD 결합 로더 (AUROC 계산용)"""
    combined_dataset = OODDataset(
        id_loader.dataset,
        ood_loader.dataset
    )
    
    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == "__main__":
    # 데이터 로더 테스트
    print("Loading CIFAR-10...")
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_root="./data",
        batch_size=128
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 샘플 확인
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    print("\nLoading SVHN (OOD)...")
    svhn_loader = get_svhn_loader(data_root="./data", batch_size=128)
    print(f"SVHN batches: {len(svhn_loader)}")
