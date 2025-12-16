# MBEE: Multi-Branch Epistemic Ensemble

**EDL의 단점을 극복하는 효율적 앙상블 기반 Epistemic Uncertainty 추정**

## 프로젝트 구조

```
mbee_project/
├── configs/
│   └── default_config.py      # 설정 파일
├── data/
│   └── datasets.py            # CIFAR-10, SVHN 데이터 로더
├── models/
│   ├── mbee.py                # MBEE 모델
│   └── edl.py                 # EDL 모델 (비교용)
├── losses/
│   └── diversity_losses.py    # NCL, OR, FDL 다양성 손실
├── utils/
│   └── metrics.py             # AUROC, ECE 등 평가 메트릭
├── experiments/
│   ├── train.py               # 학습 스크립트
│   └── evaluate_ood.py        # OOD 평가 스크립트
└── README.md
```

## 설치

```bash
pip install torch torchvision numpy scikit-learn matplotlib tqdm
```

## 빠른 시작

### 1. MBEE 모델 학습

```bash
cd mbee_project/experiments

# MBEE 학습 (다양성 손실 포함)
python train.py --model mbee --epochs 100

# EDL 학습 (비교용)
python train.py --model edl --epochs 100

# MBEE 학습 (다양성 손실 제외 - 소거 실험)
python train.py --model mbee --epochs 100 --no_diversity
```

### 2. OOD Detection 평가

```bash
python evaluate_ood.py \
    --mbee_path checkpoints/mbee_xxx/best_model.pth \
    --edl_path checkpoints/edl_xxx/best_model.pth \
    --save_dir evaluation_results
```

## 핵심 실험

### 실험 1: EDL vs MBEE OOD Detection
- **목적**: MBEE의 epistemic uncertainty가 EDL보다 OOD 탐지에 효과적임을 입증
- **지표**: AUROC, FPR95, Cohen's d

### 실험 2: 다양성 손실의 효과 (소거 실험)
- **목적**: 다양성 손실이 분기 간 다양성과 uncertainty 품질에 필수적임을 입증
- **설정**: MBEE (full) vs MBEE (no diversity)

### 실험 3: Calibration 비교
- **목적**: MBEE가 EDL보다 더 잘 calibrated 됨을 입증
- **지표**: ECE (Expected Calibration Error)

## 주요 결과 (예상)

| 방법 | Accuracy | AUROC | FPR95 | Cohen's d | ECE |
|------|----------|-------|-------|-----------|-----|
| EDL | ~93% | ~0.85 | ~0.45 | ~1.2 | ~0.08 |
| **MBEE** | ~93% | **~0.92** | **~0.25** | **~1.8** | **~0.04** |
| MBEE (no div) | ~93% | ~0.82 | ~0.50 | ~1.0 | ~0.10 |

## 모델 아키텍처

```
Input Image
    │
    ▼
┌─────────────────┐
│ Shared Backbone │  (ResNet-18)
│    (공유)        │
└────────┬────────┘
         │
    ┌────┴────┐
    │ Diversity│  (Dropout, Perturbation, Projection)
    │ Injection│
    └────┬────┘
         │
    ┌────┼────┬────┐
    ▼    ▼    ▼    ▼
┌─────┐┌─────┐┌─────┐┌─────┐
│ B1  ││ B2  ││ B3  ││ B4  │  (K개 분기)
└──┬──┘└──┬──┘└──┬──┘└──┬──┘
   │      │      │      │
   └──────┼──────┼──────┘
          │
    ┌─────┴─────┐
    │Aggregation│  (평균, 분산 → Epistemic)
    └─────┬─────┘
          │
          ▼
    Known / Unknown
```

## 다양성 손실 함수

1. **NCL (Negative Correlation Learning)**
   - Branch 예측 간 상관 최소화
   - `L_NCL = Σ_{i≠j} corr(p_i, p_j)`

2. **OR (Orthogonality Regularization)**
   - Classifier 가중치 직교성
   - `L_OR = Σ_{i≠j} ||W_i^T W_j||²`

3. **FDL (Feature Decorrelation Loss)**
   - Branch features 간 상관 최소화
   - `L_FDL = Σ_{i≠j} ||Cov(F_i, F_j)||²`

## License

MIT License

## Citation

```bibtex
@article{mbee2025,
  title={MBEE: Multi-Branch Epistemic Ensemble for Open-World Object Detection},
  author={Kim, Yongho},
  year={2025}
}
```
