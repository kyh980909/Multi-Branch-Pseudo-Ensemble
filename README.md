# MBEE: Multi-Branch Epistemic Ensemble

**EDL의 단점을 극복하는 효율적 앙상블 기반 Epistemic Uncertainty 추정**

## 프로젝트 구조

```
mbee_project/
├── configs/
│   └── default_config.py          # 공통 설정/하이퍼파라미터
├── data/
│   └── datasets.py                # CIFAR-10/SVHN 로더
├── models/
│   ├── mbee.py                    # Multi-Branch Epistemic Ensemble
│   └── edl.py                     # Evidential Deep Learning (baseline)
├── losses/
│   └── diversity_losses.py        # NCL/OR/FDL 다양성 손실
├── utils/
│   └── metrics.py                 # AUROC, ECE 등 공통 메트릭
├── experiments/
│   ├── train.py                   # 단일 모델 학습 스크립트
│   ├── evaluate_ood.py            # 지정 모델에 대한 OOD/Calibration 평가
│   ├── evaluate_all.py            # 학습 로그(history.json) 요약
│   ├── summarize_ood_results.py   # evaluate_ood 결과 요약
│   ├── run_train_model.sh         # 하이퍼파라미터 스윕 예시
│   └── run_evaluate_all_checkpoints.sh # 모든 체크포인트 평가 자동화
└── README.md
```

## 설치

```bash
pip install torch torchvision numpy scikit-learn matplotlib tqdm
```

## 빠른 시작

### 1. 학습

```bash
cd mbee_project/experiments

# MBEE 기본 학습 (다양성 손실 포함)
python train.py --model mbee --epochs 100

# MBEE 다양성 소거 실험
python train.py --model mbee --no_diversity

# EDL 베이스라인 학습
python train.py --model edl --epochs 100

# 여러 하이퍼 조합을 연속 실행하려면
bash run_train_model.sh
```

### 2. 평가

```bash
# 단일 체크포인트 평가 (MBEE)
python evaluate_ood.py --mbee_path checkpoints/mbee_YYYYMMDD_xxxxxx/best_model.pth \
                       --save_dir experiments/evaluation_results/mbee_baseline

# 단일 체크포인트 평가 (EDL)
python evaluate_ood.py --edl_path checkpoints/edl_YYYYMMDD_xxxxxx/best_model.pth \
                       --save_dir experiments/evaluation_results/edl_baseline

# 체크포인트 전체 일괄 평가
bash run_evaluate_all_checkpoints.sh

# 평가 로그 요약
python summarize_ood_results.py --results-root experiments/evaluation_results/all_models

# 학습 기록(history.json) 요약
python evaluate_all.py --checkpoints_dir experiments/checkpoints
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

## 주요 결과

실험중

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
