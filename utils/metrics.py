"""
평가 메트릭:
- AUROC: OOD detection 성능
- FPR95: 95% TPR에서의 FPR
- ECE: Expected Calibration Error
- Cohen's d: Known vs Unknown 분리도
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt


def compute_auroc(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    score_type: str = "uncertainty"
) -> float:
    """
    AUROC 계산 (OOD detection)
    
    Args:
        id_scores: In-distribution uncertainty scores
        ood_scores: Out-of-distribution uncertainty scores
        score_type: "uncertainty" (높을수록 OOD) or "confidence" (낮을수록 OOD)
    
    Returns:
        AUROC score
    """
    # Labels: 0 = ID, 1 = OOD
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores))
    ])
    scores = np.concatenate([id_scores, ood_scores])
    
    # uncertainty: 높을수록 OOD -> 그대로 사용
    # confidence: 낮을수록 OOD -> 부호 반전
    if score_type == "confidence":
        scores = -scores
    
    return roc_auc_score(labels, scores)


def compute_fpr_at_tpr(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    tpr_threshold: float = 0.95,
    score_type: str = "uncertainty"
) -> float:
    """
    특정 TPR에서의 FPR 계산 (FPR95)
    
    Args:
        id_scores: In-distribution uncertainty scores
        ood_scores: Out-of-distribution uncertainty scores
        tpr_threshold: Target TPR (default: 0.95)
        score_type: "uncertainty" or "confidence"
    
    Returns:
        FPR at given TPR
    """
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores))
    ])
    scores = np.concatenate([id_scores, ood_scores])
    
    if score_type == "confidence":
        scores = -scores
    
    fpr, tpr, _ = roc_curve(labels, scores)
    
    # TPR >= threshold인 지점에서 최소 FPR 찾기
    idx = np.where(tpr >= tpr_threshold)[0]
    if len(idx) == 0:
        return 1.0
    return fpr[idx[0]]


def compute_aupr(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    score_type: str = "uncertainty"
) -> float:
    """
    AUPR (Area Under Precision-Recall curve) 계산
    """
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores))
    ])
    scores = np.concatenate([id_scores, ood_scores])
    
    if score_type == "confidence":
        scores = -scores
    
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Expected Calibration Error (ECE)
    
    Args:
        confidences: [N] predicted confidence (max softmax)
        predictions: [N] predicted classes
        labels: [N] true labels
        n_bins: number of bins for calibration
    
    Returns:
        ECE score (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # 해당 bin에 속하는 샘플들
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            # Average confidence in this bin
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # ECE contribution
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    
    return ece


def compute_mce(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Maximum Calibration Error (MCE)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return mce


def compute_cohens_d(
    id_scores: np.ndarray,
    ood_scores: np.ndarray
) -> float:
    """
    Cohen's d: ID와 OOD 분포 간 분리도
    
    d = (mean_ood - mean_id) / pooled_std
    
    높을수록 두 분포가 잘 분리됨
    """
    mean_id = id_scores.mean()
    mean_ood = ood_scores.mean()
    
    std_id = id_scores.std()
    std_ood = ood_scores.std()
    
    n_id = len(id_scores)
    n_ood = len(ood_scores)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((n_id - 1) * std_id ** 2 + (n_ood - 1) * std_ood ** 2) / (n_id + n_ood - 2)
    )
    
    if pooled_std < 1e-10:
        return 0.0
    
    return (mean_ood - mean_id) / pooled_std


def compute_all_ood_metrics(
    id_uncertainties: np.ndarray,
    ood_uncertainties: np.ndarray
) -> Dict[str, float]:
    """
    모든 OOD detection 메트릭 계산
    
    Args:
        id_uncertainties: ID samples의 uncertainty
        ood_uncertainties: OOD samples의 uncertainty
    
    Returns:
        Dict with all metrics
    """
    return {
        'auroc': compute_auroc(id_uncertainties, ood_uncertainties),
        'aupr': compute_aupr(id_uncertainties, ood_uncertainties),
        'fpr95': compute_fpr_at_tpr(id_uncertainties, ood_uncertainties, 0.95),
        'cohens_d': compute_cohens_d(id_uncertainties, ood_uncertainties),
        'id_mean': id_uncertainties.mean(),
        'id_std': id_uncertainties.std(),
        'ood_mean': ood_uncertainties.mean(),
        'ood_std': ood_uncertainties.std()
    }


def plot_uncertainty_histogram(
    id_uncertainties: np.ndarray,
    ood_uncertainties: np.ndarray,
    title: str = "Uncertainty Distribution",
    save_path: str = None
) -> plt.Figure:
    """
    ID vs OOD uncertainty 히스토그램 시각화
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(id_uncertainties, bins=50, alpha=0.6, label='In-Distribution', color='blue', density=True)
    ax.hist(ood_uncertainties, bins=50, alpha=0.6, label='Out-of-Distribution', color='red', density=True)
    
    ax.set_xlabel('Uncertainty', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    
    # Cohen's d 표시
    d = compute_cohens_d(id_uncertainties, ood_uncertainties)
    ax.text(0.95, 0.95, f"Cohen's d = {d:.2f}", transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_reliability_diagram(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: str = None
) -> plt.Figure:
    """
    Reliability diagram (calibration plot) 시각화
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    accuracies = []
    avg_confidences = []
    counts = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            acc = (predictions[in_bin] == labels[in_bin]).mean()
            conf = confidences[in_bin].mean()
            accuracies.append(acc)
            avg_confidences.append(conf)
            counts.append(in_bin.sum())
        else:
            accuracies.append(0)
            avg_confidences.append(bin_centers[i])
            counts.append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    ax1.bar(bin_centers, accuracies, width=1/n_bins, alpha=0.7, edgecolor='black', label='Accuracy')
    ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # ECE 표시
    ece = compute_ece(confidences, predictions, labels, n_bins)
    ax1.text(0.05, 0.95, f'ECE = {ece:.4f}', transform=ax1.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Confidence histogram
    ax2.hist(confidences, bins=n_bins, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Confidence Distribution', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    method_name: str = "Method",
    save_path: str = None
) -> plt.Figure:
    """ROC curve 시각화"""
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f'{method_name} (AUROC = {auroc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve for OOD Detection', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # 메트릭 테스트
    print("Testing metrics...")
    
    # Dummy data
    np.random.seed(42)
    id_uncertainties = np.random.beta(2, 5, 1000)  # 낮은 uncertainty
    ood_uncertainties = np.random.beta(5, 2, 1000)  # 높은 uncertainty
    
    # OOD metrics
    metrics = compute_all_ood_metrics(id_uncertainties, ood_uncertainties)
    print("\nOOD Detection Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Calibration metrics
    confidences = np.random.uniform(0.5, 1.0, 1000)
    predictions = np.random.randint(0, 10, 1000)
    labels = np.random.randint(0, 10, 1000)
    
    ece = compute_ece(confidences, predictions, labels)
    mce = compute_mce(confidences, predictions, labels)
    print(f"\nCalibration Metrics:")
    print(f"  ECE: {ece:.4f}")
    print(f"  MCE: {mce:.4f}")
    
    # 시각화 테스트
    print("\nGenerating plots...")
    plot_uncertainty_histogram(id_uncertainties, ood_uncertainties, 
                               save_path="uncertainty_hist.png")
    plot_reliability_diagram(confidences, predictions, labels,
                            save_path="reliability_diagram.png")
    print("Plots saved!")
