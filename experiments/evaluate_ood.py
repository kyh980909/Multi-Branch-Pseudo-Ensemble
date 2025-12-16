"""
OOD Detection 평가 스크립트

EDL vs MBEE 비교:
- AUROC
- FPR95
- Cohen's d (분리도)
- Calibration (ECE)

사용법:
    python evaluate_ood.py --mbee_path checkpoints/mbee/.../best_model.pth \
                           --edl_path checkpoints/edl/.../best_model.pth
"""

import os
import sys
import argparse
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import Config
from data.datasets import get_cifar10_loaders, get_svhn_loader
from models.mbee import MBEEForCIFAR
from models.edl import EDLForCIFAR
from utils.metrics import (
    compute_all_ood_metrics,
    compute_ece,
    plot_uncertainty_histogram,
    plot_reliability_diagram,
    plot_roc_curve
)


@torch.no_grad()
def extract_uncertainties(
    model,
    data_loader,
    device: str,
    model_type: str
):
    """
    데이터로더에서 불확실성 추출
    
    Returns:
        uncertainties: [N] uncertainty scores
        confidences: [N] confidence scores
        predictions: [N] predicted labels
        labels: [N] true labels (있는 경우)
    """
    model.eval()
    
    all_uncertainties = []
    all_confidences = []
    all_predictions = []
    all_labels = []
    
    for images, labels in tqdm(data_loader, desc="Extracting"):
        images = images.to(device)
        
        if model_type == "mbee":
            outputs = model(images)
            uncertainty = outputs['epistemic']
            confidence = outputs['mean_probs'].max(dim=-1)[0]
            predictions = outputs['prediction']
        else:  # edl
            outputs = model(images)
            uncertainty = outputs['epistemic']
            confidence = outputs['probs'].max(dim=-1)[0]
            predictions = outputs['prediction']
        
        all_uncertainties.append(uncertainty.cpu().numpy())
        all_confidences.append(confidence.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_labels.append(labels.numpy())
    
    return {
        'uncertainties': np.concatenate(all_uncertainties),
        'confidences': np.concatenate(all_confidences),
        'predictions': np.concatenate(all_predictions),
        'labels': np.concatenate(all_labels)
    }


def load_model(model_path: str, model_type: str, device: str):
    """저장된 모델 로드"""
    if model_type == "mbee":
        model = MBEEForCIFAR(num_classes=10, num_branches=4)
    else:
        model = EDLForCIFAR(num_classes=10)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_single_model(
    model,
    model_type: str,
    model_name: str,
    id_loader,
    ood_loaders: dict,
    device: str,
    save_dir: str
):
    """단일 모델 평가"""
    results = {'model': model_name, 'type': model_type}
    
    # ID 데이터 추출
    print(f"\nEvaluating {model_name}...")
    print("  Extracting ID (CIFAR-10) features...")
    id_results = extract_uncertainties(model, id_loader, device, model_type)
    
    # ID accuracy
    id_accuracy = (id_results['predictions'] == id_results['labels']).mean() * 100
    results['id_accuracy'] = id_accuracy
    print(f"  ID Accuracy: {id_accuracy:.2f}%")
    
    # Calibration (ECE)
    ece = compute_ece(
        id_results['confidences'],
        id_results['predictions'],
        id_results['labels']
    )
    results['ece'] = ece
    print(f"  ECE: {ece:.4f}")
    
    # OOD 평가
    results['ood'] = {}
    
    for ood_name, ood_loader in ood_loaders.items():
        print(f"  Extracting OOD ({ood_name}) features...")
        ood_results = extract_uncertainties(model, ood_loader, device, model_type)
        
        # OOD metrics
        ood_metrics = compute_all_ood_metrics(
            id_results['uncertainties'],
            ood_results['uncertainties']
        )
        results['ood'][ood_name] = ood_metrics
        
        print(f"    AUROC: {ood_metrics['auroc']:.4f}")
        print(f"    FPR95: {ood_metrics['fpr95']:.4f}")
        print(f"    Cohen's d: {ood_metrics['cohens_d']:.4f}")
        
        # 시각화
        fig = plot_uncertainty_histogram(
            id_results['uncertainties'],
            ood_results['uncertainties'],
            title=f"{model_name}: CIFAR-10 vs {ood_name.upper()}",
            save_path=os.path.join(save_dir, f"{model_name}_{ood_name}_histogram.png")
        )
        plt.close(fig)
    
    # Reliability diagram
    fig = plot_reliability_diagram(
        id_results['confidences'],
        id_results['predictions'],
        id_results['labels'],
        title=f"{model_name}: Reliability Diagram",
        save_path=os.path.join(save_dir, f"{model_name}_reliability.png")
    )
    plt.close(fig)
    
    return results, id_results


def compare_models(results_list: list, save_dir: str):
    """모델 비교 테이블 생성"""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    # Header
    print(f"\n{'Model':<20} {'Accuracy':<12} {'ECE':<12} {'AUROC':<12} {'FPR95':<12} {'Cohens d':<12}")
    print("-"*80)
    
    comparison = []
    
    for result in results_list:
        model_name = result['model']
        accuracy = result['id_accuracy']
        ece = result['ece']
        
        # SVHN 결과 사용 (주요 OOD)
        if 'svhn' in result['ood']:
            ood = result['ood']['svhn']
            auroc = ood['auroc']
            fpr95 = ood['fpr95']
            cohens_d = ood['cohens_d']
        else:
            auroc = fpr95 = cohens_d = 0.0
        
        print(f"{model_name:<20} {accuracy:<12.2f} {ece:<12.4f} {auroc:<12.4f} {fpr95:<12.4f} {cohens_d:<12.4f}")
        
        comparison.append({
            'model': model_name,
            'accuracy': accuracy,
            'ece': ece,
            'auroc': auroc,
            'fpr95': fpr95,
            'cohens_d': cohens_d
        })
    
    # Save comparison
    with open(os.path.join(save_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    
    print("\n" + "="*60)
    
    return comparison


def plot_comparison_roc(
    id_uncertainties_dict: dict,
    ood_uncertainties_dict: dict,
    save_path: str
):
    """여러 모델의 ROC curve 비교"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (model_name, id_unc) in enumerate(id_uncertainties_dict.items()):
        ood_unc = ood_uncertainties_dict[model_name]
        
        labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
        scores = np.concatenate([id_unc, ood_unc])
        
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores)
        
        ax.plot(fpr, tpr, color=colors[i % len(colors)], 
                label=f'{model_name} (AUROC = {auroc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curves for OOD Detection (CIFAR-10 vs SVHN)', fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate OOD detection")
    parser.add_argument("--mbee_path", type=str, default=None, help="Path to MBEE model")
    parser.add_argument("--edl_path", type=str, default=None, help="Path to EDL model")
    parser.add_argument("--mbee_no_div_path", type=str, default=None, 
                        help="Path to MBEE without diversity (ablation)")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="./evaluation_results")
    
    args = parser.parse_args()
    
    if not any([args.mbee_path, args.edl_path, args.mbee_no_div_path]):
        raise SystemExit("Please provide at least one model path (mbee, edl, or mbee_no_div).")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Data loaders
    print("Loading data...")
    _, _, id_loader = get_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size
    )
    
    ood_loaders = {
        'svhn': get_svhn_loader(
            data_root=args.data_root,
            batch_size=args.batch_size
        )
    }
    
    # Evaluate models
    all_results = []
    id_uncertainties = {}
    ood_uncertainties = {}
    
    # MBEE
    if args.mbee_path:
        print("\nLoading MBEE model...")
        mbee_model = load_model(args.mbee_path, "mbee", device)
        mbee_results, mbee_id = evaluate_single_model(
            mbee_model, "mbee", "MBEE",
            id_loader, ood_loaders, device, args.save_dir
        )
        all_results.append(mbee_results)
        id_uncertainties["MBEE"] = mbee_id['uncertainties']
        
        # Get OOD uncertainties for MBEE
        ood_results = extract_uncertainties(mbee_model, ood_loaders['svhn'], device, "mbee")
        ood_uncertainties["MBEE"] = ood_results['uncertainties']
    
    # EDL (if provided)
    if args.edl_path:
        print("\nLoading EDL model...")
        edl_model = load_model(args.edl_path, "edl", device)
        edl_results, edl_id = evaluate_single_model(
            edl_model, "edl", "EDL",
            id_loader, ood_loaders, device, args.save_dir
        )
        all_results.append(edl_results)
        id_uncertainties["EDL"] = edl_id['uncertainties']
        
        ood_results = extract_uncertainties(edl_model, ood_loaders['svhn'], device, "edl")
        ood_uncertainties["EDL"] = ood_results['uncertainties']
    
    # MBEE without diversity (ablation, if provided)
    if args.mbee_no_div_path:
        print("\nLoading MBEE (no diversity) model...")
        mbee_no_div = load_model(args.mbee_no_div_path, "mbee", device)
        no_div_results, no_div_id = evaluate_single_model(
            mbee_no_div, "mbee", "MBEE (no div)",
            id_loader, ood_loaders, device, args.save_dir
        )
        all_results.append(no_div_results)
        id_uncertainties["MBEE (no div)"] = no_div_id['uncertainties']
        
        ood_results = extract_uncertainties(mbee_no_div, ood_loaders['svhn'], device, "mbee")
        ood_uncertainties["MBEE (no div)"] = ood_results['uncertainties']
    
    # Comparison
    comparison = compare_models(all_results, args.save_dir)
    
    # ROC comparison plot
    if len(id_uncertainties) > 1:
        plot_comparison_roc(
            id_uncertainties,
            ood_uncertainties,
            os.path.join(args.save_dir, "roc_comparison.png")
        )
        print(f"\nROC comparison saved to {args.save_dir}/roc_comparison.png")
    
    # Save all results
    with open(os.path.join(args.save_dir, "full_results.json"), "w") as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        json.dump(convert(all_results), f, indent=2)
    
    print(f"\nAll results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
