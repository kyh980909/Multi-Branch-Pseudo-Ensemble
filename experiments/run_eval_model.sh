# EDL 학습
python evaluate_ood.py --model edl --edl_lambda 0.01 --epochs 100 --wandb_project mbpe-edl-compare
python evaluate_ood.py --model edl --edl_lambda 0.1 --epochs 100 --wandb_project mbpe-edl-compare
python evaluate_ood.py --model edl --edl_lambda 0.5 --epochs 100 --wandb_project mbpe-edl-compare
python evaluate_ood.py --model edl --edl_lambda 1.0 --epochs 100 --wandb_project mbpe-edl-compare
python evaluate_ood.py --model edl --edl_lambda 5.0 --epochs 100 --wandb_project mbpe-edl-compare

# MBPE 학습 
# 전체 λ 동시 조절
python evaluate_ood.py --model mbpe --lambda_ncl 0.05 --lambda_or 0.05 --lambda_fdl 0.05 --epochs 100 --wandb_project mbpe-edl-compare
python evaluate_ood.py --model mbpe --lambda_ncl 0.1 --lambda_or 0.1 --lambda_fdl 0.1 --epochs 100 --wandb_project mbpe-edl-compare
python evaluate_ood.py --model mbpe --lambda_ncl 0.3 --lambda_or 0.3 --lambda_fdl 0.3 --epochs 100 --wandb_project mbpe-edl-compare
python evaluate_ood.py --model mbpe --lambda_ncl 0.5 --lambda_or 0.5 --lambda_fdl 0.5 --epochs 100 --wandb_project mbpe-edl-compare