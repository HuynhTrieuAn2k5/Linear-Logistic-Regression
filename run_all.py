"""VS Code main entrypoint: python run_all.py"""
from src.analysis import run_all
import pandas as pd


def main():
    results = run_all()
    l, d = results['linear'], results['logistic']
    print('\n========== LINEAR REGRESSION / HOUSING ==========')
    print(f"Train/Test: {l['train_size']}/{l['test_size']} | Encoded features: {l['feature_count_encoded']}")
    print(f"MAE = {l['mae']:,.2f} | RMSE = {l['rmse']:,.2f} | R2 = {l['r2']:.4f}")
    print('\n========== LOGISTIC REGRESSION / DIABETES ==========')
    print(f"Train/Test: {d['train_size']}/{d['test_size']}")
    print(f"Confusion Matrix = [[{d['tn']}, {d['fp']}], [{d['fn']}, {d['tp']}]]")
    print(f"Accuracy = {d['accuracy']:.4f} | Precision = {d['precision']:.4f} | Recall = {d['recall']:.4f}")
    print(f"F1 = {d['f1']:.4f} | ROC-AUC = {d['roc_auc']:.4f}")
    print('\nTHRESHOLD TUNING')
    print(pd.DataFrame(d['thresholds']).round(4).to_string(index=False))
    print('\nSaved charts and tables in outputs/')


if __name__ == '__main__':
    main()
