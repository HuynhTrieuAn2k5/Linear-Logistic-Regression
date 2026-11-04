"""Lab 04: Regression experiments; all preprocessing fit only on training data.

Run from any working directory using `python run_all.py` at project root.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, recall_score,
    r2_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUTPUT = ROOT / 'outputs'
FIGURES = OUTPUT / 'figures'
BINARY_COLS = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
ZERO_INVALID_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
SEED = 42


def setup_output() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='notebook')
    plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 170, 'font.family': 'DejaVu Sans'})


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, bbox_inches='tight')
    plt.close()


def linear_experiment() -> dict:
    """Multiple linear regression using one-hot and training-only scaling."""
    setup_output()
    df = pd.read_csv(RAW / 'Housing.csv')
    X_raw = df.drop(columns=['price']).copy()
    y = df['price']

    for col in BINARY_COLS:
        X_raw[col] = X_raw[col].map({'yes': 1, 'no': 0})
    X_encoded = pd.get_dummies(X_raw, columns=['furnishingstatus'], drop_first=True, dtype=int)
    if X_encoded.isna().any().any():
        raise ValueError('Some housing categorical values were not encoded')

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=SEED)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # NEVER fit on test!
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    prediction = model.predict(X_test_scaled)
    residuals = y_test.to_numpy() - prediction

    coef = pd.Series(model.coef_, index=X_encoded.columns, name='coefficient')
    coef_frame = coef.sort_values(ascending=False).rename_axis('feature').reset_index()
    coef_frame.to_csv(OUTPUT / 'housing_coefficients.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'actual': y_test.to_numpy(), 'predicted': prediction, 'residual': residuals}).to_csv(
        OUTPUT / 'housing_predictions.csv', index=False, encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    sns.scatterplot(x=prediction, y=residuals, alpha=.72, s=34, ax=ax)
    ax.axhline(0, color='firebrick', ls='--', lw=1.5)
    ax.set(xlabel='Gia nha du doan (price)', ylabel='Phan du (thuc te - du doan)', title='Hoi quy tuyen tinh: Phan tan phan du')
    save_figure('01_housing_residuals.png')

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    sns.scatterplot(x=y_test.to_numpy(), y=prediction, alpha=.72, s=37, ax=ax)
    low = float(min(y_test.min(), prediction.min()))
    high = float(max(y_test.max(), prediction.max()))
    ax.plot([low, high], [low, high], color='firebrick', ls='--', label='Du doan hoan hao')
    ax.set(xlabel='Gia nha thuc te (price)', ylabel='Gia nha du doan (price)', title='Thuc te so voi du doan')
    ax.legend(loc='best')
    save_figure('02_housing_actual_vs_pred.png')

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ranked = coef.sort_values()
    ranked.plot.barh(ax=ax, color=['#bf6d71' if v < 0 else '#48749a' for v in ranked])
    ax.axvline(0, color='#222', lw=.8)
    ax.set(xlabel='He so tren dac trung da StandardScaler', title='He so hoi quy co dau (+/-)')
    save_figure('03_housing_coefficients.png')

    return {
        'rows': int(len(df)), 'columns': int(df.shape[1]),
        'train_size': int(len(X_train)), 'test_size': int(len(X_test)),
        'feature_count_encoded': int(X_encoded.shape[1]),
        'mae': float(mean_absolute_error(y_test, prediction)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, prediction))),
        'r2': float(r2_score(y_test, prediction)),
        'train_r2': float(r2_score(y_train, model.predict(X_train_scaled))),
        'price_mean': float(y.mean()),
        'residual_mean': float(np.mean(residuals)),
        'residual_std': float(np.std(residuals, ddof=1)),
        'coefficients': {k: float(v) for k, v in coef.sort_values(ascending=False).items()},
        'intercept_scaled': float(model.intercept_),
        'test_indices': [int(i) for i in X_test.index],
        'train_scaler_means': {c: float(v) for c, v in zip(X_encoded.columns, scaler.mean_)},
    }


def logistic_experiment() -> dict:
    """Logistic regression with stratification and train-only median imputation."""
    setup_output()
    df = pd.read_csv(RAW / 'diabetes.csv')
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    zero_counts = {col: int((df[col] == 0).sum()) for col in ZERO_INVALID_COLS}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_train = X_train.copy()
    X_test = X_test.copy()
    medians = {}
    for col in ZERO_INVALID_COLS:
        med = float(X_train.loc[X_train[col] != 0, col].median())
        if np.isnan(med):
            raise ValueError(f'No valid training values in {col}')
        medians[col] = med
        X_train[col] = X_train[col].replace(0, med)
        X_test[col] = X_test[col].replace(0, med)
    if any((X_train[c] == 0).any() or (X_test[c] == 0).any() for c in ZERO_INVALID_COLS):
        raise AssertionError('An invalid zero value remains after imputation')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=SEED)  # L2 is the default
    model.fit(X_train_scaled, y_train)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    roc_auc = float(roc_auc_score(y_test, probabilities))

    threshold_records = []
    for th in (.3, .4, .5, .6):
        candidate = (probabilities >= th).astype(int)
        tn_th, fp_th, fn_th, tp_th = [int(v) for v in confusion_matrix(y_test, candidate, labels=[0, 1]).ravel()]
        threshold_records.append({
            'threshold': th, 'TN': tn_th, 'FP': fp_th, 'FN': fn_th, 'TP': tp_th,
            'precision': float(precision_score(y_test, candidate, zero_division=0)),
            'recall': float(recall_score(y_test, candidate, zero_division=0)),
            'f1': float(f1_score(y_test, candidate, zero_division=0)),
            'accuracy': float(accuracy_score(y_test, candidate)),
        })
    pd.DataFrame(threshold_records).to_csv(OUTPUT / 'diabetes_thresholds.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'actual': y_test.to_numpy(), 'probability': probabilities, 'predicted_at_0.5': pred}).to_csv(
        OUTPUT / 'diabetes_predictions.csv', index=False, encoding='utf-8-sig')
    pd.Series(medians, name='median_train').rename_axis('feature').to_csv(
        OUTPUT / 'diabetes_imputation_medians.csv', encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(5.8, 4.7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Du doan 0', 'Du doan 1'],
                yticklabels=['Thuc te 0', 'Thuc te 1'], ax=ax, annot_kws={'size': 15})
    ax.set(xlabel='Nhãn du doan', ylabel='Nhãn thuc te', title='Ma tran nham lan (nguong 0.5)')
    save_figure('04_diabetes_confusion_matrix.png')

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    ax.plot(fpr, tpr, lw=2.2, label=f'Logistic Regression (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], ls='--', c='gray', lw=1, label='Doan ngau nhien')
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate (Recall)', title='ROC Curve - Diabetes', xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc='lower right')
    save_figure('05_diabetes_roc.png')

    thresholds_df = pd.DataFrame(threshold_records)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(thresholds_df['threshold'], thresholds_df['recall'], marker='o', linewidth=2.0, label='Recall')
    ax.plot(thresholds_df['threshold'], thresholds_df['precision'], marker='s', linewidth=2.0, label='Precision')
    ax.set(xlabel='Nguong xac suat', ylabel='Gia tri chi so', title='Threshold tuning: Precision - Recall', ylim=(0, 1))
    ax.set_xticks(thresholds_df['threshold'])
    ax.legend()
    save_figure('06_diabetes_threshold_tradeoff.png')

    # Illustrative C sensitivity: train on same training set, inspect fixed held-out test.
    c_records = []
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        candidate_model = LogisticRegression(C=c, solver='lbfgs', max_iter=1000, random_state=SEED)  # L2 default
        candidate_model.fit(X_train_scaled, y_train)
        train_prob = candidate_model.predict_proba(X_train_scaled)[:, 1]
        test_prob = candidate_model.predict_proba(X_test_scaled)[:, 1]
        c_records.append({
            'C': c, 'coef_l2_norm': float(np.linalg.norm(candidate_model.coef_)),
            'train_auc': float(roc_auc_score(y_train, train_prob)),
            'test_auc': float(roc_auc_score(y_test, test_prob)),
        })
    pd.DataFrame(c_records).to_csv(OUTPUT / 'diabetes_C_sensitivity.csv', index=False, encoding='utf-8-sig')

    return {
        'rows': int(len(df)), 'columns': int(df.shape[1]),
        'train_size': int(len(X_train)), 'test_size': int(len(X_test)),
        'overall_positive': int(y.sum()), 'overall_negative': int((y == 0).sum()),
        'train_positive': int(y_train.sum()), 'test_positive': int(y_test.sum()),
        'zero_counts': zero_counts, 'train_medians': medians,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'accuracy': float(accuracy_score(y_test, pred)),
        'precision': float(precision_score(y_test, pred)),
        'recall': float(recall_score(y_test, pred)),
        'f1': float(f1_score(y_test, pred)),
        'roc_auc': roc_auc,
        'classification_report': classification_report(y_test, pred, target_names=['Lop 0 (khong benh)', 'Lop 1 (co benh)'], output_dict=True),
        'thresholds': threshold_records,
        'c_sensitivity': c_records,
        'model_coefficients': {col: float(w) for col, w in zip(X.columns, model.coef_[0])},
        'test_indices': [int(i) for i in X_test.index],
    }


def run_all() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {'linear': linear_experiment(), 'logistic': logistic_experiment()}
    (OUTPUT / 'experiment_results.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    return results


if __name__ == '__main__':
    summary = run_all()
    line = summary['linear']
    log = summary['logistic']
    print(f"HOUSING: n_train={line['train_size']} n_test={line['test_size']}; "
          f"MAE={line['mae']:,.2f}; RMSE={line['rmse']:,.2f}; R2={line['r2']:.4f}")
    print(f"DIABETES: n_train={log['train_size']} n_test={log['test_size']}; "
          f"Accuracy={log['accuracy']:.4f}; Precision={log['precision']:.4f}; "
          f"Recall={log['recall']:.4f}; F1={log['f1']:.4f}; ROC-AUC={log['roc_auc']:.4f}")
    print('Confusion matrix [[TN, FP], [FN, TP]]: ', [[log['tn'], log['fp']], [log['fn'], log['tp']]])
    print(pd.DataFrame(log['thresholds']).round(4).to_string(index=False))
    print('All results saved to:', OUTPUT)
