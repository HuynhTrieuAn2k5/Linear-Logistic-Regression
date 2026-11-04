"""Regenerate standalone teaching notebooks (runs from notebooks/ in VS Code)."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parent
N = root/'notebooks'
N.mkdir(exist_ok=True)

def m(x): return nbf.v4.new_markdown_cell(x)
def c(x): return nbf.v4.new_code_cell(x.strip())

linear = [
 m('# Lab 04 — Bài toán 1: Hồi quy tuyến tính đa biến (Housing)\n\n**Chạy trên VS Code:** Mở thư mục dự án, chọn Python kernel của `ml_lab_env`, bấm **Run All**. Notebook đọc dữ liệu theo đường dẫn tương đối từ `notebooks/`.\n\nQuy trình: đọc dữ liệu → mã hóa → chia 80/20 → chuẩn hóa chỉ trên Train → LinearRegression → MAE/RMSE/R² → biểu đồ & phân tích hệ số.'),
 m('## 1. Import và quan sát dữ liệu'),
 c('''from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path('../data/raw/Housing.csv')
df = pd.read_csv(DATA)
print('Kích thước dữ liệu:', df.shape)
print('Giá trị thiếu:', df.isna().sum().sum())
display(df.head())'''),
 m('## 2. Tách mục tiêu và chuyển đổi biến định tính\n\n`yes/no` → 1/0, `furnishingstatus` → One-Hot (bỏ nhóm tham chiếu để tránh bẫy biến giả).'),
 c('''X_raw = df.drop(columns=['price']).copy()
y = df['price']
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    X_raw[col] = X_raw[col].map({'yes': 1, 'no': 0})
X_encoded = pd.get_dummies(X_raw, columns=['furnishingstatus'], drop_first=True, dtype=int)
assert not X_encoded.isna().any().any()
print('Số đặc trưng sau mã hóa:', X_encoded.shape[1])
display(X_encoded.head())'''),
 m('## 3. Chia 80% Train – 20% Test\n\n`random_state=42` đảm bảo kết quả có thể tái lập.'),
 c('''X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)
print('Train:', X_train.shape, 'Test:', X_test.shape)'''),
 m('## 4. Chuẩn hóa an toàn (không Data Leakage)\n\n**Chỉ** `fit_transform` trên Train; dùng `transform` trên Test.'),
 c('''scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('Trung bình area trên tập Train:', X_train['area'].mean())
print('scaler.mean_[area]:', scaler.mean_[X_train.columns.get_loc('area')])'''),
 m('## 5. Huấn luyện và tính MAE, RMSE, R²'),
 c('''model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f'MAE = {mae:,.2f}')
print(f'RMSE = {rmse:,.2f}')
print(f'R² = {r2:.4f}')'''),
 m('## 6. Biểu đồ phần dư và thực tế–dự đoán\n\nPhần dư `y_test - y_pred` nên dao động quanh 0. Kiểm tra có hình phễu/đường cong hay không.'),
 c('''residuals = y_test.to_numpy() - y_pred
fig, ax = plt.subplots(figsize=(7, 4))
sns.scatterplot(x=y_pred, y=residuals, ax=ax, alpha=.7)
ax.axhline(0, color='red', linestyle='--')
ax.set(xlabel='Predicted price', ylabel='Residual (Actual - Predicted)', title='Housing: Residual plot')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
sns.scatterplot(x=y_test.to_numpy(), y=y_pred, ax=ax, alpha=.7)
lo = min(y_test.min(), y_pred.min()); hi = max(y_test.max(), y_pred.max())
ax.plot([lo, hi], [lo, hi], 'r--')
ax.set(xlabel='Actual price', ylabel='Predicted price', title='Housing: Actual vs Predicted')
plt.tight_layout()
plt.show()'''),
 m('## 7. Trọng số hồi quy\n\nCác hệ số nằm trên **thang đặc trưng chuẩn hóa**, vì vậy độ lớn phản ánh mức thay đổi dự báo theo một độ lệch chuẩn (khi các biến khác cố định). Không suy diễn quan hệ nhân quả.'),
 c('''coef = pd.Series(model.coef_, index=X_encoded.columns).sort_values(ascending=False)
display(coef.to_frame('He so (scaled)'))
coef.sort_values().plot(kind='barh', figsize=(9, 5), title='Standardized feature coefficients')
plt.axvline(0, color='black', linewidth=.8)
plt.tight_layout()
plt.show()'''),
 m('## 8. Tóm tắt\n\nCùng cấu hình kiểm thử được thực hiện trong `run_all.py`. Các hình, CSV và JSON được xuất tại thư mục `outputs/`.\n\n> **Lưu ý:** Đơn vị tiền tệ được mô tả là USD trong đề bài, nhưng CSV không có trường xác nhận đơn vị; mã nguồn giữ nguyên đơn vị số của cột `price`.')
]
logistic = [
 m('# Lab 04 — Bài toán 2: Hồi quy Logistic (Diabetes)\n\n**Chạy trên VS Code:** Chọn Python kernel của `ml_lab_env` rồi **Run All**. Quy trình: stratified split → bù khuyết các số 0 bất hợp lý bằng median Train → StandardScaler Train → Logistic Regression L2 → Confusion Matrix / ROC-AUC → threshold tuning.'),
 m('## 1. Đọc dữ liệu, quan sát các số 0 không hợp lệ'),
 c('''from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

df = pd.read_csv(Path('../data/raw/diabetes.csv'))
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print('Kích thước:', df.shape)
print('Phân bố Outcome:\\n', df['Outcome'].value_counts())
print('Số 0 bất hợp lý:\\n', (df[zero_cols] == 0).sum())
display(df.head())'''),
 m('## 2. Chia theo tỷ lệ nhãn bằng `stratify=y`'),
 c('''X = df.drop(columns=['Outcome'])
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train = X_train.copy()
X_test = X_test.copy()
print('Train/Test:', len(X_train), len(X_test))
print('Tỷ lệ bệnh Train/Test:', y_train.mean().round(4), y_test.mean().round(4))'''),
 m('## 3. Bù giá trị 0 bằng median chỉ tính từ Train\n\nCác giá trị 0 trong `Pregnancies` là hợp lệ nên **không thay** cột này.'),
 c('''train_medians = {}
for col in zero_cols:
    train_medians[col] = X_train.loc[X_train[col] != 0, col].median()
    X_train[col] = X_train[col].replace(0, train_medians[col])
    X_test[col] = X_test[col].replace(0, train_medians[col])
print('Median từ Train:', train_medians)
print('Số 0 còn lại ở các cột y sinh trên Train:', (X_train[zero_cols] == 0).sum().sum())
print('Số 0 còn lại ở các cột y sinh trên Test:', (X_test[zero_cols] == 0).sum().sum())'''),
 m('## 4. Chuẩn hóa train-only, huấn luyện Logistic Regression với L2 (mặc định)'),
 c('''scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)
print('Đã hội tụ sau:', int(model.n_iter_[0]), 'lần lặp')'''),
 m('## 5. Confusion Matrix, Precision, Recall, F1 và ROC-AUC'),
 c('''cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print('Confusion matrix:\\n', cm)
print(classification_report(y_test, y_pred, target_names=['Lớp 0 (Khỏe)', 'Lớp 1 (Bệnh)']))
print('ROC-AUC =', round(roc_auc_score(y_test, y_proba), 4))
fig, ax = plt.subplots(figsize=(5.8, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['Actual 0', 'Actual 1'], cbar=False, ax=ax)
ax.set(title='Confusion Matrix (threshold=0.5)', xlabel='Prediction', ylabel='Actual')
plt.tight_layout()
plt.show()'''),
 m('## 6. Đường cong ROC'),
 c('''fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
ax.plot([0, 1], [0, 1], 'k--')
ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC curve')
ax.legend()
plt.tight_layout()
plt.show()'''),
 m('## 7. Thực nghiệm ngưỡng 0.3; 0.4; 0.5; 0.6\n\n**Chú ý:** Bảng trên tập Test nhằm minh họa. Trong ứng dụng thật, phải chọn ngưỡng trên tập Validation hoặc cross-validation rồi mới đánh giá một lần trên Test.'),
 c('''rows = []
for th in [0.3, 0.4, 0.5, 0.6]:
    candidate = (y_proba >= th).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(y_test, candidate, labels=[0, 1]).ravel()
    rows.append({'Threshold': th, 'TN': tn_, 'FP': fp_, 'FN': fn_, 'TP': tp_,
                 'Precision': precision_score(y_test, candidate, zero_division=0),
                 'Recall': recall_score(y_test, candidate, zero_division=0),
                 'F1': f1_score(y_test, candidate, zero_division=0)})
threshold_table = pd.DataFrame(rows)
display(threshold_table.round(4))
threshold_table.plot(x='Threshold', y=['Precision', 'Recall'], marker='o', figsize=(7, 4))
plt.ylim(0, 1)
plt.title('Trade-off giữa Precision và Recall')
plt.tight_layout()
plt.show()'''),
 m('## 8. Tác động của siêu tham số C\n\nTrong `LogisticRegression`, tham số C là nghịch đảo của độ mạnh điều chuẩn L2. Giảm C làm co trọng số, tăng C cho trọng số lớn hơn. Đây chỉ là so sánh mô tả, **không dùng Test để chọn C**.'),
 c('''comparisons = []
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    candidate = LogisticRegression(C=C, solver='lbfgs', max_iter=1000, random_state=42)
    candidate.fit(X_train_scaled, y_train)
    comparisons.append({'C': C, 'L2 norm(coef)': np.linalg.norm(candidate.coef_),
                        'Train AUC': roc_auc_score(y_train, candidate.predict_proba(X_train_scaled)[:, 1]),
                        'Test AUC': roc_auc_score(y_test, candidate.predict_proba(X_test_scaled)[:, 1])})
display(pd.DataFrame(comparisons).round(4))'''),
 m('## 9. Kết luận\n\nMô hình logistic này chỉ là **bài thực hành học máy**, không dùng để tự chẩn đoán hoặc thay thế chuyên gia y tế. Các số liệu chạy lại được tạo trong `outputs/`.')
]
for fname, cells in [('01_linear_regression.ipynb',linear),('02_logistic_regression.ipynb',logistic)]:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {'display_name':'Python 3','language':'python','name':'python3'}
    nb.metadata.language_info = {'name':'python','version':'3.10+'}
    nbf.write(nb, N / fname)
    print('CREATED',N/fname)
