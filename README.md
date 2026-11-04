# LAB 04 — Linear Regression & Logistic Regression (VS Code)

**Mã sinh viên:** 65130011  
**Sinh viên:** Huỳnh Triệu An  
**Nguồn bài tập:** `Lab04-Linear Regression  Logistic Regression.pdf` do giảng viên cung cấp.  
**Dataset:** `data/raw/Housing.csv`, `data/raw/diabetes.csv`.

## 1. Cấu trúc thư mục

```text
Lab04_65130011_Huynh_Trieu_An/
├── data/
│   ├── raw/
│   │   ├── Housing.csv
│   │   └── diabetes.csv
│   └── processed/
├── src/
│   ├── __init__.py
│   └── analysis.py                # Toàn bộ logic huấn luyện + biểu đồ
├── notebooks/
│   ├── 01_linear_regression.ipynb
│   └── 02_logistic_regression.ipynb
├── outputs/
│   ├── figures/                  # 6 biểu đồ PNG
│   ├── experiment_results.json
│   └── *.csv                     # bảng dự đoán, hệ số, threshold, C
├── reports/
│   ├── 65130011_Huynh_Trieu_An_BaoCao_ThucHanh04_Regression.docx
│   └── Bao_cao_Lab04_65130011.pdf
├── tests/test_workflow.py
├── run_all.py                     # Chạy tất cả từ VS Code
├── make_notebooks.py             # Tạo lại sổ tay
├── requirements.txt
└── README.md
```

## 2. Thiết lập VS Code trên Windows

1. Cài Python **3.10 trở lên** và VS Code; cài các extension **Python** và **Jupyter**.
2. Vào **File → Open Folder** và chọn thư mục `Lab04_65130011_Huynh_Trieu_An`.
3. Mở **Terminal → New Terminal**, thực hiện:

```powershell
py -3 -m venv ml_lab_env
.\ml_lab_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.
```

Nếu Terminal là **Command Prompt** dùng `ml_lab_env\Scripts\activate.bat`; nếu PowerShell không cho kích hoạt, chọn Command Prompt hoặc gọi trực tiếp `.\ml_lab_env\Scripts\python.exe` (không cần thay đổi ExecutionPolicy).

4. Chọn môi trường: **Ctrl + Shift + P → Python: Select Interpreter → ml_lab_env**.
5. Chạy toàn bộ hai bài bằng lệnh:

```powershell
python run_all.py
```

6. Để chạy sổ tay: mở `notebooks/01_linear_regression.ipynb` và `notebooks/02_logistic_regression.ipynb`, chọn kernel `ml_lab_env` rồi chọn **Run All**. Nếu cần, cài thêm `ipykernel` bằng `python -m pip install ipykernel`.
7. Kết quả, hình ảnh được ghi trong `outputs/`; file báo cáo ở `reports/`.

## 3. Lệnh trên macOS/Linux

```bash
python3 -m venv ml_lab_env
source ml_lab_env/bin/activate
python -m pip install -r requirements.txt
python run_all.py
```

## 4. Kiểm thử

Tại thư mục dự án:

```bash
python -m unittest discover -s tests -v
```

## 5. Thiết lập thực nghiệm

- **Housing**: chuyển `yes/no` → 1/0, one-hot `furnishingstatus` (drop-first), chia 80/20 với seed 42, scaler fit trên train, hồi quy tuyến tính, MAE/RMSE/R² và Residual Plot.
- **Diabetes**: `stratify=y`, thay những số 0 phi lý tại `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` bằng trung vị **của train**, scaler fit trên train, Logistic Regression L2 (`C=1`, `lbfgs`), Confusion Matrix, báo cáo phân loại, ROC-AUC, threshold 0.3–0.6.
- Cả hai phần **không học đặc trưng của tập kiểm thử** trước thời điểm đánh giá.
- Threshold trên test chỉ để **minh họa yêu cầu bài thực hành**; khi triển khai thực tế phải tinh chỉnh bằng validation/CV và giữ test độc lập.
- Các con số và đồ thị của báo cáo được lấy trực tiếp từ dữ liệu đã chạy với lệnh `python run_all.py`.

txt