import os
import time
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score
)

from imblearn.datasets import fetch_datasets
from imblearn.under_sampling import (
    RandomUnderSampler,
    ClusterCentroids,
    CondensedNearestNeighbour,
    OneSidedSelection,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    TomekLinks,
    NeighbourhoodCleaningRule,
    InstanceHardnessThreshold
)
from imblearn.combine import SMOTEENN, SMOTETomek

# ==========================================
# 1. HẰNG SỐ VÀ ĐƯỜNG DẪN DÙNG CHUNG
# ==========================================
RANDOM_STATE = 42          # Hạt giống chuẩn cho chia tập dữ liệu và mô hình
DATA_RANDOM_STATE = 81     # Hạt giống tạo dữ liệu 2D giống Chương 3 trong sách


def get_data_dir():
    """Trả về đường dẫn thư mục data/ nằm ngay trong repo."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


# ==========================================
# 2. HÀM TẠO & TIỀN XỬ LÝ DỮ LIỆU CHUNG
# ==========================================
def load_synthetic_data(class_sep=0.79, n_samples=10000, test_size=0.2):
    """
    Tạo dữ liệu synthetic 2D tỷ lệ 1:99 chuẩn Chương 3 (dùng class_sep=0.79 cho
    vẽ đồ thị 2D của TV2, TV3 và class_sep=0.5 cho benchmark của TV4).
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_classes=2,
        flip_y=0,
        n_clusters_per_class=2,
        class_sep=class_sep,
        weights=[0.99],
        random_state=DATA_RANDOM_STATE
    )
    X = pd.DataFrame(X, columns=["feature_1", "feature_2"])
    y = pd.Series(y, name="target")

    # Chia train/test với stratify=y trước khi undersampling để tránh data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, X, y


def load_real_data(dataset_name="thyroid_sick", test_size=0.2):
    """
    Đọc dataset thực tế từ file nén trong thư mục data/ (nếu chưa có sẽ tự tải
    qua imblearn và lưu lại vào data/ để cả nhóm dùng offline).
    """
    data_dir = get_data_dir()
    cache_file = os.path.join(data_dir, f"{dataset_name}.npz")

    if os.path.exists(cache_file):
        loaded = np.load(cache_file)
        X, y = loaded["X"], loaded["y"]
    else:
        dataset = fetch_datasets(data_home=data_dir)[dataset_name]
        X, y = dataset.data, dataset.target
        y = np.where(y == 1, 1, 0)
        np.savez_compressed(cache_file, X=X, y=y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


# ==========================================
# 3. DANH SÁCH THUẬT TOÁN & MODEL DÙNG CHUNG
# ==========================================
def get_all_undersamplers(include_slow=True):
    """
    Trả về dict chứa tất cả các kỹ thuật Undersampling & Hybrid trong Chương 3
    đã được đồng bộ RANDOM_STATE để TV2, TV3, TV4 gọi trực tiếp.
    """
    samplers = {
        "NoSampling": None,
        # --- Nhóm của Thành viên 2 (Lấy mẫu đều & Mẫu dễ/xa biên) ---
        "RandomUS": RandomUnderSampler(random_state=RANDOM_STATE),
        "OSS": OneSidedSelection(random_state=RANDOM_STATE, n_seeds_S=10),
        # --- Nhóm của Thành viên 3 (Làm sạch biên & Lọc nhiễu) ---
        "ENN": EditedNearestNeighbours(),
        "RENN": RepeatedEditedNearestNeighbours(),
        "AllKNN": AllKNN(),
        "TomekLinks": TomekLinks(),
        "NCR": NeighbourhoodCleaningRule(),
        "IHT": InstanceHardnessThreshold(
            estimator=LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            random_state=RANDOM_STATE
        ),
        # --- Nhóm của Thành viên 4 (Kết hợp Lai) ---
        "SMOTEENN": SMOTEENN(random_state=RANDOM_STATE),
        "SMOTETomek": SMOTETomek(random_state=RANDOM_STATE),
    }

    if include_slow:
        # Hai thuật toán có độ phức tạp tính toán cao (thuộc phần TV2)
        samplers["ClusterCentroids"] = ClusterCentroids(random_state=RANDOM_STATE)
        samplers["CNN"] = CondensedNearestNeighbour(random_state=RANDOM_STATE)

    return samplers


def get_benchmark_models():
    """Trả về dict chứa 2 mô hình chuẩn (Logistic Regression & Random Forest)."""
    return {
        "LogisticRegression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1
        )
    }


# ==========================================
# 4. HÀM VẼ ĐỒ THỊ DÙNG CHUNG (TV2, TV3, LEADER aka Huydepchai dùng)
# ==========================================
def plot_resampled_comparison(X_orig, y_orig, X_res, y_res, method_name):
    """
    Vẽ biểu đồ 2D song song Trước và Sau khi Undersampling (Dành cho TV2 & TV3).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(
        x=X_orig.iloc[:, 0] if isinstance(X_orig, pd.DataFrame) else X_orig[:, 0],
        y=X_orig.iloc[:, 1] if isinstance(X_orig, pd.DataFrame) else X_orig[:, 1],
        hue=y_orig, style=y_orig, markers={0: "v", 1: "X"}, ax=axes[0]
    )
    axes[0].set_title(f"Original Dataset: {dict(Counter(y_orig))}")
    axes[0].set_xlabel("feature_1")
    axes[0].set_ylabel("feature_2")
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-5, 4)

    sns.scatterplot(
        x=X_res.iloc[:, 0] if isinstance(X_res, pd.DataFrame) else X_res[:, 0],
        y=X_res.iloc[:, 1] if isinstance(X_res, pd.DataFrame) else X_res[:, 1],
        hue=y_res, style=y_res, markers={0: "v", 1: "X"}, ax=axes[1]
    )
    axes[1].set_title(f"Undersampled using {method_name}: {dict(Counter(y_res))}")
    axes[1].set_xlabel("feature_1")
    axes[1].set_ylabel("feature_2")
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-5, 4)

    plt.tight_layout()
    return fig


def plot_decision_boundary(X, y, clf, classifier_name, ax=None):
    """
    Hàm vẽ đường biên quyết định (Decision Boundary) chuẩn cho toàn nhóm.
    Chỉ áp dụng cho dữ liệu có 2 đặc trưng (2D).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    DecisionBoundaryDisplay.from_estimator(
        clf, X_arr, response_method="predict", cmap="Wistia", alpha=0.5, ax=ax
    )
    sns.scatterplot(
        x=X_arr[:, 0], y=X_arr[:, 1], hue=y, style=y,
        markers={0: "v", 1: "X"}, palette={0: "orange", 1: "navy"}, ax=ax
    )
    ax.set_title(f"2-class classification using {classifier_name}")
    ax.set_xlabel("feature_1")
    ax.set_ylabel("feature_2")
    return ax


def plot_pr_curve(clf, X_test, y_test, name, ax=None):
    """Hàm vẽ đường cong Precision-Recall chuẩn sử dụng PrecisionRecallDisplay."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    disp = PrecisionRecallDisplay.from_estimator(
        clf, X_test, y_test, ax=ax, name=name
    )
    ax.set_title(f"Precision-Recall Curve ({name})")
    return disp


# ==========================================
# 5. HÀM ĐO ĐẠC THỰC NGHIỆM (TV4 & LEADER aka Huydepchai dùng)
# ==========================================
def evaluate_sampler_and_model(
    sampler, clf, X_train, y_train, X_test, y_test,
    method_name="NoSampling", model_name="Model", threshold=0.5
):
    """
    Chạy thực nghiệm tự động đo Training Time (gồm cả thời gian lấy mẫu + huấn luyện)
    và trả về dict chứa PR-AUC, F2, Precision, Recall.
    """
    start_time = time.perf_counter()

    if sampler is not None:
        X_res, y_res = sampler.fit_resample(X_train, y_train)
    else:
        X_res, y_res = X_train, y_train

    clf.fit(X_res, y_res)
    training_time = time.perf_counter() - start_time

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "Method": method_name,
        "Model": model_name,
        "Threshold": round(threshold, 3),
        "PR-AUC": round(average_precision_score(y_test, y_prob), 4),
        "F2": round(fbeta_score(y_test, y_pred, beta=2.0, zero_division=0), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "Training Time (s)": round(training_time, 4),
        "Resampled Shape": dict(Counter(y_res))
    }