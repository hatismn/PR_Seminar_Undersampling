import os
import sys

# Đảm bảo Python nhận diện được thư mục gốc của project khi chạy pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import (
    load_synthetic_data,
    load_real_data,
    get_all_undersamplers,
    get_benchmark_models,
    evaluate_sampler_and_model
)


def test_load_synthetic_data():
    """Kiểm tra hàm tạo dữ liệu giả lập 2D và chia train/test."""
    X_tr, X_te, y_tr, y_te, X, y = load_synthetic_data(n_samples=1000, test_size=0.2)
    assert X_tr.shape == (800, 2)
    assert X_te.shape == (200, 2)
    assert len(y_tr) == 800
    assert len(y_te) == 200


def test_load_real_data():
    """Kiểm tra đọc dữ liệu thực tế thyroid_sick đã lưu trong thư mục data/."""
    X_tr, X_te, y_tr, y_te = load_real_data("thyroid_sick", test_size=0.2)
    assert X_tr.shape[0] > 0
    assert X_te.shape[0] > 0
    assert set(y_tr).issubset({0, 1})


def test_samplers_and_evaluation_pipeline():
    """Kiểm tra pipeline chạy lấy mẫu và đo đạc các chỉ số chuẩn của nhóm."""
    X_tr, X_te, y_tr, y_te, _, _ = load_synthetic_data(n_samples=1000, test_size=0.2)
    samplers = get_all_undersamplers(include_slow=False)
    models = get_benchmark_models()

    # Chạy thử nghiệm nhanh trên RandomUS và LogisticRegression
    res = evaluate_sampler_and_model(
        sampler=samplers["RandomUS"],
        clf=models["LogisticRegression"],
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        method_name="RandomUS",
        model_name="LogisticRegression"
    )

    expected_keys = {
        "Method", "Model", "Threshold", "PR-AUC",
        "F2", "Precision", "Recall", "Training Time (s)", "Resampled Shape"
    }
    assert expected_keys.issubset(res.keys())
    assert 0.0 <= res["PR-AUC"] <= 1.0
    assert 0.0 <= res["F2"] <= 1.0