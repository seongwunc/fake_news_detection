import json
import re
import string
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import gaussian_kde
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# Use paths relative to this script so the project folder can be moved.
BASE_DIR = Path(__file__).resolve().parent
TRUE_PATH = BASE_DIR / "data" / "true.csv"
FAKE_PATH = BASE_DIR / "data" / "fake.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "fake_news_detection"
RANDOM_STATE = 42


def log_step(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def get_font(size: int = 16):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_axes(draw, x0, y0, x1, y1, title, xlabel, ylabel):
    font = get_font(15)
    small = get_font(12)
    draw.rectangle([x0, y0, x1, y1], outline="#333333", width=2)
    draw.text(((x0 + x1) / 2, 18), title, fill="#111111", anchor="mt", font=get_font(22))
    draw.text(((x0 + x1) / 2, y1 + 42), xlabel, fill="#111111", anchor="mt", font=font)
    draw.text((22, (y0 + y1) / 2), ylabel, fill="#111111", anchor="mm", font=small)


def save_target_distribution(counts: pd.Series) -> Path:
    path = OUTPUT_DIR / "target_distribution.png"
    img = Image.new("RGB", (920, 650), "white")
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 110, 90, 850, 540
    draw_axes(draw, x0, y0, x1, y1, "Target Label Distribution", "Target (0 = Fake, 1 = True)", "Count")
    max_count = max(counts.max(), 1)
    colors = {0: "#D55E00", 1: "#0072B2"}
    labels = {0: "Fake (0)", 1: "True (1)"}
    for i, target in enumerate([0, 1]):
        count = int(counts.get(target, 0))
        bar_w = 180
        cx = x0 + 220 + i * 300
        bar_h = int((count / max_count) * (y1 - y0 - 45))
        draw.rectangle([cx - bar_w // 2, y1 - bar_h, cx + bar_w // 2, y1], fill=colors[target])
        draw.text((cx, y1 + 12), labels[target], fill="#111111", anchor="mt", font=get_font(14))
        draw.text((cx, y1 - bar_h - 12), f"{count:,}", fill="#111111", anchor="mb", font=get_font(15))
    img.save(path)
    return path


def save_word_count_kde(df: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "word_count_kde.png"
    img = Image.new("RGB", (1000, 650), "white")
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 105, 90, 920, 540
    draw_axes(draw, x0, y0, x1, y1, "Word Count KDE by Target", "Word Count", "Density")
    xmax = float(df["word_count"].quantile(0.99))
    xs = np.linspace(0, xmax, 300)
    curves = {}
    max_y = 0.0
    for target in [0, 1]:
        values = df.loc[df["target"] == target, "word_count"].clip(upper=xmax).to_numpy(dtype=float)
        kde = gaussian_kde(values)
        ys = kde(xs)
        curves[target] = ys
        max_y = max(max_y, float(ys.max()))
    colors = {0: "#D55E00", 1: "#0072B2"}
    for target, ys in curves.items():
        pts = []
        for x, y in zip(xs, ys):
            px = x0 + (x / xmax) * (x1 - x0)
            py = y1 - (y / max_y) * (y1 - y0 - 20)
            pts.append((px, py))
        draw.line(pts, fill=colors[target], width=4)
    for tick in np.linspace(0, xmax, 6):
        px = x0 + (tick / xmax) * (x1 - x0)
        draw.line([(px, y1), (px, y1 + 6)], fill="#333333", width=1)
        draw.text((px, y1 + 10), f"{tick:.0f}", fill="#111111", anchor="mt", font=get_font(11))
    draw.rectangle([690, 105, 890, 165], outline="#CCCCCC", fill="white")
    draw.line([(710, 125), (755, 125)], fill=colors[0], width=4)
    draw.text((765, 116), "Fake (0)", fill="#111111", font=get_font(13))
    draw.line([(710, 148), (755, 148)], fill=colors[1], width=4)
    draw.text((765, 139), "True (1)", fill="#111111", font=get_font(13))
    img.save(path)
    return path


def save_confusion_matrix(cm: np.ndarray, name: str) -> Path:
    path = OUTPUT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    img = Image.new("RGB", (780, 680), "white")
    draw = ImageDraw.Draw(img)
    draw.text((390, 28), f"Confusion Matrix - {name}", fill="#111111", anchor="mt", font=get_font(21))
    x0, y0, cell = 190, 130, 190
    max_v = max(int(cm.max()), 1)
    for r in range(2):
        for c in range(2):
            v = int(cm[r, c])
            shade = int(245 - 155 * (v / max_v))
            fill = (shade, shade + 8 if shade < 247 else shade, 255)
            draw.rectangle([x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell, y0 + (r + 1) * cell], fill=fill, outline="#333333", width=2)
            draw.text((x0 + c * cell + cell / 2, y0 + r * cell + cell / 2), f"{v:,}", fill="#111111", anchor="mm", font=get_font(28))
    labels = ["Fake (0)", "True (1)"]
    for i, label in enumerate(labels):
        draw.text((x0 + i * cell + cell / 2, y0 - 16), label, fill="#111111", anchor="mb", font=get_font(14))
        draw.text((x0 - 18, y0 + i * cell + cell / 2), label, fill="#111111", anchor="rm", font=get_font(14))
    draw.text((x0 + cell, y0 + 2 * cell + 52), "Predicted", fill="#111111", anchor="mt", font=get_font(16))
    draw.text((82, y0 + cell), "Actual", fill="#111111", anchor="mm", font=get_font(16))
    img.save(path)
    return path


def save_roc_curve(roc_rows) -> Path:
    path = OUTPUT_DIR / "roc_curve_comparison.png"
    img = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 105, 95, 805, 585
    draw_axes(draw, x0, y0, x1, y1, "ROC Curve Comparison", "False Positive Rate", "True Positive Rate")
    for tick in np.linspace(0, 1, 6):
        px = x0 + tick * (x1 - x0)
        py = y1 - tick * (y1 - y0)
        draw.line([(px, y1), (px, y1 + 6)], fill="#333333")
        draw.line([(x0 - 6, py), (x0, py)], fill="#333333")
        draw.text((px, y1 + 10), f"{tick:.1f}", fill="#111111", anchor="mt", font=get_font(11))
        draw.text((x0 - 10, py), f"{tick:.1f}", fill="#111111", anchor="rm", font=get_font(11))
    draw.line([(x0, y1), (x1, y0)], fill="#888888", width=2)
    colors = ["#0072B2", "#D55E00"]
    for i, (name, fpr, tpr, auc) in enumerate(roc_rows):
        pts = [(x0 + f * (x1 - x0), y1 - t * (y1 - y0)) for f, t in zip(fpr, tpr)]
        draw.line(pts, fill=colors[i], width=4)
        ly = 120 + i * 32
        draw.line([(570, ly), (620, ly)], fill=colors[i], width=4)
        draw.text((630, ly - 10), f"{name} (AUC = {auc:.4f})", fill="#111111", font=get_font(13))
    img.save(path)
    return path


def remove_news_prefix(text: str) -> str:
    text = str(text)
    patterns = [
        r"^\s*[A-Z][A-Z\s.,'-]{2,80}\s+\(Reuters\)\s*[-—]\s*",
        r"^\s*\([Rr]euters\)\s*[-—]\s*",
        r"^\s*[A-Z][A-Za-z\s.,'-]{2,80}\s+\(Reuters\)\s*[-—]\s*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, count=1)
    return text


stemmer = PorterStemmer()
stop_words = set(ENGLISH_STOP_WORDS)
punct_digit_re = re.compile(r"[^A-Za-z\s]+")


def clean_text(text: str) -> str:
    text = remove_news_prefix(text)
    text = punct_digit_re.sub(" ", text)
    tokens = [
        stemmer.stem(token)
        for token in text.lower().split()
        if token not in stop_words and len(token) > 1
    ]
    return " ".join(tokens)


def caps_ratio(text: str) -> float:
    words = re.findall(r"\b[A-Za-z]+\b", str(text))
    if not words:
        return 0.0
    caps = sum(1 for word in words if len(word) > 1 and word.isupper())
    return caps / len(words)


def excl_ratio(text: str) -> float:
    text = str(text)
    words = re.findall(r"\b\w+\b", text)
    denom = max(len(words), 1)
    return text.count("!") / denom


def main() -> None:
    started = time.time()
    log_step("开始运行虚假新闻检测流程")
    log_step(f"数据目录: {BASE_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_step("读取 true.csv 和 fake.csv")
    true_df = pd.read_csv(TRUE_PATH)
    fake_df = pd.read_csv(FAKE_PATH)
    true_df["target"] = 1
    fake_df["target"] = 0
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    log_step(f"数据读取完成，共 {len(df):,} 条样本")

    df["combined_text"] = (
        df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
    )
    log_step("执行数据质量检查")
    missing_values = df.isna().sum().rename("missing_count").to_frame()
    duplicate_count = int(df.duplicated(subset=["title", "text"]).sum())
    quality = missing_values.copy()
    quality.loc["duplicate_title_text_rows", "missing_count"] = duplicate_count
    quality.to_csv(OUTPUT_DIR / "data_quality_summary.csv")

    log_step("计算 word_count、caps_ratio 和 excl_ratio")
    df["word_count"] = df["combined_text"].str.split().str.len()
    df["caps_ratio"] = df["combined_text"].map(caps_ratio)
    df["excl_ratio"] = df["combined_text"].str.count("!") / df["word_count"].clip(lower=1)

    log_step("生成标签分布图和单词长度 KDE 图")
    target_distribution_path = save_target_distribution(df["target"].value_counts())
    word_count_kde_path = save_word_count_kde(df)

    log_step("清洗文本并进行 Porter 词干提取，这一步耗时较长")
    df["clean_text"] = df["combined_text"].map(clean_text)
    df[["title", "subject", "date", "target", "word_count", "caps_ratio", "excl_ratio", "clean_text"]].head(20).to_csv(
        OUTPUT_DIR / "processed_sample.csv", index=False
    )

    log_step("划分训练集和测试集")
    X_train_text, X_test_text, y_train, y_test, train_idx, test_idx = train_test_split(
        df["clean_text"],
        df["target"],
        df.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    log_step("拟合 TF-IDF 向量器")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    style_cols = ["caps_ratio", "excl_ratio"]
    scaler = StandardScaler()
    X_train_style = scaler.fit_transform(df.loc[train_idx, style_cols])
    X_test_style = scaler.transform(df.loc[test_idx, style_cols])
    X_train = hstack([X_train_tfidf, csr_matrix(X_train_style)])
    X_test = hstack([X_test_tfidf, csr_matrix(X_test_style)])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=1),
        "Calibrated LinearSVC": CalibratedClassifierCV(
            LinearSVC(random_state=RANDOM_STATE),
            method="sigmoid",
            cv=3,
        ),
    }

    reports = {}
    roc_rows = []
    for name, model in models.items():
        log_step(f"训练模型: {name}")
        model.fit(X_train, y_train)
        log_step(f"评估模型: {name}")
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        reports[name] = classification_report(y_test, pred, output_dict=True, digits=4)
        with open(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_classification_report.txt", "w") as f:
            f.write(classification_report(y_test, pred, digits=4))

        cm = confusion_matrix(y_test, pred)
        save_confusion_matrix(cm, name)

        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        roc_rows.append((name, fpr, tpr, auc))

    log_step("生成 ROC 曲线对比图")
    roc_path = save_roc_curve(roc_rows)

    log_step("汇总模型指标并保存结果")
    metrics = []
    for name, report in reports.items():
        metrics.append(
            {
                "model": name,
                "accuracy": report["accuracy"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"],
            }
        )
    pd.DataFrame(metrics).to_csv(OUTPUT_DIR / "model_metrics_summary.csv", index=False)
    with open(OUTPUT_DIR / "classification_reports.json", "w") as f:
        json.dump(reports, f, indent=2)

    summary = {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "duplicate_title_text_rows": duplicate_count,
        "target_distribution_path": str(target_distribution_path),
        "word_count_kde_path": str(word_count_kde_path),
        "roc_curve_path": str(roc_path),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    with open(OUTPUT_DIR / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    log_step(f"运行结束，用时 {time.time() - started:.2f} 秒")
    log_step(f"结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
