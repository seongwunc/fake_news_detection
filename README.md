# 虚假新闻检测任务复现说明

## 项目简介

本项目围绕 `true.csv` 与 `fake.csv` 构建虚假新闻二分类实验。流程包括数据读取、标签构造、全局随机洗牌、数据质量检查、文体特征提取、文本清洗、TF-IDF 向量化、特征拼接、逻辑回归与校准 LinearSVC 模型训练，以及分类报告、混淆矩阵和 ROC 曲线输出。

## 文件说明

- `fake_news_detection_analysis.ipynb`：主实验 notebook，已加入注释和代码解释。
- `run_fake_news_detection.py`：一键执行脚本，适合直接在终端运行并生成全部结果文件。
- `requirements.txt`：依赖清单，已补齐版本号。

- `true.csv`：真实新闻数据。
- `fake.csv`：虚假新闻数据。
- `2400016608_陈湘媛_虚假新闻检测任务.docx`：最终报告文档。

## 推荐目录结构

请将下列文件放在同一目录下：

```text
案例二/
├── data/
    ├── true.csv
    ├── fake.csv
├── fake_news_detection_analysis.ipynb
├── run_fake_news_detection.py
├── requirements.txt
├── README.md
└── 2400016608_陈湘媛_虚假新闻检测任务.docx
```

## 一键运行步骤

在终端执行：

```bash
cd /Users/sylviachan/Desktop/案例二
python3 -m pip install -r requirements.txt
python3 run_fake_news_detection.py
```

运行完成后，脚本会在 `outputs/fake_news_detection/` 中生成数据质量表、分类报告、模型指标表和全部图表。

## Notebook 运行方式

如需逐步查看代码逻辑，可打开 `fake_news_detection_analysis.ipynb`，从上到下依次运行所有单元。Notebook 已改为相对路径读取数据，因此 `true.csv` 和 `fake.csv` 必须与 notebook 位于同一目录。

## 主要结果

代码运行结果显示，合并数据集共有 `44898` 条新闻样本，各主要字段缺失值为 `0`，按 `title` 与 `text` 完全相同口径统计的重复样本数为 `5793` 条。逻辑回归测试准确率为 `0.9852`，校准 LinearSVC 测试准确率为 `0.9909`。

## 输出文件

主要输出包括：

- `outputs/fake_news_detection/data_quality_summary.csv`
- `outputs/fake_news_detection/model_metrics_summary.csv`
- `outputs/fake_news_detection/logistic_regression_classification_report.txt`
- `outputs/fake_news_detection/calibrated_linearsvc_classification_report.txt`
- `outputs/fake_news_detection/target_distribution.png`
- `outputs/fake_news_detection/word_count_kde.png`
- `outputs/fake_news_detection/confusion_matrix_logistic_regression.png`
- `outputs/fake_news_detection/confusion_matrix_calibrated_linearsvc.png`
- `outputs/fake_news_detection/roc_curve_comparison.png`
