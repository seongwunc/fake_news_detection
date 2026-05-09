# 虚假新闻检测任务复现说明

## 项目简介

本项目围绕虚假新闻二分类实验展开。流程包括数据读取、标签构造、全局随机洗牌、数据质量检查、文体特征提取、文本清洗、TF-IDF 向量化、特征拼接、逻辑回归与校准 LinearSVC 模型训练，以及分类报告、混淆矩阵和 ROC 曲线输出。

## 数据来源

由于数据集文件体积较大，本仓库不直接包含原始 CSV 文件。请通过以下链接下载数据并放入 `data/` 目录下：

* **数据集名称**：Fake News Detection
* **下载地址**：[Kaggle - Fake News Detection](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection)
* **文件要求**：下载后请确保包含 `true.csv`（真实新闻）与 `fake.csv`（虚假新闻）。

## 实验环境与结果摘要

> **环境声明**：本实验基于 **Python 3.12.6**，核心依赖为 **Pandas 2.2.3**、**NumPy 2.1.3**、**Scikit-learn 1.6.1**、**NLTK 3.9.1** 与 **Seaborn 0.13.2**。

本研究结合 TF-IDF 内容特征与文体风格特征（大写占比、感叹号频率）进行建模。实验显示：

* **逻辑回归 (Logistic Regression)**：准确率为 **0.9852**。
* **校准 LinearSVC**：准确率为 **0.9909**。

## 文件说明

* `fake_news_detection_analysis.ipynb`：主实验 Notebook，包含详细的代码解释与可视化分析。
* `run_fake_news_detection.py`：一键执行脚本，适合直接在终端运行。
* `requirements.txt`：项目依赖清单。
* `2400016608_陈湘媛_虚假新闻检测任务.docx`：最终报告文档。

## 推荐目录结构

在运行代码前，请手动创建 `data/` 文件夹并存放数据文件：

```text
.
├── data/
│   ├── true.csv                 # 需从 Kaggle 下载
│   └── fake.csv                 # 需从 Kaggle 下载
├── fake_news_detection_analysis.ipynb
├── run_fake_news_detection.py
├── requirements.txt
├── README.md
└── 2400016608_陈湘媛_虚假新闻检测任务.docx

```

## 运行步骤

### 1. 环境准备

确保已安装所需的 Python 包：

```bash
python3 -m pip install -r requirements.txt

```

### 2. 执行脚本

在项目根目录下运行，脚本会自动读取 `data/` 中的文件：

```bash
python3 run_fake_news_detection.py

```

### 3. Notebook 运行

如需逐步查看逻辑，可使用 Jupyter 打开 `fake_news_detection_analysis.ipynb`。代码已配置为相对路径读取，只要确保数据存放在 `./data/` 即可正常运行。

## 输出文件

运行完成后，结果将保存在 `outputs/fake_news_detection/` 目录下：

* **统计表**：数据质量统计 (`data_quality_summary.csv`)、模型指标汇总。
* **报告**：两种模型的详细分类报告 (`.txt`)。
* **图表**：标签分布图、单词长度 KDE 图、混淆矩阵及 ROC 曲线对比图。

## 关键技术点

* **防止特征泄露**：使用正则匹配剔除文本中的 Reuters 等机构前缀。
* **特征融合**：将 5000 维 TF-IDF 矩阵与文体统计特征进行水平拼接。
* **随机性控制**：统一使用 `RANDOM_STATE = 42` 以保证实验可复现。
