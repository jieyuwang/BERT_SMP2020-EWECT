# 微博情感分析项目流程说明

本项目实现了从原始数据到模型训练、预测、评估、可视化的全自动流程，适合论文实验与复现。

## 目录结构
- `data/clean/`：清洗、采样、划分后的数据
- `data/cleanImage/`：类别分布、文本长度等可视化图片
- `model/`：训练日志、最优模型、训练过程曲线图
- `data/data_new_output.xlsx`：已有真实标签的数据
- `data/data_weibo_model_output.xlsx`：批量预测结果
- `data/data_new.xlsx`:最原始的数据

## 1. 数据处理与采样

运行：
```bash
python 1_process_data.py
```
- 读取原始Excel，清洗文本，统计类别分布和文本长度。
- 每类采样1000条，8:1:1划分为train/val/test。
- 输出清洗数据和分布图片。

## 2. 模型训练

运行：
```bash
python 2_finetune_weibo.py --train_path data/clean/common_train.txt --val_path data/clean/common_val.txt --output_dir model
```
- 基于BERT微调，自动保存验证F1最优模型。
- 训练日志、损失/准确率/F1曲线自动保存。

## 3. 批量预测

运行：
```bash
python 3_predict_and_save.py --model_path model/weiboBest.pt --input_xlsx data/data_new_output.xlsx --output_xlsx data/data_weibo_model_output.xlsx
```
- 支持多进程，适合Mac CPU环境。
- 预测结果写入新Excel。

## 4. 评估与可视化

运行：
```bash
python 4_evaluate_metrics.py --input_file data/data_weibo_model_output.xlsx --true_label_column sentiment --pred_label_column sentiment_pred --output_dir data/metrics
```
- 输出准确率、精确率、召回率、F1、混淆矩阵等详细指标。
- 自动生成混淆矩阵图片和详细评估报告。

## 5. 真实标签批量推理与评估（可选）

如需对带有真实标签的Excel进行**批量推理+自动评估**，可使用：
```bash
python weiboInference.py --excel_path data/data_new.xlsx --output_path data/data_new_output.xlsx --text_column txt --true_label_column sentiment --evaluate
```
- 支持多进程高效推理。
- 自动对比真实标签与预测标签，输出详细评估指标和混淆矩阵。
- 适合直接处理带真实值的原始数据，便于复现实验和论文分析。

## 依赖环境

建议使用conda新建python=3.8环境，安装requirements.txt依赖。  
如遇matplotlib中文乱码，建议在脚本开头添加：
```python
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
```

## 复现与论文实验建议

- 采样均衡，便于小类学习与论文对比。
- 训练/推理/评估全流程自动化，日志与可视化丰富。
- 评估指标与可视化图片可直接用于论文。

---

如需详细参数说明或遇到环境问题，请查阅各脚本开头注释或联系作者。 