# 情感分析模型评估指南

本指南介绍如何使用 `weiboInference.py` 和 `evaluate_metrics.py` 来计算情感分析模型的评估指标。

## 功能概述

### 支持的评估指标

1. **准确率 (Accuracy)**: 正确预测的样本数 / 总样本数
2. **精确率 (Precision)**: 每个类别的正确预测数 / 该类别的总预测数
3. **召回率 (Recall)**: 每个类别的正确预测数 / 该类别的真实样本数
4. **F1分数 (F1-Score)**: 2 × (精确率 × 召回率) / (精确率 + 召回率)
5. **混淆矩阵 (Confusion Matrix)**: 显示预测结果与真实标签的对比
6. **各类别详细指标**: 每个情感类别的单独评估结果

### 支持的标签格式

- **英文标签**: `happy`, `angry`, `sad`, `fear`, `surprise`, `neutral`
- **中文标签**: `高兴`, `愤怒`, `悲伤`, `恐惧`, `惊讶`, `中性`
- **标签变体**: `happiness/joy`, `anger/mad`, `sadness/sorrow` 等

## 使用方法

### 方法1: 使用 weiboInference.py 进行推理并评估

```bash
python weiboInference.py \
    --excel_path your_data.xlsx \
    --output_path output.xlsx \
    --text_column txt \
    --true_label_column true_label \
    --evaluate
```

**参数说明**:
- `--excel_path`: 输入Excel文件路径
- `--output_path`: 输出Excel文件路径
- `--text_column`: 文本数据列名
- `--true_label_column`: 真实标签列名（用于评估）
- `--evaluate`: 启用评估模式

### 方法2: 使用独立评估脚本

```bash
python evaluate_metrics.py \
    --input_file output.xlsx \
    --true_label_column true_label \
    --pred_label_column sentiment \
    --output_dir ./results \
    --file_type excel
```

**参数说明**:
- `--input_file`: 包含真实标签和预测标签的文件
- `--true_label_column`: 真实标签列名
- `--pred_label_column`: 预测标签列名
- `--output_dir`: 输出目录
- `--file_type`: 文件类型（excel 或 csv）

## 输入数据格式

### Excel文件格式示例

| txt | true_label |
|-----|------------|
| 今天天气真好，心情愉快！ | happy |
| 工作压力太大了，很沮丧 | sad |
| 我对这个结果很愤怒 | angry |
| 这个决定让我很害怕 | fear |
| 突然听到这个消息，我很震惊 | surprise |
| 今天心情很平静 | neutral |

### 输出文件格式

运行后会生成以下文件：

1. **输出Excel文件**: 包含原始文本、真实标签、预测标签和预测正确性
2. **confusion_matrix.png**: 混淆矩阵可视化图
3. **evaluation_results.json**: 详细的评估指标JSON文件
4. **detailed_results.xlsx**: 逐条预测结果

## 评估指标详解

### 整体指标

- **准确率**: 整体预测准确程度
- **精确率-加权平均**: 考虑各类别样本数量的精确率
- **召回率-加权平均**: 考虑各类别样本数量的召回率
- **F1分数-加权平均**: 考虑各类别样本数量的F1分数
- **精确率-宏平均**: 各类别精确率的简单平均
- **召回率-宏平均**: 各类别召回率的简单平均
- **F1分数-宏平均**: 各类别F1分数的简单平均

### 各类别指标

对每个情感类别（happy, angry, sad, fear, surprise, neutral）计算：
- 精确率
- 召回率
- F1分数
- 准确率
- 支持数（样本数量）

## 代码示例

### 直接使用 EmotionEvaluator 类

```python
from evaluate_metrics import EmotionEvaluator
import pandas as pd

# 读取数据
df = pd.read_excel('your_data.xlsx')
true_labels = df['true_label'].tolist()
pred_labels = df['pred_label'].tolist()

# 创建评估器
evaluator = EmotionEvaluator()

# 计算指标
metrics = evaluator.calculate_metrics(true_labels, pred_labels)

# 打印结果
evaluator.print_metrics(metrics)

# 绘制混淆矩阵
evaluator.plot_confusion_matrix(metrics, 'confusion_matrix.png')

# 保存结果
evaluator.save_metrics(metrics, 'evaluation_results.json')
```

### 自定义标签映射

```python
# 自定义标签映射
custom_label_map = {
    0: 'positive',
    1: 'negative',
    2: 'neutral'
}

evaluator = EmotionEvaluator(custom_label_map)
```

## 输出示例

### 控制台输出

```
============================================================
情感分析模型评估结果
============================================================

整体评估指标:
准确率 (Accuracy): 0.8500
精确率 (Precision) - 加权平均: 0.8523
召回率 (Recall) - 加权平均: 0.8500
F1分数 (F1-Score) - 加权平均: 0.8511
精确率 (Precision) - 宏平均: 0.8333
召回率 (Recall) - 宏平均: 0.8333
F1分数 (F1-Score) - 宏平均: 0.8333

各类别详细指标:
类别           精确率    召回率    F1分数   准确率    支持数  
------------------------------------------------------------
happy         0.9000   0.9000   0.9000   0.9000   100     
angry         0.8000   0.8000   0.8000   0.8000   80      
sad           0.8500   0.8500   0.8500   0.8500   90      
fear          0.7500   0.7500   0.7500   0.7500   60      
surprise      0.7000   0.7000   0.7000   0.7000   50      
neutral       0.9000   0.9000   0.9000   0.9000   120     
```

### JSON输出示例

```json
{
  "accuracy": 0.85,
  "precision_weighted": 0.8523,
  "recall_weighted": 0.85,
  "f1_weighted": 0.8511,
  "precision_macro": 0.8333,
  "recall_macro": 0.8333,
  "f1_macro": 0.8333,
  "precision_per_class": {
    "happy": 0.9,
    "angry": 0.8,
    "sad": 0.85,
    "fear": 0.75,
    "surprise": 0.7,
    "neutral": 0.9
  },
  "recall_per_class": {
    "happy": 0.9,
    "angry": 0.8,
    "sad": 0.85,
    "fear": 0.75,
    "surprise": 0.7,
    "neutral": 0.9
  },
  "f1_per_class": {
    "happy": 0.9,
    "angry": 0.8,
    "sad": 0.85,
    "fear": 0.75,
    "surprise": 0.7,
    "neutral": 0.9
  }
}
```

## 注意事项

1. **数据格式**: 确保输入数据的标签格式正确，支持多种标签变体
2. **模型文件**: 使用 `weiboInference.py` 时需要确保模型文件存在
3. **内存使用**: 处理大量数据时注意内存使用情况
4. **标签一致性**: 真实标签和预测标签的格式需要保持一致

## 故障排除

### 常见问题

1. **列名不存在**: 检查Excel文件中的列名是否正确
2. **标签格式不匹配**: 确保标签格式统一
3. **模型文件缺失**: 确保模型文件路径正确
4. **内存不足**: 减少批处理大小或使用更少的并行进程

### 调试建议

1. 使用小数据集进行测试
2. 检查输入数据的格式
3. 查看详细的错误信息
4. 使用 `example_usage.py` 进行演示

## 扩展功能

### 添加新的评估指标

可以在 `EmotionEvaluator` 类中添加新的评估指标：

```python
def calculate_custom_metric(self, true_labels, pred_labels):
    # 自定义评估指标计算
    pass
```

### 支持新的标签格式

可以扩展 `normalize_labels` 方法来支持新的标签格式：

```python
def normalize_labels(self, labels):
    # 添加新的标签映射
    label_variants.update({
        'new_label': 'standard_label'
    })
```

## 联系信息

如有问题或建议，请联系开发团队。 