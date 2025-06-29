#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感分析评估使用示例
展示如何使用weiboInference.py和evaluate_metrics.py进行模型评估
"""

import pandas as pd
import numpy as np
from evaluate_metrics import EmotionEvaluator
import subprocess
import sys
import os


def create_sample_data():
    """
    创建示例数据用于演示
    """
    # 创建示例数据
    sample_data = {
        'txt': [
            '今天天气真好，心情愉快！',
            '这个电影太棒了，我很开心',
            '工作压力太大了，很沮丧',
            '这个结果让我很失望',
            '突然听到这个消息，我很震惊',
            '这个决定让我很害怕',
            '我对这个结果很愤怒',
            '今天心情很平静',
            '这个笑话太好笑了',
            '我对未来感到担忧'
        ],
        'true_label': [
            'happy', 'happy', 'sad', 'sad', 'surprise',
            'fear', 'angry', 'neutral', 'happy', 'fear'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_excel('sample_data.xlsx', index=False)
    print("示例数据已保存为 sample_data.xlsx")
    return df


def run_inference_with_evaluation():
    """
    运行推理并计算评估指标
    """
    print("=== 步骤1: 创建示例数据 ===")
    df = create_sample_data()
    
    print("\n=== 步骤2: 运行情感分析推理 ===")
    # 注意：这里需要确保模型文件存在
    cmd = [
        'python', 'weiboInference.py',
        '--excel_path', 'sample_data.xlsx',
        '--output_path', 'sample_data_output.xlsx',
        '--text_column', 'txt',
        '--true_label_column', 'true_label',
        '--evaluate'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("推理完成")
        if result.stdout:
            print("输出:", result.stdout)
        if result.stderr:
            print("错误:", result.stderr)
    except Exception as e:
        print(f"运行推理失败: {e}")
        print("请确保模型文件存在且路径正确")
    
    print("\n=== 步骤3: 使用独立评估脚本 ===")
    # 使用独立的评估脚本
    cmd_eval = [
        'python', 'evaluate_metrics.py',
        '--input_file', 'sample_data_output.xlsx',
        '--true_label_column', 'true_label',
        '--pred_label_column', 'sentiment',
        '--output_dir', './',
        '--file_type', 'excel'
    ]
    
    try:
        result = subprocess.run(cmd_eval, capture_output=True, text=True)
        print("评估完成")
        if result.stdout:
            print("输出:", result.stdout)
        if result.stderr:
            print("错误:", result.stderr)
    except Exception as e:
        print(f"运行评估失败: {e}")


def demonstrate_evaluator_usage():
    """
    演示EmotionEvaluator类的直接使用
    """
    print("\n=== 演示EmotionEvaluator直接使用 ===")
    
    # 创建示例数据
    true_labels = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral'] * 5
    pred_labels = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral'] * 4 + ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral']
    
    # 创建评估器
    evaluator = EmotionEvaluator()
    
    # 计算指标
    metrics = evaluator.calculate_metrics(true_labels, pred_labels)
    
    # 打印结果
    evaluator.print_metrics(metrics)
    
    # 绘制混淆矩阵
    evaluator.plot_confusion_matrix(metrics, 'demo_confusion_matrix.png')
    
    # 保存结果
    evaluator.save_metrics(metrics, 'demo_evaluation_results.json')
    
    print("\n演示完成！")


def show_usage_instructions():
    """
    显示使用说明
    """
    print("\n" + "="*60)
    print("情感分析评估使用说明")
    print("="*60)
    
    print("\n1. 使用weiboInference.py进行推理并评估:")
    print("   python weiboInference.py \\")
    print("       --excel_path your_data.xlsx \\")
    print("       --output_path output.xlsx \\")
    print("       --text_column txt \\")
    print("       --true_label_column true_label \\")
    print("       --evaluate")
    
    print("\n2. 使用独立评估脚本:")
    print("   python evaluate_metrics.py \\")
    print("       --input_file output.xlsx \\")
    print("       --true_label_column true_label \\")
    print("       --pred_label_column sentiment \\")
    print("       --output_dir ./results")
    
    print("\n3. 支持的标签格式:")
    print("   - 英文: happy, angry, sad, fear, surprise, neutral")
    print("   - 中文: 高兴, 愤怒, 悲伤, 恐惧, 惊讶, 中性")
    print("   - 变体: happiness/joy, anger/mad, sadness/sorrow等")
    
    print("\n4. 输出的评估指标:")
    print("   - 准确率 (Accuracy)")
    print("   - 精确率 (Precision) - 加权/宏/微平均")
    print("   - 召回率 (Recall) - 加权/宏/微平均")
    print("   - F1分数 (F1-Score) - 加权/宏/微平均")
    print("   - 混淆矩阵")
    print("   - 各类别详细指标")
    
    print("\n5. 输出文件:")
    print("   - confusion_matrix.png: 混淆矩阵图")
    print("   - evaluation_results.json: 详细评估结果")
    print("   - detailed_results.xlsx: 逐条预测结果")


if __name__ == '__main__':
    print("情感分析评估示例")
    print("="*40)
    
    # 显示使用说明
    show_usage_instructions()
    
    # 演示评估器使用
    demonstrate_evaluator_usage()
    
    # 尝试运行完整流程（需要模型文件）
    print("\n是否尝试运行完整推理流程？(需要模型文件)")
    response = input("输入 'y' 继续，其他键跳过: ")
    
    if response.lower() == 'y':
        run_inference_with_evaluation()
    else:
        print("跳过完整流程演示")
    
    print("\n示例完成！") 