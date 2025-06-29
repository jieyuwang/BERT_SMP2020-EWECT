#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感分析模型评估脚本
计算准确率、精确率、召回率、F1分数等评估指标
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix, roc_auc_score
)
import argparse
import json
from collections import Counter
import os

matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

print(matplotlib.matplotlib_fname())
print(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))

class EmotionEvaluator:
    def __init__(self, label_map=None):
        """
        初始化评估器
        
        Args:
            label_map: 标签映射字典，如 {0: 'happy', 1: 'angry', ...}
        """
        self.label_map = label_map or {
            0: 'happy', 1: 'angry', 2: 'sad', 
            3: 'fear', 4: 'surprise', 5: 'neutral'
        }
        self.label_to_id = {v: k for k, v in self.label_map.items()}
        
    def normalize_labels(self, labels):
        """
        标准化标签格式
        
        Args:
            labels: 标签列表
            
        Returns:
            标准化后的标签列表
        """
        def normalize_single_label(l):
            l = str(l).lower().strip()
            if l in self.label_to_id:
                return l
            # 处理可能的标签变体
            label_variants = {
                'happiness': 'happy', 'joy': 'happy', '高兴': 'happy',
                'anger': 'angry', 'mad': 'angry', '愤怒': 'angry',
                'sadness': 'sad', 'sorrow': 'sad', '悲伤': 'sad',
                'fearful': 'fear', 'scared': 'fear', '恐惧': 'fear',
                'surprised': 'surprise', 'shocked': 'surprise', '惊讶': 'surprise',
                'neutral': 'neutral', 'normal': 'neutral', '中性': 'neutral'
            }
            return label_variants.get(l, l)
        
        return [normalize_single_label(l) for l in labels]
    
    def calculate_metrics(self, true_labels, pred_labels):
        """
        计算评估指标
        
        Args:
            true_labels: 真实标签列表
            pred_labels: 预测标签列表
            
        Returns:
            包含各种评估指标的字典
        """
        # 标准化标签
        true_labels = self.normalize_labels(true_labels)
        pred_labels = self.normalize_labels(pred_labels)
        
        # 基础指标
        accuracy = accuracy_score(true_labels, pred_labels)
        
        # 加权平均指标
        precision_weighted, recall_weighted, f1_weighted, support_weighted = \
            precision_recall_fscore_support(true_labels, pred_labels, 
                                          average='weighted', zero_division=0)
        
        # 宏平均指标
        precision_macro, recall_macro, f1_macro, support_macro = \
            precision_recall_fscore_support(true_labels, pred_labels, 
                                          average='macro', zero_division=0)
        
        # 微平均指标
        precision_micro, recall_micro, f1_micro, support_micro = \
            precision_recall_fscore_support(true_labels, pred_labels, 
                                          average='micro', zero_division=0)
        
        # 每个类别的详细指标
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(true_labels, pred_labels, 
                                          labels=list(self.label_map.values()),
                                          zero_division=0)
        
        # 生成分类报告
        report = classification_report(true_labels, pred_labels, 
                                     target_names=list(self.label_map.values()),
                                     zero_division=0, output_dict=True)
        
        # 生成混淆矩阵
        conf_matrix = confusion_matrix(true_labels, pred_labels, 
                                     labels=list(self.label_map.values()))
        
        # 计算每个类别的准确率
        class_accuracy = {}
        for i, label in enumerate(self.label_map.values()):
            if label in true_labels:
                mask = np.array(true_labels) == label
                class_accuracy[label] = accuracy_score(
                    np.array(true_labels)[mask], 
                    np.array(pred_labels)[mask]
                )
            else:
                class_accuracy[label] = 0.0
        
        # 统计标签分布
        true_dist = Counter(true_labels)
        pred_dist = Counter(pred_labels)
        
        return {
            'accuracy': accuracy,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_per_class': dict(zip(self.label_map.values(), precision_per_class)),
            'recall_per_class': dict(zip(self.label_map.values(), recall_per_class)),
            'f1_per_class': dict(zip(self.label_map.values(), f1_per_class)),
            'support_per_class': dict(zip(self.label_map.values(), support_per_class)),
            'class_accuracy': class_accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': report,
            'true_label_distribution': dict(true_dist),
            'pred_label_distribution': dict(pred_dist),
            'support_weighted': support_weighted,
            'support_macro': support_macro,
            'support_micro': support_micro
        }
    
    def print_metrics(self, metrics):
        """
        打印评估指标
        
        Args:
            metrics: 评估指标字典
        """
        print("\n" + "="*60)
        print("情感分析模型评估结果")
        print("="*60)
        
        print(f"\n整体评估指标:")
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"精确率 (Precision) - 加权平均: {metrics['precision_weighted']:.4f}")
        print(f"召回率 (Recall) - 加权平均: {metrics['recall_weighted']:.4f}")
        print(f"F1分数 (F1-Score) - 加权平均: {metrics['f1_weighted']:.4f}")
        print(f"精确率 (Precision) - 宏平均: {metrics['precision_macro']:.4f}")
        print(f"召回率 (Recall) - 宏平均: {metrics['recall_macro']:.4f}")
        print(f"F1分数 (F1-Score) - 宏平均: {metrics['f1_macro']:.4f}")
        print(f"精确率 (Precision) - 微平均: {metrics['precision_micro']:.4f}")
        print(f"召回率 (Recall) - 微平均: {metrics['recall_micro']:.4f}")
        print(f"F1分数 (F1-Score) - 微平均: {metrics['f1_micro']:.4f}")
        
        print(f"\n各类别详细指标:")
        print(f"{'类别':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'准确率':<8} {'支持数':<8}")
        print("-" * 60)
        for label in self.label_map.values():
            print(f"{label:<12} {metrics['precision_per_class'][label]:<8.4f} "
                  f"{metrics['recall_per_class'][label]:<8.4f} "
                  f"{metrics['f1_per_class'][label]:<8.4f} "
                  f"{metrics['class_accuracy'][label]:<8.4f} "
                  f"{metrics['support_per_class'][label]:<8.0f}")
        
        print(f"\n标签分布:")
        print(f"真实标签分布: {metrics['true_label_distribution']}")
        print(f"预测标签分布: {metrics['pred_label_distribution']}")
        
        print(f"\n详细分类报告:")
        for label in self.label_map.values():
            report = metrics['classification_report'][label]
            print(f"{label:<12} 精确率: {report['precision']:.4f} 召回率: {report['recall']:.4f} F1: {report['f1-score']:.4f} 支持数: {report['support']}")
    
    def plot_confusion_matrix(self, metrics, save_path='confusion_matrix.png'):
        """
        绘制混淆矩阵
        
        Args:
            metrics: 评估指标字典
            save_path: 保存路径
        """
        conf_matrix = np.array(metrics['confusion_matrix'])
        # 自动创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=list(self.label_map.values()), 
                   yticklabels=list(self.label_map.values()))
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"混淆矩阵图已保存为 {save_path}")
    
    def convert_to_builtin_type(self, obj):
        """
        将numpy类型转换为Python原生类型，用于JSON序列化
        
        Args:
            obj: 要转换的对象
            
        Returns:
            转换后的对象
        """
        if isinstance(obj, dict):
            return {k: self.convert_to_builtin_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_builtin_type(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_metrics(self, metrics, save_path='evaluation_results.json'):
        """
        保存评估结果到JSON文件
        
        Args:
            metrics: 评估指标字典
            save_path: 保存路径
        """
        # 转换numpy类型为Python原生类型
        metrics = self.convert_to_builtin_type(metrics)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"评估结果已保存为 {save_path}")


def main():
    parser = argparse.ArgumentParser(description='情感分析模型评估')
    parser.add_argument('--input_file', type=str, default="data/data_weibo_model_output.xlsx", help='输入文件路径（Excel或CSV）')
    parser.add_argument('--true_label_column', type=str, default="sentiment", help='真实标签列名')
    parser.add_argument('--pred_label_column', type=str, default="sentiment_pred", help='预测标签列名')
    parser.add_argument('--output_dir', type=str, default='./data/metrics', help='输出目录')
    parser.add_argument('--file_type', type=str, default='excel', choices=['excel', 'csv'], help='文件类型')
    
    args = parser.parse_args()
    
    # 读取数据
    try:
        if args.file_type == 'excel':
            df = pd.read_excel(args.input_file)
        else:
            df = pd.read_csv(args.input_file)
        print(f"成功读取文件，共{len(df)}条数据")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 检查列是否存在
    if args.true_label_column not in df.columns:
        print(f"错误: 真实标签列 '{args.true_label_column}' 不存在")
        print(f"可用列名: {', '.join(df.columns)}")
        return
    
    if args.pred_label_column not in df.columns:
        print(f"错误: 预测标签列 '{args.pred_label_column}' 不存在")
        print(f"可用列名: {', '.join(df.columns)}")
        return
    
    # 获取标签数据
    true_labels = df[args.true_label_column].astype(str).tolist()
    pred_labels = df[args.pred_label_column].astype(str).tolist()
    
    # 创建评估器
    evaluator = EmotionEvaluator()
    
    # 计算评估指标
    metrics = evaluator.calculate_metrics(true_labels, pred_labels)
    
    # 打印结果
    evaluator.print_metrics(metrics)
    
    # 绘制混淆矩阵
    evaluator.plot_confusion_matrix(metrics, f"{args.output_dir}/confusion_matrix.png")
    
    # 保存评估结果
    evaluator.save_metrics(metrics, f"{args.output_dir}/evaluation_results.json")
    
    # 保存详细结果到Excel
    results_df = pd.DataFrame({
        'true_label': true_labels,
        'pred_label': pred_labels,
        'is_correct': [t == p for t, p in zip(true_labels, pred_labels)]
    })
    results_df.to_excel(f"{args.output_dir}/detailed_results.xlsx", index=False)
    print(f"详细结果已保存为 {args.output_dir}/detailed_results.xlsx")


if __name__ == '__main__':
    main() 