import pandas as pd
import re
import json
import os
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

def remove_url(text):
    """去除文本中的URL"""
    return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', str(text), flags=re.MULTILINE)

def clean_text(text):
    """可扩展的文本清洗函数，目前仅去除URL"""
    text = remove_url(text)
    text = str(text).strip()
    return text

def plot_label_dist(data, save_path, title):
    labels = [item['label'] for item in data]
    counter = Counter(labels)
    plt.figure(figsize=(8,5))
    plt.bar(counter.keys(), counter.values(), color='skyblue')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'类别分布图已保存到 {save_path}')

def plot_text_length(data, save_path, title):
    lengths = [len(item['content']) for item in data]
    plt.figure(figsize=(8,5))
    plt.hist(lengths, bins=30, color='orange', edgecolor='black')
    plt.xlabel('Text Length')
    plt.ylabel('Count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'文本长度分布图已保存到 {save_path}')

def process_excel(input_path):
    df = pd.read_excel(input_path)
    # 只保留txt和sentiment字段
    if 'txt' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError('输入文件缺少txt或sentiment字段')
    data = []
    for idx, row in df.iterrows():
        content = clean_text(row['txt'])
        label = str(row['sentiment']).strip()
        data.append({
            'id': idx + 1,  # ID从1开始
            'content': content,
            'label': label
        })
    return data

def sample_per_class(data, n=1000, seed=42):
    random.seed(seed)
    class_map = defaultdict(list)
    for item in data:
        class_map[item['label']].append(item)
    sampled = []
    for label, items in class_map.items():
        if len(items) >= n:
            sampled_items = random.sample(items, n)
        else:
            # 不足n条时，随机复制补足
            sampled_items = items * (n // len(items)) + random.sample(items, n % len(items))
        sampled.extend(sampled_items)
    return sampled

def split_data(data, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(data)
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    return train, val, test

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f'保存 {path}，共{len(data)}条')

if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    clean_dir = os.path.join(base_dir, 'data', 'clean')
    clean_image_dir = os.path.join(base_dir, 'data', 'cleanImage')

    os.makedirs(clean_dir, exist_ok=True)
    input_path = os.path.join(base_dir, 'data', 'data_new_output.xlsx')
    all_path = os.path.join(clean_dir, 'common_eval_labeled.txt')
    train_path = os.path.join(clean_dir, 'common_train.txt')
    val_path = os.path.join(clean_dir, 'common_val.txt')
    test_path = os.path.join(clean_dir, 'common_test.txt')

    # 1. 读取和清洗数据
    data = process_excel(input_path)
    save_json(data, all_path)
    # 可视化原始类别分布和文本长度
    plot_label_dist(data, os.path.join(clean_image_dir, 'common_label_dist_before.png'), 'Label Distribution (Before Sampling)')
    plot_text_length(data, os.path.join(clean_image_dir, 'common_text_length.png'), 'Text Length Distribution (All Data)')

    # 2. 采样每类1000条
    sampled = sample_per_class(data, n=1000)
    # 可视化采样后类别分布
    plot_label_dist(sampled, os.path.join(clean_image_dir, 'common_label_dist_after.png'), 'Label Distribution (After Sampling)')

    # 3. 8:1:1划分
    train, val, test = split_data(sampled, train_ratio=0.8, val_ratio=0.1)
    save_json(train, train_path)
    save_json(val, val_path)
    save_json(test, test_path) 