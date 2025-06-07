import torch
import argparse
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.multiprocessing as mp
from functools import partial
import os


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=140):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized = self.tokenizer(text, padding='max_length', truncation=True,
                                   max_length=self.max_length, return_tensors='pt')
        # 返回扁平化的张量
        return {k: v.squeeze(0) for k, v in tokenized.items()}


def process_chunk(args, tokenizer, model, label_map, device, texts, chunk_idx):
    """处理一个数据块的情感分析"""
    print(f"进程 {os.getpid()} 开始处理块 {chunk_idx}，设备: {device}")

    # 设置当前进程的设备
    model.to(device)
    model.eval()

    # 创建数据加载器
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    results = []
    for batch in tqdm(dataloader, desc=f"块 {chunk_idx} 进度"):
        # 只将张量类型的数据移到设备
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

        with torch.no_grad():
            outputs = model(**batch)

        # 获取预测结果
        preds = np.argmax(outputs.logits.detach().cpu().numpy(), axis=1).tolist()
        results.extend(preds)

    print(f"进程 {os.getpid()} 完成处理块 {chunk_idx}")
    return results


def main():
    parser = argparse.ArgumentParser(description='并行情感分析')
    parser.add_argument('--input', type=str, default='所以注定我这辈子是做不了商人妈蛋', help='输入文本')
    parser.add_argument('--device', default='cpu', type=str, help='使用设备: cpu或cuda')
    parser.add_argument('--model_name', default='bert-base-chinese', type=str, help='HuggingFace模型名称')
    parser.add_argument('--model_path', default='../workspace/wb/best.pt', type=str, help='模型路径')
    parser.add_argument('--num_labels', default=6, type=int, help='分类标签数量')
    parser.add_argument('--excel_path', default='jingjinji_text.xlsx', type=str, help='输入Excel文件路径')
    parser.add_argument('--output_path', default='jingjinji_text_output.xlsx', type=str, help='输出Excel文件路径')
    parser.add_argument('--text_column', default='txt', type=str, help='文本数据列名')
    parser.add_argument('--batch_size', default=16, type=int, help='每个进程的批处理大小')
    parser.add_argument('--num_processes', default=4, type=int, help='并行进程数')
    parser.add_argument('--chunksize', default=1000, type=int, help='每个进程处理的行数')
    args = parser.parse_args()

    # 标签映射
    label = {0: 'happy', 1: 'angry', 2: 'sad', 3: 'fear', 4: 'surprise', 5: 'neutral'}

    # 加载模型和分词器
    print(f'加载模型: {args.model_name} ...')
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
    print(f'加载检查点: {args.model_path} ...')
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 读取Excel文件
    try:
        df = pd.read_excel(args.excel_path)
        print(f"成功读取Excel文件，共{len(df)}条数据")
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return

    # 确保列名存在
    if args.text_column not in df.columns:
        print(f"错误: 指定的列名 '{args.text_column}' 不存在于Excel文件中")
        print(f"可用列名: {', '.join(df.columns)}")
        return

    # 获取所有文本
    texts = df[args.text_column].astype(str).tolist()

    # 创建进程池
    mp.set_start_method('spawn', force=True)
    num_processes = min(args.num_processes, len(texts))

    # 确定设备分配
    if args.device == 'cuda' and torch.cuda.is_available():
        devices = [torch.device(f'cuda:{i % torch.cuda.device_count()}') for i in range(num_processes)]
    else:
        devices = [torch.device('cpu')] * num_processes

    # 分割文本为块
    chunks = [texts[i:i + args.chunksize] for i in range(0, len(texts), args.chunksize)]

    # 创建部分应用函数
    worker_func = partial(process_chunk, args, tokenizer, model, label)

    # 启动进程池
    print(f"使用 {num_processes} 个进程进行并行处理")
    with mp.Pool(processes=num_processes) as pool:
        results = []
        for i, chunk in enumerate(chunks):
            device_idx = i % len(devices)
            # 修正参数顺序
            results.append(pool.apply_async(worker_func, (devices[device_idx], chunk, i)))

        # 收集结果
        all_preds = []
        for result in results:
            all_preds.extend(result.get())

    # 将预测结果转换为标签
    sentiment_labels = [label[pred] for pred in all_preds]

    # 将结果添加到DataFrame
    df['sentiment'] = sentiment_labels

    # 保存结果
    try:
        df.to_excel(args.output_path, index=False)
        print(f"成功将结果保存到 {args.output_path}")
    except Exception as e:
        print(f"保存结果失败: {e}")


if __name__ == '__main__':
    main()