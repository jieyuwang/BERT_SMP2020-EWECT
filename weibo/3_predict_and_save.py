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

LABELS = ['happy', 'angry', 'sad', 'fear', 'surprise', 'neutral']
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=140):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {k: v.squeeze(0) for k, v in tokenized.items()}

def process_chunk(args, tokenizer, model, device, texts, chunk_idx):
    print(f"进程 {os.getpid()} 开始处理块 {chunk_idx}，设备: {device}")
    model.to(device)
    model.eval()
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    results = []
    for batch in tqdm(dataloader, desc=f"块 {chunk_idx} 进度"):
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        with torch.no_grad():
            outputs = model(**batch)
        preds = np.argmax(outputs.logits.detach().cpu().numpy(), axis=1).tolist()
        results.extend(preds)
    print(f"进程 {os.getpid()} 完成处理块 {chunk_idx}")
    return results

def main():
    parser = argparse.ArgumentParser(description='多进程BERT情感预测')
    parser.add_argument('--model_name', default='bert-base-chinese', type=str, help='HuggingFace模型名称')
    parser.add_argument('--model_path', default='model/weiboBest.pt', type=str, help='模型权重路径')
    parser.add_argument('--input_xlsx', default='data/data_new_output.xlsx', type=str, help='输入Excel文件')
    parser.add_argument('--output_xlsx', default='data/data_weibo_model_output.xlsx', type=str, help='输出Excel文件')
    parser.add_argument('--text_column', default='txt', type=str, help='文本列名')
    parser.add_argument('--batch_size', default=32, type=int, help='批处理大小')
    parser.add_argument('--num_processes', default=4, type=int, help='并行进程数')
    parser.add_argument('--chunksize', default=1000, type=int, help='每个进程处理的行数')
    args = parser.parse_args()

    # 加载模型和分词器
    print(f'加载模型: {args.model_name} ...')
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(LABELS))
    print(f'加载权重: {args.model_path} ...')
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 读取Excel
    df = pd.read_excel(args.input_xlsx)
    if args.text_column not in df.columns:
        raise ValueError(f"输入文件缺少 {args.text_column} 字段")
    texts = df[args.text_column].astype(str).tolist()

    # 多进程分块
    mp.set_start_method('spawn', force=True)
    num_processes = min(args.num_processes, len(texts))
    devices = [torch.device(f'cuda:{i % torch.cuda.device_count()}') if torch.cuda.is_available() else torch.device('cpu') for i in range(num_processes)]
    chunks = [texts[i:i + args.chunksize] for i in range(0, len(texts), args.chunksize)]
    worker_func = partial(process_chunk, args, tokenizer, model)
    print(f"使用 {num_processes} 个进程进行并行预测")
    with mp.Pool(processes=num_processes) as pool:
        results = []
        for i, chunk in enumerate(chunks):
            device_idx = i % len(devices)
            results.append(pool.apply_async(worker_func, (devices[device_idx], chunk, i)))
        all_preds = []
        for result in results:
            all_preds.extend(result.get())
    # 标签映射
    sentiment_labels = [id2label[pred] for pred in all_preds]
    df['sentiment_pred'] = sentiment_labels
    df.to_excel(args.output_xlsx, index=False)
    print(f'预测完成，结果已保存到 {args.output_xlsx}')

if __name__ == '__main__':
    main() 