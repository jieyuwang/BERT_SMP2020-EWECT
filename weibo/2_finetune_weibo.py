import os
import json
import torch
import argparse
from loguru import logger
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

LABELS = ['happy', 'angry', 'sad', 'fear', 'surprise', 'neutral']
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

class WeiboDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=140):
        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['content']
        label = label2id[item['label']]
        tokenized = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length)
        return {
            'input_ids': torch.tensor(tokenized['input_ids']),
            'token_type_ids': torch.tensor(tokenized['token_type_ids']),
            'attention_mask': torch.tensor(tokenized['attention_mask']),
            'labels': torch.tensor(label)
        }

def evaluate(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Eval'):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            label = batch['labels'].cpu().numpy()
            preds.extend(pred)
            trues.extend(label)
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='macro')
    return total_loss / len(dataloader), acc, f1

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_set = WeiboDataset(args.train_path, tokenizer)
    val_set = WeiboDataset(args.val_path, tokenizer)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(LABELS))
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    best_f1 = 0
    log_file = os.path.join(args.output_dir, 'train_log.txt')
    logger.add(log_file, rotation='1 week', retention='30 days', enqueue=True)
    logger.info(f"训练参数: batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}, model={args.model_name}")
    # 记录指标
    train_losses, val_losses, val_accs, val_f1s = [], [], [], []
    for epoch in range(1, args.epochs+1):
        model.train()
        total_steps_epoch = len(train_loader)
        pbar = tqdm(enumerate(train_loader, 1), total=total_steps_epoch, desc=f'Epoch {epoch}/{args.epochs}')
        total_loss = 0
        for step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            avg_loss = total_loss / step
            pbar.set_postfix({'loss': avg_loss, 'step': f'{step}/{total_steps_epoch}'})
        avg_train_loss = total_loss/total_steps_epoch
        train_losses.append(avg_train_loss)
        writer.add_scalar('Train/loss', avg_train_loss, epoch)
        logger.info(f'Epoch {epoch}/{args.epochs} 训练损失: {avg_train_loss:.4f}')
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        writer.add_scalar('Val/loss', val_loss, epoch)
        writer.add_scalar('Val/acc', val_acc, epoch)
        writer.add_scalar('Val/f1', val_f1, epoch)
        logger.info(f'Epoch {epoch}/{args.epochs} 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}, 验证F1: {val_f1:.4f}')
        print(f'Epoch {epoch}/{args.epochs}: 训练损失={avg_train_loss:.4f}, 验证损失={val_loss:.4f}, 验证准确率={val_acc:.4f}, 验证F1={val_f1:.4f}')
        # 保存最优模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(args.output_dir, 'weiboBest.pt')
            torch.save({'state_dict': model.state_dict()}, save_path)
            logger.info(f'保存最优模型到 {save_path}')
    writer.close()
    logger.info('训练结束')
    # 训练过程可视化
    plt.figure(figsize=(10,6))
    plt.plot(range(1, args.epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs+1), val_losses, label='Val Loss')
    plt.plot(range(1, args.epochs+1), val_accs, label='Val Acc')
    plt.plot(range(1, args.epochs+1), val_f1s, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Training & Validation Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, 'train_val_metrics.png')
    plt.savefig(fig_path, dpi=300)
    print(f'训练过程曲线图已保存到 {fig_path}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/clean/common_train.txt')
    parser.add_argument('--val_path', type=str, default='data/clean/common_val.txt')
    parser.add_argument('--model_name', type=str, default='bert-base-chinese')
    parser.add_argument('--output_dir', type=str, default='model')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args) 