import gradio as gr
import torch
import argparse
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Start Web Demo')
    parser.add_argument('--device', default='cpu', type=str, help='cpu or cuda')
    parser.add_argument('--model_name', default='bert-base-chinese', type=str,
                        help='huggingface transformer model name')
    parser.add_argument('--model_path', default='workspace/wb/best.pt', type=str, help='model path')
    parser.add_argument('--num_labels', default=6, type=int, help='fine-tune num labels')

    args = parser.parse_args()
    return args


class Runer():
    def __init__(self, label, device, model_name, model_path, *args, **kwargs):
        self.device = device
        self.label = label
        self.model_name = model_name
        self.num_labels = len(label.keys())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self.build_model(model_name, model_path, self.num_labels)

    def build_model(self, model_name, model_path, num_labels):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        print(f'Loading checkpoint: {model_path} ...')
        checkpoint = torch.load(model_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(f'missing_keys: {missing_keys}\n'
              f'===================================================================\n')
        print(f'unexpected_keys: {unexpected_keys}\n'
              f'===================================================================\n')
        model.eval()
        model.to(self.device)
        return model

    def infer(self, input_text):
        token = self.tokenizer(input_text, padding='max_length', truncation=True, max_length=140)
        input_ids = torch.tensor(token['input_ids'], device=self.device).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_ids)
        pred = torch.nn.functional.softmax(output.logits).detach().cpu().numpy()[0]
        return {self.label[i]: float(pred[i]) for i in range(self.num_labels)}

    def run(self):
        with gr.Blocks(title="情感分类模型演示") as demo:
            gr.Markdown("# 文本情感分类模型")
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="输入文本",
                        placeholder="请输入需要分析情感的文本...",
                        lines=5
                    )
                    predict_btn = gr.Button("预测", variant="primary")
                with gr.Column():
                    # 直接使用gr.Label作为输出组件
                    output = gr.Label(label="情感分析结果", num_top_classes=len(self.label))

            predict_btn.click(
                fn=self.infer,
                inputs=input_text,
                outputs=output
            )

            gr.Examples(
                examples=[
                    ["你是个什么东西，垃圾"],
                    ["复旦大学学风真好啊！"],
                    ["这部电影让我感动得哭了"],
                    ["今天天气真糟糕，心情也不好"],
                    ["这个消息太让人惊讶了！"]
                ],
                inputs=input_text
            )

        demo.launch(server_name='0.0.0.0', server_port=7860, share = True)


if __name__ == '__main__':
    args = vars(parse_args())
    args['label'] = {0: 'happy', 1: 'angry', 2: 'sad', 3: 'fear', 4: 'surprise', 5: 'neutral'}
    runer = Runer(**args)
    runer.run()