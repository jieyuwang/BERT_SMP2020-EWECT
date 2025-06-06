import pandas as pd
import os
from argparse import ArgumentParser


def determine_label(text):
    """根据文本内容确定标签"""
    # 检查文本中是否包含与台风相关的关键词
    typhoon_keywords = ["台风", "杜苏芮", "登陆", "防汛", "防台风", "应急", "灾害", "预警"]
    emergency_keywords = ["防御措施", "转移安置", "安全检查", "抢险救援", "应急管理"]

    # 初始化标签列表
    labels = []

    # 判断是否与台风相关
    if any(keyword in text for keyword in typhoon_keywords):
        labels.append("自然灾害预警")
        labels.append("台风预警")

    # 判断是否与应急管理相关
    if any(keyword in text for keyword in emergency_keywords):
        labels.append("应急管理")
        labels.append("防御措施")

    # 如果没有找到匹配的关键词，添加默认标签
    if not labels:
        labels.append("其他")

    # 将标签列表转换为用逗号分隔的字符串
    return ", ".join(labels)


def process_excel(input_file, output_file=None, model_path=None, num_labels=6):
    """处理Excel文件并添加标签"""
    try:
        # 读取Excel文件
        df = pd.read_excel(input_file)
        print(f"成功读取Excel文件，包含 {len(df)} 行数据")

        # 检查数据中是否包含'txt'列
        if 'txt' not in df.columns:
            raise ValueError("Excel文件中缺少'txt'列，无法进行文本分析")

        # 为每一行数据确定标签
        df['标签'] = df['txt'].apply(determine_label)
        print("标签添加完成")

        # 如果未指定输出文件，则在原文件名基础上添加"_with_labels"后缀
        if output_file is None:
            file_name, file_ext = os.path.splitext(input_file)
            output_file = f"{file_name}_with_labels{file_ext}"

        # 保存处理后的Excel文件
        df.to_excel(output_file, index=False)
        print(f"已将结果保存到: {output_file}")

        return output_file

    except Exception as e:
        print(f"处理Excel文件时出错: {str(e)}")
        return None


def main():
    """主函数：解析命令行参数并处理Excel文件"""
    parser = ArgumentParser(description='处理Excel数据并添加标签')
    parser.add_argument('--input', type=str, required=True, help='输入Excel文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出Excel文件路径')
    parser.add_argument('--model_path', type=str, default='workspace/wb/best.pt', help='模型路径')
    parser.add_argument('--num_labels', type=int, default=6, help='标签数量')
    parser.add_argument('--device', type=str, default='cpu', help='设备类型 (cpu/cuda)')

    args = parser.parse_args()

    print(f"开始处理Excel文件: {args.input}")
    process_excel(args.input, args.output, args.model_path, args.num_labels)
    print("处理完成")


if __name__ == "__main__":
    main()