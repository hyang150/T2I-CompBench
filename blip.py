
import os
import re
import json
import argparse
import spacy
from tqdm import tqdm
from BLIPvqa_eval.BLIP.train_vqa_func import VQA_main  # 假设该函数可以对指定目录进行评估

def create_annotation_for_subfolder(subfolder_path, dataset_label):
    """
    遍历指定子文件夹下的图片，解析文件名得到 prompt 信息，
    构造 annotation 字典列表，返回列表。
    假设图片文件名格式为:
      {prompt_index}_{prompt_text}_{img_index:05d}.png
    例如: "1_a_book_hidden_by_a_wallet_00000.png"
    """
    nlp = spacy.load("en_core_web_sm")
    annotations = []
    
    file_names = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.png')]
    file_names.sort(key=lambda x: int(re.search(r'_(\d{5})\.png$', x).group(1)))  # 按最后5位数字排序

    for file_name in file_names:
        # 文件名示例: "1_a_book_hidden_by_a_wallet_00000.png"
        # 利用正则拆分：前面第一部分为 prompt index，接下来的部分为 prompt_text，最后为图片序号
        match = re.match(r"^(\d+)_([a-zA-Z0-9_]+)_(\d{5})\.png$", file_name)
        if not match:
            print(f"文件 {file_name} 格式不符合要求，跳过")
            continue
        
        prompt_idx = match.group(1)  # 字符串
        prompt_text_underscore = match.group(2)  # 如 "a_book_hidden_by_a_wallet"
        img_index = match.group(3)  # 如 "00000"
        
        # 将下划线替换为空格还原 prompt 文本
        prompt_text = prompt_text_underscore.replace("_", " ")
        # 构造问题文本，例如加上问号
        question = f"{prompt_text}?"
        
        # 构造 annotation 字典
        annotation = {
            "image": os.path.join(subfolder_path, file_name),
            "question_id": int(img_index),  # 可选择用图片序号作为 question_id
            "question": question,
            "dataset": dataset_label  # 使用子文件夹名称或其他标识
        }
        annotations.append(annotation)
    return annotations

def save_annotations(annotations, out_path):
    """将 annotations 列表保存为 JSON 文件"""
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)
    print(f"保存 annotation 文件：{out_path}")

def evaluate_blip_vqa(annotation_folder, vqa_out_folder):
    """
    调用 BLIP_vqa 的评估函数 VQA_main，
    参数分别为 annotation 文件夹和输出路径。
    这里假设 VQA_main 会读取 annotation_folder 下的 JSON 文件（如 vqa_test.json）
    并输出结果到 vqa_out_folder 下的 "result/vqa_result.json"
    """
    # 调用 BLIP_vqa 方法（具体实现视你的代码而定）
    score_dict = VQA_main(annotation_folder, os.path.join(annotation_folder, "VQA"))
    return score_dict

def main():
    parser = argparse.ArgumentParser(description="BLIP-VQA Evaluation for Generated Images")
    parser.add_argument("--image_root", type=str, required=True,
                        help="生成图片的根目录（包含各子类别文件夹）")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="prompt txt 文件所在目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="保存生成的 annotation 和评估结果的输出目录")
    args = parser.parse_args()

    image_root = args.image_root
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历 image_root 下所有子文件夹
    subfolders = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]
    all_annotations = []
    # 遍历每个子文件夹
    for subf in subfolders:
        subfolder_path = os.path.join(image_root, subf)
        # 假设对应的 txt 文件名为 subf + ".txt"（例如 color_val.txt）
        txt_file = os.path.join(dataset_dir, subf + ".txt")
        if not os.path.exists(txt_file):
            print(f"未找到对应 txt 文件: {txt_file}，跳过子文件夹 {subf}")
            continue
        
        # 这里我们并不需要逐行读取 txt，因为文件名中已经包含 prompt 索引和文本，
        # 但如果需要验证或进一步处理，可在这里加载 txt 文件。
        # 这里只记录子文件夹名称作为 dataset 标签
        dataset_label = subf
        print(f"处理子文件夹 {subf} ...")
        annotations = create_annotation_for_subfolder(subfolder_path, dataset_label)
        # 保存该子文件夹的 annotation 文件
        annotation_out_path = os.path.join(output_dir, f"{subf}_vqa_test.json")
        save_annotations(annotations, annotation_out_path)
        all_annotations.extend(annotations)
        
        # 对该子文件夹调用 BLIP_VQA 评估（假设每个子文件夹独立评估）
        # 这里我们使用 annotation_out_path 所在目录作为 annotation_folder
        annotation_folder = os.path.join(output_dir, subf + "_annotation_blip")
        os.makedirs(annotation_folder, exist_ok=True)
        # 将 annotation_out_path 复制到 annotation_folder 中，命名为 vqa_test.json
        os.system(f"cp {annotation_out_path} {os.path.join(annotation_folder, 'vqa_test.json')}")
        # 调用 BLIP_VQA 评估函数，输出结果到 annotation_folder/VQA/result/
        vqa_result = evaluate_blip_vqa(annotation_folder, os.path.join(annotation_folder, "VQA"))
        print(f"子文件夹 {subf} BLIP-VQA 评估结果: {vqa_result}")

    # 如果希望将所有子文件夹的评估结果合并，也可以在此处做汇总处理
    # 此处仅打印所有 annotation 数量
    print(f"总共处理 {len(all_annotations)} 张图片的 annotation 信息。")

if __name__ == "__main__":
    main()
