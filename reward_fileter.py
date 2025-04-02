import os
import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm

def select_high_score_images(scores_csv_path, output_dir, threshold=0.7):
    """
    从评分数据中选择高分图像并将它们复制到输出目录
    
    参数:
        scores_csv_path: 包含图像路径和BLIPVQA评分的CSV文件路径
        output_dir: 用于存储高分图像的输出目录
        threshold: 选择样本的分数阈值 (0.0-1.0)
    
    返回:
        selected_data: 包含所选图像信息的DataFrame
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载评分数据
    print(f"加载评分数据从 {scores_csv_path}")
    data = pd.read_csv(scores_csv_path)
    
    # 确保数据包含必要的列
    required_columns = ['image_path', 'prompt', 'blipvqa_score']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"CSV文件缺少必要的列: {col}")
    
    # 筛选高分图像
    selected_data = data[data['blipvqa_score'] >= threshold].copy()
    print(f"选择了 {len(selected_data)} 个样本，总共 {len(data)} 个样本 (阈值: {threshold})")
    
    # 创建数据统计摘要
    print("\n分数分布统计:")
    print(data['blipvqa_score'].describe())
    print("\n所选样本分数分布统计:")
    print(selected_data['blipvqa_score'].describe())
    
    # 复制选定的图像到输出目录
    print(f"\n正在复制所选图像到 {output_dir}")
    for idx, row in tqdm(selected_data.iterrows(), total=len(selected_data)):
        src_path = row['image_path']
        # 获取文件名并创建目标路径
        filename = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, filename)
        
        # 复制图像文件
        try:
            shutil.copy2(src_path, dst_path)
            # 更新DataFrame中的图像路径为新路径
            selected_data.at[idx, 'image_path'] = dst_path
        except Exception as e:
            print(f"复制 {src_path} 时出错: {e}")
    
    # 保存筛选后的数据到新的CSV文件
    selected_csv_path = os.path.join(output_dir, 'selected_high_score_images.csv')
    selected_data.to_csv(selected_csv_path, index=False)
    print(f"已将筛选后的数据保存到 {selected_csv_path}")
    
    return selected_data

def analyze_score_distribution(scores_csv_path):
    """
    分析分数分布，帮助确定适当的阈值
    """
    data = pd.read_csv(scores_csv_path)
    
    # 计算不同阈值下的样本数量
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    print("\n不同阈值下的样本数量:")
    for thresh in thresholds:
        count = sum(data['blipvqa_score'] >= thresh)
        percentage = (count / len(data)) * 100
        print(f"阈值 {thresh}: {count} 样本 ({percentage:.2f}%)")
    
    # 绘制分数分布直方图
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(data['blipvqa_score'], bins=20, alpha=0.7)
        plt.xlabel('BLIPVQA 分数')
        plt.ylabel('样本数量')
        plt.title('BLIPVQA 分数分布')
        
        # 添加垂直线表示不同阈值
        for thresh in thresholds:
            plt.axvline(x=thresh, color='r', linestyle='--', alpha=0.3)
            plt.text(thresh, plt.ylim()[1]*0.9, f'T={thresh}', rotation=90)
        
        plt.tight_layout()
        plt.savefig('score_distribution.png')
        print("\n分数分布直方图已保存到 'score_distribution.png'")
    except ImportError:
        print("未安装matplotlib，跳过绘制直方图")

if __name__ == "__main__":
    # 设置输入输出路径
    scores_csv_path = "path_to_your_scored_images.csv"  # 修改为您的CSV文件路径
    output_dir = "selected_high_quality_images"  # 输出目录
    
    # 分析分数分布（可选，帮助确定适当的阈值）
    analyze_score_distribution(scores_csv_path)
    
    # 选择得分高于0.75的图像（您可以根据分布分析调整此阈值）
    selected_data = select_high_score_images(
        scores_csv_path=scores_csv_path,
        output_dir=output_dir,
        threshold=0.75  # 根据您的数据分布调整此阈值
    )
    
    print(f"\n完成! {len(selected_data)} 个高质量样本已准备好用于微调。")