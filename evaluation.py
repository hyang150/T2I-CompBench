"""
处理嵌套目录结构的3-in-1评估脚本
该脚本自动处理包含多个分类子目录的生成图像，并运行T2I-CompBench的3-in-1评估
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
import re

# 设置环境变量解决MKL冲突
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description="嵌套目录结构的3-in-1评估工具")
    parser.add_argument(
        "--t2i_compbench_dir",
        type=str,
        required=True,
        help="T2I-CompBench目录的路径"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="包含分类子目录的生成图像根目录的路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="评估结果输出目录（默认为T2I-CompBench/examples）"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="all",
        help="要评估的类别，以逗号分隔。例如：'color,shape,texture'。默认为'all'（评估所有类别）"
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="跳过评估步骤，直接生成报告（如果已有评估结果）"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=0,
        help="每个类别评估的图像数量上限，0表示不限制"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="清理所有Zone.Identifier文件"
    )
    return parser.parse_args()

def run_command(cmd, desc=None, env=None):
    """运行命令并检查执行状态，可设置环境变量"""
    if desc:
        print(f"正在{desc}...")
    
    try:
        # 合并当前环境变量和新设置的环境变量
        current_env = os.environ.copy()
        if env:
            current_env.update(env)
        
        # 使用环境变量运行命令
        subprocess.run(cmd, check=True, shell=True, env=current_env)
        if desc:
            print(f"✅ {desc}成功")
        return True
    except subprocess.CalledProcessError as e:
        if desc:
            print(f"❌ {desc}失败: {e}")
        return False

def clean_zone_identifier_files(directory):
    """清理目录中的Zone.Identifier文件"""
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(":Zone.Identifier") or ".zone.identifier" in file.lower():
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    print(f"无法删除文件 {file_path}: {e}")
    
    if count > 0:
        print(f"已删除 {count} 个Zone.Identifier文件")
    return count

def is_valid_image_file(file_path):
    """检查文件是否为有效的图像文件"""
    if not os.path.exists(file_path):
        return False
    
    # 检查文件名是否包含Zone.Identifier
    if ":Zone.Identifier" in file_path or ".zone.identifier" in file_path.lower():
        return False
    
    # 检查文件扩展名
    valid_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
    if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
        return False
    
    # 检查文件大小，图像文件通常至少有几KB
    if os.path.getsize(file_path) < 1024:  # 小于1KB的文件可能不是图像
        return False
    
    return True

def get_image_categories(root_dir):
    """获取所有图像类别文件夹"""
    categories = []
    for item in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, item)):
            categories.append(item)
    return categories

def copy_images_from_categories(src_root, dest_dir, categories=None, subset=0):
    """从分类目录中复制所有图像到目标目录"""
    print(f"正在从分类目录复制图像...")
    
    # 清理目标目录中的Zone.Identifier文件
    clean_zone_identifier_files(dest_dir)
    
    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)
    
    # 清空目标目录中的现有图像
    for existing_file in os.listdir(dest_dir):
        file_path = os.path.join(dest_dir, existing_file)
        if is_valid_image_file(file_path):
            os.remove(file_path)
    
    # 确定要处理的分类
    if categories is None or categories[0].lower() == "all":
        categories = get_image_categories(src_root)
    
    # 跟踪图像ID以避免冲突
    next_id = 0
    copied_count = 0
    skipped_count = 0
    category_stats = {}
    
    # 为每个类别创建映射文件
    category_map = {}
    
    # 复制每个类别中的图像
    for category in categories:
        category_path = os.path.join(src_root, category)
        if not os.path.isdir(category_path):
            print(f"警告: 类别目录 '{category}' 不存在，已跳过")
            continue
        
        # 递归查找该类别及其子目录中的所有图像
        all_images = []
        for root, _, files in os.walk(category_path):
            for file in files:
                file_path = os.path.join(root, file)
                if is_valid_image_file(file_path):
                    all_images.append(file_path)
        
        if not all_images:
            print(f"警告: 在类别 '{category}' 中没有找到有效的图像文件，已跳过")
            continue
        
        # 可选限制每个类别的图像数量
        if subset > 0 and len(all_images) > subset:
            print(f"在类别 '{category}' 中找到 {len(all_images)} 个图像文件，但仅使用前 {subset} 个")
            all_images = all_images[:subset]
        else:
            print(f"在类别 '{category}' 中找到 {len(all_images)} 个图像文件")
        
        category_stats[category] = len(all_images)
        
        # 为当前类别创建ID映射
        category_map[category] = {}
        
        # 复制文件并重命名
        for img_path in tqdm(all_images, desc=f"复制 {category} 图像"):
            # 提取原始文件名（不带路径）
            original_filename = os.path.basename(img_path)
            base_name, ext = os.path.splitext(original_filename)
            
            # 清理base_name，替换非字母数字字符为下划线
            base_name = re.sub(r'[^\w]', '_', base_name)
            
            # 生成新的文件名，使用递增的ID
            new_filename = f"{base_name}_{str(next_id).zfill(6)}{ext}"
            dest_path = os.path.join(dest_dir, new_filename)
            
            try:
                # 保存原始文件名到ID的映射
                category_map[category][next_id] = original_filename
                
                # 复制文件
                shutil.copy2(img_path, dest_path)
                next_id += 1
                copied_count += 1
            except Exception as e:
                print(f"复制文件 {img_path} 时出错: {e}")
                skipped_count += 1
    
    # 保存类别到ID的映射，用于结果分析
    map_file = os.path.join(os.path.dirname(dest_dir), "category_id_map.json")
    with open(map_file, "w") as f:
        json.dump(category_map, f, indent=2)
    
    print(f"✅ 总共复制了 {copied_count} 个图像文件，跳过了 {skipped_count} 个文件")
    print(f"类别统计: {category_stats}")
    print(f"类别到ID的映射已保存到: {map_file}")
    
    # 最后再次清理可能的Zone.Identifier文件
    clean_zone_identifier_files(dest_dir)
    
    return True

def setup_directories(t2i_dir, output_dir):
    """创建必要的目录结构"""
    dirs = [
        output_dir,
        os.path.join(output_dir, "samples"),
        os.path.join(output_dir, "annotation_blip"),
        os.path.join(output_dir, "labels"),
        os.path.join(output_dir, "labels", "annotation_obj_detection_2d"),
        os.path.join(output_dir, "labels", "annotation_obj_detection_3d"),
        os.path.join(output_dir, "annotation_num"),
        os.path.join(output_dir, "annotation_clip"),
        os.path.join(output_dir, "annotation_3_in_1"),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def run_evaluations(t2i_dir, output_dir):
    """运行所有评估方法"""
    # 清理任何可能的Zone.Identifier文件
    samples_dir = os.path.join(output_dir, "samples")
    clean_zone_identifier_files(samples_dir)
    
    # 设置环境变量解决MKL冲突
    env_vars = {
        "MKL_THREADING_LAYER": "GNU",
        "MKL_SERVICE_FORCE_INTEL": "1"
    }
    
    # 1. BLIP-VQA用于属性绑定
    blip_cmd = f"cd {os.path.join(t2i_dir, 'BLIPvqa_eval')} && python BLIP_vqa.py --out_dir={output_dir}"
    if not run_command(blip_cmd, "运行BLIP-VQA属性绑定评估", env_vars):
        return False
    
    # 2. UniDet用于空间关系和数量
    unidet_2d_cmd = f"cd {os.path.join(t2i_dir, 'UniDet_eval')} && python 2D_spatial_eval.py --complex=True"
    if not run_command(unidet_2d_cmd, "运行UniDet 2D空间关系评估", env_vars):
        return False
    
    unidet_num_cmd = f"cd {os.path.join(t2i_dir, 'UniDet_eval')} && python numeracy_eval.py"
    if not run_command(unidet_num_cmd, "运行UniDet数量评估", env_vars):
        return False
    
    unidet_3d_cmd = f"cd {os.path.join(t2i_dir, 'UniDet_eval')} && python 3D_spatial_eval.py"
    if not run_command(unidet_3d_cmd, "运行UniDet 3D空间关系评估", env_vars):
        return False
    
    # 3. CLIPScore用于非空间关系
    clip_cmd = f"cd {os.path.join(t2i_dir, 'CLIPScore_eval')} && python CLIP_similarity.py --outpath={output_dir}/ --complex=True"
    if not run_command(clip_cmd, "运行CLIPScore非空间关系评估", env_vars):
        return False
    
    # 4. 3-in-1评估
    three_in_one_cmd = f"cd {os.path.join(t2i_dir, '3_in_1_eval')} && python 3_in_1.py --outpath={output_dir}/"
    if not run_command(three_in_one_cmd, "运行3-in-1综合评估", env_vars):
        return False
    
    return True

def load_results(file_path):
    """加载评估结果JSON文件"""
    if not os.path.exists(file_path):
        print(f"警告: {file_path} 不存在")
        return {}
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # 转换为字典方便查找
        result_dict = {}
        for item in data:
            question_id = item["question_id"]
            if isinstance(question_id, str) and question_id.isdigit():
                question_id = int(question_id)
            answer = float(item["answer"])
            result_dict[question_id] = answer
        
        return result_dict
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return {}

def load_category_map(output_dir):
    """加载类别ID映射文件"""
    map_file = os.path.join(output_dir, "category_id_map.json")
    if not os.path.exists(map_file):
        print(f"警告: 类别映射文件 {map_file} 不存在")
        return {}
    
    try:
        with open(map_file, "r") as f:
            category_map = json.load(f)
            
            # 将字符串键转换为整数键
            converted_map = {}
            for category, id_dict in category_map.items():
                converted_map[category] = {}
                for id_str, filename in id_dict.items():
                    converted_map[category][int(id_str)] = filename
            
            return converted_map
    except Exception as e:
        print(f"读取类别映射文件时出错: {e}")
        return {}

def get_category_for_id(question_id, category_map):
    """根据ID查找对应的类别"""
    for category, id_dict in category_map.items():
        if question_id in id_dict:
            return category
    return "未知"

def generate_report(output_dir):
    """生成评估报告"""
    print("正在生成评估报告...")
    examples_dir = Path(output_dir)
    
    # 加载所有评估结果
    blip_results = load_results(examples_dir / "annotation_blip" / "vqa_result.json")
    unidet_2d_results = load_results(examples_dir / "labels" / "annotation_obj_detection_2d" / "vqa_result.json")
    unidet_3d_results = load_results(examples_dir / "labels" / "annotation_obj_detection_3d" / "vqa_result.json")
    unidet_num_results = load_results(examples_dir / "annotation_num" / "vqa_result.json")
    clip_results = load_results(examples_dir / "annotation_clip" / "vqa_result.json")
    combined_results = load_results(examples_dir / "annotation_3_in_1" / "vqa_result.json")
    
    # 加载类别映射
    category_map = load_category_map(examples_dir)
    
    # 获取所有图像ID
    all_ids = set()
    for result_dict in [blip_results, unidet_2d_results, unidet_3d_results, 
                       unidet_num_results, clip_results, combined_results]:
        all_ids.update(result_dict.keys())
    
    # 创建评估报告DataFrame
    data = []
    for qid in all_ids:
        sample_file = None
        for file in os.listdir(examples_dir / "samples"):
            if file.endswith(f"_{str(qid).zfill(6)}.png") or file.endswith(f"_{str(qid)}.png"):
                sample_file = file
                break
        
        # 获取图像所属类别
        category = get_category_for_id(qid, category_map)
        
        data.append({
            "ID": qid,
            "文件名": sample_file,
            "类别": category,
            "BLIP-VQA (属性)": blip_results.get(qid, np.nan),
            "UniDet-2D (空间)": unidet_2d_results.get(qid, np.nan),
            "UniDet-3D (空间)": unidet_3d_results.get(qid, np.nan),
            "UniDet (数量)": unidet_num_results.get(qid, np.nan),
            "CLIPScore": clip_results.get(qid, np.nan),
            "3-in-1 (综合)": combined_results.get(qid, np.nan)
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values(["类别", "ID"])
    
    # 计算整体统计信息
    stats = {
        "平均分": df.mean(numeric_only=True),
        "最高分": df.max(numeric_only=True),
        "最低分": df.min(numeric_only=True),
        "标准差": df.std(numeric_only=True)
    }
    stats_df = pd.DataFrame(stats)
    
    # 计算按类别分组的统计信息
    category_grouped = df.groupby("类别")
    category_stats = pd.DataFrame()
    
    for metric in ["BLIP-VQA (属性)", "UniDet-2D (空间)", "UniDet-3D (空间)", 
                  "UniDet (数量)", "CLIPScore", "3-in-1 (综合)"]:
        for stat_name, stat_func in [("平均分", "mean"), ("标准差", "std"), ("数量", "count")]:
            col_name = f"{metric}_{stat_name}"
            category_stats[col_name] = category_grouped[metric].agg(stat_func)
    
    # 保存详细奖励结果（用于GORS微调）
    reward_dir = examples_dir / "reward"
    os.makedirs(reward_dir, exist_ok=True)
    
    reward_data = []
    for qid, score in combined_results.items():
        reward_data.append({"question_id": qid, "answer": f"{score:.4f}"})
    
    with open(reward_dir / "vqa_result.json", "w") as f:
        json.dump(reward_data, f, indent=2)
    
    # 保存报告
    report_path = examples_dir / "evaluation_report.csv"
    df.to_csv(report_path, index=False)
    
    stats_path = examples_dir / "evaluation_stats.csv"
    stats_df.to_csv(stats_path)
    
    category_stats_path = examples_dir / "category_evaluation_stats.csv"
    category_stats.to_csv(category_stats_path)
    
    # 保存高奖励分数图像ID列表，用于筛选样本
    threshold = 0.7  # 可以设置不同的阈值
    high_reward_ids = {qid for qid, score in combined_results.items() if score >= threshold}
    high_reward_df = df[df["ID"].isin(high_reward_ids)]
    high_reward_path = examples_dir / f"high_reward_samples_t{threshold}.csv"
    high_reward_df.to_csv(high_reward_path, index=False)
    
    # 打印统计摘要
    print("\n===== 评估统计摘要 =====")
    print(f"总评估图像数: {len(df)}")
    print("\n平均评分:")
    for col, val in stats["平均分"].items():
        print(f"  {col}: {val:.4f}")
    
    print(f"\n高奖励图像 (分数 >= {threshold}): {len(high_reward_ids)}张")
    
    print("\n类别统计:")
    for category in df["类别"].unique():
        category_df = df[df["类别"] == category]
        print(f"  {category} (共{len(category_df)}张图像):")
        if "3-in-1 (综合)" in category_df.columns:
            mean_score = category_df["3-in-1 (综合)"].mean()
            print(f"    3-in-1平均分: {mean_score:.4f}")
            high_reward_count = len(category_df[category_df["3-in-1 (综合)"] >= threshold])
            print(f"    高奖励图像数: {high_reward_count}张")
    
    print(f"\n详细报告已保存至: {report_path}")
    print(f"整体统计已保存至: {stats_path}")
    print(f"类别统计已保存至: {category_stats_path}")
    print(f"高奖励样本列表已保存至: {high_reward_path}")
    print(f"GORS微调用奖励文件已保存至: {reward_dir / 'vqa_result.json'}")
    
    return True

def main():
    args = parse_args()
    
    # 设置目录
    t2i_dir = os.path.abspath(args.t2i_compbench_dir)
    image_dir = os.path.abspath(args.image_dir)
    output_dir = args.output_dir if args.output_dir else os.path.join(t2i_dir, "examples")
    output_dir = os.path.abspath(output_dir)
    
    # 检查目录是否存在
    if not os.path.isdir(t2i_dir):
        print(f"错误: T2I-CompBench目录 '{t2i_dir}' 不存在")
        return 1
    
    if not os.path.isdir(image_dir):
        print(f"错误: 图像目录 '{image_dir}' 不存在")
        return 1
    
    # 处理类别参数
    categories = args.categories.split(",") if args.categories != "all" else ["all"]
    
    # 创建必要的目录
    setup_directories(t2i_dir, output_dir)
    
    # 如果指定了清理选项，先清理所有Zone.Identifier文件
    if args.clean:
        print("正在清理所有Zone.Identifier文件...")
        clean_count = clean_zone_identifier_files(image_dir)
        clean_count += clean_zone_identifier_files(output_dir)
        print(f"总共清理了 {clean_count} 个Zone.Identifier文件")
    
    # 从嵌套目录复制图像文件
    if not copy_images_from_categories(image_dir, os.path.join(output_dir, "samples"), categories, args.subset):
        return 1
    
    # 确保没有Zone.Identifier文件
    samples_dir = os.path.join(output_dir, "samples")
    clean_zone_identifier_files(samples_dir)
    
    # 运行评估
    if not args.skip_evaluation:
        if not run_evaluations(t2i_dir, output_dir):
            print("评估过程中出现错误，请检查上述输出信息")
            return 1
    else:
        print("已跳过评估步骤，直接生成报告...")
    
    # 生成报告
    if not generate_report(output_dir):
        print("生成报告过程中出现错误")
        return 1
    
    print("评估流程全部完成！")
    return 0

if __name__ == "__main__":
    sys.exit(main())