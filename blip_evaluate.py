import os
import json
import shutil
import subprocess
import pandas as pd
import re
import glob
import tempfile
import pathlib

# 定义文件路径
project_dir = "T2I-CompBench"
samples_dir = os.path.join(project_dir, "examples", "samples")
blipvqa_dir = os.path.join(project_dir, "BLIPvqa_eval")
blipvqa_result_dir = os.path.join(project_dir, "examples", "annotation_blip")
csv_file_path = "generated_images_log.csv"

def is_image_file(filename):
    """检查文件是否为图像文件"""
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

def is_identifier_file(filename):
    """检查是否为identifier文件（非图像文件）"""
    return not is_image_file(filename)

def clean_samples_directory():
    """删除samples目录中的非图像文件"""
    print("正在清理samples目录...")
    files_removed = 0
    
    for filename in os.listdir(samples_dir):
        file_path = os.path.join(samples_dir, filename)
        if os.path.isfile(file_path) and is_identifier_file(filename):
            os.remove(file_path)
            files_removed += 1
    
    print(f"已删除 {files_removed} 个非图像文件")
    
    # 检查剩余的图像文件数量
    image_files = [f for f in os.listdir(samples_dir) if is_image_file(f)]
    print(f"samples目录中剩余 {len(image_files)} 个图像文件")
    return len(image_files) > 0

def extract_question_id(image_path):
    """从图像路径提取question_id"""
    try:
        filename = os.path.basename(image_path)
        
        # 尝试匹配格式1: 描述_000000.png
        match1 = re.search(r'_(\d{6})\.', filename)
        if match1:
            return int(match1.group(1))
        
        # 尝试匹配格式2: 描述_00000_000000.png
        match2 = re.search(r'_(\d{5})_(\d{6})\.', filename)
        if match2:
            # 组合两个ID部分，或者选择其中一个作为ID
            return int(match2.group(2))  # 使用第二个数字部分作为ID
        
        return None
    except Exception as e:
        print(f"提取question_id时出错: {e}, 文件: {image_path}")
        return None

def get_all_sample_images():
    """获取样本目录中的所有图像文件列表"""
    images = []
    for filename in os.listdir(samples_dir):
        if is_image_file(filename):
            images.append(os.path.join(samples_dir, filename))
    return images

def preprocess_image_files():
    """预处理图像文件名，确保它们符合BLIP_vqa.py的要求"""
    print("正在预处理图像文件...")
    
    # 创建临时目录用于存储预处理后的图像
    temp_dir = os.path.join(samples_dir, "temp_processed")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(samples_dir) if is_image_file(f)]
    print(f"找到 {len(image_files)} 个图像文件需要预处理")
    
    # 创建映射字典，用于记录原始文件名到新文件名的映射
    filename_mapping = {}
    
    # 检查图像文件的命名模式
    pattern1_count = 0  # 格式1: 描述_000000.png
    pattern2_count = 0  # 格式2: 描述_00000_000000.png
    other_count = 0     # 其他格式
    
    for filename in image_files:
        if re.search(r'_\d{6}\.', filename):
            pattern1_count += 1
        elif re.search(r'_\d{5}_\d{6}\.', filename):
            pattern2_count += 1
        else:
            other_count += 1
    
    print(f"图像文件命名模式统计:")
    print(f"  格式1 (描述_000000.png): {pattern1_count}")
    print(f"  格式2 (描述_00000_000000.png): {pattern2_count}")
    print(f"  其他格式: {other_count}")
    
    # 处理每个图像文件
    for idx, filename in enumerate(image_files):
        original_path = os.path.join(samples_dir, filename)
        
        # 判断文件名格式
        is_pattern1 = re.search(r'_\d{6}\.', filename) is not None
        is_pattern2 = re.search(r'_\d{5}_\d{6}\.', filename) is not None
        
        if is_pattern1:
            # 已经符合格式1，直接复制
            new_filename = filename
        elif is_pattern2:
            # 格式2，提取最后六位数字作为ID
            match = re.search(r'_\d{5}_(\d{6})\.', filename)
            if match:
                id_part = match.group(1)
                # 提取主要部分（不包括ID和扩展名）
                base_parts = filename.split('_')[:-2]  # 去掉最后两个ID部分
                base_name = '_'.join(base_parts)
                extension = os.path.splitext(filename)[1]
                # 创建新文件名
                new_filename = f"{base_name}_{id_part}{extension}"
            else:
                # 如果无法提取ID，使用默认编号
                base_name = os.path.splitext(filename)[0]
                extension = os.path.splitext(filename)[1]
                new_filename = f"{base_name}_{idx:06d}{extension}"
        else:
            # 不符合任何已知格式，重命名为符合格式的文件名
            base_name = os.path.splitext(filename)[0]
            extension = os.path.splitext(filename)[1]
            new_filename = f"{base_name}_{idx:06d}{extension}"
        
        # 复制到临时目录
        new_path = os.path.join(temp_dir, new_filename)
        shutil.copy2(original_path, new_path)
        
        # 记录映射关系
        filename_mapping[filename] = new_filename
    
    # 备份原始samples目录
    backup_dir = os.path.join(os.path.dirname(samples_dir), "samples_backup")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(samples_dir, backup_dir, ignore=lambda src, names: ["temp_processed"])
    print(f"已创建原始samples目录备份: {backup_dir}")
    
    # 移动临时目录中的文件到samples目录
    # 首先删除原始目录中的图像文件
    for filename in image_files:
        file_path = os.path.join(samples_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # 然后移动临时目录中的文件到samples目录
    for filename in os.listdir(temp_dir):
        shutil.move(os.path.join(temp_dir, filename), os.path.join(samples_dir, filename))
    
    # 删除临时目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # 保存文件名映射到JSON文件
    mapping_file = os.path.join(os.path.dirname(samples_dir), "filename_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(filename_mapping, f, indent=2)
    
    print(f"预处理完成，已处理 {len(filename_mapping)} 个文件")
    print(f"文件名映射已保存到: {mapping_file}")
    
    return filename_mapping

def run_blipvqa_evaluation():
    """运行BLIPvqa评估"""
    print("正在运行BLIPvqa评估...")
    
    # 确保输出目录存在
    os.makedirs(blipvqa_result_dir, exist_ok=True)
    
    # 切换到BLIPvqa_eval目录
    original_dir = os.getcwd()
    os.chdir(blipvqa_dir)
    
    # 运行评估脚本，设置环境变量以解决MKL问题
    env = os.environ.copy()
    # 设置MKL变量以修复兼容性问题
    env["MKL_THREADING_LAYER"] = "GNU"
    # 或强制使用INTEL (如果上面方法不起作用)
    # env["MKL_SERVICE_FORCE_INTEL"] = "1"
    
    try:
        out_dir = "../examples/"
        subprocess.run(["python", "BLIP_vqa.py", f"--out_dir={out_dir}"], env=env, check=True)
        print("BLIPvqa评估完成")
        success = True
    except subprocess.CalledProcessError as e:
        print(f"BLIPvqa评估失败: {e}")
        success = False
    
    # 返回原始目录
    os.chdir(original_dir)
    return success

def load_blipvqa_results():
    """加载BLIPvqa评估结果"""
    result_file = os.path.join(blipvqa_result_dir, "vqa_result.json")
    
    if not os.path.exists(result_file):
        print(f"警告: 结果文件 {result_file} 不存在")
        return {}
    
    try:
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # 将结果转换为字典格式，question_id作为键
        results_dict = {item["question_id"]: float(item["answer"]) for item in results}
        print(f"已加载 {len(results_dict)} 条评估结果")
        return results_dict
    except Exception as e:
        print(f"加载结果文件时出错: {e}")
        return {}

def update_csv_with_results(results_dict, filename_mapping=None):
    """将评估结果更新到CSV文件，包括处理不在CSV中的新图片"""
    if not results_dict:
        print("没有评估结果可更新")
        return
    
    try:
        # 获取样本目录中的所有图像
        all_sample_images = get_all_sample_images()
        print(f"样本目录中共有 {len(all_sample_images)} 个图像文件")
        
        # 尝试加载CSV文件
        try:
            df = pd.read_csv(csv_file_path)
            print(f"已加载CSV文件，包含 {len(df)} 行")
            csv_exists = True
        except FileNotFoundError:
            print(f"CSV文件 {csv_file_path} 不存在，将创建新文件")
            df = pd.DataFrame(columns=['prompt_file', 'image_index', 'prompt_text', 'seed', 'image_path', 'blipvqa_score'])
            csv_exists = False
        
        # 添加新列用于存储评估分数
        if 'blipvqa_score' not in df.columns:
            df['blipvqa_score'] = None
        
        # 创建反向文件名映射（如果存在）
        reverse_mapping = {}
        if filename_mapping:
            reverse_mapping = {v: k for k, v in filename_mapping.items()}
        
        # 更新CSV中现有图像的评估分数
        updated_count = 0
        if csv_exists:
            for idx, row in df.iterrows():
                image_path = row['image_path']
                orig_filename = os.path.basename(image_path)
                
                # 检查图像路径是否使用了新的文件名
                processed_filename = None
                if orig_filename in filename_mapping:
                    processed_filename = filename_mapping[orig_filename]
                
                # 从文件名中提取question_id（针对新文件名或原始文件名）
                question_id = None
                if processed_filename:
                    question_id = extract_question_id(processed_filename)
                if question_id is None:
                    question_id = extract_question_id(orig_filename)
                
                if question_id is not None and question_id in results_dict:
                    df.at[idx, 'blipvqa_score'] = results_dict[question_id]
                    updated_count += 1
        
        # 处理不在CSV中的新图像
        new_image_count = 0
        csv_image_paths = df['image_path'].tolist() if csv_exists else []
        
        for image_path in all_sample_images:
            image_filename = os.path.basename(image_path)
            original_filename = image_filename
            
            # 如果有反向映射，尝试获取原始文件名
            if image_filename in reverse_mapping:
                original_filename = reverse_mapping[image_filename]
            
            # 如果图像路径不在CSV中
            if image_path not in csv_image_paths and original_filename not in csv_image_paths and image_filename not in csv_image_paths:
                question_id = extract_question_id(image_filename)
                
                if question_id is not None and question_id in results_dict:
                    # 从文件名提取提示信息
                    prompt_text = os.path.basename(image_filename).split('_')[0]
                    if not prompt_text or prompt_text == image_filename:
                        prompt_text = "未知提示"
                    
                    # 添加新行
                    new_row = {
                        'prompt_file': 'unknown',
                        'image_index': question_id,
                        'prompt_text': prompt_text,
                        'seed': -1,  # 未知种子
                        'image_path': original_filename,  # 使用原始文件名
                        'blipvqa_score': results_dict[question_id]
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    new_image_count += 1
        
        # 保存更新后的CSV文件
        df.to_csv(csv_file_path, index=False)
        print(f"已更新 {updated_count} 个现有图像的评估分数")
        print(f"已添加 {new_image_count} 个新图像及其评估分数")
        
        # 计算并显示评估分数统计信息
        if 'blipvqa_score' in df.columns and not df['blipvqa_score'].isna().all():
            mean_score = df['blipvqa_score'].mean()
            median_score = df['blipvqa_score'].median()
            min_score = df['blipvqa_score'].min()
            max_score = df['blipvqa_score'].max()
            
            print(f"评估分数统计:")
            print(f"  平均分: {mean_score:.4f}")
            print(f"  中位数: {median_score:.4f}")
            print(f"  最低分: {min_score:.4f}")
            print(f"  最高分: {max_score:.4f}")
    
    except Exception as e:
        print(f"更新CSV文件时出错: {e}")
        import traceback
        traceback.print_exc()

def restore_original_filenames(filename_mapping):
    """恢复图像到原始文件名"""
    print("正在恢复原始文件名...")
    
    # 反转映射字典，用于还原文件名
    reverse_mapping = {v: k for k, v in filename_mapping.items()}
    
    # 创建临时目录
    temp_dir = os.path.join(samples_dir, "temp_restore")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 获取当前samples目录中的所有图像文件
    current_files = [f for f in os.listdir(samples_dir) if is_image_file(f)]
    
    # 处理每个文件
    restored_count = 0
    for filename in current_files:
        if filename in reverse_mapping:
            # 获取原始文件名
            original_filename = reverse_mapping[filename]
            
            # 复制到临时目录
            shutil.copy2(
                os.path.join(samples_dir, filename),
                os.path.join(temp_dir, original_filename)
            )
            restored_count += 1
        else:
            # 保持文件名不变
            shutil.copy2(
                os.path.join(samples_dir, filename),
                os.path.join(temp_dir, filename)
            )
    
    # 清空samples目录
    for filename in current_files:
        os.remove(os.path.join(samples_dir, filename))
    
    # 移动临时目录中的文件到samples目录
    for filename in os.listdir(temp_dir):
        shutil.move(os.path.join(temp_dir, filename), os.path.join(samples_dir, filename))
    
    # 删除临时目录
    shutil.rmtree(temp_dir)
    
    print(f"已恢复 {restored_count} 个文件到原始文件名")

def main():
    """主函数"""
    print("开始执行T2I-CompBench评估流程")
    
    # 步骤1: 清理samples目录
    if not clean_samples_directory():
        print("samples目录中没有图像文件，程序终止")
        return
    
    # 步骤2: 预处理图像文件
    filename_mapping = preprocess_image_files()
    
    # 获取图像总数
    all_sample_images = get_all_sample_images()
    print(f"处理后，samples目录中有 {len(all_sample_images)} 个图像文件待评估")
    
    # 步骤3: 运行BLIPvqa评估
    if not run_blipvqa_evaluation():
        print("BLIPvqa评估失败，程序终止")
        return
    
    # 步骤4: 加载评估结果
    results_dict = load_blipvqa_results()
    
    # 步骤5: 更新CSV文件，处理新增的图像
    update_csv_with_results(results_dict, filename_mapping)
    
    # 步骤6: 恢复原始文件名
    restore_original_filenames(filename_mapping)
    
    print("评估流程完成")

if __name__ == "__main__":
    main()