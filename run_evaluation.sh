#!/bin/bash

# 自动化评估脚本 - 使用T2I-CompBench的3-in-1评估方法对生成图像进行评估

# 设置路径变量
T2I_COMPBENCH_DIR="T2I-CompBench"  # 请修改为您的T2I-CompBench目录路径
GENERATED_IMAGES_DIR="csci_544\T2I-CompBench\examples\samples"  # 请修改为您的生成图像目录路径
EXAMPLES_DIR="${T2I_COMPBENCH_DIR}/examples"

# 创建临时目录结构
mkdir -p "${EXAMPLES_DIR}/samples"

# 复制所有生成的图像到T2I-CompBench的examples/samples目录
echo "正在复制图像文件到评估目录..."
cp "${GENERATED_IMAGES_DIR}"/*.png "${EXAMPLES_DIR}/samples/"

# 确保评估结果目录存在
mkdir -p "${EXAMPLES_DIR}/annotation_blip"
mkdir -p "${EXAMPLES_DIR}/labels/annotation_obj_detection_2d"
mkdir -p "${EXAMPLES_DIR}/labels/annotation_obj_detection_3d"
mkdir -p "${EXAMPLES_DIR}/annotation_num"
mkdir -p "${EXAMPLES_DIR}/annotation_clip"
mkdir -p "${EXAMPLES_DIR}/annotation_3_in_1"

# 检查命令执行状态的函数
check_execution() {
    if [ $? -ne 0 ]; then
        echo "错误: $1 执行失败"
        exit 1
    else
        echo "✅ $1 执行成功"
    fi
}

# 运行所有评估
echo "正在运行BLIP-VQA属性绑定评估..."
cd "${T2I_COMPBENCH_DIR}/BLIPvqa_eval"
python BLIP_vqa.py --out_dir="${EXAMPLES_DIR}/"
check_execution "BLIP-VQA评估"


echo "正在运行UniDet 2D空间关系评估..."
cd "${T2I_COMPBENCH_DIR}/UniDet_eval"
python 2D_spatial_eval.py --complex=True
check_execution "UniDet 2D空间关系评估"

echo "正在运行UniDet数量评估..."
python numeracy_eval.py
check_execution "UniDet数量评估"

echo "正在运行UniDet 3D空间关系评估..."
python 3D_spatial_eval.py
check_execution "UniDet 3D空间关系评估"

echo "正在运行CLIPScore非空间关系评估..."
cd "${T2I_COMPBENCH_DIR}/CLIPScore_eval"
python CLIP_similarity.py --outpath="${EXAMPLES_DIR}/" --complex=True
check_execution "CLIPScore评估"

echo "正在运行3-in-1综合评估..."
cd "${T2I_COMPBENCH_DIR}/3_in_1_eval"
python 3_in_1.py --outpath="${EXAMPLES_DIR}/"
check_execution "3-in-1评估"

echo "所有评估已完成！结果保存在 ${EXAMPLES_DIR}/annotation_3_in_1/vqa_result.json"

# 汇总所有评估结果并生成报告
echo "正在生成评估报告..."
cd "${T2I_COMPBENCH_DIR}"
python - <<EOF
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

# 结果目录
examples_dir = Path("${EXAMPLES_DIR}")

# 读取各评估结果
def load_results(file_path):
    if not os.path.exists(file_path):
        print(f"警告: {file_path} 不存在")
        return {}
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    result_dict = {}
    for item in data:
        question_id = item["question_id"]
        answer = float(item["answer"])
        result_dict[question_id] = answer
    
    return result_dict

# 加载所有评估结果
blip_results = load_results(examples_dir / "annotation_blip" / "vqa_result.json")
unidet_2d_results = load_results(examples_dir / "labels" / "annotation_obj_detection_2d" / "vqa_result.json")
unidet_3d_results = load_results(examples_dir / "labels" / "annotation_obj_detection_3d" / "vqa_result.json")
unidet_num_results = load_results(examples_dir / "annotation_num" / "vqa_result.json")
clip_results = load_results(examples_dir / "annotation_clip" / "vqa_result.json")
combined_results = load_results(examples_dir / "annotation_3_in_1" / "vqa_result.json")

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
        if file.endswith(f"_{str(qid).zfill(6)}.png"):
            sample_file = file
            break
    
    data.append({
        "ID": qid,
        "文件名": sample_file,
        "BLIP-VQA (属性)": blip_results.get(qid, np.nan),
        "UniDet-2D (空间)": unidet_2d_results.get(qid, np.nan),
        "UniDet-3D (空间)": unidet_3d_results.get(qid, np.nan),
        "UniDet (数量)": unidet_num_results.get(qid, np.nan),
        "CLIPScore": clip_results.get(qid, np.nan),
        "3-in-1 (综合)": combined_results.get(qid, np.nan)
    })

df = pd.DataFrame(data)

# 计算统计信息
stats = {
    "平均分": df.mean(numeric_only=True),
    "最高分": df.max(numeric_only=True),
    "最低分": df.min(numeric_only=True),
    "标准差": df.std(numeric_only=True)
}
stats_df = pd.DataFrame(stats)

# 保存报告
report_path = examples_dir / "evaluation_report.csv"
df.to_csv(report_path, index=False)

stats_path = examples_dir / "evaluation_stats.csv"
stats_df.to_csv(stats_path)

# 打印统计摘要
print("\n===== 评估统计摘要 =====")
print(f"总评估图像数: {len(df)}")
print("\n平均评分:")
for col, val in stats["平均分"].items():
    print(f"  {col}: {val:.4f}")

print(f"\n详细报告已保存至: {report_path}")
print(f"统计摘要已保存至: {stats_path}")
EOF

check_execution "评估报告生成"

echo "评估流程全部完成！"