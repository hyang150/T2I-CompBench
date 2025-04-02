#!/bin/bash
################################################################################
# 此脚本修改 BLIP_vqa.py，使其在生成标注时对空问题赋予默认问句，
# 从而确保每张图片都有有效问题参与 VQA 评估，进而能生成针对所有图片的评分结果。
################################################################################

# -----------------【1】基本目录设置 -----------------
T2I_COMPBENCH_DIR="/home/hyang150/csci_544/T2I-CompBench"
IMAGE_DIR="/home/hyang150/csci_544/generated_images"
OUT_DIR="$T2I_COMPBENCH_DIR/examples/annotation_blip"
BLIP_DIR="$T2I_COMPBENCH_DIR/BLIPvqa_eval"
EVAL_SCORE_DIR="$T2I_COMPBENCH_DIR/Evaluate_Score"

# -----------------【2】临时修改 BLIP_vqa.py -----------------
# 修改 Create_annotation_for_BLIP 函数中，对空问题的处理，将其默认赋值为 "What is in the image?"
# 备份原文件，并用 sed 替换那一行（注意空格和引号）
cd "$BLIP_DIR" || { echo "目录 $BLIP_DIR 不存在！"; exit 1; }
# 备份原文件
cp BLIP_vqa.py BLIP_vqa.py.bak
# 使用 sed 替换 "image_dict['question'] = ''" 为默认问句
sed -i 's/            image_dict\['"'"'question'"'"'\] = ''/            image_dict\['"'"'question'"'"'\] = "What is in the image?"/g' BLIP_vqa.py
cd - > /dev/null

# -----------------【3】复制图片并生成映射文件 -----------------
SAMPLES_DIR="$OUT_DIR/samples"
mkdir -p "$SAMPLES_DIR"
mkdir -p "$EVAL_SCORE_DIR"

rm -f "$SAMPLES_DIR"/*.png
MAPPING_FILE="$T2I_COMPBENCH_DIR/image_mapping.txt"
rm -f "$MAPPING_FILE"

echo "===== 正在复制图片到 $SAMPLES_DIR 并生成映射信息..."
find "$IMAGE_DIR" -type f -name "*.png" | while read -r img; do
  # 假设图片路径为 IMAGE_DIR/{Category}/PromptName_000xxx.png
  category=$(echo "$img" | awk -F'/' '{print $(NF-1)}')
  base=$(basename "$img")  # 例如：PromptName_000001.png
  new_name="${category}_${base}"
  cp "$img" "$SAMPLES_DIR/$new_name"
  echo "$new_name,$category,$base" >> "$MAPPING_FILE"
done
echo "===== 图片复制完成，映射信息保存在 $MAPPING_FILE"

# -----------------【4】调用 BLIP-VQA 脚本进行评估 -----------------
echo "===== 开始执行 BLIP-VQA 脚本..."
cd "$BLIP_DIR" || { echo "目录 $BLIP_DIR 不存在！"; exit 1; }
python BLIP_vqa.py --out_dir="../$(realpath --relative-to="$T2I_COMPBENCH_DIR" "$OUT_DIR")"
cd - > /dev/null

# -----------------【5】拆分统一评估结果 -----------------
RESULT_FILE="$OUT_DIR/annotation_blip/VQA/result/vqa_result.json"
if [ ! -f "$RESULT_FILE" ]; then
  echo "打分结果文件 $RESULT_FILE 不存在！"
  exit 1
fi

echo "===== 开始拆分打分结果到单个 JSON 文件..."
python << 'EOF'
import os, json

mapping_file = "/home/hyang150/csci_544/T2I-CompBench/image_mapping.txt"
result_file = "/home/hyang150/csci_544/T2I-CompBench/examples/annotation_blip/annotation_blip/VQA/result/vqa_result.json"
eval_score_dir = "/home/hyang150/csci_544/T2I-CompBench/Evaluate_Score"

# 构建映射：key 为 question_id（从原始文件名提取最后6位数字，去除前导0）
mapping = {}
with open(mapping_file, "r") as f:
    for line in f:
        new_name, category, orig = line.strip().split(",")
        try:
            num_part = orig.rsplit('_', 1)[-1].split('.')[0]
            question_id = str(int(num_part))
        except Exception:
            question_id = None
        if question_id is not None:
            mapping[question_id] = (new_name, category, orig)

with open(result_file, "r") as f:
    results = json.load(f)

for item in results:
    qid = str(item.get("question_id"))
    if qid in mapping:
        new_name, category, orig = mapping[qid]
        parts = new_name.split('_', 1)
        file_core = parts[1].rsplit('.', 1)[0] if len(parts)==2 else new_name.rsplit('.',1)[0]
        out_dir = os.path.join(eval_score_dir, category)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{file_core}.json")
        with open(out_path, "w") as out_f:
            json.dump(item, out_f)
        print(f"已写入 {out_path}")
    else:
        print(f"[警告] 未找到 question_id {qid} 的映射信息")
EOF

echo "===== 所有评估结果已拆分并保存在 $EVAL_SCORE_DIR 目录中。"
