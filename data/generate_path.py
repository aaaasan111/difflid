import os
import argparse


# 根据数据集地址生成训练图像路径的txt文件
def generate_pairs(gt_dir, haze_dir, output_txt):
    gt_files = set(f for f in os.listdir(gt_dir) if f.lower().endswith('.png'))
    gt_to_haze = {gt: [] for gt in gt_files}
    for fname in os.listdir(haze_dir):
        if not fname.lower().endswith('.png'):
            continue
        prefix = fname.split('_')[0]  # '1'
        gt_name = f"{prefix}.png"
        if gt_name not in gt_to_haze:
            print(f"警告：未找到雾图 {fname} 对应的GT图像 {gt_name}")
            continue
        gt_to_haze[gt_name].append(fname)

    with open(output_txt, 'w') as out_f:
        for gt_name in sorted(gt_to_haze.keys(), key=lambda x: int(os.path.splitext(x)[0])):
            for haze_fname in sorted(gt_to_haze[gt_name]):
                haze_path = os.path.join(haze_dir, haze_fname)
                gt_path = os.path.join(gt_dir, gt_name)
                out_f.write(f"{haze_path}|{gt_path}\n")

    print(f"配对信息已写入：{output_txt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate haze-GT image pairs for RESIDE ITS subset')
    parser.add_argument('--gt_dir', type=str, default="",
                        help='Path to GT images directory')
    parser.add_argument('--haze_dir', type=str, default="",
                        help='Path to haze images directory')
    parser.add_argument('--output', type=str, default='', help='Output txt file path')
    args = parser.parse_args()
    generate_pairs(args.gt_dir, args.haze_dir, args.output)
