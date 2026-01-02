import os
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr_ssim(image_path1, image_path2):
    # 打开两张图片
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # 转换为灰度图像
    img1_gray = img1.convert('L')
    img2_gray = img2.convert('L')

    # 转换为 NumPy 数组
    img1_array = np.array(img1_gray)
    img2_array = np.array(img2_gray)

    # 计算 PSNR
    psnr_value = psnr(img1_array, img2_array)

    # 计算 SSIM
    ssim_value = ssim(img1_array, img2_array)

    return psnr_value, ssim_value

pred_res = "output/pasd-DRSRDeg-moco-caption-500000"
# pred_res = "/data0/llj/workspace/DRSR/results/aid_our/Iso/sig1.0/SR"
gt = "/data0/llj/dataset/AID/real_world/test_HR"

# 按照类对图片group
img_group = {}
# import pdb;pdb.set_trace()
for file in os.listdir(pred_res):
    if file.endswith(".jpg") or file.endswith(".png"):
        cls_name = file.split("_")[0]
        if cls_name in img_group.keys():
            img_group[cls_name].append(file)
        else:
            img_group[cls_name] = []

# print(img_group)

# 计算每个group下的psnr和ssim
res = {}
global_psnr = []
global_ssim = []
for k,v in img_group.items():
    cur_psnr = []
    cur_ssim = []
    for file in v:
        pred_path = os.path.join(pred_res, file)
        gt_path = os.path.join(gt, file.replace(".png", ".jpg"))
        # print(pred_path, gt_path)
        psnr_value, ssim_value = calculate_psnr_ssim(pred_path, gt_path)
        cur_psnr.append(psnr_value)
        cur_ssim.append(ssim_value)
    avg_psnr = sum(cur_psnr) / len(cur_psnr)
    avg_ssim = sum(cur_ssim) / len(cur_ssim)
    res[k] = (avg_psnr, avg_ssim)
    print(f"{k} {avg_psnr} {avg_ssim}")

    global_psnr.append(avg_psnr)
    global_ssim.append(avg_ssim)

global_psnr = sum(global_psnr) / len(global_psnr)
global_ssim = sum(global_ssim) / len(global_ssim)

print(f"Overall {global_psnr} {global_ssim}")


