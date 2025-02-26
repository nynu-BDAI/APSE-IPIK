import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import CLIP

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = CLIP.load("ViT-B/32", device=device)

# 加载图像并预处理
img_path ="/mnt/Data/wangshilong/self_datasets/f30k/images/4926108699.jpg"# 替换为您的图像路径
image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

# 假设文本输入
texts = ["a small cat"]
text = CLIP.tokenize(texts).to(device)

def visualize_attention(image, text, model, device):
    model.eval()
    with torch.no_grad():
        # 前向传播得到 logits
        logits_per_image, logits_per_text = model(image, text)

    # 提取每一层的 attn_probs 并累积
    attn_probs_sum = None
    num_layers = 0
    for i, blk in enumerate(model.visual.transformer.resblocks):
        if blk.attn_probs is not None:
            # 将多头注意力权重在头维度上平均，以得到一个综合的注意力图
            attn_probs_avg = blk.attn_probs.mean(dim=1)  # 取每个块的平均注意力
            if attn_probs_sum is None:
                attn_probs_sum = attn_probs_avg
            else:
                attn_probs_sum += attn_probs_avg
            num_layers += 1

    # 取多层的平均
    attn_map = attn_probs_sum / num_layers  # shape: [num_tokens, num_tokens]

    # 假设输入图像被切分成 num_tokens 个 patch，取出第一个 token（通常是 CLS token）对其他 token 的注意力
    cls_attention = attn_map[0, 1:]  # 忽略第一个 [CLS] token 自身的注意力

    # 将 cls_attention 转换为图像大小的形状
    dim = int(cls_attention.shape[0] ** 0.5)  # 假设 patch 数是正方形
    cls_attention = cls_attention.reshape(dim, dim) #49个数，每行7个排7行

    # 转移到 CPU 并检查 NaN 或 Inf 值
    cls_attention = cls_attention.cpu().numpy()  # 转换为 NumPy 数组
    if np.isnan(cls_attention).any() or np.isinf(cls_attention).any():
        cls_attention = np.nan_to_num(cls_attention, nan=0.0, posinf=1.0, neginf=0.0)

    # 增强注意力对比度
    cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min())
    cls_attention = np.power(cls_attention, 2)  # 对比度增强，可以尝试不同的幂次

    # 确保 cls_attention 是 float32 类型
    cls_attention = cls_attention.astype(np.float32)

    # 调整大小到 224x224
    cls_attention = cv2.resize(cls_attention, (224, 224))  # 调整到与输入图像相同的大小

    # 将注意力图叠加到原始图像
    def show_cam_on_image(img, mask):
        mask = 1 - mask  # 取反操作，使得高值变为低值，低值变为高值
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    # 将图像张量转换为 numpy 并进行归一化
    image_np = image[0].permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    # 生成可视化图像
    vis_image = show_cam_on_image(image_np, cls_attention)

    # 展示图片
    plt.imshow(vis_image)
    plt.axis('off')
    plt.show()

# 调用函数进行可视化
visualize_attention(image, text, model, device)