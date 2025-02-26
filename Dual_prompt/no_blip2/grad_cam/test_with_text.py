import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import CLIP

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = CLIP.load("ViT-B/32", device=device)

# 加载图像并预处理
img_path = "/mnt/Data/wangshilong/pycode/HAT-pytorch/Dual_prompt/no_blip2/grad_cam/catdog.jpg" # 替换为您的图像路径
image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

# 假设文本输入
texts = ["a dog"]
text = CLIP.tokenize(texts).to(device)


def generate_attention_with_gradcam(image, text, model, device):
    # 获取图像和文本的输出
    logits_per_image, logits_per_text = model(image, text)
    similarity_score = logits_per_image[0, 0]

    # 计算梯度
    model.zero_grad()
    similarity_score.backward(retain_graph=True)  # 保留计算图

    # 提取图像的梯度
    if image.grad is None:
        print("Gradient is None. Ensure that image.requires_grad=True.")
    gradients = image.grad[0].cpu().numpy()  # 提取梯度并转换为 numpy 数组

    # 生成显著性图
    gradients = np.mean(gradients, axis=0)  # 对通道维度求平均
    gradients = np.maximum(gradients, 0)  # 只保留正梯度
    gradients = cv2.resize(gradients, (224, 224))  # 调整大小到 224x224

    # 归一化
    gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min())

    # 将显著性图叠加到原始图像
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    # 将图像张量转换为 numpy 并进行归一化
    image_np = image[0].permute(1, 2, 0).cpu().detach().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    # 生成可视化图像
    vis_image = show_cam_on_image(image_np, gradients)

    # 展示图片
    plt.imshow(vis_image)
    plt.axis('off')
    plt.show()

# 调用函数进行可视化
generate_attention_with_gradcam(image, text, model, device)