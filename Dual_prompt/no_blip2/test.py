# 开发时间 2024/11/2 10:57
# 开发人员:牧良逢
import os
import copy

import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from config_no_blip2 import ex_test
from Dual_prompt.no_blip2.modal import DoubleEncoder
import torch
from PIL import Image
import clip
import torchvision.transforms as transforms
import cv2

ex_test.add_config({
    'use_prompt':True,

    'train_type':'VTPT3',

    #Prompt
    'embed_dim':768,
    "prompt_pool": True,
    "pool_size": 20,
    "prompt_length": 3,
    "top_k": 3,
    'prompt_init': 'uniform',
    'prompt_key': True,
    'prompt_key_init': 'uniform',
    'batchwise_prompt': False,
    'embedding_key': 'cls',
})
def get_transform():
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


@ex_test.automain
def main(_config):
    print(_config)
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"], workers=True)
    model = DoubleEncoder(_config)
    model.cuda()
    ckpt = torch.load(_config['checkpoint'])
    model.load_state_dict(ckpt['state_dict'])
    devices=model.devices
    transform=get_transform()
    image=Image.open("/mnt/Data/wangshilong/self_datasets/f30k/images/418112325.jpg").convert('RGB')
    img=transform(image).to(devices).unsqueeze(0)
    text="'A restaurant has modern wooden tables and chairs.'"
    text_ids=clip.tokenize(text).to(devices)

    with torch.no_grad():
        '================原始图像处理================='
        instance_query_image = model.get_query_embedding(img)['cls_feature']
        image_token_embedding = model.get_query_embedding(img)['patch_emb']
        image_prompt = model.prompt(x_embed=image_token_embedding,
                                   prompt_mask=None,
                                   cls_features=instance_query_image)['batched_prompt']
        image_output1 = model.image_encoder(img, image_prompt)
        '================原始文本处理=================='
        instance_query_text = model.get_text_feature_embedding(token_id=text_ids)['text_feature']
        text_prompt = model.imagePrompt_to_textPrompt_projection(image_prompt)
        text_output1 = model.text_encoder(instance_query_text, text_prompt)
        '===============原始图像-文本结果处理============'
        text_output_original = text_output1 / text_output1.norm(dim=1, keepdim=True)
        image_output_original = image_output1 / image_output1.norm(dim=1, keepdim=True)

    def visualize_attention(image, text, model, device):
        model.eval()
        attn_probs_sum = None
        num_layers = 0
        for i, blk in enumerate(model.visual_transformer.transformer.resblocks):
            if blk.attn_probs is not None:
                # 将多头注意力权重在头维度上平均，以得到一个综合的注意力图
                attn_probs_avg = blk.attn_probs.mean(dim=1)  # 取每个块的平均注意力
                if attn_probs_sum is None:
                    attn_probs_sum = attn_probs_avg
                else:
                    attn_probs_sum += attn_probs_avg
                num_layers += 1

        # 取多层的平均
        attn_map = attn_probs_sum / num_layers  # shape: [1, num_tokens]

        # 假设输入图像被切分成 num_tokens 个 patch，取出第一个 token（通常是 CLS token）对其他 token 的注意力
        cls_attention = attn_map[0, 10:]  # 忽略第一个 [CLS] token 自身的注意力

        # 将 cls_attention 转换为图像大小的形状
        dim = int(cls_attention.shape[0] ** 0.5)  # 假设 patch 数是正方形
        cls_attention = cls_attention.reshape(dim, dim)  # 49个数，每行7个排7行

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
    visualize_attention(img, text_ids, model, devices)




