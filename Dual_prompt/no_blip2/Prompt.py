# 开发时间 2024/10/21 10:08
# 开发人员:牧良逢
import torch
import torch.nn as nn
class Prompt(nn.Module):
    """
    Prompt module for the L2P (Learning to Prompt) strategy.

    Wang, Zifeng, et al. "Learning to prompt for continual learning."
    Proceedings of the IEEE/CVF Conference on Computer Vision and \
    Pattern Recognition. 2022.

    Implementation is based on:
    - https://github.com/JH-LEE-KR/l2p-pytorch

    These prompts are added to L2P model in models.timm_vit
    """

    def __init__(
        self,
        length=5,
        embed_dim=768,
        embedding_key="mean",
        prompt_init="uniform",
        prompt_pool=False,
        prompt_key=False,
        pool_size=None,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init="uniform",
    ):
        """
        Args:
            length (int): length of the prompt. Default 5.
            embed_dim (int): embedding dimension of the prompt. Default 768.
            embedding_key (str): method to generate embedding to find key \
                                similary. Default "mean".
            prompt_init (str): initialization of the prompt pool. \
                                Default "uniform".
            prompt_pool (bool): use prompt pool or not. Default False.
            prompt_key (bool): use learnable prompt keys. Default False.
            pool_size (int): size of the pool.
            top_k (int): select the top k similar prompts.
            batchwise_prompt (bool): use prompt batchwise. Defalt False.
            prompt_key_init (str): initialization of the key pool. \
                                Default "uniform",
        """
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == "zero":
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == "uniform":
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == "zero":
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == "uniform":
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x**2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(
            torch.maximum(square_sum, torch.tensor(epsilon, device=x.device))
        )
        return x * x_inv_norm


    def forward(self,x_embed,prompt_mask=None,cls_features=None):
        """
        Args:
            x_embed: input tensor (Batch_size, Sequence_length, embed_dim)
            prompt_mask: mask to select specific prompts.
            cls_features: key features to find the close prompts
        """
        out = dict()

        if self.prompt_pool:  # 如果使用prompt池
            # 根据embedding_key来选择不同的方式计算查询特征
            if self.embedding_key == "mean":
                x_embed_mean = torch.mean(x_embed, dim=1)  # B, embed_dim
            elif self.embedding_key == "max":
                x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, embed_dim
            elif self.embedding_key == "mean_max":
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == "cls":
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, embed_dim
                else:
                    x_embed_mean = cls_features  # B, embed_dim
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            # 对 prompt_key 和 查询向量 进行 L2 归一化
            prompt_norm = self.l2_normalize(self.prompt_key, dim=1)  # Pool_size, embed_dim
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # Batch_size, embed_dim

            # 计算查询特征和提示池prompt_key之间的相似度 (Batch_size, Pool_size)
            similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool_size

            # 针对每一张图片选择top_k个最相似的prompt
            if prompt_mask is None:
                # 每张图片选取top_k个prompt
                _, idx = torch.topk(similarity, k=self.top_k, dim=1)  # B, top_k
                # print(idx)
            else:
                # 如果提供了prompt_mask，直接使用
                idx = prompt_mask  # B, top_k

            # 根据索引idx从提示池中提取对应的提示
            batched_prompt_raw = self.prompt[idx]  # B, top_k, length, embed_dim
            batch_size, top_k, length, c = batched_prompt_raw.shape

            # 将提示reshape为 (Batch_size, top_k * length, embed_dim)
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)

            out["prompt_idx"] = idx  # 输出每个图像选中的prompt索引

            # 还可以返回一些用于调试的中间信息
            out["prompt_norm"] = prompt_norm
            out["x_embed_norm"] = x_embed_norm
            out["similarity"] = similarity
            out['batched_prompt']=batched_prompt
            # 将选择的prompt和输入嵌入拼接 (Batch_size, prompt+token, embed_dim)
            out["prompted_embedding"] = torch.cat([batched_prompt, x_embed], dim=1)

        else:
            # 如果没有使用prompt池，直接返回初始化好的prompt
            if self.prompt_init == "zero":
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == "uniform":
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)

            out["prompted_embedding"] = torch.cat([batched_prompt, x_embed], dim=1)

        # 输出包含拼接后的提示嵌入和其他信息
        return out
