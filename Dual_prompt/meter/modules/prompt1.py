import torch
import torch.nn as nn


class Prompt(nn.Module):
    def __init__(self, length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',type=None ):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key  # 输入图像嵌入表示的键
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key  # default=True
        self.pool_size = pool_size  # prompt的数量
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt  # default=True
        self.type=type

        if self.prompt_pool:  # true 创建提示池
            prompt_pool_shape = (pool_size, length, embed_dim)

            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)

        # if using learnable prompt keys 如果使用可学习键
        if self.prompt_key:
            key_shape = (pool_size, embed_dim)  # prompt_size*dim
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            prompt_mean = torch.mean(self.prompt, dim=1)  # prompt.shape
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None,cls_feature=None):  # x_embed:batch*512
        out = dict()  # 字典
        if self.prompt_pool:
            # #根据self.embedding_key的值，选择不同的方式来计算x_embed的特征表示
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                x_embed_mean = cls_feature  # batch*dim
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1).to(self.device)  # Pool_size,dim
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # batch*dim


            # 计算相似度 prompt_key和x_embed_norm
            similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool_size 要确保两个tensor在相同设备上

            if prompt_mask is None:  # 检查是否提供了prompt_mask。如果没有提供prompt_mask，则说明需要自动选择最相关的提示。
                _, idx = torch.topk(similarity, k=self.top_k,
                                    dim=1)  # B, top_k  8*5 返回与每个图片最相关的五个 prompt 的索引。下划线返回的是对应的值，这里我们不需要

                if self.batchwise_prompt:  # 检查是否启用了batchwise_prompt，即是否在批量级别上选择提示，而不是为每个输入独立选择。
                    '''
                        batchwise_prompt的主要作用包括：
                        提高效率：在某些情况下，计算每个输入的最相关提示可能非常耗时，尤其是当批次大小很大或提示池很大时。通过在批次级别上选择提示，我们可以减少这种计算负担，因为这需要进行的相似度计算和排序操作更少。
                        增加稳定性：通过为整个批次选择一组共同的最相关提示，可以减少模型输出的方差，从而增加模型的稳定性。这是因为模型不会过度依赖于对单个输入特别设计的提示，而是利用一组在整个批次中普遍相关的提示。
                        促进一致性：在处理一批相似的输入时（例如，都属于同一类别的图像），选择一组共同的提示可以促进模型学习到更一致的特征表示，因为所有输入都将接收到相同的上下文信息和指导信号。
                        '''
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # 使用torch.unique提取当前bacth返回idx中所有唯一的提示索引(prompt_id)也就是该批次用到了哪些提示保存他们的索引及其统计他们的出现次数(id_counts)，sorted=True确保返回的索引是排序的。
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:  # 检查唯一提示索引的数量是否少于提示池的大小(pool_size)。如果是，需要补足到pool_size。触发条件：16个图片查询top5个prompt有一次都没查询到的prompt
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],),
                                                                     torch.min(idx.flatten()),
                                                                     device=prompt_id.device)])
                        id_counts = torch.cat(
                            [id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k)  # top_k   #找到出现次数最多的 k 个 prompt 的索引
                    major_prompt_id = prompt_id[major_idx]  # 存储 topk 个 prompt 的索引号
                    #print(f'slect id is:{major_prompt_id}')
                    #expand to batch 这里是为整个batch搞一个prompt 那就把这top5个prompt加到整个batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1)  # B, top_k   batch个图片每个图片后面加 5 个 prompt
                    '''
                    设我们有一个模型，它利用提示池来提升处理任务的能力。在这个场景中，提示池的大小（pool_size）被设定为5，意味着理想情况下，我们希望从5个不同的提示中为每个输入选择最相关的提示。
                    在某个批次处理中，基于相似度选择提示的结果（idx）如下：
                    [[1, 3], [2, 3]]
                    这意味着对于两个输入样本，我们分别找到了最相关的两个提示，索引为1和3对于第一个输入样本，索引为2和3对于第二个输入样本。
                    通过这个过程，我们得到唯一的提示索引[1, 2, 3]以及它们的出现次数[1, 1, 2]。
                    填充操作
                        由于prompt_id的数量（3个唯一提示索引）小于pool_size（5），我们需要补足到5。
                        根据上文的逻辑，我们将使用idx中的最小值进行填充，同时用0填充对应的id_counts。
                    示例解释：
                        确定填充值：在idx中，所有的提示索引是[1, 2, 3]，其中最小值为1。因此，我们将使用1作为填充值。
                    执行填充：
                        对于prompt_id，我们需要添加2个额外的1，使其长度达到5。填充后的prompt_id将是[1, 2, 3, 1, 1]。
                        对于id_counts，相应地，我们添加2个额外的0，填充后的id_counts将是[1, 1, 2, 0, 0]。
                    结果解释：
                         通过这种填充方式，我们得到了一个完整的、长度为pool_size的prompt_id和id_counts。
                         虽然填充的值并不反映真实的相似度选择结果，但它们保持了提示池的一致性和完整性，并且通过将填充的id_counts设为0，明确标识了这些填充的索引在当前批次中实际上并没有被选中。
                         这样的处理方式既保留了基于数据自然选择出的提示索引的有效信息，又通过一种相对保守的方法补足了不足的部分，从而使模型能够在后续步骤中更稳定、更一致地利用提示池。
                         '''
            else:
                idx = prompt_mask  # B, top_k

            batched_prompt_raw = self.prompt[idx]  # B, top_k, length, C    8*5*5*512
            '''根据索引idx（每个样本的top_k个最相关提示的索引）从提示池中选择提示。
            batched_prompt_raw的形状为[B, top_k, length, C]，其中B是批次大小，top_k是选中的提示数量，length是每个提示的长度，C是嵌入维度。
            '''
            batch_size, top_k, length, c = batched_prompt_raw.shape

            # 批次的 prompt 实际上是被展平了。
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)  # B, top_k * length, C

            out['prompt_idx'] = idx  # 每张图片对应相关的5个prompt索引

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            '''计算选中的提示键与输入嵌入之间的相似度。
            首先，根据idx从归一化的提示键中选择对应的键batched_key_norm，
            然后计算这些键与输入嵌入x_embed_norm的点积，最后求取平均相似度reduce_sim。
            '''
            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx]  # B, top_k, C    16*5*768 16张图片每张有五个相关的768维的prompt
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C  #在第一维度前加一个维度
            sim = batched_key_norm * x_embed_norm  # B, top_k, C  16*5*768  规范化之后的逐元素相乘凸显出提示关注图像的地方，便于优化模型训练
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # Scalar

            out['reduce_sim'] = reduce_sim
        else:  # prompt_pool = False
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))  #
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1).to(self.device)

        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]  # 25
        out['batched_prompt']=batched_prompt
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)  # prompted_embedding.shap : 16*221*768
        # batch*26*512
        return out
