import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import clip
from . import swin_transformer as swin
from . import objectives1, meter_utils1
from .bert import BertModel
from .prompt1 import Prompt

def freeze_layers(model, bool):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = bool


class DoubleEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model, self.preprocess = clip.load(name='ViT-B/32')
        freeze_layers(self.model, False)
        for module in self.model.modules():
            module.to(torch.float32)
        self.freeze_positional_embedding()
        self.eval_bool = False
        #self.cross_attn = CrossAttention(embed_dim=512, num_heads=4)
        self.datasets = ['coco', 'f30k', 'iaprtc12', 'ec', 'rsicd']

        self.image_prompts = nn.ModuleDict({
            dataset: Prompt(length=config['prompt_length'], embed_dim=512, embedding_key=config['embedding_key'],
                            prompt_init=config['prompt_init'], prompt_pool=config['prompt_pool'],
                            prompt_key=config['prompt_key'], pool_size=config['pool_size'], top_k=config['top_k'],
                            batchwise_prompt=config['batchwise_prompt'], prompt_key_init=config['prompt_key_init'],
                            type='specialized') for dataset in self.datasets
        })

        self.text_prompts = nn.ModuleDict({
            dataset: Prompt(length=config['prompt_length'], embed_dim=512, embedding_key=config['embedding_key'],
                            prompt_init=config['prompt_init'], prompt_pool=config['prompt_pool'],
                            prompt_key=config['prompt_key'], pool_size=config['pool_size'], top_k=config['top_k'],
                            batchwise_prompt=config['batchwise_prompt'], prompt_key_init=config['prompt_key_init'],
                            type='specialized') for dataset in self.datasets
        })

        self.general_image_prompt = Prompt(length=config['prompt_length'], embed_dim=512,
                                           embedding_key=config['embedding_key'],
                                           prompt_init=config['prompt_init'], prompt_pool=config['prompt_pool'],
                                           prompt_key=config['prompt_key'], pool_size=config['pool_size'],
                                           top_k=config['top_k'],
                                           batchwise_prompt=config['batchwise_prompt'],
                                           prompt_key_init=config['prompt_key_init'],
                                           type='general')

        self.general_text_prompt = Prompt(length=config['prompt_length'], embed_dim=512,
                                          embedding_key=config['embedding_key'],
                                          prompt_init=config['prompt_init'], prompt_pool=config['prompt_pool'],
                                          prompt_key=config['prompt_key'], pool_size=config['pool_size'],
                                          top_k=config['top_k'],
                                          batchwise_prompt=config['batchwise_prompt'],
                                          prompt_key_init=config['prompt_key_init'],
                                          type='general')

        self.scale = 768 ** -0.5
        self.image_proj = nn.Parameter(scale * torch.randn(512, 768))
        self.image_proj_back = nn.Parameter(scale * torch.randn(768, 512))


    def freeze_positional_embedding(self):
        for name, param in self.model.named_parameters():
            if 'positional_embedding' in name:
                print(f"Freezing {name}")
                param.requires_grad = False
    def set_current_dataset(self, dataset_name):
        self.current_dataset = dataset_name
        for name, param in self.named_parameters():
            if 'prompt' in name:
                if dataset_name in name or 'general' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            #print(f"Name: {name}, requires_grad: {param.requires_grad}, Shape: {param.shape}")

    def infer1(self, batch, image_token_type_idx=1, img=None):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey].cuda()

        text_ids = batch["text_ids"].cuda()

        text_output1 = self.model.encode_text(text_ids)
        image_output1 = self.model.encode_image(img)

        # 得到通用提示池提示
        general_i_prompt = self.general_image_prompt(image_output1)['batched_prompt']
        general_t_prompt = self.general_text_prompt(text_output1)['batched_prompt']

        # 使用当前数据集的提示池处理图像和文本
        specialized_i_prompt = self.image_prompts[self.current_dataset](image_output1)['batched_prompt']
        specialized_t_prompt = self.text_prompts[self.current_dataset](text_output1)['batched_prompt']

        text_output1=torch.unsqueeze(text_output1,dim=1)
        image_output1=torch.unsqueeze(image_output1,dim=1)

        'cross-attention'
        #text_output1=text_output1.permute(1, 0, 2)
        #image_output1=image_output1.permute(1, 0, 2)
        #cross_feature_image,cross_feature_text=self.cross_attn(image_output1,text_output1)

        'com_attention'
        fix_feature=torch.cat([image_output1,text_output1],dim=1)
        cross_feature_text,_=self.func_attention(fix_feature,text_output1,self.config["lambda_softmax"])
        cross_feature_image,_ = self.func_attention(fix_feature,image_output1,self.config["lambda_softmax"])

        '''cross_feature_image=cross_feature_image.permute(1,0,2)
        cross_feature_text=cross_feature_text.permute(1,0,2)'''

        combined_text_input=torch.cat([cross_feature_text,specialized_t_prompt,general_t_prompt],dim=1)
        combined_image_input=torch.cat([cross_feature_image,specialized_i_prompt,general_i_prompt],dim=1)


        text_output2 = self.second_encode_text(combined_text_input)
        image_output2 = self.second_encode_image(combined_image_input)


        image_output2 = image_output2 / image_output2.norm(dim=1, keepdim=True)
        text_output2 = text_output2 / text_output2.norm(dim=1, keepdim=True)


        image_output1 = image_output1 / image_output1.norm(dim=1, keepdim=True)
        text_output1 = text_output1 / text_output1.norm(dim=1, keepdim=True)

        ret = {
            'text_output1': text_output1,
            'image_output1': image_output1,
            'text_output2': text_output2,
            'image_output2': image_output2,

        }
        return ret


    def forward(self, batch):
        ret = dict()
        if "irtr" in self.current_tasks:
            ret.update(objectives1.compute_irtr_my(self, batch))
        return ret

    def training_step(self, batch, batch_idx):
        self.eval_bool = True
        meter_utils1.set_task(self)
        output = self(batch)
        #print(self.general_image_prompt.prompt.data)
        #print(self.image_prompts['f30k'].prompt.data)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        '''for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"Gradient of {name}: {param.grad}")
            else:
                print(f"Gradient of {name}: None")'''
        return total_loss

    def training_epoch_end(self, outs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outs):
        if self.current_epoch != 0:
            if self.config['exp_name'] == "finetune_irtr_iaprtc12":
                meter_utils1.epoch_eval_irtr_nn(self)
            elif self.config['exp_name'] == 'finetune_irtr_ec':
                meter_utils1.epoch_eval_irtr_nn(self)
            else:
                meter_utils1.epoch_eval_irtr(self)

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outs):
        if self.config['exp_name'] == "finetune_irtr_iaprtc12":
            meter_utils1.epoch_eval_irtr_nn(self, is_test=True)
        elif self.config['exp_name'] == 'finetune_irtr_ec':
            meter_utils1.epoch_eval_irtr_nn(self, is_test=True)
        else:
            meter_utils1.epoch_eval_irtr(self, is_test=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
    '''def image_combine_prompt(self, image_cls, type='general'):
        if type == 'general':
            combine_image_input = self.general_image_prompt(image_cls)['prompted_embedding']
        else:
            combine_image_input = self.image_prompts[self.current_dataset](image_cls)['prompted_embedding']
        return combine_image_input

    def text_combine_prompt(self, text_cls, type='general'):
        if type == 'general':
            combine_text_input = self.general_text_prompt(text_cls)['prompted_embedding']
        else:
            combine_text_input = self.text_prompts[self.current_dataset](text_cls)['prompted_embedding']
        return combine_text_input'''

    def second_encode_image(self, combine_image_input):
        assert torch.isfinite(combine_image_input).all(), "Input tensor contains NaN or Inf"
        assert torch.isfinite(self.image_proj).all(), "Projection matrix contains NaN or Inf"
        projected_input = combine_image_input @ self.image_proj  # 512-->768
        adapt_transformer_feature = projected_input.permute(1, 0, 2)  # Transformer的标准输入形式。
        x = self.model.visual.transformer(adapt_transformer_feature)
        x = x.permute(1, 0, 2)
        x = self.model.visual.ln_post(x)
        x = x @ self.image_proj_back
        x = x[:, 0, :]
        return x

    def second_encode_text(self, combine_text_input):
        batch, num_tokens, dim = combine_text_input.shape
        expand_combine_text_input = torch.zeros(batch, self.model.context_length, dim)
        expand_combine_text_input[:, :num_tokens, :] = combine_text_input
        adapt_transformer_feature = expand_combine_text_input.permute(1, 0, 2).cuda()
        x = self.model.transformer(adapt_transformer_feature)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x).type(self.model.dtype)
        x = x[:, 0, :]
        return x

    def l2norm(self,X, dim, eps=1e-8):
        """L2-normalize columns of X
        """
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
        X = torch.div(X, norm)
        return X

    def func_attention(self,query, context, smooth, eps=1e-8):
        """
        query: (n_context, queryL, d)
        context: (n_context, sourceL, d)
        """
        batch_size_q, queryL = query.size(0), query.size(1)
        batch_size, sourceL = context.size(0), context.size(1)

        # Get attention
        # --> (batch, d, queryL)
        queryT = torch.transpose(query, 1, 2)

        # (batch, sourceL, d)(batch, d, queryL)
        # --> (batch, sourceL, queryL)
        attn = torch.bmm(context, queryT)

        attn = nn.LeakyReLU(0.1)(attn)
        attn = self.l2norm(attn, 2)
        # --> (batch, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        attn = nn.Softmax(1)(attn * smooth)
        # --> (batch*queryL, sourceL)
        attn = attn.view(batch_size * queryL, sourceL)
        #attn = nn.Softmax(1)(attn * smooth)
        # --> (batch, queryL, sourceL)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> (batch, sourceL, queryL)
        attnT = torch.transpose(attn, 1, 2).contiguous()
        # print(attnT.shape)

        # pic = attnT[0][0].view(28, 28)
        # print(pic)
        # plt.matshow(pic.data.cpu().numpy(), cmap=plt.cm.Blues)
        # plt.savefig('3.jpg')
        # assert 1==0

        # --> (batch, d, sourceL)
        contextT = torch.transpose(context, 1, 2)
        # (batch x d x sourceL)(batch x sourceL x queryL)
        # --> (batch, d, queryL)
        weightedContext = torch.bmm(contextT, attnT)
        # --> (batch, queryL, d)
        weightedContext = torch.transpose(weightedContext, 1, 2) #58*2*512

        return weightedContext, attnT


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim,num_heads)

    def forward(self, img_feats, text_feats):
        # Image as query, text as key/value
        attn_output_img_query, _ = self.multihead_attn(img_feats, text_feats, text_feats)
        # Text as query, image as key/value
        attn_output_text_query, _ = self.multihead_attn(text_feats, img_feats, img_feats)
        return attn_output_img_query, attn_output_text_query





