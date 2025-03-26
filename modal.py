# 开发时间 2024/7/25 17:03
# 开发人员:牧良逢
import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
# import clip
from grad_cam import CLIP
from . import objectives_no_blip2, meter_utils_no_blip2
from transformers import Blip2Processor, Blip2ForConditionalGeneration,AutoProcessor
from Prompt import Prompt
from clip_encoders import CustomImageEncoder,CustomTextEncoder,CustomTextEncoder_textwithprompt
class DoubleEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.devices ='cuda' if torch.cuda.is_available() else 'cpu'
        self.save_hyperparameters()
        self.model, self.preprocess = CLIP.load(name='ViT-B/32', device=self.devices)
        for module in self.model.modules():
            module.to(torch.float32)
        self.logit_scale=self.model.logit_scale
        self.datasets = ['coco', 'f30k', 'iaprtc12', 'ec', 'rsicd']
        self.visual_transformer = self.model.visual
        self.image_encoder = CustomImageEncoder(self.visual_transformer).to(self.devices)
        self.text_encoder =CustomTextEncoder(self.model,self.device,dtype=self.model.dtype)
        self.text_encoder_attention=CustomTextEncoder_textwithprompt(self.model,self.device,dtype=self.model.dtype)
        self.prompt=Prompt(length=self.config['prompt_length'],
                           embed_dim=self.config['embed_dim'],
                           embedding_key=self.config['embedding_key'],
                           prompt_init=self.config['prompt_init'],
                           prompt_pool=self.config['prompt_pool'],
                           prompt_key=self.config['prompt_key'],
                           pool_size=self.config['pool_size'],
                           top_k=self.config['top_k'],
                           batchwise_prompt=self.config['batchwise_prompt'],
                           prompt_key_init=self.config['prompt_key_init'])

        self.prompt_text = Prompt(length=self.config['prompt_length'],
                             embed_dim=512,
                             embedding_key=self.config['embedding_key'],
                             prompt_init=self.config['prompt_init'],
                             prompt_pool=self.config['prompt_pool'],
                             prompt_key=self.config['prompt_key'],
                             pool_size=self.config['pool_size'],
                             top_k=self.config['top_k'],
                             batchwise_prompt=self.config['batchwise_prompt'],
                             prompt_key_init=self.config['prompt_key_init'])

        self.imagePrompt_to_textPrompt_projection = nn.Linear(768, 512)
        self.text_prompt_attention_layer=nn.MultiheadAttention(embed_dim=512, num_heads=8,batch_first=True)

        #uncertainty
        self.K=self.config['K_prototype']
        self.tau=self.config['tau']
        self.embed_dim=self.model.visual.output_dim

        # vision prototype
        self.v_prototype = nn.parameter.Parameter(torch.zeros(self.K, self.embed_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.v_prototype)

    def encode_text_with_grad(self, text):
        text_feature=self.model.encode_text(text)
        return text_feature
    def encode_image_with_grad(self,image):
        image_features=self.model.encode_image(image)
        return image_features
    def set_current_dataset(self, dataset_name):
        self.current_dataset = dataset_name

    def infer1(self, batch, image_token_type_idx=1, img=None):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"

            img = batch[imgkey].cuda()
            text_ids = batch["original_text_ids"].cuda()
            #use Deepseek
            if self.training:
                global_text_ids=batch["global_text_ids"].cuda()
                background_text_ids=batch["background_text_ids"].cuda()
                entity_text_ids=batch["entity_text_ids"].cuda()
            else:
                global_text_ids = None
                background_text_ids = None
                entity_text_ids = None

            if self.config['train_type']=='FFT':
                # print('当前任务是FFT（fully fine-tune）')
                #原始文本处理
                text_output1 =self.encode_text_with_grad(text_ids)
                #原始图像处理
                #image_output1=self.encode_image_with_grad(img)
                image_output1=self.get_query_embedding(img)['feature']

                if self.training:
                    '===============全局文本处理===================='
                    global_text_output1 = self.encode_text_with_grad(global_text_ids)
                    global_t_alpha, global_tu, global_vv_logits = self.text_uncertainty_modeling(global_text_output1)
                    '===============背景文本处理===================='
                    background_text_output1 = self.encode_text_with_grad(background_text_ids)
                    background_t_alpha, background_tu, background_vv_logits = self.text_uncertainty_modeling(
                        background_text_output1)
                    '===============实体文本处理===================='
                    entity_text_output1 = self.encode_text_with_grad(entity_text_ids)
                    entity_t_alpha, entity_tu, entity_vv_logits = self.text_uncertainty_modeling(entity_text_output1)

                    text_output_global = global_text_output1 / global_text_output1.norm(dim=1, keepdim=True)
                    text_output_background = background_text_output1 / background_text_output1.norm(dim=1, keepdim=True)
                    text_output_entity = entity_text_output1 / entity_text_output1.norm(dim=1, keepdim=True)
                else:
                    global_text_output1 = None
                    background_text_output1 = None
                    entity_text_output1 = None
                    global_t_alpha, global_tu, global_vv_logits = None, None, None
                    background_t_alpha, background_tu, background_vv_logits = None, None, None
                    entity_t_alpha, entity_tu, entity_vv_logits = None, None, None

                text_output_original = text_output1 / text_output1.norm(dim=1, keepdim=True)
                image_output_original = image_output1 / image_output1.norm(dim=1, keepdim=True)
                ret = {
                    'text_output_original': text_output_original,
                    'image_output_original': image_output_original
                }
                if self.training:
                    ret.update({
                        'text_output_global': text_output_global,
                        'text_output_background': text_output_background,
                        'text_output_entity': text_output_entity,
                        'global_t_alpha': global_t_alpha,
                        'global_tu': global_tu,
                        'global_vv_logits': global_vv_logits,
                        'background_t_alpha': background_t_alpha,
                        'background_tu': background_tu,
                        'background_vv_logits': background_vv_logits,
                        'entity_t_alpha': entity_t_alpha,
                        'entity_tu': entity_tu,
                        'entity_vv_logits': entity_vv_logits,
                    })
                return ret


            #use prompt
            else:
                #VPT
                if self.config['train_type']=="VPT":
                    #print('当前任务是VPT')
                    '================原始图像==================='
                    instance_query_image=self.get_query_embedding(img)['cls_feature']
                    image_token_embedding=self.get_query_embedding(img)['patch_emb']
                    image_prompt=self.prompt(x_embed=image_token_embedding,prompt_mask=None,cls_features=instance_query_image)['batched_prompt']
                    image_output1=self.image_encoder(img,image_prompt)
                    '================原始文本===================='
                    text_output1 = self.encode_text_with_grad(text_ids)
                    if self.training:
                      '===============全局文本处理===================='
                      global_text_output1=self.encode_text_with_grad(global_text_ids)
                      global_t_alpha, global_tu, global_vv_logits=self.text_uncertainty_modeling(global_text_output1)
                      '===============背景文本处理===================='
                      background_text_output1 = self.encode_text_with_grad(background_text_ids)
                      background_t_alpha, background_tu, background_vv_logits=self.text_uncertainty_modeling(background_text_output1)
                      '===============实体文本处理===================='
                      entity_text_output1=self.encode_text_with_grad(entity_text_ids)
                      entity_t_alpha,entity_tu,entity_vv_logits=self.text_uncertainty_modeling(entity_text_output1)

                      text_output_global = global_text_output1 / global_text_output1.norm(dim=1, keepdim=True)
                      text_output_background = background_text_output1 / background_text_output1.norm(dim=1,keepdim=True)
                      text_output_entity = entity_text_output1 / entity_text_output1.norm(dim=1, keepdim=True)
                    else:
                        global_text_output1 = None
                        background_text_output1 = None
                        entity_text_output1 = None
                        global_t_alpha, global_tu, global_vv_logits = None, None, None
                        background_t_alpha, background_tu, background_vv_logits = None, None, None
                        entity_t_alpha, entity_tu, entity_vv_logits = None, None, None

                    text_output_original = text_output1 / text_output1.norm(dim=1, keepdim=True)
                    image_output_original = image_output1 / image_output1.norm(dim=1, keepdim=True)
                    ret = {
                        'text_output_original': text_output_original,
                        'image_output_original': image_output_original
                    }
                    if self.training:
                        ret.update({
                            'text_output_global': text_output_global,
                            'text_output_background': text_output_background,
                            'text_output_entity': text_output_entity,
                            'global_t_alpha': global_t_alpha,
                            'global_tu': global_tu,
                            'global_vv_logits': global_vv_logits,
                            'background_t_alpha': background_t_alpha,
                            'background_tu': background_tu,
                            'background_vv_logits': background_vv_logits,
                            'entity_t_alpha': entity_t_alpha,
                            'entity_tu': entity_tu,
                            'entity_vv_logits': entity_vv_logits,
                        })
                    return ret

                #V-TPT Function1
                elif self.config['train_type'] =='VTPT1':
                    #print('当前任务是V-T Prompt Tune Func1')
                    '==============原始图像==============='
                    instance_query_image = self.get_query_embedding(img)['cls_feature']
                    image_token_embedding = self.get_query_embedding(img)['patch_emb']
                    image_prompt =self.prompt(x_embed=image_token_embedding, prompt_mask=None,cls_features=instance_query_image)['batched_prompt']
                    image_output1 = self.image_encoder(img, image_prompt)
                    '=============原始文本================='
                    instance_query_text=self.get_text_feature_embedding(token_id=text_ids)['text_feature']
                    text_token_embedding=self.get_text_feature_embedding(token_id=text_ids)['token_embedding']
                    text_prompt=self.prompt_text(x_embed=text_token_embedding,prompt_mask=None,cls_features=instance_query_text)['batched_prompt']
                    text_output1=self.text_encoder(instance_query_text,text_prompt)

                    if self.training:
                        '===============全局文本处理===================='
                        instance_query_globaltext = self.get_text_feature_embedding(token_id=global_text_ids)['text_feature']
                        global_text_token_embedding=self.get_text_feature_embedding(token_id=global_text_ids)['token_embedding']
                        global_text_prompt = self.prompt_text(x_embed=global_text_token_embedding,prompt_mask=None,cls_features=instance_query_globaltext)['batched_prompt']
                        global_text_output1 = self.text_encoder(instance_query_globaltext, global_text_prompt)

                        global_t_alpha, global_tu, global_vv_logits = self.text_uncertainty_modeling(global_text_output1)

                        '===============背景文本处理===================='
                        instance_query_backgroundtext = self.get_text_feature_embedding(token_id=background_text_ids)['text_feature']
                        background_text_token_embedding=self.get_text_feature_embedding(token_id=background_text_ids)['token_embedding']
                        background_text_prompt = self.prompt_text(x_embed=background_text_token_embedding,prompt_mask=None,cls_features=instance_query_backgroundtext)['batched_prompt']
                        background_text_output1 = self.text_encoder(instance_query_backgroundtext,background_text_prompt)

                        background_t_alpha, background_tu, background_vv_logits = self.text_uncertainty_modeling(background_text_output1)

                        '===============实体文本处理===================='
                        instance_query_entitytext = self.get_text_feature_embedding(token_id=entity_text_ids)['text_feature']
                        entity_text_token_embedding=self.get_text_feature_embedding(token_id=entity_text_ids)['token_embedding']
                        entity_text_prompt = self.prompt_text(x_embed=entity_text_token_embedding,prompt_mask=None,cls_features=instance_query_entitytext)['batched_prompt']
                        entity_text_output1 = self.text_encoder(instance_query_entitytext, entity_text_prompt)

                        entity_t_alpha, entity_tu, entity_vv_logits = self.text_uncertainty_modeling(entity_text_output1)

                        text_output_global = global_text_output1 / global_text_output1.norm(dim=1, keepdim=True)
                        text_output_background = background_text_output1 / background_text_output1.norm(dim=1,keepdim=True)
                        text_output_entity = entity_text_output1 / entity_text_output1.norm(dim=1, keepdim=True)
                    else:
                        global_text_output1 = None
                        background_text_output1 = None
                        entity_text_output1 = None
                        global_t_alpha, global_tu, global_vv_logits = None, None, None
                        background_t_alpha, background_tu, background_vv_logits = None, None, None
                        entity_t_alpha, entity_tu, entity_vv_logits = None, None, None


                    text_output_original = text_output1 / text_output1.norm(dim=1, keepdim=True)
                    image_output_original = image_output1 / image_output1.norm(dim=1, keepdim=True)

                    ret = {
                        'text_output_original': text_output_original,
                        'image_output_original': image_output_original
                            }
                    if self.training:
                        ret.update({
                            'text_output_global': text_output_global,
                            'text_output_background': text_output_background,
                            'text_output_entity': text_output_entity,
                            'global_t_alpha': global_t_alpha,
                            'global_tu': global_tu,
                            'global_vv_logits': global_vv_logits,
                            'background_t_alpha': background_t_alpha,
                            'background_tu': background_tu,
                            'background_vv_logits': background_vv_logits,
                            'entity_t_alpha': entity_t_alpha,
                            'entity_tu': entity_tu,
                            'entity_vv_logits': entity_vv_logits,
                        })

                    return ret

                #V-TPT Function2
                elif self.config['train_type'] =='VTPT2':
                    #print('当前任务是V-T Prompt Tune Func2')
                    instance_query_image = self.get_query_embedding(img)['cls_feature']
                    image_token_embedding = self.get_query_embedding(img)['patch_emb']
                    image_prompt =self.prompt(x_embed=image_token_embedding,
                                              prompt_mask=None,
                                              cls_features=instance_query_image)['batched_prompt']
                    image_output1 =self.image_encoder(img, image_prompt)

                    text_token_embedding = self.get_text_feature_embedding(token_id=text_ids)['token_embedding']
                    instance_query_text = self.get_text_feature_embedding(token_id=text_ids)['text_feature']
                    text_prompt = self.prompt_text(x_embed=text_token_embedding, prompt_mask=None, cls_features=instance_query_text)['batched_prompt']
                    text_attention_prompt=self.text_attention_prompt(text_token_embedding,text_prompt)
                    text_output1=self.text_encoder_attention(text_attention_prompt)

                    text_output_original = text_output1 / text_output1.norm(dim=1, keepdim=True)
                    image_output_original = image_output1 / image_output1.norm(dim=1, keepdim=True)
                    ret = {
                        'text_output_original': text_output_original,
                        'image_output_original': image_output_original
                    }
                    return ret
                #文本和图像共享提示
                elif self.config['train_type']=='VTPT3':
                    #print('当前任务是V-T Prompt Tune Func3')
                    '================原始图像处理================='
                    instance_query_image = self.get_query_embedding(img)['cls_feature']
                    image_token_embedding = self.get_query_embedding(img)['patch_emb']
                    image_prompt = self.prompt(x_embed=image_token_embedding,
                                               prompt_mask=None,
                                               cls_features=instance_query_image)['batched_prompt']
                    image_output1 = self.image_encoder(img, image_prompt)
                    '================原始文本处理=================='
                    instance_query_text = self.get_text_feature_embedding(token_id=text_ids)['text_feature']
                    text_prompt=self.imagePrompt_to_textPrompt_projection(image_prompt)
                    text_output1 = self.text_encoder(instance_query_text, text_prompt)
                    #text_output1=instance_query_text
                    if self.training:
                      '===============全局文本处理===================='
                      instance_query_globaltext=self.get_text_feature_embedding(token_id=global_text_ids)['text_feature']
                      global_text_prompt=self.imagePrompt_to_textPrompt_projection(image_prompt)
                      global_text_output1=self.text_encoder(instance_query_globaltext,global_text_prompt)

                      global_t_alpha, global_tu, global_vv_logits=self.text_uncertainty_modeling(global_text_output1)

                      '===============背景文本处理===================='
                      instance_query_backgroundtext = self.get_text_feature_embedding(token_id=background_text_ids)['text_feature']
                      background_text_prompt = self.imagePrompt_to_textPrompt_projection(image_prompt)
                      background_text_output1 = self.text_encoder(instance_query_backgroundtext, background_text_prompt)

                      background_t_alpha, background_tu, background_vv_logits=self.text_uncertainty_modeling(background_text_output1)

                      '===============实体文本处理===================='
                      instance_query_entitytext=self.get_text_feature_embedding(token_id=entity_text_ids)['text_feature']
                      entity_text_prompt=self.imagePrompt_to_textPrompt_projection(image_prompt)
                      entity_text_output1=self.text_encoder(instance_query_entitytext,entity_text_prompt)

                      entity_t_alpha,entity_tu,entity_vv_logits=self.text_uncertainty_modeling(entity_text_output1)

                      text_output_global = global_text_output1 / global_text_output1.norm(dim=1, keepdim=True)
                      text_output_background = background_text_output1 / background_text_output1.norm(dim=1,keepdim=True)
                      text_output_entity = entity_text_output1 / entity_text_output1.norm(dim=1, keepdim=True)
                    else:
                        global_text_output1 = None
                        background_text_output1 = None
                        entity_text_output1 = None
                        global_t_alpha, global_tu, global_vv_logits = None, None, None
                        background_t_alpha, background_tu, background_vv_logits = None, None, None
                        entity_t_alpha, entity_tu, entity_vv_logits = None, None, None

                    '===============原始图像-文本结果处理============'
                    text_output_original = text_output1 / text_output1.norm(dim=1, keepdim=True)
                    image_output_original = image_output1 / image_output1.norm(dim=1, keepdim=True)

                    ret = {
                        'text_output_original': text_output_original,
                        'image_output_original': image_output_original,
                    }

                    if self.training:

                        ret.update({
                            'text_output_global': text_output_global,
                            'text_output_background': text_output_background,
                            'text_output_entity': text_output_entity,
                            'global_t_alpha': global_t_alpha,
                            'global_tu': global_tu,
                            'global_vv_logits': global_vv_logits,
                            'background_t_alpha': background_t_alpha,
                            'background_tu': background_tu,
                            'background_vv_logits': background_vv_logits,
                            'entity_t_alpha': entity_t_alpha,
                            'entity_tu': entity_tu,
                            'entity_vv_logits': entity_vv_logits,
                        })

                    return ret



    def forward(self, batch):
        ret = dict()
        if "irtr" in self.current_tasks:
            ret.update(objectives_no_blip2.compute_irtr_my(self, batch))
        return ret

    def training_step(self, batch, batch_idx):
        self.eval_bool = False
        meter_utils_no_blip2.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        self.log('total_loss', total_loss)
        return total_loss

    def training_epoch_end(self, outs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outs):
        if self.current_epoch != 0:
            if self.config['exp_name'] == "finetune_irtr_iaprtc12":
                meter_utils_no_blip2.epoch_eval_irtr_nn(self)
            elif self.config['exp_name'] == 'finetune_irtr_ec':
                meter_utils_no_blip2.epoch_eval_irtr_nn(self)
            else:
                meter_utils_no_blip2.epoch_eval_irtr(self)
        else:
                  meter_utils_no_blip2.epoch_eval_irtr(self) #是否在第一轮之前就先测一下确认一下模型base line

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outs):
        if self.config['exp_name'] == "finetune_irtr_iaprtc12":
            meter_utils_no_blip2.epoch_eval_irtr_nn(self, is_test=True)
        elif self.config['exp_name'] == 'finetune_irtr_ec':
            meter_utils_no_blip2.epoch_eval_irtr_nn(self, is_test=True)
        else:
            meter_utils_no_blip2.epoch_eval_irtr(self, is_test=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.config['learning_rate'],weight_decay=0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=1e-7)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }

    def get_query_embedding(self, x: torch.Tensor):
        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        patch_emb = x
        x = self.model.visual.ln_post(x[:, 0, :])
        ret={
            'patch_emb':patch_emb,
            'cls_feature':x,
            'feature':x @ self.model.visual.proj
        }
        return ret

    def get_text_feature_embedding(self, token_id: torch.Tensor):
        x = self.model.encode_text(token_id)
        y=self.model.token_embedding(token_id)
        ret={'token_embedding':y,
             'text_feature':x
             }
        return ret

    def text_attention_prompt (self, token_embedding: torch.Tensor, prompt: torch.Tensor):
        attention_layer = self.text_prompt_attention_layer
        attn_output, _ = attention_layer(token_embedding, prompt, prompt)
        return attn_output




    def uncertainty_compute(self, sims): #sims为实例和原型的相似度
        K = sims.size(1) #原型的数量，sim矩阵的第二维度（batch*K）K
        E = torch.exp(sims / self.tau) #证据向量
        #E = sims
        #E[E < 0] = 0
        alpha = E + 1 #计算Dirichlet分布参数alpha，其中E是证据向量
        S = torch.sum(alpha, dim=1, keepdim=True) #证据向量的和
        evi = K / S #整体不确定性质量fai
        u=1-evi #不确定性评分
        return sims, u


    def text_uncertainty_modeling(self, text_feats):
        logit_scale = self.model.logit_scale.exp()
        v_prototype = self.v_prototype / self.v_prototype.norm(dim=-1,keepdim=True)
        thub_logits = logit_scale * torch.matmul(text_feats, v_prototype.t())
        sims, tu = self.uncertainty_compute(thub_logits)

        vv_logits = logit_scale * torch.matmul(v_prototype, v_prototype.t())

        return sims,tu, vv_logits

