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
        #load_blip
        #self.blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl",
        #                                                                 torch_dtype=torch.float16).to(self.devices)
        # for param in self.blip_model.parameters():
        #     param.requires_grad = False
        #
        #self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.blip_model=None
        self.blip_processor=None

        # self.model, self.preprocess = clip.load(name='ViT-B/32',device=self.devices)
        self.model, self.preprocess = CLIP.load(name='ViT-B/32', device=self.devices)
        for module in self.model.modules():
            module.to(torch.float32)
        # for param in self.model.parameters():
        #     param.requires_grad = False

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
            text_ids = batch["text_ids"].cuda()

            if self.config['train_type']=='FFT':
                # print('当前任务是FFT')
                #原始文本处理
                text_output1 =self.encode_text_with_grad(text_ids)
                #原始图像处理
                image_output1=self.encode_image_with_grad(img)
                #原始文本特征
                text_output_original = text_output1/ text_output1.norm(dim=1, keepdim=True)
                #原始图像特征
                image_output_original=image_output1 / image_output1.norm(dim=1, keepdim=True)
                ret = {
                    'text_output_original': text_output_original,
                    'image_output_original': image_output_original
                }
                return ret
            #use prompt
            else:
                #VPT
                if self.config['train_type']=="VPT":
                    #print('当前任务是VPT')

                    text_output1 = self.encode_text_with_grad(text_ids)

                    instance_query_image=self.get_query_embedding(img)['cls_feature']
                    image_token_embedding=self.get_query_embedding(img)['patch_emb']
                    image_prompt=self.prompt(x_embed=image_token_embedding,prompt_mask=None,cls_features=instance_query_image)['batched_prompt']
                    image_output1=self.image_encoder(img,image_prompt)
                    text_output_original = text_output1 / text_output1.norm(dim=1, keepdim=True)
                    image_output_original = image_output1 / image_output1.norm(dim=1, keepdim=True)
                    ret = {
                        'text_output_original': text_output_original,
                        'image_output_original': image_output_original
                    }
                    return ret
                #V-TPT Function1
                elif self.config['train_type'] =='VTPT1':
                    #print('当前任务是V-T Prompt Tune Func1')
                    instance_query_image = self.get_query_embedding(img)['cls_feature']
                    image_token_embedding = self.get_query_embedding(img)['patch_emb']
                    image_prompt =self.prompt(x_embed=image_token_embedding, prompt_mask=None, cls_features=instance_query_image)['batched_prompt']
                    image_output1 = self.image_encoder(img, image_prompt)

                    instance_query_text=self.get_text_feature_embedding(token_id=text_ids)['text_feature']
                    text_token_embedding=self.get_text_feature_embedding(token_id=text_ids)['token_embedding']
                    text_prompt=self.prompt_text(x_embed=text_token_embedding,prompt_mask=None,cls_features=instance_query_text)['batched_prompt']
                    text_output1=self.text_encoder(instance_query_text,text_prompt)

                    text_output_original = text_output1 / text_output1.norm(dim=1, keepdim=True)
                    image_output_original = image_output1 / image_output1.norm(dim=1, keepdim=True)
                    ret = {
                        'text_output_original': text_output_original,
                        'image_output_original': image_output_original
                    }
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
                    '===============原始图像-文本结果处理============'
                    text_output_original = text_output1 / text_output1.norm(dim=1, keepdim=True)
                    image_output_original = image_output1 / image_output1.norm(dim=1, keepdim=True)
                    #image_output_original=text_output1 / text_output1.norm(dim=1, keepdim=True)


                    #
                    # '''===============BLIP2{全局}文本生成==============='''
                    # image_blip_global_pixel_values = batch['image_blip_global_pixel_values'].cuda()
                    # image_blip_global_input_ids = batch['image_blip_global_input_ids'].cuda()
                    # image_blip_global_attention_mask = batch['image_blip_global_attention_mask'].cuda()
                    # # generated_ids_global = self.blip_model.generate(pixel_values=image_blip_global_pixel_values,
                    # #                                                 input_ids=image_blip_global_input_ids,
                    # #                                                 attention_mask=image_blip_global_attention_mask,
                    # #                                                 max_new_tokens=51)
                    # # generated_global_text = self.blip_processor.batch_decode(generated_ids_global, skip_special_tokens=True)
                    # generated_global_text=self.generate_blip_text(image_blip_global_pixel_values,image_blip_global_input_ids,image_blip_global_attention_mask)
                    # '===============BLIP2{背景}文本生成==============='
                    # image_blip_background_pixel_values = batch['image_blip_background_pixel_values'].cuda()
                    # image_blip_background_input_ids = batch['image_blip_background_input_ids'].cuda()
                    # image_blip_background_attention_mask = batch['image_blip_background_attention_mask'].cuda()
                    # # generated_ids_background = self.blip_model.generate(pixel_values=image_blip_background_pixel_values,
                    # #                                                     input_ids=image_blip_background_input_ids,
                    # #                                                     attention_mask=image_blip_background_attention_mask,
                    # #                                                     max_new_tokens=51)
                    # # generated_background_text = self.blip_processor.batch_decode(generated_ids_background,skip_special_tokens=True)
                    # generated_background_text=self.generate_blip_text(image_blip_background_pixel_values,image_blip_background_input_ids,image_blip_background_attention_mask)
                    # '===============BLIP2{实体}文本生成==============='
                    # image_blip_entries_pixel_values = batch['image_blip_entries_pixel_values'].cuda()
                    # image_blip_entries_input_ids = batch['image_blip_entries_input_ids'].cuda()
                    # image_blip_entries_attention_mask = batch['image_blip_entries_attention_mask'].cuda()
                    # # generated_ids_entries = self.blip_model.generate(pixel_values=image_blip_entries_pixel_values,
                    # #                                                  input_ids=image_blip_entries_input_ids,
                    # #                                                  attention_mask=image_blip_entries_attention_mask,
                    # #                                                  max_new_tokens=51)
                    # # generated_entries_text = self.blip_processor.batch_decode(generated_ids_entries, skip_special_tokens=True)
                    # generated_entries_text=self.generate_blip_text(image_blip_entries_pixel_values,image_blip_entries_input_ids,image_blip_entries_attention_mask)
                    # '============BLIP2生成三个角度文本处理（从文本-->特征）==========='
                    # text_inputs_global = clip.tokenize(generated_global_text,truncate=True).to(self.devices)
                    # text_inputs_background = clip.tokenize(generated_background_text,truncate=True).to(self.devices)
                    # text_inputs_entries = clip.tokenize(generated_entries_text,truncate=True).to(self.devices)
                    #
                    # #全局特征
                    # generated_text_features_global = self.encode_text_with_grad(text_inputs_global)
                    # # 背景特征
                    # generated_text_features_background = self.encode_text_with_grad(text_inputs_background)
                    # # 实体特征
                    # generated_text_features_entries = self.encode_text_with_grad(text_inputs_entries)
                    # #
                    # generated_text_features_global = generated_text_features_global / generated_text_features_global.norm(dim=1, keepdim=True)
                    # generated_text_features_background = generated_text_features_background / generated_text_features_background.norm(dim=1, keepdim=True)
                    # generated_text_features_entries = generated_text_features_entries / generated_text_features_entries.norm(dim=1, keepdim=True)
                    ret = {
                        'text_output_original': text_output_original,
                        'image_output_original': image_output_original,
                        # 'text_output_blip_global':generated_text_features_global,
                        # 'text_output_blip_background':generated_text_features_background,
                        # 'text_output_blip_entries':generated_text_features_entries
                    }
                    return ret

    def forward(self, batch):
        ret = dict()
        if "irtr" in self.current_tasks:
            ret.update(objectives_no_blip2.compute_irtr_my(self, batch))
        return ret

    def training_step(self, batch, batch_idx):
        self.eval_bool = True
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
            meter_utils_no_blip2.epoch_eval_irtr(self)

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5,eta_min=1e-8)
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
            'cls_feature':x
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


    def generate_blip_text(self, pixel_values, input_ids, attention_mask):
        if self.blip_model is None:
            # 动态加载 BLIP2
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl",
                                                                            torch_dtype=torch.float16).to(self.devices)
            self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")

            # 确保 BLIP2 不参与训练
            for param in self.blip_model.parameters():
                param.requires_grad = False

        # 使用 BLIP2 生成文本
        generated_ids = self.blip_model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            #max_new_tokens=51
        )
        generated_text = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_text
