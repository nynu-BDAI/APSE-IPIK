# 开发时间 2024/7/25 17:03
# 开发人员:牧良逢
import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import clip
from . import swin_transformer as swin
from . import objectives1, meter_utils1
from .bert import BertModel
from .prompt1 import Prompt
from .clip_encoders import CustomImageEncoder
from transformers import Blip2Processor, Blip2ForConditionalGeneration,AutoProcessor

class DoubleEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.devices ='cuda' if torch.cuda.is_available() else 'cpu'

        self.save_hyperparameters()
        #load_blip
        self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl",torch_dtype=torch.float16).to(self.devices)
        for param in self.blip_model.parameters():
            param.requires_grad = False

        self.model, self.preprocess = clip.load(name='ViT-B/32')
        for module in self.model.modules():
            module.to(torch.float32)
        self.logit_scale=self.model.logit_scale
        #self.logit_scale.requires_grad=False
        self.visual_transformer = self.model.visual
        self.image_encoder = CustomImageEncoder(self.visual_transformer).to(self.devices)
        for param in self.image_encoder.parameters():
            param.requires_grad = True

        self.datasets = ['coco', 'f30k', 'iaprtc12', 'ec', 'rsicd']

        '''self.image_prompts = nn.ModuleDict({
            dataset: Prompt(length=config['prompt_length'], embed_dim=768, embedding_key=config['embedding_key'],
                            prompt_init=config['prompt_init'], prompt_pool=config['prompt_pool'],
                            prompt_key=config['prompt_key'], pool_size=config['pool_size'], top_k=config['top_k'],
                            batchwise_prompt=config['batchwise_prompt'], prompt_key_init=config['prompt_key_init'],
                            type='specialized') for dataset in self.datasets
        })'''

        '''self.text_prompts = nn.ModuleDict({
            dataset: Prompt(length=config['prompt_length'], embed_dim=768, embedding_key=config['embedding_key'],
                            prompt_init=config['prompt_init'], prompt_pool=config['prompt_pool'],
                            prompt_key=config['prompt_key'], pool_size=config['pool_size'], top_k=config['top_k'],
                            batchwise_prompt=config['batchwise_prompt'], prompt_key_init=config['prompt_key_init'],
                            type='specialized') for dataset in self.datasets
        })'''

        self.general_image_prompt = Prompt(length=config['prompt_length'], embed_dim=768,
                                           embedding_key=config['embedding_key'],
                                           prompt_init=config['prompt_init'], prompt_pool=config['prompt_pool'],
                                           prompt_key=config['prompt_key'], pool_size=config['pool_size'],
                                           top_k=config['top_k'],
                                           batchwise_prompt=config['batchwise_prompt'],
                                           prompt_key_init=config['prompt_key_init'],
                                           type='general')

        '''self.general_text_prompt = Prompt(length=config['prompt_length'], embed_dim=768,
                                          embedding_key=config['embedding_key'],
                                          prompt_init=config['prompt_init'], prompt_pool=config['prompt_pool'],
                                          prompt_key=config['prompt_key'], pool_size=config['pool_size'],
                                          top_k=config['top_k'],
                                          batchwise_prompt=config['batchwise_prompt'],
                                          prompt_key_init=config['prompt_key_init'],
                                          type='general')'''

    def encode_text_with_grad(self, text):
        text_feature=self.model.encode_text(text)
        return text_feature
    def encode_image_with_grad(self,image):
        image_features=self.model.encode_image(image)
        return image_features
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


        #全局信息
        image_blip_global_pixel_values=batch['image_blip_global_pixel_values'].cuda()
        image_blip_global_input_ids=batch['image_blip_global_input_ids'].cuda()
        image_blip_global_attention_mask=batch['image_blip_global_attention_mask'].cuda()
        generated_ids = self.blip_model.generate(pixel_values=image_blip_global_pixel_values,
                                                 input_ids=image_blip_global_input_ids,
                                                 attention_mask=image_blip_global_attention_mask)
        generated_global_text = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)

        #背景信息
        image_blip_background_pixel_values = batch['image_blip_background_pixel_values'].cuda()
        image_blip_background_input_ids=batch['image_blip_background_input_ids'].cuda()
        image_blip_background_attention_mask=batch['image_blip_background_attention_mask'].cuda()
        generated_ids = self.blip_model.generate(pixel_values=image_blip_background_pixel_values,
                                                 input_ids=image_blip_background_input_ids,
                                                 attention_mask=image_blip_background_attention_mask)
        generated_background_text = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)

        #实体信息
        image_blip_entries_pixel_values = batch['image_blip_entries_pixel_values'].cuda()
        image_blip_entries_input_ids = batch['image_blip_entries_input_ids'].cuda()
        image_blip_entries_attention_mask = batch['image_blip_entries_attention_mask'].cuda()
        generated_ids = self.blip_model.generate(pixel_values=image_blip_entries_pixel_values,
                                                 input_ids=image_blip_entries_input_ids,
                                                 attention_mask=image_blip_entries_attention_mask)
        generated_entries_text = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)


        '''=========blip2生成全局文本==========
        generated_texts_id = []
        img_blip = batch['blip2_image']
        for i in img_blip:
            generated_ids=self.blip_model.generate(**i).view(-1).tolist()
            generated_texts_id.append(generated_ids)
        generated_texts=self.blip_processor.batch_decode(generated_texts_id,skip_special_tokens=True)'''


        '''=========blip2生成背景文本==========
        generated_background_texts_id = []
        img_blip_background = batch['blip2_image_background']
        for i in img_blip_background:
            generated_ids_background = self.blip_model.generate(**i).view(-1).tolist()
            generated_background_texts_id.append(generated_ids_background)
        generated_background_texts = self.blip_processor.batch_decode(generated_background_texts_id,
                                                                      skip_special_tokens=True)'''


        '''=========blip2生成实体文本==========
        generated_entries_texts_id = []
        img_blip_entries = batch['blip2_image_entries']
        for i in img_blip_entries:
            generated_ids_entries = self.blip_model.generate(**i).view(-1).tolist()
            generated_entries_texts_id.append(generated_ids_entries)
        generated_entries_texts = self.blip_processor.batch_decode(generated_entries_texts_id, skip_special_tokens=True)


        text_inputs = clip.tokenize(generated_texts).to(self.devices)
        text_background_inputs = clip.tokenize(generated_background_texts).to(self.devices)
        text_entries_inputs = clip.tokenize(generated_entries_texts).to(self.devices)'''

        ''''=========================构建输入=============================='
        #全局输入
        inputs={'pixel_values':img_blip}
        #背景输入
        background_inputs={'pixel_values':img_blip_background}
        #实体输入
        entries_inputs = {'pixel_values':img_blip_entries}
        '=========================生成阶段============================='
        #生成全局信息
        generated_ids = self.blip_model.generate(**inputs)
        generated_text = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
        #print(generated_text)
        #生成背景信息
        generated_ids_background=self.blip_model.generate(**background_inputs)
        generated_text_background=self.blip_processor.batch_decode(generated_ids_background,skip_special_tokens=True)
        #print(generated_text_background)
        #生成实体信息
        generated_ids_entries = self.blip_model.generate(**entries_inputs)
        generated_text_entries = self.blip_processor.batch_decode(generated_ids_entries, skip_special_tokens=True)
        #print(generated_text_entries)'''

        #CLIP_tokenize
        text_inputs_global = clip.tokenize(generated_global_text).to(self.devices)
        text_inputs_background=clip.tokenize(generated_background_text).to(self.devices)
        text_inputs_entries=clip.tokenize(generated_entries_text).to(self.devices)

        # 全局特征
        generated_text_features_global = self.encode_text_with_grad(text_inputs_global)
        # 背景特征
        generated_text_features_background = self.encode_text_with_grad(text_inputs_background)
        # 实体特征
        generated_text_features_entries = self.encode_text_with_grad(text_inputs_entries)


        #原始文本处理
        text_ids = batch["text_ids"].cuda()
        text_output1 =self.encode_text_with_grad(text_ids)

        # 构建prompt_qurry
        image_output1=self.encode_image_with_grad(img)
        image_output1_768=self.image_vit_768(img) #得到未经过映射后的特征
        image_cls_feature=image_output1_768['cls_feature']
        patch_emb=image_output1_768['patch_emb']

        #获取相关的prompt
        general_i_prompt = self.general_image_prompt(patch_emb,prompt_mask=None,cls_feature=image_cls_feature)['batched_prompt']
        #specialized_i_prompt = self.image_prompts[self.current_dataset](patch_emb,prompt_mask=None,cls_feature=image_cls_feature)['batched_prompt']

        #prefix prompt
        #combined_image_input=torch.cat([specialized_i_prompt,general_i_prompt],dim=1) #batch*len*dim
        #image_output2=self.image_encoder(img,combined_image_input)
        image_output2 = self.image_encoder(img, general_i_prompt)
        #image_output2 = self.image_encoder(img, self.vis_initial_prefix)

        #原始文本特征
        text_output2 = text_output1
        #image_output2 = self.second_encode_image(images=img,combine_image_input=combined_image_input)
        image_output2 = image_output2 / image_output2.norm(dim=1, keepdim=True)
        text_output2 = text_output2 / text_output2.norm(dim=1, keepdim=True)
        image_output1 = image_output1 / image_output1.norm(dim=1, keepdim=True)
        text_output1 = text_output1 / text_output1.norm(dim=1, keepdim=True)
        generated_text_features_global=generated_text_features_global/generated_text_features_global.norm(dim=1,keepdim=True)
        generated_text_features_background = generated_text_features_background / generated_text_features_background.norm(dim=1,keepdim=True)
        generated_text_features_entries = generated_text_features_entries / generated_text_features_entries.norm(dim=1,keepdim=True)

        ret = {
            'text_output1': text_output1,
            'image_output1': image_output1,
            'text_output_clip': text_output2,
            'image_output_clip': image_output2,
            'text_output_blip_global':generated_text_features_global,
            'image_output_blip_global':image_output2,
            'text_output_blip_background':generated_text_features_background,
            'image_output_blip_background':image_output2,
            'text_output_blip_entries': generated_text_features_entries,
            'image_output_blip_entries':image_output2
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
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.log(f'grad_norm/{name}', grad_norm)


        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr)

        self.log('total_loss', total_loss)
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
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.config['learning_rate'],weight_decay=0.2)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5,eta_min=1e-6)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,5,gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }

    ''' def second_encode_image(self, images,combine_image_input):
        x =self.model.visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.model.visual.ln_post(x[:, 0, :])

        if self.model.visual.proj is not None:
            x = x @ self.model.visual.proj

        return x'''

    def image_vit_768(self, x: torch.Tensor):
        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        patch_emb=x
        x = torch.cat([self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.visual.ln_post(x[:, 0, :])
        ret={
            'patch_emb':patch_emb,
            'cls_feature':x
        }
        return ret

    '''def generate_caption(self,image):
        # 生成描述
        output_ids = self.blip_model.generate(pixel_values=image)
        caption = self.blip_processor.decode(output_ids[0], skip_special_tokens=True)
        return caption'''