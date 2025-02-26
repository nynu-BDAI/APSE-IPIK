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
        #self.freeze_clip_parameters()
        self.eval_bool = False

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

        '''self.scale = 768 ** -0.5
        self.image_proj = nn.Parameter(self.scale*torch.randn(512, 768))
        self.image_proj_back = nn.Parameter(self.scale*torch.randn(768, 512))'''
        '''self.image_proj=Proj_MLP(512,1024,768)
        self.image_proj_back=Proj_MLP(768,1024,512)'''
    '''def freeze_clip_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad is True:
                print(f"Freezing {name}")
                param.requires_grad = False'''
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

        text_output1 = self.model.encode_text(text_ids) #text_ids:{batch*77}
        image_output1 = self.model.encode_image(img)

        # 得到通用提示池提示
        general_i_prompt = self.general_image_prompt(image_output1)['batched_prompt']
        general_t_prompt = self.general_text_prompt(text_output1)['batched_prompt']

        # 使用当前数据集的提示池处理图像和文本
        specialized_i_prompt = self.image_prompts[self.current_dataset](image_output1)['batched_prompt']
        specialized_t_prompt = self.text_prompts[self.current_dataset](text_output1)['batched_prompt']

        text_output1=torch.unsqueeze(text_output1,dim=1)
        image_output1=torch.unsqueeze(image_output1,dim=1)

        combined_text_input=torch.cat([specialized_t_prompt,general_t_prompt,text_output1],dim=1)
        combined_image_input=torch.cat([specialized_i_prompt,general_i_prompt,image_output1],dim=1)


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

    '''def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }'''
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.config['learning_rate'])
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=1e-6)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,10,gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }


    def second_encode_image(self, combine_image_input):
        #assert torch.isfinite(combine_image_input).all(), "Input tensor contains NaN or Inf"
        #assert torch.isfinite(self.image_proj).all(), "Projection matrix contains NaN or Inf"
        projected_input = combine_image_input @ self.image_proj  # 512-->768
        #projected_input = self.image_proj(combine_image_input)
        adapt_transformer_feature = projected_input.permute(1, 0, 2)  # Transformer的标准输入形式。
        x = self.model.visual.transformer(adapt_transformer_feature)
        x = x.permute(1, 0, 2)
        x = self.model.visual.ln_post(x)
        x = x @ self.image_proj_back
        #x =self.image_proj_back(x)
        #print(self.config['top_k']*self.config['prompt_length']*2)
        x = x[:,self.config['top_k']*self.config['prompt_length']*2, :]
        return x

    def second_encode_text(self, combine_text_input):
         batch, num_tokens, dim = combine_text_input.shape
         expand_combine_text_input = torch.zeros(batch, self.model.context_length, dim)
         expand_combine_text_input[:, :num_tokens, :] = combine_text_input
         adapt_transformer_feature = expand_combine_text_input.permute(1, 0, 2).cuda()
         x = self.model.transformer(adapt_transformer_feature)
         x = x.permute(1, 0, 2)
         x = self.model.ln_final(x).type(self.model.dtype)
         x = x[:, self.config['top_k'] * self.config['prompt_length'] * 2, :]
         return x


    '''def second_encode_image(self, combine_image_input):
        batch, num_tokens, dim = combine_image_input.shape
        expand_combine_image_input = torch.zeros(batch, self.model.context_length, dim)
        expand_combine_image_input[:, :num_tokens, :] = combine_image_input
        x = expand_combine_image_input
        x = x.cuda()
        #x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2).cuda()
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x).type(self.model.dtype)
        x = x[:, self.config['top_k']*self.config['prompt_length']*2, :]
        return x
    def second_encode_text(self, combine_text_input):
        batch, num_tokens, dim = combine_text_input.shape
        expand_combine_text_input = torch.zeros(batch, self.model.context_length, dim)
        expand_combine_text_input[:, :num_tokens, :] = combine_text_input
        x = expand_combine_text_input
        x = x.cuda()
        # x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x).type(self.model.dtype)
        x = x[:, self.config['top_k'] * self.config['prompt_length'] * 2, :]
        return x'''



import torch.nn.functional as F
class Proj_MLP(nn.Module):
    def __init__(self, input_size=512, hidden_size=1024, output_size=768):
        super(Proj_MLP, self).__init__()
        # 第一层MLP：输入到隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 第二层MLP：隐藏层到输出层
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 通过第一层，并应用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二层，完成维度转换
        x = self.fc2(x)
        return x