import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from .dist_utils_no_blip2 import all_gather
import torch.nn.functional as F
from my_HAT_PL.evaluation1 import i2t_SCAN, t2i_SCAN,i2t_SCAN_NN,t2i_SCAN_NN
import sys


def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret

def compute_itm(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret

def compute_snli(pl_module, batch):
    infer = pl_module.infer(
        batch, mask_text=False, mask_image=False, 
    )
    snli_logits = pl_module.snli_classifier(infer["cls_feats"])

    snli_labels = batch["labels"]
    snli_labels = torch.tensor(snli_labels).to(pl_module.device).long()
    snli_loss = F.cross_entropy(snli_logits, snli_labels.view(-1))

    ret = {
        "snli_loss": snli_loss,
        "snli_logits": snli_logits,
        "snli_labels": snli_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_snli_loss")(ret["snli_loss"])
        acc = getattr(pl_module, f"{phase}_snli_accuracy")(
            ret["snli_logits"], ret["snli_labels"]
        )
        pl_module.log(f"snli/{phase}/loss", loss)
        pl_module.log(f"snli/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_snli_loss")(
                F.cross_entropy(
                    ret["snli_logits"][dev_batches], ret["snli_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_snli_accuracy")(
                ret["snli_logits"][dev_batches], ret["snli_labels"][dev_batches]
            )
            pl_module.log(f"snli/dev/loss", dev_loss)
            pl_module.log(f"snli/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_snli_loss")(
                F.cross_entropy(
                    ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_snli_accuracy")(
                ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]
            )
            pl_module.log(f"snli/test/loss", test_loss)
            pl_module.log(f"snli/test/accuracy", test_acc)

    return ret

def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels.view(-1))

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret


def compute_irtr(pl_module, batch):
    is_training_phase = pl_module.training

    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret

def compute_contrastiveLoss(im, s, margin):
    # compute image-sentence score matrix
        scores = im.mm(s.t())
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def focal_equal(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as equal
    sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
    attn: (batch, queryL, sourceL)
    """
    funcF = attn * sourceL - torch.sum(attn, dim=-1, keepdim=True)
    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn

def func_attention(query, context, smooth, eps=1e-8):
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
    attn = l2norm(attn, 2)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax(1)(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    #print(attnT.shape)

    #pic = attnT[0][0].view(28, 28)
    #print(pic)
    #plt.matshow(pic.data.cpu().numpy(), cmap=plt.cm.Blues)
    #plt.savefig('3.jpg')
    #assert 1==0

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


'''#BFAN
def func_attention(query, context, smooth, eps=1e-8):
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
    attn = l2norm(attn, 2)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax(1)(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    #BFAN

    funcH = focal_equal(attn, batch_size, queryL, sourceL)

    tmp_attn = funcH * attn
    attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
    re_attn = tmp_attn / attn_sum



    # --> (batch, sourceL, queryL)
    re_attnT = torch.transpose(re_attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, re_attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, re_attnT'''


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def xattn_score_i2t(images, captions):
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)

    # 计算余弦相似度
    for i in range(n_caption):
        cap_i = captions[i].unsqueeze(0)  # (1, d)
        cap_i_expand = cap_i.repeat(n_image, 1)  # (n_image, d)

        # 计算图像和扩展字幕之间的余弦相似度
        sim = F.cosine_similarity(images, cap_i_expand, dim=1)  # (n_image,)
        similarities.append(sim.unsqueeze(1))  # (n_image, 1)

    similarities = torch.cat(similarities, 1) # (n_image, n_caption)

    return similarities


def xattn_score_t2i(images, captions):
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)

    # 计算余弦相似度
    for i in range(n_caption):
        cap_i = captions[i].unsqueeze(0)  # (1, d)
        cap_i_expand = cap_i.repeat(n_image, 1)  # (n_image, d)

        # 计算图像和扩展字幕之间的余弦相似度
        sim = F.cosine_similarity(images, cap_i_expand, dim=1)  # (n_image,)
        similarities.append(sim.unsqueeze(1))  # (n_image, 1)

    similarities = torch.cat(similarities, 1)  # (n_image, n_caption)
    return similarities.T

def uni_score(images,captions):
    similarities = torch.mm(images, captions.T)
    image_norms = images.norm(dim=1, keepdim=True)  # (n_image, 1)
    caption_norms = captions.norm(dim=1, keepdim=True)  # (n_caption, 1)
    norm_matrix = torch.mm(image_norms, caption_norms.T)  # (n_image, n_caption)
    # 计算余弦相似度
    similarities = similarities / (norm_matrix + 1e-8)  # 避免除以0
    return similarities
def compute_SCAN(im, s, margin, direction,weight=None):
        # compute image-sentence score matrix
        if direction == 't2i':
            scores = xattn_score_t2i(im, s)

        elif direction == 'i2t':
            scores = xattn_score_i2t(im, s)

        elif direction=='unify':
            scores=uni_score(im,s)

        else:
            raise ValueError("unknown first norm type")
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

        if weight is not None:
            cost_s = cost_s*weight
            cost_im = cost_im*weight

        return cost_s.sum() + cost_im.sum()


def compute_irtr_my(pl_module, batch):
    # dont use Deepseek
    #images,original_caption,original_input_ids,ids=batch
    #use Deepseek
    images,original_caption,original_input_ids,global_caption,global_input_ids,background_caption,background_input_ids,entity_caption,entity_input_ids,ids=batch

    infer = pl_module.infer1(
            batch={'image':images,
                   'original_text_ids':original_input_ids,

                   #use Deepseek
                   'global_text_ids':global_input_ids,
                   'background_text_ids':background_input_ids,
                   'entity_text_ids':entity_input_ids
                  }
    )
    #print("Infer method called successfully")

    image_feats = infer['image_output_original']  # (B 49 H) (when infer use befor pool)  after pool:  (B H)
    text_feats = infer['text_output_original']  # (B 32 H)#                                           (B H)

    #add use deepseek
    text_global_feats=infer['text_output_global']
    text_background_feats=infer['text_output_background']
    text_entity_feats=infer['text_output_entity']

    #uncertainty
    global_vv_logits=infer['global_vv_logits']
    background_vv_logits=infer['background_vv_logits']
    entity_vv_logits=infer['entity_vv_logits']

    global_tu=infer['global_tu']
    background_tu=infer['background_tu']
    entity_tu=infer['entity_tu']

    global_weight=f(global_tu)
    background_weight=f(background_tu)
    entity_weight=f(entity_tu)

    global_t_alpha=infer['global_t_alpha']
    background_t_alpha=infer['background_t_alpha']
    entity_t_alpha=infer['entity_t_alpha']

    #CLIP loss
    logit_scale = pl_module.logit_scale.exp()
    logits_per_image = logit_scale * image_feats @ text_feats.t()
    logits_per_text = logits_per_image.t()
    batch_size = image_feats.size(0)
    ground_truth = torch.arange(batch_size, device=image_feats.device)
    loss_i2t = F.cross_entropy(logits_per_image, ground_truth)
    loss_t2i = F.cross_entropy(logits_per_text, ground_truth)
    loss_clip = (loss_t2i + loss_i2t) / 2

    #Triple Loss
    loss_triple=compute_SCAN(image_feats,text_feats,margin=pl_module.hparams.config["margin"], direction=pl_module.hparams.config["direction"],weight=None)
    #add use deepseek
    loss_triple1=compute_SCAN(image_feats,text_global_feats,margin=pl_module.hparams.config["margin"], direction=pl_module.hparams.config["direction"],weight=global_weight)
    loss_triple2=compute_SCAN(image_feats,text_background_feats,margin=pl_module.hparams.config["margin"], direction=pl_module.hparams.config["direction"],weight=background_weight)
    loss_triple3=compute_SCAN(image_feats,text_entity_feats,margin=pl_module.hparams.config["margin"], direction=pl_module.hparams.config["direction"],weight=entity_weight)
    # print(f'loss_clip={loss_clip}')
    # print(f'loss_triple={loss_triple}')
    # print(f'loss_triple1={loss_triple1}')
    # print(f'loss_triple2={loss_triple2}')

    #Var Loss
    loss_var = VarianceLoss()
    global_loss_var=loss_var(global_vv_logits)
    background_loss_var=loss_var(background_vv_logits)
    entity_loss_var=loss_var(entity_vv_logits)
    loss_Var=global_loss_var+background_loss_var+entity_loss_var

    #Uncertainty Loss
    loss_uct=UncertaintyAwareLoss()

    global_loss_uct=loss_uct(logits_per_text,global_t_alpha)
    background_loss_uct=loss_uct(logits_per_text,background_t_alpha)
    entity_loss_uct=loss_uct(logits_per_text,entity_t_alpha)

    loss_Uct=global_loss_uct+background_loss_uct+entity_loss_uct


    # print(f'loss_clip={loss_clip}')
    # print(f'loss_triple={loss_triple}')
    # print(f'loss_triple1={loss_triple1}')
    # print(f'loss_triple2={loss_triple2}')
    # print(f'loss_triple3={loss_triple3}')
    # print(f'loss_Var={loss_Var}')
    # print(f'loss_Uct{loss_Uct}')

    ret = {
        #"irtr_loss": loss_clip+loss_triple+0.01*loss_triple1+0.01*loss_triple2+0.01*loss_triple3+0.1*loss_Var+100*loss_Uct
        "irtr_loss": loss_clip + loss_triple
    }

    return ret

def shard_xattn_i2t(images, captions, shard_size=16):

    n_im_shard = (len(images) - 1) // shard_size + 1 #图像分类多少和小阶，每个小阶128个图像特征
    n_cap_shard = (len(captions) - 1) // shard_size + 1 #文本分了多少个小阶，每个小阶128个文本特征

    d = np.zeros((len(images), len(captions))) #初始化一个矩阵用于存放最终的结果：batch*batch
    for i in range(n_im_shard): #遍历每一个图像阶段，每次shard_size个image_feature
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i, j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im = torch.from_numpy(images[im_start:im_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            sim = xattn_score_i2t(im, s)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d #返回相似度矩阵

def shard_xattn_t2i(images, captions, shard_size=128):

    n_im_shard = (len(images) - 1) // shard_size + 1  # 图像分类多少和小阶，每个小阶128个图像特征
    n_cap_shard = (len(captions) - 1) // shard_size + 1  # 文本分了多少个小阶，每个小阶128个文本特征

    d = np.zeros((len(images), len(captions)))  # 初始化一个矩阵用于存放最终的结果：1000*5000
    for i in range(n_im_shard):  # 遍历每一个图像阶段，每次128个image_feature
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i, j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im = torch.from_numpy(images[im_start:im_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            sim = xattn_score_i2t(im, s).T
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d  # 返回相似度矩阵

def shard_xattn_uni(images, captions, shard_size=128):

    n_im_shard = (len(images) - 1) // shard_size + 1  # 图像分类多少和小阶，每个小阶128个图像特征
    n_cap_shard = (len(captions) - 1) // shard_size + 1  # 文本分了多少个小阶，每个小阶128个文本特征

    d = np.zeros((len(images), len(captions)))  # 初始化一个矩阵用于存放最终的结果：1000*5000
    for i in range(n_im_shard):  # 遍历每一个图像阶段，每次128个image_feature
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_uni batch (%d,%d)' % (i, j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im = torch.from_numpy(images[im_start:im_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            sim = uni_score(im, s)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d  # 返回相似度矩阵

@torch.no_grad()
def compute_irtr_val(pl_module):
    val_dataloader = pl_module.trainer.datamodule.val_dataloader()
    img_embs = None
    cap_embs = None

    #for images,caption,image_blip_global_pixel_values,image_blip_global_input_ids,image_blip_global_attention_mask,image_blip_background_pixel_values,image_blip_background_input_ids,image_blip_background_attention_mask,image_blip_entries_pixel_values,image_blip_entries_input_ids,image_blip_entries_attention_mask,input_ids,ids in val_dataloader:
    for images,caption,input_ids,ids in val_dataloader:

        # make sure val logger is used
        # compute the embeddings
        infer = pl_module.infer1(batch={
                                           'image':images,
                                           'caption':caption,
                                           'original_text_ids':input_ids,

                                        # 'image_blip_global_pixel_values': image_blip_global_pixel_values,
                                        # 'image_blip_global_input_ids': image_blip_global_input_ids,
                                        # 'image_blip_global_attention_mask': image_blip_global_attention_mask,
                                        #
                                        # 'image_blip_background_pixel_values': image_blip_background_pixel_values,
                                        # 'image_blip_background_input_ids': image_blip_background_input_ids,
                                        # 'image_blip_background_attention_mask': image_blip_background_attention_mask,
                                        #
                                        # 'image_blip_entries_pixel_values': image_blip_entries_pixel_values,
                                        # 'image_blip_entries_input_ids': image_blip_entries_input_ids,
                                        # 'image_blip_entries_attention_mask': image_blip_entries_attention_mask,
                                                                  })

        img_emb = infer['image_output_original']
        cap_emb = infer['text_output_original']

        if img_embs is None:
            img_embs = np.zeros((len(val_dataloader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(val_dataloader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    if pl_module.hparams.config["direction"] == 'i2t':
        sims = shard_xattn_i2t(img_embs, cap_embs, shard_size=128)
    elif pl_module.hparams.config["direction"]=='t2i':
        sims = shard_xattn_t2i(img_embs, cap_embs, shard_size=128)
    else:
        sims = shard_xattn_uni(img_embs, cap_embs, shard_size=128)
    sims_all = sims


    (r1, r5, r10, r20, r50, r70, r100, medr, meanr) = i2t_SCAN(sims_all)
    (r1i, r5i, r10i, r20i, r50i, r70i, r100i, medri, meanri) = t2i_SCAN(sims_all)


    pl_module.log('best_irtr', (r1+r1i))

    return (r1, r5, r10, r20, r50, r70, r100, r1i, r5i, r10i, r20i, r50i, r70i, r100i)

'==============================iaprtc12val/test========================'
@torch.no_grad()
def compute_irtr_val_nn(pl_module):
    val_dataloader = pl_module.trainer.datamodule.val_dataloader()
    img_embs = None
    cap_embs = None
    for (images, input_ids, ids) in val_dataloader:
        # make sure val logger is used
        # compute the embeddings
        infer = pl_module.infer1(batch={'image':images,
                                        'text_ids':input_ids,
                                                             })

        img_emb = infer['image_output2']
        cap_emb = infer['text_output2']

        if img_embs is None:
            img_embs = np.zeros((len(val_dataloader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(val_dataloader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

    #img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    if pl_module.hparams.config["direction"] == 'i2t':
        sims = shard_xattn_i2t(img_embs, cap_embs, shard_size=128)
    elif pl_module.hparams.config["direction"] == 'unify':
        sims = shard_xattn_i2t(img_embs, cap_embs, shard_size=128)
    else:
        sims = shard_xattn_uni(img_embs, cap_embs, shard_size=128)
    sims_all = sims


    (r1, r5, r10, r20, r50, r70, r100, medr, meanr) = i2t_SCAN_NN(sims_all)
    (r1i, r5i, r10i, r20i, r50i, r70i, r100i, medri, meanri) = t2i_SCAN_NN(sims_all)
    pl_module.log('best_irtr', (r1+r1i))
    return (r1, r5, r10, r20, r50, r70, r100, r1i, r5i, r10i, r20i, r50i, r70i, r100i)

'=======================iaprtc12test=========================='
@torch.no_grad()
def compute_irtr_test_nn(pl_module):
    val_dataloader = pl_module.trainer.datamodule.test_dataloader()
    img_embs = None
    cap_embs = None
    for (images, input_ids, ids) in val_dataloader:
        # make sure val logger is used
        # compute the embeddings
        infer = pl_module.infer1(batch={'image':images,
                                        'text_ids':input_ids,
                                                             })
        #取output1:纯正的clip输出用于zero-shot
        img_emb = infer['image_output2']
        cap_emb = infer['text_output2']

        if img_embs is None:
            img_embs = np.zeros((len(val_dataloader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(val_dataloader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

    #img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    if pl_module.hparams.config["direction"] == 'i2t':
        sims = shard_xattn_i2t(img_embs, cap_embs, shard_size=128)
    elif pl_module.hparams.config["direction"] == 'unify':
        sims = shard_xattn_i2t(img_embs, cap_embs, shard_size=128)
    else:
        sims = shard_xattn_uni(img_embs, cap_embs, shard_size=128)
    sims_all = sims


    (r1, r5, r10, r20, r50, r70, r100, medr, meanr) = i2t_SCAN_NN(sims_all)
    (r1i, r5i, r10i, r20i, r50i, r70i, r100i, medri, meanri) = t2i_SCAN_NN(sims_all)
    pl_module.log('best_irtr', (r1+r1i))
    return (r1, r5, r10, r20, r50, r70, r100, r1i, r5i, r10i, r20i, r50i, r70i, r100i)

@torch.no_grad()
def compute_irtr_test(pl_module):
    val_dataloader = pl_module.trainer.datamodule.test_dataloader()
    img_embs = None
    cap_embs = None
    # for images,caption,image_blip_global_pixel_values,image_blip_global_input_ids,image_blip_global_attention_mask,image_blip_background_pixel_values,image_blip_background_input_ids,image_blip_background_attention_mask,image_blip_entries_pixel_values,image_blip_entries_input_ids,image_blip_entries_attention_mask,input_ids,ids  in val_dataloader:
    for images,caption,input_ids,ids  in val_dataloader:

        # make sure val logger is used
        # compute the embeddings
        infer = pl_module.infer1(batch={
                        'image': images,
                        'caption': caption,
                        'original_text_ids':input_ids,
                        #
                        # 'image_blip_global_pixel_values': image_blip_global_pixel_values,
                        # 'image_blip_global_input_ids': image_blip_global_input_ids,
                        # 'image_blip_global_attention_mask': image_blip_global_attention_mask,
                        #
                        # 'image_blip_background_pixel_values': image_blip_background_pixel_values,
                        # 'image_blip_background_input_ids': image_blip_background_input_ids,
                        # 'image_blip_background_attention_mask': image_blip_background_attention_mask,
                        #
                        # 'image_blip_entries_pixel_values': image_blip_entries_pixel_values,
                        # 'image_blip_entries_input_ids': image_blip_entries_input_ids,
                        # 'image_blip_entries_attention_mask': image_blip_entries_attention_mask,

                                                             })
        img_emb = infer['image_output_original']
        cap_emb = infer['text_output_original']
        if img_embs is None:
            img_embs = np.zeros((len(val_dataloader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(val_dataloader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)]) #一个图像对应5个文本，则使用这一行
    if pl_module.hparams.config["direction"] == 'i2t':
        sims = shard_xattn_i2t(img_embs, cap_embs, shard_size=128)
    elif pl_module.hparams.config["direction"] == 't2i':
        sims = shard_xattn_t2i(img_embs, cap_embs, shard_size=128)
    else:
        sims = shard_xattn_uni(img_embs, cap_embs, shard_size=128)
    sims_all = sims

    (r1, r5, r10, r20, r50, r70, r100, medr, meanr) = i2t_SCAN(sims_all)
    (r1i, r5i, r10i, r20i, r50i, r70i, r100i, medri, meanri) = t2i_SCAN(sims_all)

    pl_module.log('best_irtr', (r1 + r1i))

    return (r1, r5, r10, r20, r50, r70, r100, r1i, r5i, r10i, r20i, r50i, r70i, r100i)

@torch.no_grad()
def compute_irtr_test_zero_shot(pl_module):
    val_dataloader = pl_module.trainer.datamodule.test_dataloader()
    img_embs = None
    cap_embs = None
    for (images, input_ids, ids) in val_dataloader:
        # make sure val logger is used
        # compute the embeddings
        infer = pl_module.infer1(batch={'image':images,
                                        'text_ids':input_ids,
                                                             })

        img_emb = infer['image_output2']
        cap_emb = infer['text_output2']

        if img_embs is None:
            img_embs = np.zeros((len(val_dataloader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(val_dataloader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

    #img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    if pl_module.hparams.config["direction"] == 'i2t':
        sims = shard_xattn_i2t(img_embs, cap_embs, shard_size=128)
    elif pl_module.hparams.config["direction"] == 'unify':
        sims = shard_xattn_i2t(img_embs, cap_embs, shard_size=128)
    else:
        sims = shard_xattn_uni(img_embs, cap_embs, shard_size=128)
    sims_all = sims


    (r1, r5, r10, r20, r50, r70, r100, medr, meanr) = i2t_SCAN(sims_all)
    (r1i, r5i, r10i, r20i, r50i, r70i, r100i, medri, meanri) = t2i_SCAN(sims_all)


    pl_module.log('best_irtr', (r1+r1i))

    return (r1, r5, r10, r20, r50, r70, r100, r1i, r5i, r10i, r20i, r50i, r70i, r100i)



@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    #TODO: speed up the process by caching text/image features
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append((_b['image'][0], _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _im, _iid = img_batch

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            im = _im.repeat(fblen, 1, 1, 1).to(device=txt_batch['text_ids'].device)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        img=im,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    try:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
            if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
        )
    except:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["gqa_test"].id2answer
            if "gqa_test" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["gqa"].id2answer
        )
        vqa_logits = output["vqa_logits"]
        vqa_preds = vqa_logits.argmax(dim=-1)
        vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
        questions = batch["text"]
        qids = batch["qid"]
        return {"qids": qids, "preds": vqa_preds, "gqa": True}
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds, "gqa": False}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    gqa = False
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]
        gqa = out['gqa']

    rets = list()
    for qid, pred in zip(qids, preds):
        if gqa:
            rets.append({"questionId": qid, "prediction": pred})
        else:
            rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")

class VarianceLoss(nn.Module):
    """
    Compute Variance Loss
    """
    def __init__(self):
        super(VarianceLoss, self).__init__()
        self.mse = nn.MSELoss(reduce=True, size_average=True)

    # sims (K,K)
    def forward(self, vv): #vv图像原型之间的相似度矩阵
        K = vv.size(0)
        label = torch.zeros(vv.shape).cuda()
        mask = 1 - torch.eye(K).cuda()
        vv_m = mask * vv
        loss = self.mse(vv_m, label)  #意味着使非对角线元素（不同原型之间的相似度）接近 0

        return loss

class UncertaintyAwareLoss(nn.Module):
    """
    Compute UncertaintyAwareLoss
    """
    def __init__(self, tau=5):
        super(UncertaintyAwareLoss, self).__init__()
        self.tau = tau
        self.mse = nn.MSELoss(reduce=True, size_average=True)
        self.relu = nn.ReLU(inplace=True)

    # sims (K,K)
    def forward(self, sims, sim_K, lambda_=0.00005):
        BS = sims.size(0)
        K = sim_K.size(1)
        mask = 1 - torch.eye(BS).cuda() #对角线为0，其余为1
        soft_label = (sims * mask).mean(1, keepdim=True) #列上面的相似度均值
        E = torch.exp(sim_K / self.tau)
        alpha = E + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        U = K / S  #整体不确定性
        scale =  (1 - U).mean() / soft_label.mean() #计算一个缩放因子 scale，将 soft_label 的量级与确定性 1 - U 对齐
        # ce loss
        loss = self.mse(1 - U, scale * soft_label)
        return loss

def f(u,k=1.0):
    weight=torch.exp(-k*u)
    return weight