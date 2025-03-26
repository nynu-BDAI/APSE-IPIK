import json
from faulthandler import is_enabled

import clip
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json as jsonmod
import matplotlib.pyplot as plt

from transformers import Blip2Processor,Blip2ForConditionalGeneration,AutoProcessor
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler
import six

from Get_text import ImageDescriptionGenerator

import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def convert_to_feature(raw, tokenizer):
    line = convert_to_unicode(raw)
    tokens_id = tokenizer(line, truncate=True)
    #tokens_id = tokenizer(line,)# truncate=True 自动截断
    return tokens_id

def get_transform( split_name, crop_size):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(crop_size)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(crop_size)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform

def get_paths(path, name='coco', use_restval=True):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:

    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...     A+=D['images'][i]['sentids'][:5]
    ...

    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    if 'coco' == name:
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json')
        }

        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }

        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }

        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }

        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        #ids['val'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        #ids['test'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))[:1000]
        ids['trainrestval'] = (
            ids['train'],
            np.load(os.path.join(capdir, 'coco_restval_ids.npy')))

        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']

    elif 'f8k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr8k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif 'f30k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif 'iaprtc12' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'annotations_complete_eng')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif 'rsicd' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_rsicd.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif 'ec' == name:
        roots['train'] = {'img': os.path.join(path, 'T2I_train.img.tsv'),
                          'cap': os.path.join(path, 'Translated_T2I_train.text.tsv')}
        roots['val'] = {'img': os.path.join(path, 'T2I_val.img.tsv'),
                        'cap': os.path.join(path, 'Translated_T2I_val.text.tsv')}
        roots['test'] = {'img': os.path.join(path, 'T2I_val.img.tsv'),
                         'cap': os.path.join(path, 'Translated_T2I_val.text.tsv')}
        ids = {'train': None, 'val': None, 'test': None}
    return roots, ids


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, max_len,useDeepseek,ids=None,split=None,):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        self.split=split
        self.cache_dir = os.path.join(os.path.dirname("/mnt/Data/wangshilong/self_datasets/coco/images/train2014"), "caption_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.root=root  #tuple类型包含('/mnt/Data/wangshilong/self_datasets/coco/images/train2014', '/mnt/Data/wangshilong/self_datasets/coco/images/val2014')
        self.json=json  #同上
        self.transform=get_transform(self.split,224)
        self.max_len = max_len #77
        self.tokenizer=clip.tokenize
        self.devices='cuda' if torch.cuda.is_available() else 'cpu'

        # when using `restval`, two json files are needed
        if isinstance(self.json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(self.json),)
            self.root = (root,)

        # if ids provided by get_paths, use split-specific ids
        if ids is None:
            self.ids = list(self.coco.anns.keys())
        else:
            self.ids = ids  # [40987,36541,.....]是一个列表存样本id

        # if `restval` data is to be used, record the break point for ids
        if isinstance(self.ids, tuple):
            self.bp = len(self.ids[0]) #断点位置break point，用于区分train里的ids和val中的ids
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)

        #VLM配置
        self.API_KEY = "sk-ahlkqzrgdmdlimaqdfvbeweykcfrlchbglpqodpnadjsgeec"
        self.BASE_URL="https://api.siliconflow.cn/v1"
        self.TextGenerator=ImageDescriptionGenerator(api_key=self.API_KEY,base_url=self.BASE_URL)

        # 添加速率限制参数
        self.requests_per_minute = 500
        self.last_request_time = 0
        self.min_request_interval = 60.0 / self.requests_per_minute

        self.useDeepseek=useDeepseek

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """

        root, original_caption, img_id, path, image = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)

        if self.split=="train" and self.useDeepseek is True:
            # 根据图像ID创建缓存文件名
            cache_filename = os.path.join(self.cache_dir, f"{img_id}_captions.json")

            # 首先尝试从缓存加载
            captions = self.load_from_cache(cache_filename)
            if captions is None:
                global_text_query="What does this image describe from a global perspective?Answer is limited to 30 words"
                background_text_query="What information is depicted in the background of this image?Answer is limited to 30 words"
                entity_text_query="What entities are present in this image? Answer is limited to 30 words"

                global_caption=self.getImageDecription(path,global_text_query)
                background_caption=self.getImageDecription(path,background_text_query)
                entity_caption=self.getImageDecription(path,entity_text_query)

                captions = {
                    "global": global_caption,
                    "background": background_caption,
                    "entity": entity_caption
                }

                self.save_to_cache(cache_filename, captions)
            # original
            original_input_ids = convert_to_feature(original_caption, self.tokenizer)
            original_input_ids = torch.as_tensor(original_input_ids, dtype=torch.long)

            # global
            global_caption = captions["global"]
            global_input_ids = convert_to_feature(global_caption, self.tokenizer)
            global_input_ids = torch.as_tensor(global_input_ids, dtype=torch.long)

            # background
            background_caption = captions["background"]
            background_input_ids = convert_to_feature(background_caption, self.tokenizer)
            background_input_ids = torch.as_tensor(background_input_ids, dtype=torch.long)

            # entity
            entity_caption = captions["entity"]
            entity_input_ids = convert_to_feature(entity_caption, self.tokenizer)
            entity_input_ids = torch.as_tensor(entity_input_ids, dtype=torch.long)

            return image, original_caption, original_input_ids,global_caption,global_input_ids, background_caption, background_input_ids, entity_caption, entity_input_ids, index

        else:
            original_input_ids = convert_to_feature(original_caption, self.tokenizer)
            original_input_ids = torch.as_tensor(original_input_ids, dtype=torch.long)

            return image, original_caption, original_input_ids, index


    def get_raw_item(self, index):
        if index < self.bp:  # index<= 5000
            coco = self.coco[0]
            root = self.root[0]
        else:
            coco = self.coco[1]
            root = self.root[1]
        ann_id = self.ids[index]  # self.ids是图像文本对样本id的列表,ann_id相当于是样本id
        caption = coco.anns[ann_id]['caption']  # 取得对应id的文本
        img_id = coco.anns[ann_id]['image_id']  # 取得对应id的imag_id
        path = coco.loadImgs(img_id)[0]['file_name']
        path=os.path.join(root, path)
        image = Image.open(path).convert('RGB')

        # plt.imshow(image)
        # plt.axis('off')  # 隐藏坐标轴
        # plt.show()
        return root, caption, img_id, path, image #root:'/mnt/Data/wangshilong/self_datasets/coco/images/val2014'

    def __len__(self):
        return len(self.ids)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def getImageDecription(self,image_path,text_query):

        # 实现速率限制
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)

        # 添加随机抖动以避免同步请求
        time.sleep(random.uniform(0.1, 0.5))

        image_to_base64=self.TextGenerator.image_to_base64(image_path)
        get_Decription=self.TextGenerator.get_image_description(image_to_base64,text_query)

        # 更新最后请求时间
        self.last_request_time = time.time()

        return get_Decription

    def load_from_cache(self, cache_file):
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None

    def save_to_cache(self, cache_file, captions):
        try:
            with open(cache_file, 'w') as f:
                json.dump(captions, f)
        except Exception as e:
            print(f"保存缓存失败: {e}")
class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split, vocab, max_len, use_deepseek):
        self.cache_dir = os.path.join(os.path.dirname("/mnt/Data/wangshilong/self_datasets/f30k/images"),"f30k_caption_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.useDeepseek=use_deepseek
        self.max_len = max_len
        self.transform=get_transform(split,224)
        self.tokenizer = clip.tokenize

        self.root = root
        self.vocab = vocab
        self.split = split
        self.dataset = jsonmod.load(open(json, 'r'))['images'] #["sentids":[],"imageid":0,"sentences":[5个句子信息],"split":"train","filename":"xxxx.jpg"]
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))] #[0,1]第一个图像和第一个文本描述

        # VLM配置
        self.API_KEY = "sk-ahlkqzrgdmdlimaqdfvbeweykcfrlchbglpqodpnadjsgeec"
        self.BASE_URL = "https://api.siliconflow.cn/v1"
        self.TextGenerator = ImageDescriptionGenerator(api_key=self.API_KEY, base_url=self.BASE_URL)

        # 添加速率限制参数
        self.requests_per_minute = 500
        self.last_request_time = 0
        self.min_request_interval = 60.0 / self.requests_per_minute

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn"""
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index] #[x,y] x对应图像的id，y对应该图像的第几个文本描述
        img_id = ann_id[0]
        original_caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']
        image_filename=os.path.splitext(path)[0]
        image_path=os.path.join(self.root,path)
        image = Image.open(os.path.join(root, path)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.split=="train" and self.useDeepseek is True:

            cache_filename = os.path.join(self.cache_dir, f"{image_filename}_captions.json")

            # 首先尝试从缓存加载
            captions = self.load_from_cache(cache_filename)
            # plt.imshow(image)
            # plt.axis('off')
            # plt.show()
            if captions is None:
                global_text_query = "What does this image describe from a global perspective?Answer is limited to 30 words"
                background_text_query = "What information is depicted in the background of this image?Answer is limited to 30 words"
                entry_text_query = "What entities are present in this image? Answer is limited to 30 words"

                global_caption = self.getImageDecription(image_path, global_text_query)
                background_caption = self.getImageDecription(image_path, background_text_query)
                entity_caption = self.getImageDecription(image_path, entry_text_query)

                captions = {
                    "global": global_caption,
                    "background": background_caption,
                    "entry": entity_caption
                }
                self.save_to_cache(cache_filename, captions)

            # original
            original_input_ids = convert_to_feature(original_caption, self.tokenizer)
            original_input_ids = torch.as_tensor(original_input_ids, dtype=torch.long)

            # global
            global_caption=captions["global"]
            global_input_ids = convert_to_feature(global_caption, self.tokenizer)
            global_input_ids = torch.as_tensor(global_input_ids, dtype=torch.long)

            # background
            background_caption=captions["background"]
            background_input_ids = convert_to_feature(background_caption, self.tokenizer)
            background_input_ids = torch.as_tensor(background_input_ids, dtype=torch.long)

            # entity
            entity_caption=captions["entity"]
            entity_input_ids = convert_to_feature(entity_caption, self.tokenizer)
            entity_input_ids = torch.as_tensor(entity_input_ids, dtype=torch.long)

            return image, original_caption, original_input_ids,global_caption,global_input_ids, background_caption, background_input_ids, entity_caption, entity_input_ids, index

        else: #test,val

            original_input_ids = convert_to_feature(original_caption, self.tokenizer)
            original_input_ids = torch.as_tensor(original_input_ids, dtype=torch.long)

            return image,original_caption,original_input_ids,index



    def __len__(self):
        return len(self.ids)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def getImageDecription(self, image_path, text_query):

        # 实现速率限制
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)

        # 添加随机抖动以避免同步请求
        time.sleep(random.uniform(0.1, 0.5))

        image_to_base64 = self.TextGenerator.image_to_base64(image_path)
        get_Decription = self.TextGenerator.get_image_description(image_to_base64, text_query)

        # 更新最后请求时间
        self.last_request_time = time.time()

        return get_Decription

    def load_from_cache(self, cache_file):
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None

    def save_to_cache(self, cache_file, captions):
        try:
            with open(cache_file, 'w') as f:
                json.dump(captions, f)
        except Exception as e:
            print(f"保存缓存失败: {e}")

#No_useDeepseek-vl2 用于测试和验证
def collate_fn_bert(data):
    image,original_caption,original_input_ids,index=zip(*data)
    original_input_ids = torch.stack(original_input_ids, 0)
    original_input_ids = torch.squeeze(original_input_ids, dim=1)
    image = torch.stack(image, dim=0)
    ids = np.array(index)
    return image,original_caption,original_input_ids,ids

def collate_fn_bert_useDeepseek(data):
    image, original_caption, original_input_ids,global_caption,global_input_ids, background_caption, background_input_ids, entity_caption, entity_input_ids, index=zip(*data)

    original_input_ids = torch.stack(original_input_ids, 0)
    original_input_ids = torch.squeeze(original_input_ids, dim=1)

    global_input_ids=torch.stack(global_input_ids,0)
    global_input_ids=torch.squeeze(global_input_ids,dim=1)

    background_input_ids=torch.stack(background_input_ids,0)
    background_input_ids=torch.squeeze(background_input_ids,dim=1)

    entity_input_ids=torch.stack(entity_input_ids,0)
    entity_input_ids=torch.squeeze(entity_input_ids,dim=1)

    image = torch.stack(image, dim=0)
    ids = np.array(index)

    return image,original_caption,original_input_ids,global_caption,global_input_ids,background_caption,background_input_ids,entity_caption,entity_input_ids,ids

class F30kDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        super().__init__()

        # _config['data_root'], _config['datasets'], vocab, _config['max_text_len'], _config['tokenizer'], _config['image_size'], _config['per_gpu_batchsize'], _config['num_workers'])

        self.data_path = os.path.join(_config["data_root"], _config['datasets'])
        self.datasets = _config['datasets']
        self.vocab = None

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]

        self.setup_flag = False
        self.dist = dist

        self.roots, self.ids = get_paths(self.data_path, self.datasets)
        #change!
        self.useDeepSeek=True
    def set_train_dataset(self):
        self.train_dataset = FlickrDataset(root=self.roots['train']['img'],
                                           split='train',
                                           json=self.roots['train']['cap'],
                                           vocab=self.vocab,
                                           max_len=self.max_text_len,
                                           use_deepseek=self.useDeepSeek)

    def set_val_dataset(self):
        self.val_dataset = FlickrDataset(root=self.roots['val']['img'],
                                         split='val',
                                         json=self.roots['val']['cap'],
                                         vocab=self.vocab,
                                         max_len=self.max_text_len,
                                         use_deepseek=False)

    def set_test_dataset(self):
        self.test_dataset = FlickrDataset(root=self.roots['test']['img'],
                                          split='test',
                                          json=self.roots['test']['cap'],
                                          vocab=self.vocab,
                                          max_len=self.max_text_len,
                                          use_deepseek=False)
        #sub_size=5000
        #self.test_dataset=torch.utils.data.Subset(self.test_dataset,range(sub_size))
    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()
            self.setup_flag = True

        '''if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None'''

    def train_dataloader(self):
        if self.useDeepSeek is False:
            # from torch.utils.data import ConcatDataset
            # dataset=ConcatDataset([self.train_dataset,self.val_dataset])
            loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                 #dataset=dataset,
                                                 batch_size=self.batch_size,
                                                 # sampler=self.train_sampler,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers,
                                                 collate_fn=collate_fn_bert)
            # print(len(self.train_dataset))
            # print(f'train batches: {len(loader)}')
            # print(f'train samples: {len(loader) * self.batch_size}')
            return loader

        else:
            loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                 # dataset=dataset,
                                                 batch_size=self.batch_size,
                                                 # sampler=self.train_sampler,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers,
                                                 collate_fn=collate_fn_bert_useDeepseek)
            return loader



    def val_dataloader(self):

            loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                 batch_size=self.batch_size,
                                                 # sampler=self.val_sampler,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers,
                                                 collate_fn=collate_fn_bert)

            return loader




    def test_dataloader(self):

            loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                 batch_size=self.batch_size,
                                                 # sampler=self.val_sampler,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers,
                                                 collate_fn=collate_fn_bert)

            return loader



class MscocoDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        super().__init__()

        # _config['data_root'], _config['datasets'], vocab, _config['max_text_len'], _config['tokenizer'], _config['image_size'], _config['per_gpu_batchsize'], _config['num_workers'])

        self.data_path = os.path.join(_config["data_root"], _config['datasets'])#'/mnt/Data/wangshilong/self_datasets/coco'
        self.datasets = _config['datasets'] #coco
        self.vocab = None

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]

        # self.tokenizer = _config["tokenizer"]

        self.setup_flag = False
        self.dist = dist

        self.roots, self.ids = get_paths(self.data_path, self.datasets)
        # ids是一个列表里面存了每个图文对的id
        self.i=1
        # change
        self.useDeepSeek=True
    def set_train_dataset(self):
        self.train_dataset = CocoDataset(root=self.roots['train']['img'],
                                         json=self.roots['train']['cap'],
                                         max_len=self.max_text_len,
                                         useDeepseek=self.useDeepSeek,
                                         ids=self.ids['train'],
                                         split='train'
                                         )


    def set_val_dataset(self):
        self.val_dataset = CocoDataset(root=self.roots['val']['img'],
                                       json=self.roots['val']['cap'],
                                       max_len=self.max_text_len,
                                       useDeepseek=False,
                                       ids=self.ids['val'],
                                       split='val')

    def set_test_dataset(self):
        self.test_dataset = CocoDataset(root=self.roots['test']['img'],
                                        json=self.roots['test']['cap'],
                                        max_len=self.max_text_len,
                                        useDeepseek=False,
                                        ids=self.ids['test'],
                                        split='test')

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()
            # print("Train dataset self.bp:", self.train_dataset.bp)
            # print("Val dataset self.bp:", self.val_dataset.bp)
            # print("Test dataset self.bp:", self.test_dataset.bp)
            self.setup_flag = True

        '''if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None'''
    def train_dataloader(self):
        if self.useDeepSeek is False:
            loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                 batch_size=self.batch_size,
                                                 # sampler=self.train_sampler,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers,
                                                 collate_fn=collate_fn_bert)
            # print(len(self.train_dataset))
            # print(f'train batches: {len(loader)}')
            # print(f'train samples: {len(loader) * self.batch_size}')
            return loader
        else:
            loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                 batch_size=self.batch_size,
                                                 # sampler=self.train_sampler,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers,
                                                 collate_fn=collate_fn_bert_useDeepseek)
            return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                             batch_size=self.batch_size,
                                             # sampler=self.val_sampler,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=self.num_workers,
                                             collate_fn=collate_fn_bert)
        # print(len(self.val_dataset))
        # print(f'val batches: {len(loader)}')
        # print(f'val samples: {len(loader) * self.batch_size}')
        return loader

    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                             batch_size=self.batch_size,
                                             # sampler=self.val_sampler,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=self.num_workers,
                                             collate_fn=collate_fn_bert)
        # print(len(self.test_dataset))
        # print(f'test batches: {len(loader)}')
        # print(f'test samples: {len(loader) * self.batch_size}')
        return loader


'===================================iaprtc12==========================================='
import re


class IAPRTC12Dataset(data.Dataset):
    ''' def __init__(self, root, json, split, vocab, max_len, transform=None):

        #
        self.model, _ = clip.load('ViT-B/32', device='cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = max_len
        self.tokenizer = clip.tokenize
        #

        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]'''

    def __init__(self, root, ann, max_len):
        """
        初始化数据集
        :param images_root_dir: 图像文件夹根路径
        :param annotations_root_dir: 注释文件夹根路径
        :param transform: 图像预处理变换
        """
        _, self.transform = clip.load('ViT-B/32', device='cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = max_len
        self.images_root_dir = root  # /mnt/Data/wangshilong/self_datasets/iapric12/images  里面存的都是子文件夹里面都是图像
        self.images_root_dir_match = '/mnt/Data/wangshilong/self_datasets/iaprtc12'
        self.annotations_root_dir = ann  # /mnt/Data/wangshilong/self_datasets/iapric12/ann....eng 里面存的都是子文件夹里面存的图像信息

        self.data = self._load_data()
        self.tokenizer = clip.tokenize

    def _load_data(self):
        """
        加载数据（图像路径和对应的注释）
        """
        data = []
        for root_dir, sub_dirs, _ in os.walk(self.annotations_root_dir):
            for sub_dir in sub_dirs:
                sub_dir_path = os.path.join(root_dir, sub_dir)
                for file in os.listdir(sub_dir_path):
                    if file.endswith('.eng'):
                        annotation_file_path = os.path.join(sub_dir_path, file)
                        content = self._read_file(annotation_file_path)
                        if content:
                            image_path = self._extract_tag_content(content, 'IMAGE')
                            description = self._extract_tag_content(content, 'DESCRIPTION')
                            if image_path and description:
                                full_image_path = os.path.join(self.images_root_dir_match, image_path)
                                if os.path.exists(full_image_path):
                                    data.append((full_image_path, description))
        return data

    def _read_file(self, filepath):
        """
        读取文件内容，处理编码错误
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                return None

    def _extract_tag_content(self, content, tag):
        """
        提取指定标签的内容
        """
        pattern = fr'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, caption = self.data[index]  # 返回一个元组，对应图像文本
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        is_clip = True
        if is_clip:
            input_ids = convert_to_feature(caption, self.tokenizer)
            input_ids = torch.as_tensor(input_ids, dtype=torch.long)
            return image, input_ids, index

    ''' def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        is_clip = True
        if is_clip:
            input_ids= convert_to_feature(caption, self.max_len, self.tokenizer)
            input_ids = torch.as_tensor(input_ids, dtype=torch.long)
            return image, input_ids, index
        else:
            tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            return image, target, index, img_id

    def __len__(self):
        return len(self.ids)'''


from torch.utils.data import Subset


class IAPRTC12DataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        super().__init__()

        self.data_path = os.path.join(_config["data_root"], _config['datasets'])  # /.../iaptc12
        self.datasets = _config['datasets']

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]

        self.setup_flag = False
        self.dist = dist

        self.roots, self.ids = get_paths(self.data_path, self.datasets)

    def set_train_dataset(self):
        self.train_dataset = IAPRTC12Dataset(root=self.roots['train']['img'],
                                             ann=self.roots['train']['cap'],
                                             max_len=self.max_text_len,
                                             )

        indices = list(range(len(self.train_dataset)))
        train_indices = indices[:15000]
        self.train_dataset = Subset(self.train_dataset, train_indices)
        return self.train_dataset

    def set_val_dataset(self):
        self.val_dataset = IAPRTC12Dataset(root=self.roots['val']['img'],
                                           ann=self.roots['val']['cap'],
                                           max_len=self.max_text_len,
                                           )
        indices = list(range(len(self.val_dataset)))
        val_indices = indices[15000:]
        self.val_dataset = Subset(self.val_dataset, val_indices)
        self.val_dataset.indices = list(range(len(self.val_dataset.indices)))
        return self.val_dataset

    def set_test_dataset(self):
        self.test_dataset = IAPRTC12Dataset(root=self.roots['test']['img'],
                                            ann=self.roots['test']['cap'],
                                            max_len=self.max_text_len,
                                            )
        indices = list(range(len(self.test_dataset)))
        test_indices = indices[15000:]
        self.test_dataset = Subset(self.test_dataset, test_indices)
        self.test_dataset.indices = list(range(len(self.test_dataset.indices)))  # 重置索引
        return self.test_dataset

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.setup_flag = True

        '''if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None'''

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                             batch_size=self.batch_size,
                                             # sampler=self.train_sampler,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=self.num_workers,
                                             collate_fn=collate_fn_bert)
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                             batch_size=self.batch_size,
                                             # sampler=self.val_sampler,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=self.num_workers,
                                             collate_fn=collate_fn_bert)
        return loader

    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                             batch_size=self.batch_size,
                                             # sampler=self.val_sampler,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=self.num_workers,
                                             collate_fn=collate_fn_bert)
        return loader


'===========================================RSICD_dataset=========================================='


class RSICDDataset(data.Dataset):
    """
    Dataset loader for RSICD dataset.
    """

    def __init__(self, root, json, split, max_len):
        # 加载CLIP模型
        self.model, self.transform = clip.load('ViT-B/32', device='cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = max_len
        self.tokenizer = clip.tokenize
        self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.root = root

        self.split = split

        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if 'split' in d and d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn"""

        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        qusetion_background = 'Question: {What is the background of this image like?} Answer:'
        question_entries = 'Question: {List all the objects included in the image.} Answer:'
        image_blip = self.blip_processor(image, return_tensors="pt")
        image_blip_background = self.blip_processor(image, qusetion_background, return_tensors="pt")
        image_blip_entries = self.blip_processor(image, question_entries, return_tensors='pt')

        image_blip = image_blip['pixel_values']
        image_blip_background = image_blip_background['pixel_values']
        image_blip_entries = image_blip_entries['pixel_values']
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        is_clip = True
        if is_clip:
            input_ids = self.tokenizer(caption)
            input_ids = torch.as_tensor(input_ids, dtype=torch.long)
            return image,image_blip,image_blip_background,image_blip_entries, input_ids, index

    def __len__(self):
        return len(self.ids)


class RSICDDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        super().__init__()

        self.data_path = os.path.join(_config["data_root"], _config['datasets'])  # /.../iaptc12
        self.datasets = _config['datasets']

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]

        self.setup_flag = False
        self.dist = dist

        self.roots, self.ids = get_paths(self.data_path, self.datasets)

    def set_train_dataset(self):
        self.train_dataset = RSICDDataset(root=self.roots['train']['img'],
                                          json=self.roots['train']['cap'],
                                          split='train',
                                          max_len=self.max_text_len,
                                          )

    def set_val_dataset(self):
        self.val_dataset = RSICDDataset(root=self.roots['val']['img'], #
                                        json=self.roots['val']['cap'],
                                        split='val',
                                        max_len=self.max_text_len,
                                        )
        '''sub_size=5000
        self.val_dataset=torch.utils.data.Subset(self.val_dataset,range(sub_size))'''

    def set_test_dataset(self):
        self.test_dataset = RSICDDataset(root=self.roots['test']['img'],
                                         json=self.roots['test']['cap'],
                                         split='test',
                                         max_len=self.max_text_len,
                                         )

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.setup_flag = True

        '''if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None'''

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                             batch_size=self.batch_size,
                                             # sampler=self.train_sampler,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=self.num_workers,
                                             collate_fn=collate_fn_bert)
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                             batch_size=self.batch_size,
                                             # sampler=self.val_sampler,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=self.num_workers,
                                             collate_fn=collate_fn_bert)
        return loader

    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                             batch_size=self.batch_size,
                                             # sampler=self.val_sampler,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=self.num_workers,
                                             collate_fn=collate_fn_bert)
        return loader


'================================EC_datasets==============================='
import os
import json as jsonmod
import torch
from torch.utils.data import Dataset
from PIL import Image
import clip
import pandas as pd
import base64
from io import BytesIO


class ECommerceDataset(Dataset):
    def __init__(self, text_tsv, img_tsv, max_len):
        _, self.transform = clip.load('ViT-B/32', device='cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = max_len
        self.text_tsv = text_tsv
        self.img_tsv = img_tsv

        self.tokenizer = clip.tokenize

        # Load text and image data
        self.text_data = self._load_text_data()
        self.image_data = self._load_image_data()

        # Combine text and image data based on img_id
        self.data = self._merge_data()

    def _load_text_data(self):
        return pd.read_csv(self.text_tsv, sep='\t', header=None, names=['img_id', 'text', 'Translated'])

    def _load_image_data(self):
        return pd.read_csv(self.img_tsv, sep='\t', header=None, names=['img_id', 'image'])

    def _merge_data(self):
        merged_data = pd.merge(self.text_data, self.image_data, on='img_id')
        # 移除无法解码的图像
        valid_indices = []
        for i, row in merged_data.iterrows():
            try:
                self._decode_image(row['image'])
                valid_indices.append(i)
            except Exception as e:
                print(f"Skipping invalid image for img_id {row['img_id']}: {e}")
        return merged_data.iloc[valid_indices]

    def _decode_image(self, base64_str):
        try:
            image_data = base64.urlsafe_b64decode(base64_str)
            image = Image.open(BytesIO(image_data)).convert('RGB')
            '''image_path=os.path.join('/mnt/Data/wangshilong',"1.png")
            image.save(image_path)'''
            return image
        except Exception as e:
            print(f"Error decoding image: {e}")
            raise e  # 重新抛出异常以便在_merge_data方法中捕获

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img_id, caption, base64_image = row['img_id'], row['Translated'], row['image']
        image = self._decode_image(base64_image)

        if self.transform:
            image = self.transform(image)

        input_ids = clip.tokenize(caption, truncate=True)
        return image, input_ids, index


class ECommerceDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        super().__init__()
        self.data_path = os.path.join(_config["data_root"], _config['datasets'])
        self.datasets = _config['datasets']

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]

        self.setup_flag = False
        self.dist = dist

        self.roots, self.ids = get_paths(self.data_path, self.datasets)

    def set_train_dataset(self):
        self.train_dataset = ECommerceDataset(
            text_tsv=self.roots['train']['cap'],
            img_tsv=self.roots['train']['img'],
            max_len=self.max_text_len,

        )

    def set_val_dataset(self):
        self.val_dataset = ECommerceDataset(
            text_tsv=self.roots['val']['cap'],
            img_tsv=self.roots['val']['img'],
            max_len=self.max_text_len,

        )

    def set_test_dataset(self):
        self.test_dataset = ECommerceDataset(
            text_tsv=self.roots['test']['cap'],
            img_tsv=self.roots['test']['img'],
            max_len=self.max_text_len,

        )

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()
            self.setup_flag = True

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn_bert
        )
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn_bert
        )
        return loader

    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn_bert
        )
        return loader


