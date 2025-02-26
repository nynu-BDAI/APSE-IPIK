# 开发时间 2024/7/15 18:13
# 开发人员:牧良逢
from sacred import Experiment

ex = Experiment("DOUBLEencoder")
ex_test=Experiment("Test")

def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "vcr": 0,
        "vcr_qar": 0,
        "nlvr2": 0,
        "irtr": 0,
        "contras": 0,
        "snli": 0,
    }
    ret.update(d)
    return ret

@ex.config
def config():
    exp_name = "finetune_irtr"
    seed = 0
    datasets = " "
    loss_names = _loss_names({"irtr": 1})
    batch_size = 64  # use this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    margin = 0.1 # use
    task_ids=None

    # Image setting
    #train_transform_keys = ["clip"]
    #val_transform_keys = ["clip"]
    image_size = 224
    patch_size = 32
    #draw_false_image = 1
    image_only = False

    # Text Setting
    #vqav2_label_size = 3129
    max_text_len = 77
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    #draw_false_text = 0

    # Transformer Setting
    num_top_layer = 6
    input_image_embed_size = 1024
    input_text_embed_size = 768
    vit = "swin_base_patch4_window7_224_in22k"
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate =1e-5 #use
    lr_update = 10
    weight_decay = 0.2
    decay_power = 1
    max_epoch = 5#use
    max_steps = None
    warmup_steps = 2000
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0 # use 多少次验证一次

    test_only =False
    checkpoint = '/mnt/Data/wangshilong/pycode/HAT-pytorch/Dual_prompt/runs/last.ckpt'

    # below params varies with the environment
    data_root = '/mnt/Data/wangshilong/self_datasets'
    log_dir = "/mnt/Data/wangshilong/pycode/HAT-pytorch/Dual_prompt/continual_result"
    per_gpu_batchsize =64# you should define this manually with per_gpu_batch_size=#
    num_gpus =4
    num_nodes = 1
    load_path = ""
    num_workers =8
    precision = 16

    #SCAN
    direction = 'i2t'
    lambda_softmax = 0.4

@ex_test.config
def config():
    exp_name = "finetune_irtr"
    seed = 0
    datasets = " "
    loss_names = _loss_names({"irtr": 1})
    batch_size = 64  # use this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    margin = 0.1 # use
    task_ids=None

    # Image setting
    #train_transform_keys = ["clip"]
    #val_transform_keys = ["clip"]
    image_size = 224
    patch_size = 32
    #draw_false_image = 1
    image_only = False

    # Text Setting
    #vqav2_label_size = 3129
    max_text_len = 77
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    #draw_false_text = 0

    # Transformer Setting
    num_top_layer = 6
    input_image_embed_size = 1024
    input_text_embed_size = 768
    vit = "swin_base_patch4_window7_224_in22k"
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate =1e-5 #use
    lr_update = 10
    weight_decay = 0.2
    decay_power = 1
    max_epoch = 50#use
    max_steps = None
    warmup_steps = 2000
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0 # use 多少次验证一次

    test_only =False
    checkpoint = '/mnt/Data/wangshilong/pycode/HAT-pytorch/Dual_prompt/runs/last.ckpt'

    # below params varies with the environment
    data_root = '/mnt/Data/wangshilong/self_datasets'
    log_dir = "/mnt/Data/wangshilong/pycode/HAT-pytorch/Dual_prompt/continual_result"
    per_gpu_batchsize =64 # you should define this manually with per_gpu_batch_size=#
    num_gpus =2
    num_nodes = 1
    load_path = ""
    num_workers =8
    precision = 16

    #SCAN
    direction = 'i2t'
    lambda_softmax = 0.4




