# Path to the downloaded CLIP official weights.
# See: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/clip.py#L30
# CLIP_VIT_B16_PATH = 'D:\pythonProject\st-adapter-main\ViT-B-16.pt'
CLIP_VIT_B16_PATH = '/home/zhangliye/ymh/st-adapter/ViT-B-16.pt'
CLIP_VIT_L14_PATH = ''

# Whether cuDNN should be temporarily disable for 3D depthwise convolution.
# For some PyTorch builds the built-in 3D depthwise convolution may be much
# faster than the cuDNN implementation. You may experiment with your specific
# environment to find out the optimal option.
DWCONV3D_DISABLE_CUDNN = True

# Configuration of datasets. The required fields are listed for Something-something-v2 (ssv2)
# and Kinetics-400 (k400). Fill in the values to use the datasets, or add new datasets following
# these examples.
DATASETS = {
    'ssv2': dict(
        TRAIN_ROOT='',
        VAL_ROOT='',
        TRAIN_LIST='',
        VAL_LIST='',
        NUM_CLASSES=174,
    ),
    'UCF101': dict(
        TRAIN_ROOT='/home/zhangliye/ymh/st-adapter/data/ucf101/train',
        VAL_ROOT='/home/zhangliye/ymh/st-adapter/data/ucf101/val',
        TRAIN_LIST='/home/zhangliye/ymh/st-adapter/data/ucf101/train.csv',
        VAL_LIST='/home/zhangliye/ymh/st-adapter/data/ucf101/val.csv',
        NUM_CLASSES=101,
    ),
    'k400': dict(
        # TRAIN_ROOT='D:\pythonProject\Transformer\\vit-shift_org\zkping\\vit-shift\data\Kinetics400_mmlab',
        # VAL_ROOT='D:\pythonProject\Transformer\\vit-shift_org\zkping\\vit-shift\data\Kinetics400_mmlab',
        # TRAIN_LIST='D:\pythonProject\Transformer\\vit-shift_org\zkping\\vit-shift\data\Kinetics400_mmlab\\train.csv',
        # VAL_LIST='D:\pythonProject\Transformer\\vit-shift_org\zkping\\vit-shift\data\Kinetics400_mmlab\\val.csv',
        # NUM_CLASSES=400,
        TRAIN_ROOT='/home/zhangliye/ymh/k400/zkping/vit-shift/data/Kinetics400_mmlab',
        VAL_ROOT='/home/zhangliye/ymh/k400/zkping/vit-shift/data/Kinetics400_mmlab',
        TRAIN_LIST='/home/zhangliye/ymh/k400/zkping/vit-shift/data/Kinetics400_mmlab/train.csv',
        VAL_LIST='/home/zhangliye/ymh/k400/zkping/vit-shift/data/Kinetics400_mmlab/val.csv',
        NUM_CLASSES=400,
    ),
    'hmdb51': dict(
        # TRAIN_ROOT='D:\pythonProject\Transformer\\vit-shift_org\zkping\\vit-shift\data\Kinetics400_mmlab',
        # VAL_ROOT='D:\pythonProject\Transformer\\vit-shift_org\zkping\\vit-shift\data\Kinetics400_mmlab',
        # TRAIN_LIST='D:\pythonProject\Transformer\\vit-shift_org\zkping\\vit-shift\data\Kinetics400_mmlab\\train.csv',
        # VAL_LIST='D:\pythonProject\Transformer\\vit-shift_org\zkping\\vit-shift\data\Kinetics400_mmlab\\val.csv',
        # NUM_CLASSES=400,
        TRAIN_ROOT='/home/zhangliye/ymh/st-adapter/data/hmdb51/train',
        VAL_ROOT='/home/zhangliye/ymh/st-adapter/data/hmdb51/test',
        TRAIN_LIST='/home/zhangliye/ymh/st-adapter/data/hmdb51/train.csv',
        VAL_LIST='/home/zhangliye/ymh/st-adapter/data/hmdb51/test.csv',
        NUM_CLASSES=51,
    ),
}
gpu_id=0