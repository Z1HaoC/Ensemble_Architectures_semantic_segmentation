import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import itertools

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 数据集类
class Dataset(torch.utils.data.Dataset):
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian',
               'bicyclist', 'unlabelled']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # 类别到索引的映射
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # 读取图像
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码并选择需要的类别
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        masks = [(mask == v) for v in self.class_values]
        mask = np.dstack(masks).astype('float')

        # 数据增强
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # 预处理
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


# 评价指标
def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    if union == 0:
        return 1
    dice = (2.0 * intersection) / union
    return dice


def iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    if union == 0:
        return 1
    return intersection / union


# 训练集增强（移除了TileInterpolationAugment）
def get_training_augmentation():
    return albu.Compose([
        albu.Resize(512, 512),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.2,
            rotate_limit=10,
            shift_limit=0.05,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        albu.RandomBrightnessContrast(p=0.3),
    ])


# 验证集增强（简化为仅调整大小）
def get_validation_augmentation():
    return albu.Compose([
        albu.Resize(512, 512),
    ])


# 预处理
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32') if len(x.shape) == 3 else x.astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# 数据路径
DATA_DIR = './CamVid/'
x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')
x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')
x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian',
           'bicyclist', 'unlabelled']
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# 初始化数据集
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataset_vis = Dataset(
    x_test_dir,
    y_test_dir,
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=0)

# 定义基模型
model_unet = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
).to(DEVICE)

model_deeplabv3 = smp.DeepLabV3(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
).to(DEVICE)

model = smp.UnetPlusPlus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
    decoder_attention_type='scse',
).to(DEVICE)

# 定义损失和指标
class_weights = torch.tensor([
    1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 5.0, 5.0, 1.0, 5.0, 5.0, 1.0
]).to(DEVICE)

# 修改损失函数为JaccardLoss + CrossEntropyLoss
loss = smp.utils.losses.JaccardLoss() + smp.utils.losses.CrossEntropyLoss(weight=class_weights)
metrics = [smp.utils.metrics.IoU(threshold=0.5)]

# 定义训练和验证的 Epoch
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
    device=DEVICE,

)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,

)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    train_epoch.optimizer,
    mode='max',
    factor=0.1,
    patience=3,
    min_lr=1e-6
)

# 训练 U-Net++
model = model.to(DEVICE)
max_score = -float('inf')  # 初始化 max_score

for i in range(100):
    print('\nEpoch (U-Net++):', i)
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    scheduler.step(valid_logs['iou_score'])

    # 保存最佳模型
    if valid_logs['iou_score'] > max_score:
        max_score = valid_logs['iou_score']
        torch.save(model.state_dict(), 'best_model_unetplusplus_params_JaccardLoss.pth')
        print('U-Net++ Model saved!')

    # 减小学习率
    if i == 25:
        train_epoch.optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease U-Net++ decoder learning rate to 1e-5!')

model.cpu()
torch.cuda.empty_cache()

# 训练 U-Net
train_epoch_unet = smp.utils.train.TrainEpoch(
    model_unet,
    loss=loss,
    metrics=metrics,
    optimizer=torch.optim.Adam(model_unet.parameters(), lr=0.0001),
    device=DEVICE,

)
valid_epoch_unet = smp.utils.train.ValidEpoch(
    model_unet,
    loss=loss,
    metrics=metrics,
    device=DEVICE,

)
model_unet = model_unet.to(DEVICE)
max_score_unet = -float('inf')  # 初始化 max_score_unet

for i in range(100):
    print('\nEpoch (U-Net):', i)
    train_logs_unet = train_epoch_unet.run(train_loader)
    valid_logs_unet = valid_epoch_unet.run(valid_loader)

    scheduler.step(valid_logs_unet['iou_score'])

    if valid_logs_unet['iou_score'] > max_score_unet:
        max_score_unet = valid_logs_unet['iou_score']
        torch.save(model_unet.state_dict(), 'best_model_unet_params_JaccardLoss.pth')
        print('U-Net Model saved!')

    if i == 25:
        train_epoch_unet.optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease U-Net decoder learning rate to 1e-5!')

model_unet.cpu()
torch.cuda.empty_cache()

# 训练 DeepLabV3
train_epoch_deeplabv3 = smp.utils.train.TrainEpoch(
    model_deeplabv3,
    loss=loss,
    metrics=metrics,
    optimizer=torch.optim.Adam(model_deeplabv3.parameters(), lr=0.0001),
    device=DEVICE,

)
valid_epoch_deeplabv3 = smp.utils.train.ValidEpoch(
    model_deeplabv3,
    loss=loss,
    metrics=metrics,
    device=DEVICE,

)
model_deeplabv3 = model_deeplabv3.to(DEVICE)
max_score_deeplabv3 = -float('inf')  # 初始化 max_score_deeplabv3

for i in range(100):
    print('\nEpoch (DeepLabV3):', i)
    train_logs_deeplabv3 = train_epoch_deeplabv3.run(train_loader)
    valid_logs_deeplabv3 = valid_epoch_deeplabv3.run(valid_loader)

    scheduler.step(valid_logs_deeplabv3['iou_score'])

    if valid_logs_deeplabv3['iou_score'] > max_score_deeplabv3:
        max_score_deeplabv3 = valid_logs_deeplabv3['iou_score']
        torch.save(model_deeplabv3.state_dict(), 'best_model_deeplabv3_params_JaccardLoss.pth')
        print('DeepLabV3 Model saved!')

    if i == 25:
        train_epoch_deeplabv3.optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease DeepLabV3 decoder learning rate to 1e-5!')

model_deeplabv3.cpu()
torch.cuda.empty_cache()

# 保存基模型的预测结果
save_dir = './model_predictions'
os.makedirs(save_dir, exist_ok=True)

# 保存 U-Net++ 的预测结果
model.load_state_dict(torch.load('best_model_unetplusplus_params_JaccardLoss.pth'))
model.to(DEVICE)
model.eval()
unetplusplus_predictions = []
unetplusplus_labels = []
with torch.no_grad():
    for images, masks in train_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        unetplusplus_predictions.append(outputs.cpu())
        unetplusplus_labels.append(masks.cpu())

torch.save(unetplusplus_predictions, os.path.join(save_dir, 'unetplusplus_predictions_JaccardLoss.pth'))
torch.save(unetplusplus_labels, os.path.join(save_dir, 'labels.pth'))

# 保存 U-Net 的预测结果
model_unet.load_state_dict(torch.load('best_model_unet_params_JaccardLoss.pth'))
model_unet.to(DEVICE)
model_unet.eval()
unet_predictions = []
with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(DEVICE)
        outputs = model_unet(images)
        unet_predictions.append(outputs.cpu())
torch.save(unet_predictions, os.path.join(save_dir, 'unet_predictions_JaccardLoss.pth'))

# 保存 DeepLabV3 的预测结果
model_deeplabv3.load_state_dict(torch.load('best_model_deeplabv3_params_JaccardLoss.pth'))
model_deeplabv3.to(DEVICE)
model_deeplabv3.eval()
deeplabv3_predictions = []
with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(DEVICE)
        outputs = model_deeplabv3(images)
        deeplabv3_predictions.append(outputs.cpu())
torch.save(deeplabv3_predictions, os.path.join(save_dir, 'deeplabv3_predictions_JaccardLoss.pth'))

model.cpu()
model_unet.cpu()
model_deeplabv3.cpu()
torch.cuda.empty_cache()


class MetaModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # 初始特征融合
        self.entry_conv = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes * 3, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )  # 正确闭合

        # 多尺度残差块（包含空洞卷积）
        self.res_block1 = ResidualBlock(256, 256, dilation=1)
        self.res_block2 = ResidualBlock(256, 256, dilation=2)
        self.res_block3 = ResidualBlock(256, 256, dilation=4)

        # 通道注意力模块
        self.attention = ChannelAttention(256)

        # 特征精炼模块
        self.refinement = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, num_classes, kernel_size=1)
        )

        # 跳跃连接
        self.skip_conv = torch.nn.Conv2d(num_classes * 3, num_classes, kernel_size=1)

    def forward(self, inputs):
        # 初始特征提取 [B, 3*C, H, W] -> [B, 256, H, W]
        x = self.entry_conv(inputs)

        # 多尺度残差处理
        x = self.res_block1(x)  # dilation=1
        x = self.res_block2(x)  # dilation=2
        x = self.res_block3(x)  # dilation=4

        # 通道注意力
        x = self.attention(x)

        # 特征精炼 [B, 256, H, W] -> [B, C, H, W]
        x = self.refinement(x)

        # 跳跃连接融合原始输入
        skip = self.skip_conv(inputs)  # [B, 3*C, H, W] -> [B, C, H, W]
        return torch.sigmoid(x + skip)


# 残差块（带空洞卷积）
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                     padding=dilation, dilation=dilation, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                     padding=dilation, dilation=dilation, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)


# 通道注意力模块（改进版SE Block）
class ChannelAttention(torch.nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // reduction),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channels // reduction, channels),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        scale = avg_out + max_out
        return x * scale.view(b, c, 1, 1)


# 加载预存的预测结果和标签
save_dir = './model_predictions'
unetplusplus_predictions = torch.load(os.path.join(save_dir, 'unetplusplus_predictions_JaccardLoss.pth'))
unet_predictions = torch.load(os.path.join(save_dir, 'unet_predictions_JaccardLoss.pth'))
deeplabv3_predictions = torch.load(os.path.join(save_dir, 'deeplabv3_predictions_JaccardLoss.pth'))
meta_train_labels = torch.load(os.path.join(save_dir, 'labels.pth'))

# 将张量拼接为一个连续的张量
meta_train_data = []
meta_train_labels_list = []
for upred, u_pred, d_pred, labels in zip(unetplusplus_predictions, unet_predictions, deeplabv3_predictions,
                                         meta_train_labels):
    combined = torch.cat([upred, u_pred, d_pred], dim=1)
    meta_train_data.append(combined)
    meta_train_labels_list.append(labels)

# 确保每个批次的输入数据和标签都有相同的长度
assert len(meta_train_data) == len(meta_train_labels_list), "Input data and labels must have the same number of batches"

meta_train_data = torch.cat(meta_train_data, dim=0)
meta_train_labels = torch.cat(meta_train_labels_list, dim=0).to(DEVICE)


# 定义元模型的数据集和数据加载器
class MetaDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        assert len(data) == len(labels), "Data and labels must have the same length"

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)


meta_dataset = MetaDataset(meta_train_data, meta_train_labels)
meta_loader = torch.utils.data.DataLoader(meta_dataset, batch_size=2, shuffle=True)

# 设置混合精度训练
scaler = torch.cuda.amp.GradScaler()

# 训练元模型
meta_model = MetaModel(len(CLASSES)).to(DEVICE)
meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)

# 修改元模型的损失函数为JaccardLoss + CrossEntropyLoss
meta_criterion = smp.utils.losses.JaccardLoss() + smp.utils.losses.CrossEntropyLoss(weight=class_weights)

num_epochs_meta = 5
for epoch in range(num_epochs_meta):
    running_loss = 0.0
    meta_model.train()
    for data, labels in meta_loader:
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)

        meta_optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            outputs = meta_model(data)
            loss = meta_criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(meta_optimizer)
        scaler.update()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(meta_loader)}')

torch.save(meta_model.state_dict(), 'meta_model_JaccardLoss.pth')
