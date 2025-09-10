# Ensemble_Architectures_semantic_segmentation
[![GitHub License](https://img.shields.io/github/license/Z1HaoC/Ensemble_Architectures_semantic_segmentation)](https://github.com/Z1HaoC/Ensemble_Architectures_semantic_segmentation/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/torch-1.8.0%2B-orange)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/dataset-CamVid-green)](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)

一款基于**基础模型+元模型**的语义分割集成架构训练框架，通过融合U-Net、U-Net++、DeepLabV3等经典模型与注意力机制，提升复杂场景（如城市场景小目标分割）的语义分割精度。


## 📖 项目简介
本项目聚焦语义分割任务中的性能优化，采用“两阶段训练”思路：
1. **基础模型训练**：构建U-Net、U-Net++、DeepLabV3三种经典语义分割模型，为后续集成提供高质量特征基础；
2. **元模型训练**：在基础模型输出的特征之上，结合SE/CBAM/ECA等注意力机制与改进卷积块（深度可分离卷积、空洞卷积等），实现多模型特征的高效融合，进一步提升分割精度。

核心优势：
- 支持多基础模型并行训练与结果自动保存；
- 元模型提供4种可配置架构，适配不同场景需求；
- 全流程适配CamVid数据集，开箱即用，支持自定义数据集扩展。


## 🛠️ 环境配置
### 依赖列表
| 依赖库                  | 版本要求       | 用途说明                     |
|-------------------------|----------------|------------------------------|
| Python                  | ≥ 3.7          | 基础运行环境                 |
| PyTorch                 | ≥ 1.8.0        | 深度学习框架                 |
| segmentation-models-pytorch | ≥ 0.2.0    | 语义分割模型快速构建         |
| albumentations          | ≥ 1.1.0        | 数据增强（支持掩码同步变换） |
| opencv-python           | ≥ 4.5.0        | 图像读写与预处理             |
| numpy                   | ≥ 1.19.0       | 数值计算与数组处理           |
| matplotlib              | ≥ 3.3.0        | 训练曲线与结果可视化         |
| tqdm                    | 最新版         | 训练进度条显示               |

### 安装命令
```bash
# 优先安装PyTorch（建议根据CUDA版本选择，示例为CUDA 11.1）
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 安装剩余依赖
pip install segmentation-models-pytorch albumentations opencv-python numpy matplotlib tqdm
```

> 注：若无需GPU加速，可安装CPU版本PyTorch，具体参考[PyTorch官方文档](https://pytorch.org/get-started/locally/)。


## 📊 数据准备
### 1. 数据集获取
使用**CamVid数据集**（城市场景语义分割基准数据集），包含701张标注图像，共12个类别（如天空、建筑、道路、行人等）。  
下载地址：[CamVid官方下载页](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)  
（需注册剑桥大学账号，或通过[Kaggle镜像](https://www.kaggle.com/carlolepelaars/camvid)下载）

### 2. 目录结构
将数据集解压后，按以下结构放置在项目根目录，确保代码可正确读取图像与掩码：
```plaintext
./CamVid/
├── train/           # 训练集图像（367张，.png格式）
├── trainannot/      # 训练集掩码（367张，单通道.png，像素值对应类别索引）
├── val/             # 验证集图像（101张，.png格式）
├── valannot/        # 验证集掩码（101张，单通道.png）
├── test/            # 测试集图像（233张，.png格式）
└── testannot/       # 测试集掩码（233张，单通道.png）
```

### 3. 数据格式说明
- 图像：RGB三通道，分辨率默认360×480；
- 掩码：单通道灰度图，每个像素值代表一个类别（如`0`=天空、`1`=建筑，具体类别映射见代码中`CLASS_NAMES`）。


## 🚀 训练流程
### 阶段1：训练基础模型（必须先执行）
通过`createbasemodel.py`训练U-Net、U-Net++、DeepLabV3三种基础模型，生成模型权重与预测结果。

#### 运行命令
```bash
# 基础训练（默认参数：ResNet34编码器、batch_size=8、epochs=50）
python createbasemodel.py

# 自定义参数（示例：调整batch_size与训练轮次）
python createbasemodel.py --batch-size 16 --epochs 100 --lr 0.001
```

#### 关键参数说明
| 参数名        | 类型    | 默认值 | 说明                     |
|---------------|---------|--------|--------------------------|
| `--batch-size`| int     | 8      | 批次大小（根据GPU显存调整）|
| `--epochs`    | int     | 50     | 训练轮次                 |
| `--lr`        | float   | 0.001  | 初始学习率               |
| `--img-size`  | tuple   | 360 480| 输入图像尺寸             |
| `--save-dir`  | str     | ./weights | 模型权重保存目录         |

#### 输出结果
- 模型权重：`./weights/`目录下生成`unet_best.pth`、`unetpp_best.pth`、`deeplabv3_best.pth`；
- 预测结果：`./model_predictions/`目录下生成三种模型的验证集/测试集预测掩码（.png格式），用于后续元模型训练。


### 阶段2：训练元模型（依赖基础模型输出）
在基础模型预测结果的基础上，通过`jaccardloss.py`（脚本式）或`jaccardloss.ipynb`（交互式）训练元模型，融合多模型特征。

#### 运行命令
```bash
# 脚本式运行（推荐，适合服务器环境）
python jaccardloss.py

# 交互式运行（适合调试与可视化，需安装Jupyter）
jupyter notebook jaccardloss.ipynb
```

#### 元模型支持的4种架构
| 架构名称                     | 核心组件                          | 适用场景                     |
|------------------------------|-----------------------------------|------------------------------|
| Standard + SE Attention      | 标准卷积 + SE通道注意力           | 通用场景，平衡精度与速度     |
| Depthwise Separable + CBAM   | 深度可分离卷积 + CBAM注意力       | 轻量级部署，减少计算量       |
| Dilated Convolution + Spatial Attention | 空洞卷积 + 空间注意力       | 需捕捉大尺度特征（如道路）   |
| Group Convolution + ECA      | 分组卷积 + ECA通道注意力          | 高分辨率图像分割，保留细节   |

#### 输出结果
- 元模型权重：`./weights/meta_model_best.pth`；
- 性能指标：训练过程中实时打印Jaccard指数（IoU）、Loss，训练结束后输出测试集mIoU；
- 可视化结果：`./meta_predictions/`目录下生成元模型的预测掩码与“预测-真值”对比图。


## 📁 文件说明
| 文件名                  | 核心功能                                                                 | 备注                                 |
|-------------------------|--------------------------------------------------------------------------|--------------------------------------|
| `createbasemodel.py`    | 基础模型（U-Net/U-Net++/DeepLabV3）训练、权重保存、预测结果生成         | 必须先运行，为元模型提供输入         |
| `jaccardloss.py`        | 元模型训练脚本，支持4种架构配置，计算Jaccard Loss优化模型               | 依赖`./model_predictions/`目录        |
| `jaccardloss.ipynb`     | 元模型交互式训练 notebook，含中间结果可视化（适合调试）                 | 与`jaccardloss.py`功能一致           |
| `basemodel.py`          | 基础模型训练备份脚本，功能与`createbasemodel.py`一致                     | 可用于对比不同训练参数的效果         |
| `utils/`（若存在）      | 工具函数目录（图像读取、掩码处理、指标计算等）                           | 需确保该目录在`sys.path`中           |


## ⚠️ 注意事项
1. **GPU环境配置**：
   - 代码默认使用第1块GPU（`CUDA_VISIBLE_DEVICES=0`），多GPU训练需修改代码中设备配置（如`torch.device("cuda:1")`）；
   - 若无GPU，可添加`--cpu`参数强制使用CPU训练（但训练速度会显著变慢）。

2. **基础模型依赖检查**：
   - 元模型训练前，务必确保`./model_predictions/`目录下存在3种基础模型的预测结果（共6个文件：`unet_val_preds.npy`、`unetpp_val_preds.npy`等）；
   - 若该目录缺失文件，需重新运行`createbasemodel.py`。

3. **训练调优建议**：
   - 若出现过拟合，可减小`batch_size`、增加数据增强（在`createbasemodel.py`的`get_transforms`函数中添加翻转、旋转等）；
   - 若GPU显存不足，可降低`--img-size`（如256×256）或`--batch-size`。


## 🤝 贡献指南
1. Fork本仓库到个人账号；
2. 创建特性分支（`git checkout -b feature/your-feature`）；
3. 提交修改（`git commit -m "Add new feature: XXX"`）；
4. 推送到分支（`git push origin feature/your-feature`）；
5. 打开Pull Request，描述修改内容与用途。


## 📄 许可证
本项目采用[MIT许可证](https://github.com/Z1HaoC/Ensemble_Architectures_semantic_segmentation/blob/main/LICENSE)，允许个人与商业使用，但需保留原作者版权信息。


## ❓ 问题反馈
若遇到训练报错、数据读取失败等问题，可通过以下方式反馈：
1. 提交GitHub Issues：[https://github.com/Z1HaoC/Ensemble_Architectures_semantic_segmentation/issues](https://github.com/Z1HaoC/Ensemble_Architectures_semantic_segmentation/issues)
2. 描述问题时请附上：错误日志、环境配置（Python/PyTorch版本、GPU型号）、运行命令。
