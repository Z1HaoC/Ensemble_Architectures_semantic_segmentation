# Ensemble_Architectures_semantic_segmentation
basemodel.pycreatebasemodel.pycreatebasemodel.py# 语义分割集成架构训练指南
本项目提供了语义分割任务中基础模型与元模型的训练流程，通过组合不同架构的模型以提升分割性能。
项目简介
本项目包含两个核心训练阶段：
基础模型训练：基于 U-Net、U-Net++、DeepLabV3 等经典语义分割架构训练基础模型
元模型训练：在基础模型的输出基础上，结合注意力机制和改进卷积块训练元学习模型
环境要求
plaintext
python >= 3.7
torch >= 1.8.0
segmentation-models-pytorch >= 0.2.0
albumentations >= 1.1.0
opencv-python >= 4.5.0
numpy >= 1.19.0
matplotlib >= 3.3.0
可通过以下命令安装主要依赖：
bash
pip install torch segmentation-models-pytorch albumentations opencv-python numpy matplotlib
数据准备
下载 CamVid 数据集（城市场景语义分割数据集）
按照以下目录结构放置数据：
plaintext
./CamVid/
├── train/           # 训练集图像
├── trainannot/      # 训练集掩码
├── val/             # 验证集图像
├── valannot/        # 验证集掩码
├── test/            # 测试集图像
└── testannot/       # 测试集掩码
训练流程
1. 训练基础模型
基础模型包括 U-Net、U-Net++、DeepLabV3，使用createbasemodel.py进行训练：
bash
python createbasemodel.py
基础模型训练细节：
采用 ResNet34 作为编码器，使用 ImageNet 预训练权重
损失函数为 Jaccard Loss + 带类别权重的 CrossEntropy Loss
训练过程中会自动保存验证集性能最佳的模型参数（.pth文件）
训练完成后，基础模型的预测结果会保存至./model_predictions/目录
2. 训练元模型
元模型训练需在基础模型训练完成后进行，依赖基础模型的输出结果，使用jaccardloss.ipynb（交互式）或jaccardloss.py（脚本式）：
bash
# 脚本式运行
python jaccardloss.py

# 或使用Jupyter Notebook交互式运行
jupyter notebook jaccardloss.ipynb
元模型训练细节：
支持四种元学习架构：
Standard + SE Attention
Depthwise Separable + CBAM
Dilated Convolution + Spatial Attention
Group Convolution + ECA Attention
基于基础模型的预测结果进行训练，融合多模型特征提升性能
文件说明
createbasemodel.py：基础模型训练脚本，负责训练 U-Net、U-Net++、DeepLabV3 并保存模型参数和预测结果
jaccardloss.py/jaccardloss.ipynb：元模型训练脚本，在基础模型输出基础上训练带注意力机制的元学习模型
basemodel.py：基础模型训练的备份脚本（与 createbasemodel.py 功能类似，可作为参考）
注意事项
训练前请确保 GPU 环境配置正确（代码默认使用CUDA_VISIBLE_DEVICES=0）
基础模型训练时间较长，建议根据硬件配置调整batch_size和训练轮次
元模型训练依赖./model_predictions/目录下的基础模型输出，请确保该目录存在且包含正确文件# 语义分割集成架构训练指南
本项目提供了语义分割任务中基础模型与元模型的训练流程，通过组合不同架构的模型以提升分割性能。
项目简介
本项目包含两个核心训练阶段：
基础模型训练：基于 U-Net、U-Net++、DeepLabV3 等经典语义分割架构训练基础模型
元模型训练：在基础模型的输出基础上，结合注意力机制和改进卷积块训练元学习模型
环境要求
plaintext
python >= 3.7
torch >= 1.8.0
segmentation-models-pytorch >= 0.2.0
albumentations >= 1.1.0
opencv-python >= 4.5.0
numpy >= 1.19.0
matplotlib >= 3.3.0
可通过以下命令安装主要依赖：
bash
pip install torch segmentation-models-pytorch albumentations opencv-python numpy matplotlib
数据准备
下载 CamVid 数据集（城市场景语义分割数据集）
按照以下目录结构放置数据：
plaintext
./CamVid/
├── train/           # 训练集图像
├── trainannot/      # 训练集掩码
├── val/             # 验证集图像
├── valannot/        # 验证集掩码
├── test/            # 测试集图像
└── testannot/       # 测试集掩码
训练流程
1. 训练基础模型
基础模型包括 U-Net、U-Net++、DeepLabV3，使用createbasemodel.py进行训练：
bash
python createbasemodel.py
基础模型训练细节：
采用 ResNet34 作为编码器，使用 ImageNet 预训练权重
损失函数为 Jaccard Loss + 带类别权重的 CrossEntropy Loss
训练过程中会自动保存验证集性能最佳的模型参数（.pth文件）
训练完成后，基础模型的预测结果会保存至./model_predictions/目录
2. 训练元模型
元模型训练需在基础模型训练完成后进行，依赖基础模型的输出结果，使用jaccardloss.ipynb（交互式）或jaccardloss.py（脚本式）：
bash
# 脚本式运行
python jaccardloss.py

# 或使用Jupyter Notebook交互式运行
jupyter notebook jaccardloss.ipynb
元模型训练细节：
支持四种元学习架构：
Standard + SE Attention
Depthwise Separable + CBAM
Dilated Convolution + Spatial Attention
Group Convolution + ECA Attention
基于基础模型的预测结果进行训练，融合多模型特征提升性能
文件说明
createbasemodel.py：基础模型训练脚本，负责训练 U-Net、U-Net++、DeepLabV3 并保存模型参数和预测结果
jaccardloss.py/jaccardloss.ipynb：元模型训练脚本，在基础模型输出基础上训练带注意力机制的元学习模型
basemodel.py：基础模型训练的备份脚本（与 createbasemodel.py 功能类似，可作为参考）
注意事项
训练前请确保 GPU 环境配置正确（代码默认使用CUDA_VISIBLE_DEVICES=0）
基础模型训练时间较长，建议根据硬件配置调整batch_size和训练轮次
元模型训练依赖./model_predictions/目录下的基础模型输出，请确保该目录存在且包含正确文件
Z1HaoC/Ensemble_Architectures_semantic_segmentation
README.md
﻿
模板
