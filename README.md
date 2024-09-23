# 一个基于超过16,000张抓取照片的四分类不当图片模型  
## 更新(release)  
- 添加了第二个模型(v2)，相比于第一个模型(v1)对politic和other有更好的区分能力  
- 构建Detect类时传入`version`参数 `v1` / `v2`  


## 序言
> 总共有四个类别：卡通、色情、政治以及其他。
- 每个类别大约有4,000张图片。
> 由于数据集的敏感性，它不会公开。

## 在训练过程中，使用了`ResNet34`和`ResNet50`分类模型进行迁移学习。
### 数据集描述
- **色情**：从接近20,000张抓取的图片中清理过滤后留下的近4,000张图片。
- **卡通**：从近10,000张抓取的图片中精选出的超过4,000张图片。
- **政治**：利用了`wider_face`数据集中的一些肖像图片，以及自行抓取的敏感图片。
- **其他**：包括2,000张抓取的图片以及从Kaggle获得的宠物和场景数据集中的超过5,000张图片，这些图片都是符合规定的，可能会出现在一般的博客或网站上。

### 模型介绍
- 对`ResNet34`和`ResNet50`应用了迁移学习，并调整输出层以适应四分类的需求。

### 输出解释
- 将原始模型的输出通过softmax函数处理，得到每个类别的概率。
- 四个类别如下：
  - **卡通**：非色情的卡通图片。
  - **色情**：色情图片。
  - **政治**：政治敏感及暴力内容。
  - **其他**：一般合规图片，可能出现在各种博客和网站上。

### 模型性能
```shell
类别: 卡通  
  精确度: 0.95  
  召回率: 0.95  
类别: 其他  
  精确度: 0.92  
  召回率: 0.82  
类别: 政治  
  精确度: 0.79  
  召回率: 0.93  
类别: 色情  
  精确度: 0.95  
  召回率: 0.88  
整体准确率: 0.89  
```

### 使用方法
- `./release`
- 或者直接使用pip安装
```shell
pip install SensitiveImgDetect
```
> 如果找不到包，请切换到官方源。

### 开发
- `./dev`

# SensitiveImgDetect

该包提供了一种加载预训练PyTorch模型的方法。

## 安装

你可以使用pip来安装这个包：

```bash
pip install SensitiveImgDetect
```

## 概览

`Detect` 类是为图像分类任务设计的，允许用户使用预训练模型预测单个或多个图像的类别。本文档提供了关于如何初始化和使用该类的概述，包括示例代码片段。

## 安装

在使用`Detect`类之前，请确保已经安装了必要的库：

```bash
pip install torch pillow torchvision
```

## 初始化

要使用`Detect`类，你需要初始化它。类构造器允许你指定计算设备（CPU或GPU）。

```python
from SensitiveImgDetect import Detect 

# 初始化Detect类
detector = Detect(device='cuda')  # 如果没有CUDA请使用'cpu'
```

## 方法

### 1. 检测单张图片的类别

#### 方法: `detect_single_type`

**描述**: 预测单张图片的类别标签。

**参数**:
- `img`: 一个图像对象（PIL Image）。

**返回**: 一个字符串，表示预测的类别标签。

#### 示例用法:

```python
from PIL import Image

# 加载一张图片
img = Image.open("path_to_your_image.jpg")

# 预测类别标签
predicted_label = detector.detect_single_type(img)
print(f"这张图片的预测类别是: {predicted_label}")
```

### 2. 检测单张图片的类别概率

#### 方法: `detect_single_prob`

**描述**: 预测单张图片的类别概率。

**参数**:
- `img`: 一个图像对象（PIL Image）。

**返回**: 一个字典，包含类别标签及其相应的概率。

#### 示例用法:

```python
# 预测类别概率
predicted_probs = detector.detect_single_prob(img)
print("这张图片的类别概率:")
for class_label, probability in predicted_probs.items():
    print(f"{class_label}: {probability}")
```

### 3. 检测图片列表的类别

#### 方法: `detect_list_type`

**描述**: 预测一系列图片的类别标签。

**参数**:
- `img_list`: 图像对象列表（PIL Images）。

**返回**: 预测的类别标签列表。

#### 示例用法:

```python
# 加载多张图片
images = [Image.open("image_1.jpg"), Image.open("image_2.jpg")]

# 预测图片列表的类别标签
predicted_labels = detector.detect_list_type(images)
print("图片列表的预测类别:")
print(predicted_labels)
```

### 4. 检测图片列表的类别概率

#### 方法: `detect_list_prob`

**描述**: 预测一系列图片的类别概率。

**参数**:
- `img_list`: 图像对象列表（PIL Images）。

**返回**: 包含类别标签及其相应概率的字典列表。

#### 示例用法:

```python
# 预测图片列表的类别概率
predicted_probs_list = detector.detect_list_prob(images)
for index, probs in enumerate(predicted_probs_list):
    print(f"图片 {index + 1} 的概率:")
    for class_label, probability in probs.items():
        print(f"{class_label}: {probability}")
```  
### 补充  
> `detect_list*`相比于`detect_single*`会占用更多的内存  
  