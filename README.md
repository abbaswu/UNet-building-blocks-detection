# U-Net: Semantic segmentation with PyTorch
<a href="#"><img src="https://img.shields.io/github/actions/workflow/status/milesial/PyTorch-UNet/main.yml?logo=github&style=for-the-badge" /></a>
<a href="https://hub.docker.com/r/milesial/unet"><img src="https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge" /></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.13+-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.6+-blue.svg?logo=python&style=for-the-badge" /></a>

![input and output for a random image in the test dataset](https://i.imgur.com/GD8FcB7.png)


Customized implementation of the [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch for Kaggle's [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) from high definition images.

- [Quick start](#quick-start)
  - [Without Docker](#without-docker)
  - [With Docker](#with-docker)
- [Description](#description)
- [Usage](#usage)
  - [Docker](#docker)
  - [Training](#training)
  - [Prediction](#prediction)
- [Weights & Biases](#weights--biases)
- [Pretrained model](#pretrained-model)
- [Data](#data)

## Quick start

### Without Docker

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download the data and run training:
```bash
bash scripts/download_data.sh
python train.py --amp
```

### With Docker

1. [Install Docker 19.03 or later:](https://docs.docker.com/get-docker/)
```bash
curl https://get.docker.com | sh && sudo systemctl --now enable docker
```
2. [Install the NVIDIA container toolkit:](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
3. [Download and run the image:](https://hub.docker.com/repository/docker/milesial/unet)
```bash
sudo docker run --rm --shm-size=8g --ulimit memlock=-1 --gpus all -it milesial/unet
```

4. Download the data and run training:
```bash
bash scripts/download_data.sh
python train.py --amp
```

## Description
This model was trained from scratch with 5k images and scored a [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) of 0.988423 on over 100k test images.

It can be easily used for multiclass segmentation, portrait segmentation, medical segmentation, ...


## Usage
**Note : Use Python 3.6 or newer**

### Docker

A docker image containing the code and the dependencies is available on [DockerHub](https://hub.docker.com/repository/docker/milesial/unet).
You can download and jump in the container with ([docker >=19.03](https://docs.docker.com/get-docker/)):

```console
docker run -it --rm --shm-size=8g --ulimit memlock=-1 --gpus all milesial/unet
```


### Training

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

Automatic mixed precision is also available with the `--amp` flag. [Mixed precision](https://arxiv.org/abs/1710.03740) allows the model to use less memory and to be faster on recent GPUs by using FP16 arithmetic. Enabling AMP is recommended.


### Prediction

After training your model and saving it to `MODEL.pth`, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```console
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```
You can specify which model file to use with `--model MODEL.pth`.

## Weights & Biases

The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  Loss curves, validation curves, weights and gradient histograms, as well as predicted masks are logged to the platform.

When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it
 by setting the `WANDB_API_KEY` environment variable. If not, it will create an anonymous run which is automatically deleted after 7 days.


## Pretrained model
A [pretrained model](https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0) is available for the Carvana dataset. It can also be loaded from torch.hub:

```python
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
```
Available scales are 0.5 and 1.0.

## Data
The Carvana data is available on the [Kaggle website](https://www.kaggle.com/c/carvana-image-masking-challenge/data).

You can also download it using the helper script:

```
bash scripts/download_data.sh
```

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively (note that the `imgs` and `masks` folder should not contain any sub-folder or any other files, due to the greedy data-loader). For Carvana, images are RGB and masks are black and white.

You can use your own dataset as long as you make sure it is loaded properly in `utils/data_loading.py`.


---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)


# Vectorization building blocks of Atlas Municipal

P3: Emanuel Stüdeli, Tanja Falasca, Chunyang Gao -> [baug-ikg-010.d.ethz.ch](http://baug-ikg-010.d.ethz.ch/)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/e0ed6bc3-46b9-45bc-9721-26383d094932/2937845b-e346-498c-b915-a0afe124ac20/Untitled.png)

conda activate myenv

which python

base是conda的默认环境

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/e0ed6bc3-46b9-45bc-9721-26383d094932/e232903d-8099-462d-8109-5d7d3a922cfe/Untitled.png)

# 1. Tutorial

**generate_tiling:**

w_size: 每个小块的宽度和高度

win_size: 窗口，用于后续生成tile的大小

pad_px: 设置边缘填充的像素数, step=pad_px

img_pad: 在图片的边缘添加填充，前后都填充大小为pad_px的像素数

view_as_windows: 每个窗口的大小为 **`(win_size,win_size,3)`**，

**reconstruct_from_patches:**

patches_images: 包含所有小块的数组

step_size: 滑动窗口步长

image_size_2d: 原始完整图像的大小

i_h, i_w: 调整后图像大小

p_h, p_w：图像小块大小

**scheduler:**

学习率调度器，这个调度器会在验证集上的损失停止改善时减少学习率。**`ReduceLROnPlateau`** 调度器会监控一个指标（这里是最小化**`'min'`**），当该指标在**`patience`**指定的周期数内没有改进时，它将学习率乘以**`factor`**（这里是0.5）。如果学习率降低到**`min_lr`**（这里是0.00001）以下，则不会进一步降低。**`verbose=True`** 会打印出学习率调整的信息。

## 

# 2. Paper

Combining Deep Learning and Mathematical Morphology for Historical Map Segmentation

# 3. U-Net

sigmoid确保了输出值在0到1之间，可以被解释为概率，对于二分类问题有用，对于多分类问题则需要修改为softmax函数

### 1.train.py

data loader

dice loss

global step: 五个epoch里共有多少个loader

division_step: 分割步数，

颜色通道问题：在PyTorch中，图像数据通常以**`(C, H, W)`**的形式存储，其中**`C`**代表颜色通道数（例如，对于RGB图像，**`C`**为3），**`H`**代表图像高度，**`W`**代表图像宽度。这种格式称为“通道优先”（channel-first）格式。

然而，**`matplotlib.pyplot.imshow`**期望的图像数据格式是**`(H, W, C)`**，即“高度优先”，后跟“宽度”和“颜色通道数”，这称为“通道末尾”（channel-last）格式。

loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

**`pin_memory`**：一个布尔值，如果设置为 **`True`**，则数据在加载到 GPU 前会被锁页内存（page-locked memory）所包裹，这样能够更快地将数据传输到 GPU 上。锁页内存可以避免在数据传输时发生内存页的复制，从而提高数据传输的效率。通常情况下，在使用 GPU 训练模型时，建议将 **`pin_memory`** 设置为 **`True`**，以提高数据加载的效率。

val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
**`drop_last`** 参数是用来指定在数据集样本数量不能被批量大小整除时，是否丢弃最后一个批次。
使用 **`**`** 符号是一种将字典解包为关键字参数的方式。当你调用一个函数或方法时，可以使用 **`**`** 符号将一个字典中的键值对解包为关键字参数传递给函数。

**`wandb.init(project='U-Net', resume='allow', anonymous='must')`**：

- **`wandb.init()`** 是 WandB 库中用于初始化实验的函数。通过这个函数，你可以指定项目名称、恢复实验（如果之前有）以及是否匿名等参数。
- **`project='U-Net'`**：指定了实验所属的项目名称为 'U-Net'。这样可以将实验记录到 WandB 项目中，并在 WandB 仪表板中查看和比较实验结果。
- **`resume='allow'`**：允许恢复实验。如果之前有同名的实验存在，允许在其基础上继续记录实验结果。
- **`anonymous='must'`**：指定实验为匿名模式。在匿名模式下，实验结果不会与具体的用户账户关联，保护用户隐私。

**`experiment.config.update(...)`**：

- **`experiment`** 是 **`wandb.init()`** 函数的返回值，是一个实验对象，用于记录实验过程和结果。
- **`config`** 是实验配置的一个属性，它允许你记录实验的各种配置参数，例如模型超参数、数据预处理方式等。
- **`update()`** 方法用于更新实验配置。在这里，通过 **`update()`** 方法将一个字典中的键值对更新到实验的配置中。
- 字典中包含了一系列实验的配置参数，包括训练周期数（epochs）、批量大小（batch_size）、学习率（learning_rate）、验证集比例（val_percent）、保存检查点（save_checkpoint）、图像缩放比例（img_scale）、以及是否使用混合精度训练（amp）等。

**`logging.info()`**：

- **`logging`** 是 Python 内置的日志记录模块。
- **`info()`** 是 **`logging`** 模块提供的一种记录日志消息的方法，用于记录一般信息。
- 

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

- 使用 ReduceLROnPlateau 学习率调度器，当模型在验证集上的性能不再提升时，自动降低学习率。
- **`'max'`** 表示验证性能的目标是最大化，通常用于评估指标是 Dice score 等需要最大化的指标。
- **`patience=5`** 表示当验证性能连续 5 个周期（epochs）没有提升时，就降低学习率。

grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

- 使用 PyTorch 的 **`GradScaler`** 类来实现梯度缩放，用于混合精度训练。
- **`enabled=amp`** 控制是否启用混合精度训练，当 **`amp=True`** 时启用混合精度训练，否则禁用。

**images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)**

**`memory_format=torch.channels_last`** 指定了张量的内存格式为 channels_last，这意味着张量的最后一个维度表示通道（channel），适用于图像数据的处理。channels_last 内存格式在 GPU 上的计算效率可能更高，尤其是对于一些深度学习框架和操作。

**`with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):`**：

- **`torch.autocast`** 上下文管理器用于指定混合精度训练的上下文环境。
- **`device.type if device.type != 'mps' else 'cpu'`**：这里使用三元表达式判断当前设备类型是否为 'mps'（即混合精度训练支持的设备），如果是，则使用 CPU 进行自动混合精度训练，否则使用当前设备。
- **`enabled=amp`**：启用混合精度训练，如果 **`amp=True`**，则启用混合精度训练，否则禁用。

**`loss = criterion(masks_pred.squeeze(1), true_masks.float())`**：
**`masks_pred.squeeze(1)`** 将预测的掩膜张量的第二个维度（通道维度）压缩，因为对于二分类任务，输出的张量维度通常是 (batch_size, 1, height, width)，需要将其压缩为 (batch_size, height, width)。

**`optimizer.zero_grad(set_to_none=True)`**：

- 这一行代码用于清零优化器中所有参数的梯度。参数 **`set_to_none=True`** 表示将梯度张量设置为 **`None`**，而不是清零梯度张量的值。这种操作对于混合精度训练非常有用，可以避免清除过程中的类型转换开销。

**`grad_scaler.scale(loss).backward()`**：

- 这一行代码是混合精度训练中的关键步骤。首先，用 **`grad_scaler.scale(loss)`** 对损失值进行缩放，然后调用 **`backward()`** 方法进行反向传播计算梯度。

**`grad_scaler.unscale_(optimizer)`**：

- 这一行代码用于将优化器中的梯度值反向缩放回原始大小。在混合精度训练中，由于损失值被缩放了，所以在进行梯度裁剪和参数更新之前，需要将梯度值还原到原始大小。

**`torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)`**：

- 这一行代码用于梯度裁剪，即对梯度进行截断，以防止梯度爆炸的问题。**`gradient_clipping`** 参数指定了梯度裁剪的阈值。

**`grad_scaler.step(optimizer)`** 和 **`grad_scaler.update()`**：

- **`grad_scaler.step(optimizer)`** 用于使用优化器来更新模型参数，但在混合精度训练中，此步骤需要由梯度缩放器（**`grad_scaler`**）来完成。
- **`grad_scaler.update()`** 用于更新梯度缩放器内部的缩放因子，以便在下一步计算中使用。

**`division_step = (n_train // (5 * batch_size))`**：

- 计算一个训练周期中进行模型评估的步数。这里将训练集的大小（**`n_train`**）除以 5 倍的批次大小（**`batch_size`**），以便在训练过程中约每 1/5 个训练周期进行一次评估。

**`histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())`** 和 **`histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())`**：

- 将参数的值和梯度转换为 CPU 上的 NumPy 数组，并使用 **`wandb.Histogram`** 将其添加到 **`histograms`** 字典中。
1. **`val_score = evaluate(model, val_loader, device, amp)`**：
    - 调用 **`evaluate`** 函数对模型进行评估，得到验证集的评估分数（例如 Dice score）。
2. **`scheduler.step(val_score)`**：
    - 根据验证集的评估分数调整学习率，这里使用的是 ReduceLROnPlateau 学习率调度器。

**`logging.info('Validation Dice score: {}'.format(val_score))`**：

- 使用 Python 的 logging 模块记录验证集上的 Dice 分数。这条日志信息将被记录到日志文件中，以便后续查看和分析。

**`experiment.log({...})`**：

- 使用实验记录器（可能是 WandB）记录一系列信息，包括学习率、验证集 Dice 分数、图像数据和参数直方图等。
- **`'learning rate': optimizer.param_groups[0]['lr']`**：记录当前学习率，通过 **`optimizer.param_groups[0]['lr']`** 获取优化器中的学习率参数。
- **`'validation Dice': val_score`**：记录验证集上的 Dice 分数，这是模型性能的一个评估指标。
- **`'images': wandb.Image(images[0].cpu())`**：记录输入图像的第一个样本，使用 WandB 的 Image 类将图像数据转换为可记录的格式，并确保数据在 CPU 上。
- **`'masks'`**：记录真实掩膜和预测掩膜的图像数据。
    - **`'true': wandb.Image(true_masks[0].float().cpu())`**：记录真实掩膜的图像数据，确保数据在 CPU 上，并将数据转换为浮点型。
    - **`'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu())`**：记录预测掩膜的图像数据，首先使用 **`argmax(dim=1)`** 获取预测掩膜的类别索引，然后将数据转换为浮点型，并确保数据在 CPU 上。
    

### 2.data_loading.py

**`mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]`**：

- **`mask_dir.glob()`** 函数返回一个生成器，生成指定目录下符合指定模式的文件路径。
- **`idx + mask_suffix + '.*'`** 创建了文件名的模式，**`idx`** 是传入的索引，**`mask_suffix`** 是传入的后缀。
- **`list(...)`** 将生成器转换为列表。
- **`[0]`** 选取列表中的第一个文件路径。
- 因此，**`mask_file`** 包含了第一个匹配到的文件路径。

```python
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))
```

使用多进程池来并行处理掩码文件，以确定数据集中所有掩码的唯一值。**`partial()`** 函数用于为 **`unique_mask_values`** 函数提供部分参数，即 **`mask_dir`** 和 **`mask_suffix`**。**`tqdm()`** 用于显示进度条。最后，将结果存储在名为 **`unique`** 的列表中。

### 3.predict.py

**`net.eval()`**命令将模型设置为评估模式，这意味着所有的训练特定的操作比如Dropout将会被禁用

mask pred: 1,2,640,959

true mask: 1,640,959

nn.Crossentropy: input(3,4), target(3)

images: 2,3,500,500

labels: 2,1,500,500

outputs: 2,1,500,500

### 3. predict.py

PIL size: width, height

cv: height, width

像素是宽高比，ndarray是高宽

**1.preprocess: preprocess(mask_values, pil_img, scale, is_mask)输入的是img图像，得到的是numpy数组**

调整输入图像的大小为新的宽度和高度; mask_values unique?

如果是mask，则设置每个像素对应的类别的值

如果不是mask，则若为灰度图像，将维度扩展为3维；若为彩色图像，调整通道顺序

将像素值标准化到0到1之间

**2.predict_img： 输入的是Image.open的图像，得到的是numpy数组**

**转化为torch tensor**

img = img.unsqueeze(0)

增加一个维度（因为神经网络通常需要批处理的输入）

使用**`torch.no_grad()`**上下文管理器禁用梯度计算，这是因为在推理时不需要计算梯度，这样可以节省计算资源和内存。在这个上下文中执行模型的前向传播。

使用**`F.interpolate`**将输出调整回原始输入图像的大小。**`mode='bilinear'`**指定了双线性插值作为缩放方法。

**`(full_img.size[1], full_img.size[0])`**：这是一个元组，指定了输出张量调整后的目标尺寸。**`full_img.size[1]`**是原始输入图像的宽度，**`full_img.size[0]`**是原始输入图像的高度。请注意，**`PIL.Image`**对象的**`size`**属性返回的是**`(width, height)`**，而在PyTorch中调整张量大小时通常使用**`(height, width)`**的顺序。

如果模型具有多个类别（多类别分类任务），则通过 **`argmax(dim=1)`** 取得每个像素的预测类别索引。否则，对于二分类任务，我们使用 **`torch.sigmoid`** 将输出进行 sigmoid 函数激活，并与阈值 **`out_threshold`** 进行比较，得到二值化的掩码。

有一个二维张量时，使用 **`argmax(dim=1)`** 可以找到每一行上的最大值的索引。

将得到的掩码转换为 NumPy 数组，并返回

**3.get_output_filenames**

**`def _generate_name(fn):`** 在**`get_output_filenames`**函数内部，定义了一个辅助函数**`_generate_name`**，它接受一个文件名**`fn`**作为输入。这个辅助函数用于生成单个输出文件名。

**4.mask_to_image(mask: np.ndarray, mask_values): 输入是numpy数组，得到的是图像**

```
if isinstance(mask_values[0], list):
    out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
elif mask_values == [0, 1]:
    out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
else:
    out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
```

根据 **`mask_values`** 的不同情况，创建一个用于存储图像数据的 NumPy 数组 **`out`**。如果 **`mask_values`** 是一个嵌套列表，则创建一个三维数组；如果 **`mask_values`** 是 **`[0, 1]`**，则创建一个布尔类型的二维数组；否则创建一个二维数组，并将数据类型设为无符号整型。

例如，在语义分割任务中，一个像素可能对应于多个类别，比如路面、车辆、行人等。每个通道可以分别表示每个类别的概率分布或者像素属于该类别的置信度。

因此，如果 **`mask_values`** 是一个嵌套列表，每个子列表中的元素表示一个类别或标签的取值范围，例如 **`[[0, 0, 0], [255, 255, 255]]`** 可以表示黑色和白色两个类别。

如果掩码 **`mask`** 的维度是 3（通常表示多类别掩码），则取每个像素位置上值最大的通道索引作为掩码的值，以将多类别掩码转换为单通道掩码。

遍历 **`mask_values`** 中的每个值 **`v`**，并将掩码 **`mask`** 中值等于当前索引 **`i`** 的位置，对应的图像 **`out`** 的像素值设为 **`v`**。

将 NumPy 数组 **`out`** 转换为 PIL 图像对象，并返回。

## 4.paper code

IRIS: instance segmentation  and semantic segmentation, binary semantic segmentation 

**Combining Deep Learning and Mathematical Morphology for Historical Map Segmentation**

watershed

dgmm code:

eval_shape_detection: mask_label_image

- 通过复制 **`labels`** 数组创建一个新的 **`labels_renumbered`** 数组。
- 在 **`labels_renumbered`** 数组中，使用掩码 **`bg_mask`** 来将背景像素的值设置为 **`bg_label`**。

border_calibration: cv2.bitwise_and

after test merge EPM patches, extract using border_calibration

after watershed: evaluation using eval_shape_detection

## 5. my implementation

train_v2: use mask to mask out irrelavant regions