# LoRA、DoRA 原理与代码实现

相关链接：
- https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch
- https://lightning.ai/lightning-ai/studios/code-lora-from-scratch
- https://zhuanlan.zhihu.com/p/681502990

论文链接：
- [LoRA] https://arxiv.org/abs/2106.09685
- [DoRA] https://arxiv.org/abs/2402.09353

# LoRA 原理

**一句话描述**：LoRA 通过在模型权重更新中加入低秩矩阵实现对大模型低成本的高效微调。

预训练的 LLM 通常被称为基础模型，因为它们在各种任务中具有通用性。然而，要将预先训练好的 LLM 用于特定的数据集或任务通常需要进一步微调模型。微调允许模型适应特定的领域而不需要昂贵的预训练，但是更新所有层仍然需要昂贵的计算资源，特别是对于较大的模型。

**LoRA的主要原理**：LoRA（Low Rank Adaptation）是一种机器学习技术，假设预训练语言模型具有较低的内在维度，即使是投影到一个更小的子空间上也可以进行高效的学习。因此 LoRA 通过仅调整模型参数的一个小的低秩子集，来修改预先训练好的模型(例如 LLM )，以更好地适应特定的、通常较小的数据集。它允许对特定任务数据的大型模型进行有效的微调，大大减少了微调所需的计算成本和时间。

**LoRA的公式**：
假设有一个给定层的较大权重矩阵 $W$，在反向传播过程中，需要学习一个 $\Delta W$ 矩阵，它包含了在训练过程中我们需要更新多少原始权值以使损失函数最小化的信息。假设输入是 $x$，输出是 $h$，那么: $h = W_0x$.

在进行参数更新时，如果使用全参数量微调：$h=W_0x+△Wx $，此时权重更新定义如下：$$W_{\text {updated }}=W_0 + \Delta W$$

如果采用 LoRA，则：
$$h=W_0x+\Delta Wx=W_0x+ABx $$
$$W_{\text {updated }}=W_0+AB$$
$△W$的分解意味着可以用两个较小的低秩矩阵A和B来表示较大的矩阵。

**低秩矩阵的优势**：
假定只使用单独的一个 $\Delta W$ 矩阵，参数量为 5,000 x 10,000 (共有50M 参数)，如果使用 LoRA，选择秩为 $r=8$, 此时原来的大矩阵可分解为两个大小分别为 5000 x 8 维以及 8 x 10000 维的小矩阵 A 和 B，A 和 B 只有 80000 + 40000 = 120000 个参数，这比通过 $\Delta W$ 进行常规微调的50M 参数小400倍。

**实现过程**：
冻结预训练的模型参数，在Transfomer的每一个线性层中加入一个可训练的旁路矩阵（低秩可分离矩阵 $\Delta W$），接着将旁路输出与初始输出相加后输入到网络当中，并只训练这些新增的旁路矩阵参数。其中，低秩可分离矩阵由两个矩阵组成，第一个矩阵 $A$ 负责降维，第二个矩阵 $B$ 负责升维，中间层维度为 $r$，从而来模拟本征秩（intrinsic rank），这两个低秩矩阵能够大幅度减小参数量。

![An illustration of regular finetuning (left) and LoRA finetuning (right).](https://raw.githubusercontent.com/Samanthe-H/PicGo/master/lora%20v.s.%20regular.png)


## LoRA 层代码实现

首先初始化一个 LoRALayer, 创建低秩矩阵 A 和 B，以及缩放超参数 alpha 和秩超参数 rank:

<img src="https://raw.githubusercontent.com/Samanthe-H/PicGo/master/LoRA_1.png" alt="Illustration of the LoRA matrices A and B with rank r." style="zoom:50%;" />


```python

import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        '''
        in_dim: the input dimension of the layer we want to modify using LoRA
        out_dim: the respective output dimension of that layer
        rank: a hyperparameter that controls the inner dimension of the matrices A and B
        alpha: a scaling hyperparameter applied to the output of the low-rank adaptation
        '''
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    
```

- 对 rank 的理解：
  
    rank用来指定用于适应的低秩矩阵的秩。更小的 rank 导致更简单的低秩矩阵，这导致在适应过程中学习的参数更少。这可以导致更快的训练和潜在的计算需求减少。然而，当 rank 较小时，低秩矩阵获取任务特定信息的能力就会降低。

- 对 alpha 参数的解释：

    这个因素决定了 LoRA 层引入的对模型现有权重的更改的大小: $\alpha(x\times A\times B)$，较高的 alpha 值意味着对模型行为的较大调整，而较低的值会导致更微小的变化。

- 对 A, B 的分析：

    我们从一个随机分布中用很小的值初始化了 A，用零初始化了 B。A 的分布的标准差是由秩的平方根决定的(这个选择确保了 A 中的初始值不会太大)。在训练开始时，在通过反向传播更新 A 和 B 之前，LoRALayer 不会影响原始权重，因为如果 B = 0，则 AB = 0。

接下来，定义 LinearWithLoRA 层以及 LinearWithLoRAMerged 层，用 linear+lora_layer 替换神经网络中现有的线性（前馈）层：

<img src="https://raw.githubusercontent.com/Samanthe-H/PicGo/master/LoRA_2.png" style="zoom: 50%;" />

<img src="https://raw.githubusercontent.com/Samanthe-H/PicGo/master/LoRA_3.png" alt="LoRA_3" style="zoom: 50%;" />

将神经网络中现有的线性层替换为结合了原始线性层和 LoRALayer 的 LinearWithLoRA 层




```python

# 在代码中，当通过修改现有的 PyTorch 模型来实现 LoRA 时，实现这种线性层的 LoRA 修改的一个简单方法是
# 用 LinearWithLoRA 层替换每个线性层，该线性层结合了我们之前的 LoRALayer 实现:
class LinearWithLoRA(nn.Module):

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
# LinearWithLoRAMerged 计算下面等式的左边，LinearWithLoRA 计算等式的右边，两个函数等价
# x.(W+A.B) = x.W + x.A.B  
class LinearWithLoRAMerged(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        lora = self.lora.A @ self.lora.B # Combine LoRA matrices
        # Then combine LoRA with orig. weights
        combined_weight = self.linear.weight + self.lora.alpha*lora.T 
        return F.linear(x, combined_weight, self.linear.bias)

```


```python
# %%
import torch

torch.manual_seed(123)
layer = nn.Linear(10, 2)
x = torch.randn((1, 10))

print("Original output:", layer(x))

```

    Original output: tensor([[0.6639, 0.4487]], grad_fn=<AddmmBackward0>)



```python
# %%
layer_lora_1 = LinearWithLoRA(layer, rank=2, alpha=4)
print("LoRA output:", layer_lora_1(x))

layer_lora_2 = LinearWithLoRAMerged(layer, rank=2, alpha=4)
print("LoRA output:", layer_lora_2(x))

```

    LoRA output: tensor([[0.6639, 0.4487]], grad_fn=<AddBackward0>)
    LoRA output: tensor([[0.6639, 0.4487]], grad_fn=<AddmmBackward0>)


# DoRA 原理

DoRA (Weight-Decomposed Low-Rank Adaptation) 的灵感来自于数学原理：任何矢量都可以表示为其幅度 (表示其长度的标量值) 和方向 (表示其空间方向的单位矢量) 的乘积。开发 DoRA 的动机是基于对 LoRA 和全参数微调方法的分析和比较。DoRA 的作者发现，LoRA 总是按比例增加或减少幅度和方向更新，但缺乏对方向进行细微调整的能力，而全参数微调可以实现只对方向矩阵进行细微的改变。因此，研究人员提出了幅度和方向分量的解耦，将 LoRA 仅应用于方向分量 $V$ ，同时也允许幅度分量 $m$ 单独训练 (即对 $m$ 进行全量微调，对 $V$ 做 LoRA 微调)，以有效地减少可训练参数的数量。

DoRA 可以被看作是 LoRA 的改进或扩展,可以很容易地调整 LoRA 的代码来实现 DoRA。使用 DoRA 可以增强 LoRA 的学习能力和训练稳定性，同时避免了任何额外的推理开销。

DoRA 可以用两个步骤来描述:
1. 将预先训练好的权矩阵分解为幅度向量 (magnitude vector) $m$ 和方向矩阵 (directional matrix) $V$;
$$W_0 = m\frac{V}{\lVert V \rVert_c} = \lVert W_0 \rVert_c \frac{W_0}{\lVert W_0 \rVert_c}$$
2. 将 LoRA 应用于方向矩阵 $V$。

$$W_\text{update}= m\frac{V+\Delta V}{\lVert V+\Delta V \rVert_c} = m\frac{V+AB}{\lVert V+AB \rVert_c}$$


<div style="text-align: center;">
    <img src="pictures/DoRA_1.png" width="250" height="200">
    <img src="pictures/DoRA_2.png" width="350" height="200">
矢量分解（左）以及权重矩阵分解（右）的示意图

<div style="text-align: center;">
    <img src="pictures/DoRA_3.png" width="600" height="450">
DORA 步骤图解



如果将 DoRA 与 LoRA 进行比较，引入幅度向量 m 会增加0.01% 的参数。然而，在 LLM 和 vision transformer 基准测试中，他们发现如果 DoRA 的等级减半，那么 DoRA 的性能甚至会超过 LoRA。例如，当 DoRA 只使用普通 LoRA 的一半参数时，性能比较如下所示：

<div style="text-align: center;">
    <img src="pictures/DoRA_4.png" width="700" height="200">

###### DORA 与 LoRA 性能比较
</div>

除此之外，DoRA 对 rank 的变化更加不敏感，更具有稳健性。因此，在相对较小的 rank 下使用 DoRA 的成功率可能更高，这使得这种方法比 LoRA 更具有参数效率：
<div style="text-align: center;">
    <img src="pictures/DoRA_5.png" width="500" height="300">

###### DORA 与 LoRA 关于 rank 的稳健性比较
</div>


## DoRA 层代码实现


```python

class LinearWithDoRAMerged(nn.Module):
    """
    一个结合了Linear层和DoRA层的模块。
    该模块通过使用DoRA技术来增强线性层的表示能力。

    参数:
    - linear: 原始的线性层(nn.Linear)实例
    - rank: DoRA层的秩
    - alpha: DoRA层中的缩放参数
    """

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear  # 原始的线性层
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )  # 初始化LoRA层
        self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0, keepdim=True)
        )  # 初始化一个参数，用于调整新权重的模长

    # Code loosely inspired by    
    # https://github.com/catid/dora/blob/main/dora.py
    def forward(self, x):
        """
        前向传播过程。
        参数:
        - x: 输入特征向量

        返回:
        - 经过更新后的线性变换后的结果
        """
        lora = self.lora.A @ self.lora.B # 计算LoRA的低秩近似
        numerator = self.linear.weight + self.lora.alpha*lora.T
        denominator = numerator.norm(p=2, dim=0, keepdim=True)
        directional_component = numerator / denominator # normalize，计算方向组件
        new_weight = self.m * directional_component # 更新权重
        return F.linear(x, new_weight, self.linear.bias) # 应用新的权重进行线性变换
```

self.m 是一个可学习的幅度向量，表示归一化权重矩阵每列的大小。该参数允许模型在训练期间动态调整组合权重矩阵中每个权重向量的比例。这种额外的灵活性可以帮助模型更好地捕捉不同特征的重要性。

总之，LinearWithDoRAMerged 扩展了 LinearWithLoRAMerged 通过结合动态权重归一化和缩放来提高训练性能的概念。

若要用 DoRA 层替换线性层，可以参考代码：


```python
import torch

torch.manual_seed(123)
layer = nn.Linear(10, 2)
x = torch.randn((1, 10))

print("Original output:", layer(x))

layer_dora = LinearWithLoRAMerged(layer, rank=4, alpha=8)
print("DoRA output:", layer_dora(x))

```

    Original output: tensor([[0.6639, 0.4487]], grad_fn=<AddmmBackward0>)
    DoRA output: tensor([[0.6639, 0.4487]], grad_fn=<AddmmBackward0>)


# Example 1：在多层感知机上应用 LoRA 和 DoRA

原始代码见：[LoRA and DoRA From Scratch](https://github.com/rasbt/dora-from-scratch/tree/main)


```python
%load_ext watermark
%watermark --conda -p torch,transformers,datasets,lightning
```

    Author: Sebastian Raschka
    
    Python implementation: CPython
    Python version       : 3.11.8
    IPython version      : 8.22.2
    
    torch: 2.2.2



### 1.1 配置&读取数据


```python
import time
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
```

    /opt/anaconda3/envs/llama_factory/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
##########################
### SETTINGS
##########################

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

##########################
### MNIST DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Using downloaded and verified file: data/MNIST/raw/train-images-idx3-ubyte.gz
    Extracting data/MNIST/raw/train-images-idx3-ubyte.gz to data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to data/MNIST/raw/train-labels-idx1-ubyte.gz


    100%|██████████| 28881/28881 [00:00<00:00, 62797145.58it/s]
    
    Extracting data/MNIST/raw/train-labels-idx1-ubyte.gz to data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to data/MNIST/raw/t10k-images-idx3-ubyte.gz


​    
    100%|██████████| 1648877/1648877 [00:00<00:00, 17363610.65it/s]


    Extracting data/MNIST/raw/t10k-images-idx3-ubyte.gz to data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to data/MNIST/raw/t10k-labels-idx1-ubyte.gz


    100%|██████████| 4542/4542 [00:00<00:00, 8710804.19it/s]


    Extracting data/MNIST/raw/t10k-labels-idx1-ubyte.gz to data/MNIST/raw
    
    Image batch dimensions: torch.Size([64, 1, 28, 28])
    Image label dimensions: torch.Size([64])


### 1.2 多层感知器模型构建(不含LoRA和DoRA)

首先构造一个简单的三层感知机，该例不涉及大语言模型，只是直观展示 LoRA 和 DoRA 的实现过程以及具体对网络中的什么部分起作用。

<div style="text-align: center;">
    <img src="pictures/3-layerMP.png" width="250" height="380"> 

###### 一个简单的三层感知机（input 的维度应改为784）
</div>


```python
# %%
# Applying LoRA Layers with a small 3-layer multilayer perceptron
class MultilayerPerceptron(nn.Module):
    def __init__(self, num_features, 
        num_hidden_1, num_hidden_2, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, num_hidden_1),
            nn.ReLU(),
            nn.Linear(num_hidden_1, num_hidden_2),
            nn.ReLU(),
            nn.Linear(num_hidden_2, num_classes)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

# Hyperparameters
random_seed = 123
learning_rate = 0.005
num_epochs = 2

# Architecture
num_features = 784
num_hidden_1 = 128
num_hidden_2 = 256
num_classes = 10

torch.manual_seed(random_seed)
model_pretrained = MultilayerPerceptron(
    num_features=num_features,
    num_hidden_1=num_hidden_1,
    num_hidden_2=num_hidden_2, 
    num_classes=num_classes
)

model_pretrained.to(DEVICE)
optimizer_pretrained = torch.optim.Adam(model_pretrained.parameters(), lr=learning_rate)
print(model_pretrained)

```

    MultilayerPerceptron(
      (layers): Sequential(
        (0): Linear(in_features=784, out_features=128, bias=True)
        (1): ReLU()
        (2): Linear(in_features=128, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=10, bias=True)
      )
    )



```python
def compute_accuracy(model, data_loader, device):
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.view(-1, 28*28).to(device)
            targets = targets.to(device)
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float()/num_examples * 100


def train(num_epochs, model, optimizer, train_loader, device):

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.view(-1, 28*28).to(device)
            targets = targets.to(device)

            # FORWARD AND BACK PROP
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            if not batch_idx % 400:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))

        with torch.set_grad_enabled(False):
            print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
                  epoch+1, num_epochs,
                  compute_accuracy(model, train_loader, device)))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
```


```python
train(num_epochs, model_pretrained, optimizer_pretrained, train_loader, DEVICE)
print(f'Test accuracy: {compute_accuracy(model_pretrained, test_loader, DEVICE):.2f}%')
```

    Epoch: 001/002 | Batch 000/938 | Loss: 2.2971
    Epoch: 001/002 | Batch 400/938 | Loss: 0.2102
    Epoch: 001/002 | Batch 800/938 | Loss: 0.1499
    Epoch: 001/002 training accuracy: 95.75%
    Time elapsed: 0.22 min
    Epoch: 002/002 | Batch 000/938 | Loss: 0.0885
    Epoch: 002/002 | Batch 400/938 | Loss: 0.0328
    Epoch: 002/002 | Batch 800/938 | Loss: 0.0873
    Epoch: 002/002 training accuracy: 97.50%
    Time elapsed: 0.46 min
    Total Training Time: 0.46 min
    Test accuracy: 96.87%


### 1.3 进一步构建基于LoRA和DoRA的多层感知器

随后定义 LoRA 与 DoRA 模块，并用 LinearWithLoRA 或 LinearWithDoRA 替换多层感知机模型中的原始线性层来添加 LoRA、DoRA层:


```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class LinearWithDoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        self.m = nn.Parameter(torch.ones(1, linear.out_features))

    def forward(self, x):
        linear_output = self.linear(x)
        lora_output = self.lora(x)
        lora_output_norm = lora_output / (lora_output.norm(p=2, dim=1, keepdim=True) + 1e-9)
        dora_modification = self.m * lora_output_norm
        return linear_output + dora_modification
```


```python
import copy

model_lora = copy.deepcopy(model_pretrained)
model_dora = copy.deepcopy(model_pretrained)
```


```python
# %%
# 0,2,4 只修改网络中的三个线性层
model_lora.layers[0] = LinearWithLoRA(model_lora.layers[0], rank=4, alpha=8)
model_lora.layers[2] = LinearWithLoRA(model_lora.layers[2], rank=4, alpha=8)
model_lora.layers[4] = LinearWithLoRA(model_lora.layers[4], rank=4, alpha=8)

model_lora.to(DEVICE)
optimizer_lora = torch.optim.Adam(model_lora.parameters(), lr=learning_rate)
model_lora
```




    MultilayerPerceptron(
      (layers): Sequential(
        (0): LinearWithLoRA(
          (linear): Linear(in_features=784, out_features=128, bias=True)
          (lora): LoRALayer()
        )
        (1): ReLU()
        (2): LinearWithLoRA(
          (linear): Linear(in_features=128, out_features=256, bias=True)
          (lora): LoRALayer()
        )
        (3): ReLU()
        (4): LinearWithLoRA(
          (linear): Linear(in_features=256, out_features=10, bias=True)
          (lora): LoRALayer()
        )
      )
    )




```python
model_dora.layers[0] = LinearWithDoRA(model_dora.layers[0], rank=4, alpha=8)
model_dora.layers[2] = LinearWithDoRA(model_dora.layers[2], rank=4, alpha=8)
model_dora.layers[4] = LinearWithDoRA(model_dora.layers[4], rank=4, alpha=8)

model_dora.to(DEVICE)
optimizer_dora = torch.optim.Adam(model_dora.parameters(), lr=learning_rate)
model_dora
```




    MultilayerPerceptron(
      (layers): Sequential(
        (0): LinearWithDoRA(
          (linear): Linear(in_features=784, out_features=128, bias=True)
          (lora): LoRALayer()
        )
        (1): ReLU()
        (2): LinearWithDoRA(
          (linear): Linear(in_features=128, out_features=256, bias=True)
          (lora): LoRALayer()
        )
        (3): ReLU()
        (4): LinearWithDoRA(
          (linear): Linear(in_features=256, out_features=10, bias=True)
          (lora): LoRALayer()
        )
      )
    )



我们刚刚初始化了LoRA和DoRA层，但还没有训练LoRA层，所以有和没有初始LoRA权重的模型应该具有相同的预测性能:


```python
print(f'Test accuracy orig model: {compute_accuracy(model_pretrained, test_loader, DEVICE):.2f}%')
print(f'Test accuracy LoRA model: {compute_accuracy(model_lora, test_loader, DEVICE):.2f}%')
print(f'Test accuracy DoRA model: {compute_accuracy(model_dora, test_loader, DEVICE):.2f}%')
```

    Test accuracy orig model: 96.87%
    Test accuracy LoRA model: 96.87%
    Test accuracy DoRA model: 96.87%


### 1.4 用LoRA训练模型

首先冻结原始的线性层，只让 LoRALayer 层可训练，如下所示:


```python
# %%
def freeze_linear_layers(model):
    """
    冻结模型中的所有线性层的参数，使其不参与反向传播和参数更新。
    
    参数:
    model - 模型对象，通常是包含线性层的神经网络模型。
    
    返回值:
    无
    """
    for child in model.children():
        if isinstance(child, nn.Linear):
            # 如果子模块是线性层，则将其所有参数的requires_grad设置为False
            for param in child.parameters():
                param.requires_grad = False
        else:
            # 递归地冻结子模块中的线性层
            freeze_linear_layers(child)

freeze_linear_layers(model_lora)
for name, param in model_lora.named_parameters():
    print(f"{name}: {param.requires_grad}")

```

    layers.0.linear.weight: False
    layers.0.linear.bias: False
    layers.0.lora.A: True
    layers.0.lora.B: True
    layers.2.linear.weight: False
    layers.2.linear.bias: False
    layers.2.lora.A: True
    layers.2.lora.B: True
    layers.4.linear.weight: False
    layers.4.linear.bias: False
    layers.4.lora.A: True
    layers.4.lora.B: True


可以看到，只有低秩矩阵 A 和 B 可以训练，原始 weight 和 bias 都被冻结。


```python
# 使用Adam优化器初始化LoRA模型的参数
optimizer_lora = torch.optim.Adam(model_lora.parameters(), lr=learning_rate)
# 对LoRA模型进行训练
train(num_epochs, model_lora, optimizer_lora, train_loader, DEVICE)
# 打印LoRA微调后的测试精度
print(f'Test accuracy LoRA finetune: {compute_accuracy(model_lora, test_loader, DEVICE):.2f}%')
```

    Epoch: 001/002 | Batch 000/938 | Loss: 0.0729
    Epoch: 001/002 | Batch 400/938 | Loss: 0.2804
    Epoch: 001/002 | Batch 800/938 | Loss: 0.0878
    Epoch: 001/002 training accuracy: 97.51%
    Time elapsed: 0.25 min
    Epoch: 002/002 | Batch 000/938 | Loss: 0.0336
    Epoch: 002/002 | Batch 400/938 | Loss: 0.0264
    Epoch: 002/002 | Batch 800/938 | Loss: 0.0547
    Epoch: 002/002 training accuracy: 97.67%
    Time elapsed: 0.49 min
    Total Training Time: 0.49 min
    Test accuracy LoRA finetune: 96.98%


### 1.5 用DoRA训练模型


```python
freeze_linear_layers(model_dora)

# Check if linear layers are frozen
for name, param in model_dora.named_parameters():
    print(f"{name}: {param.requires_grad}")
```

    layers.0.m: True
    layers.0.linear.weight: False
    layers.0.linear.bias: False
    layers.0.lora.A: True
    layers.0.lora.B: True
    layers.2.m: True
    layers.2.linear.weight: False
    layers.2.linear.bias: False
    layers.2.lora.A: True
    layers.2.lora.B: True
    layers.4.m: True
    layers.4.linear.weight: False
    layers.4.linear.bias: False
    layers.4.lora.A: True
    layers.4.lora.B: True


DoRA 层与 LoRA 层的区别在于每一个 DoRA 层都多了一个可以训练的幅度向量 $m$。


```python
optimizer_dora = torch.optim.Adam(model_dora.parameters(), lr=learning_rate)
train(num_epochs, model_dora, optimizer_dora, train_loader, DEVICE)
print(f'Test accuracy DoRA finetune: {compute_accuracy(model_dora, test_loader, DEVICE):.2f}%')
```

    Epoch: 001/002 | Batch 000/938 | Loss: 0.1141
    Epoch: 001/002 | Batch 400/938 | Loss: 0.0589
    Epoch: 001/002 | Batch 800/938 | Loss: 0.0553
    Epoch: 001/002 training accuracy: 98.04%
    Time elapsed: 0.29 min
    Epoch: 002/002 | Batch 000/938 | Loss: 0.0414
    Epoch: 002/002 | Batch 400/938 | Loss: 0.0313
    Epoch: 002/002 | Batch 800/938 | Loss: 0.0114
    Epoch: 002/002 training accuracy: 98.31%
    Time elapsed: 0.56 min
    Total Training Time: 0.56 min
    Test accuracy DoRA finetune: 97.44%


在这个简单的例子中，LoRA 与 DoRA 的效果都优于原来的模型。

### 1.6 补充——将 LinearWithLoRA 换为 LinearWithLoRAMerged；DoRA 类似。


```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

    
# This LoRA code is equivalent to LinearWithLoRA
class LinearWithLoRAMerged(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        combined_weight = self.linear.weight + self.lora.alpha*lora.T
        return F.linear(x, combined_weight, self.linear.bias)

    
# This DoRA code is equivalent to LinearWithDoRA
# Code inspired by https://github.com/catid/dora/blob/main/dora.py
class LinearWithDoRAMerged(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
        
        self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0, keepdim=True))

    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        numerator = self.linear.weight + self.lora.alpha*lora.T
        denominator = numerator.norm(p=2, dim=0, keepdim=True)
        directional_component = numerator / denominator
        new_weight = self.m * directional_component
        return F.linear(x, new_weight, self.linear.bias)
```


```python
model_pretrained
```




    MultilayerPerceptron(
      (layers): Sequential(
        (0): Linear(in_features=784, out_features=128, bias=True)
        (1): ReLU()
        (2): Linear(in_features=128, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=10, bias=True)
      )
    )




```python
import copy

model_lora = copy.deepcopy(model_pretrained)
model_dora = copy.deepcopy(model_pretrained)
```


```python
model_lora.layers[0] = LinearWithLoRAMerged(model_lora.layers[0], rank=4, alpha=8)
model_lora.layers[2] = LinearWithLoRAMerged(model_lora.layers[2], rank=4, alpha=8)
model_lora.layers[4] = LinearWithLoRAMerged(model_lora.layers[4], rank=4, alpha=8)

model_lora.to(DEVICE)
optimizer_lora = torch.optim.Adam(model_lora.parameters(), lr=learning_rate)
model_lora
```




    MultilayerPerceptron(
      (layers): Sequential(
        (0): LinearWithLoRAMerged(
          (linear): Linear(in_features=784, out_features=128, bias=True)
          (lora): LoRALayer()
        )
        (1): ReLU()
        (2): LinearWithLoRAMerged(
          (linear): Linear(in_features=128, out_features=256, bias=True)
          (lora): LoRALayer()
        )
        (3): ReLU()
        (4): LinearWithLoRAMerged(
          (linear): Linear(in_features=256, out_features=10, bias=True)
          (lora): LoRALayer()
        )
      )
    )




```python
model_dora.layers[0] = LinearWithDoRAMerged(model_dora.layers[0], rank=4, alpha=8)
model_dora.layers[2] = LinearWithDoRAMerged(model_dora.layers[2], rank=4, alpha=8)
model_dora.layers[4] = LinearWithDoRAMerged(model_dora.layers[4], rank=4, alpha=8)

model_dora.to(DEVICE)
optimizer_dora = torch.optim.Adam(model_dora.parameters(), lr=learning_rate)
model_dora
```




    MultilayerPerceptron(
      (layers): Sequential(
        (0): LinearWithDoRAMerged(
          (linear): Linear(in_features=784, out_features=128, bias=True)
          (lora): LoRALayer()
        )
        (1): ReLU()
        (2): LinearWithDoRAMerged(
          (linear): Linear(in_features=128, out_features=256, bias=True)
          (lora): LoRALayer()
        )
        (3): ReLU()
        (4): LinearWithDoRAMerged(
          (linear): Linear(in_features=256, out_features=10, bias=True)
          (lora): LoRALayer()
        )
      )
    )



LoRA训练


```python
def freeze_linear_layers(model):
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively freeze linear layers in children modules
            freeze_linear_layers(child)

freeze_linear_layers(model_lora)
# Check if linear layers are frozen
for name, param in model_lora.named_parameters():
    print(f"{name}: {param.requires_grad}")
```

    layers.0.linear.weight: False
    layers.0.linear.bias: False
    layers.0.lora.A: True
    layers.0.lora.B: True
    layers.2.linear.weight: False
    layers.2.linear.bias: False
    layers.2.lora.A: True
    layers.2.lora.B: True
    layers.4.linear.weight: False
    layers.4.linear.bias: False
    layers.4.lora.A: True
    layers.4.lora.B: True



```python
optimizer_lora = torch.optim.Adam(model_lora.parameters(), lr=learning_rate)
train(num_epochs, model_lora, optimizer_lora, train_loader, DEVICE)
print(f'Test accuracy LoRA finetune: {compute_accuracy(model_lora, test_loader, DEVICE):.2f}%')
```

    Epoch: 001/002 | Batch 000/938 | Loss: 0.1039
    Epoch: 001/002 | Batch 400/938 | Loss: 0.0313
    Epoch: 001/002 | Batch 800/938 | Loss: 0.0581
    Epoch: 001/002 training accuracy: 97.71%
    Time elapsed: 0.25 min
    Epoch: 002/002 | Batch 000/938 | Loss: 0.0038
    Epoch: 002/002 | Batch 400/938 | Loss: 0.0470
    Epoch: 002/002 | Batch 800/938 | Loss: 0.0251
    Epoch: 002/002 training accuracy: 97.93%
    Time elapsed: 0.49 min
    Total Training Time: 0.49 min
    Test accuracy LoRA finetune: 97.19%


DoRA训练


```python
def freeze_linear_layers(model):
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively freeze linear layers in children modules
            freeze_linear_layers(child)

freeze_linear_layers(model_dora)
# Check if linear layers are frozen
for name, param in model_dora.named_parameters():
    print(f"{name}: {param.requires_grad}")
```

    layers.0.m: True
    layers.0.linear.weight: False
    layers.0.linear.bias: False
    layers.0.lora.A: True
    layers.0.lora.B: True
    layers.2.m: True
    layers.2.linear.weight: False
    layers.2.linear.bias: False
    layers.2.lora.A: True
    layers.2.lora.B: True
    layers.4.m: True
    layers.4.linear.weight: False
    layers.4.linear.bias: False
    layers.4.lora.A: True
    layers.4.lora.B: True



```python
optimizer_dora = torch.optim.Adam(model_dora.parameters(), lr=learning_rate)
train(num_epochs, model_dora, optimizer_dora, train_loader, DEVICE)
print(f'Test accuracy DoRA finetune: {compute_accuracy(model_dora, test_loader, DEVICE):.2f}%')
```

    Epoch: 001/002 | Batch 000/938 | Loss: 0.0507
    Epoch: 001/002 | Batch 400/938 | Loss: 0.0942
    Epoch: 001/002 | Batch 800/938 | Loss: 0.0417
    Epoch: 001/002 training accuracy: 97.79%
    Time elapsed: 0.28 min
    Epoch: 002/002 | Batch 000/938 | Loss: 0.0368
    Epoch: 002/002 | Batch 400/938 | Loss: 0.0090
    Epoch: 002/002 | Batch 800/938 | Loss: 0.1163
    Epoch: 002/002 training accuracy: 98.12%
    Time elapsed: 0.54 min
    Total Training Time: 0.54 min
    Test accuracy DoRA finetune: 97.10%


# Example 2：用 LoRA 微调 DistilBERT（最后两层）

目的：为文本分类任务通过 LoRA 微调一个小的 BERT 模型。

## 代码实现（只展示部分）

详细代码见：[02_finetune-with-lora.ipynb](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch?tab=files&layout=column&path=cloudspaces%2F01hm9hypqc6y1hrapb5prmtz0h&y=5&x=0#:~:text=02_finetune%2Dwith%2Dlora.ipynb)

初始化 DistilBERT 模型


```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)
```

我们只想训练新的 LoRA 权重，所以首先冻结所有模型参数，将所有可训练的参数设置为 param.requires_grad = False :


```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total number of trainable parameters:", count_parameters(model))


for param in model.parameters():
    param.requires_grad = False

print(model)

###########################################
#                结果展示
###########################################
# DistilBertForSequenceClassification(
#   (distilbert): DistilBertModel(
#     (embeddings): Embeddings(
#       (word_embeddings): Embedding(30522, 768, padding_idx=0)
#       (position_embeddings): Embedding(512, 768)
#       (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#       (dropout): Dropout(p=0.1, inplace=False)
#     )
#     (transformer): Transformer(
#       (layer): ModuleList(
#         (0-5): 6 x TransformerBlock(
#           (attention): MultiHeadSelfAttention(
#             (dropout): Dropout(p=0.1, inplace=False)
#             (q_lin): Linear(in_features=768, out_features=768, bias=True)
#             (k_lin): Linear(in_features=768, out_features=768, bias=True)
#             (v_lin): Linear(in_features=768, out_features=768, bias=True)
#             (out_lin): Linear(in_features=768, out_features=768, bias=True)
#           )
#           (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#           (ffn): FFN(
#             (dropout): Dropout(p=0.1, inplace=False)
#             (lin1): Linear(in_features=768, out_features=3072, bias=True)
#             (lin2): Linear(in_features=3072, out_features=768, bias=True)
#             (activation): GELUActivation()
#           )
#           (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#         )
#       )
#     )
#   )
#   (pre_classifier): Linear(in_features=768, out_features=768, bias=True)
#   (classifier): Linear(in_features=768, out_features=2, bias=True)
#   (dropout): Dropout(p=0.2, inplace=False)
# )
```

显然，该模型包含6个含线性层的 transformer 模块—— (0-5): 6 x TransformerBlock( )，此外，该模型还有两个线性输出层—— pre_classifier、classifier。

可以定义下面的赋值函数和循环，有选择地为这些线性层启用 LoRA，下面的代码中只有 lora_query 和 lora_value 为 True，说明只有 $q$ 和 $v$ 矩阵启用了 LoRA，其他矩阵参数都被冻结。在相同可训练参数下，同时调整 $q$ 和 $v$ 矩阵的效果最好。


```python

from functools import partial

# default hyperparameter choices
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False

layers = []

assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)

for layer in model.distilbert.transformer.layer:
    if lora_query:
        layer.attention.q_lin = assign_lora(layer.attention.q_lin)
    if lora_key:
        layer.attention.k_lin = assign_lora(layer.attention.k_lin)
    if lora_value:
        layer.attention.v_lin = assign_lora(layer.attention.v_lin)
    if lora_projection:
        layer.attention.out_lin = assign_lora(layer.attention.out_lin)
    if lora_mlp:
        layer.ffn.lin1 = assign_lora(layer.ffn.lin1)
        layer.ffn.lin2 = assign_lora(layer.ffn.lin2)
if lora_head:
    model.pre_classifier = assign_lora(model.pre_classifier)
    model.classifier = assign_lora(model.classifier)

print(model)
```

接下来训练 DistilBERT 模型，若要在训练期间只更新最后两层，则冻结 transformer 模块中的线性层:


```python
# freeze all
for param in model.parameters():
    param.requires_grad = False

# unfreeze output layers
for param in model.pre_classifier.parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True
```

**补充**

准确率：用 LoRA 微调（最优参数）所有层 > 微调所有层的所有参数 > 用 LoRA 微调最后两层（默认参数） > 微调最后两层的所有参数

该例中 LoRA 最优参数为（网格搜索获得）：
- lora_r: 8
- lora_alpha: 1
- lora_query: True
- lora_key: False
- lora_value: True
- lora_projection: False
- lora_mlp: True
- lora_head: False

详见：https://lightning.ai/lightning-ai/studios/code-lora-from-scratch

