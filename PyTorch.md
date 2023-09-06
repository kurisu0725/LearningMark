# PyTorch

## 数据加载

### Dataset

### DataLoader

## 图像变换Transform

### TensorBoard

### Transform

## 神经网络

### torch.nn.functional.Conv2D

```Python
def conv2d(input: Tensor,
           weight: Tensor,
           bias: Tensor | None = None,
           stride: int | Size | List[int] | Tuple[int, ...] = 1,
           padding: str = "valid",
           dilation: int | Size | List[int] | Tuple[int, ...] = 1,
           groups: int = 1) -> Tensor
```

- **input** – input tensor of shape (minibatch,in_channels,*H*,*W*)
- **weight**- 卷积核
- **stride**-卷积核移动的步长
- **padding**-填充大小

### 卷积层 Convolution Layers

### 最大池化层 下采样

降低数据维度，减少运算量

### 非线性激活

### Loss 损失函数

#### torch.nn.L1Loss

#### torch.nn.MES 均方差

#### torch.nn.CrossEntropyLoss 交叉熵

### Optimizer 优化器 

### 一些函数

#### torch.flatten

#### torch.nn.Sequential

将定义的神经网络写在Sequential中，forward直接调用即可。

```Python
class Kurisu(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x
```

