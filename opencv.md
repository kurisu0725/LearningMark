## OpenCV

### 图片

#### HSV

HSV 即使用**色相（Hue）**、**饱和度（Saturation）**、**明度（Value）**来表示色彩的一种方式，是一种在人们生活中常用的颜色系统，因为它符合人们描述颜色的方式——是什么颜色、颜色有多深、颜色有多亮。

#### **图片读取**

```cv.imread(filename, flags) -> retval``` 

```flags```为枚举类用来表示读取方式,如```cv.IMREAD_COLOR,cv.IMREAD_GRAYSCALE```。

以灰度图读取：```cv.imread(r'E:\wallpaper\kurisu\kurisu.png', cv.IMREAD_GRAYSCALE)```

**中文路径：**

```Python
def cv_imread(filePath):
    # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
    cv_img=cv.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img
```



#### **显示**

```cv.imshow(winname, mat) -> None```

样例：

```python
# editor: kurisu
# edit time: 2023/8/8 21:57
import cv2 as cv


def get_image_info(image):
    print(type(image))      # <class 'numpy.ndarray'>
    print(image.shape)      # 高度 宽度  通道数
    print(image.size)       # 像素大小
    print(image.dtype)      # 数据类型


src = cv.imread(r'E:\wallpaper\kurisu\kurisu.png')
cv.imshow("input image", src)
get_image_info(src)
cv.waitKey(0)
cv.destroyAllWindows()
```

结果：

![image-20230810163511342](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230810163511342.png)

![image-20230810163533422](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230810163533422.png)

```cv.waitKey(val)``` 等待```val```毫秒，若按键按下则继续执行程序，返回值为按下字符的ASCII码。设置为0可以让图片一直显示。

#### 写入图像

```Python
cv.imwrite(	filename, img[, params]	) -> retval
```

#### 色彩空间转换 cvtColor

样例:

```Python
import cv2 as cv

def color_space_transform(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.namedWindow('gray', cv.WINDOW_NORMAL)
    cv.imshow('gray', gray)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.namedWindow('hsv', cv.WINDOW_NORMAL)
    cv.imshow('hsv', hsv)

    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    cv.namedWindow('hls', cv.WINDOW_NORMAL)
    cv.imshow('hls', hls)

    YCrCb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    cv.namedWindow('YCrCb', YCrCb)
    cv.imshow('YCrCb', YCrCb)

    yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    cv.namedWindow('yuv', yuv)
    cv.imshow('yuv', yuv)

src = cv.imread('./test.png')
src = cv.resize(src, None, fx = 0.5, fy = 0.5)
cv.namedWindow('src', cv.WINDOW_NORMAL)
cv.imshow('src', src)
color_space_transform(src)
cv.waitKey(0)
cv.destroyAllWindows()
```

结果：

![image-20230810192222033](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230810192222033.png)

#### 追踪特定颜色 inRange

```cv2.inRange(hsv, lowerb, upperb)```

样例：

![输入](D:\Pycharm_Workplace\opencv\csdn opencv\图片\test.png)

```Python
# editor: kurisu
# edit time: 2023/8/10 19:30
# 追踪特定颜色

import cv2 as cv
import numpy as np

def tracking_colors(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # 追踪绿色
    # lower_hsv = np.array([35, 43, 46])
    # upper_hsv = np.array([77, 255, 255])
    # 追踪蓝色
    lower_hsv = np.array([100, 43, 46])
    upper_hsv = np.array([124, 255, 255])
    mask = cv.inRange(hsv, lowerb= lower_hsv, upperb= upper_hsv)
    cv.namedWindow('mask', cv.WINDOW_NORMAL)
    cv.imshow('mask', mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

src = cv.imread('./test.png')
tracking_colors(src)
```

##### 怎样找到追踪对象的HSV值？BGR2HSV

```python
# editor: kurisu
# edit time: 2023/8/18 19:00
import cv2 as cv
import numpy as np

green = np.uint8([[[0, 255, 0]]])
hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
print(hsv_green)
```

**注意：** src为三维的ndarray。





#### 像素运算

加减乘除、与非或。都是每个像素每个通道的运算。

##### 调节图片对比度和亮度 addWeighted

样例：

```Python
# editor: kurisu
# edit time: 2023/8/10 19:44
import cv2 as cv
import numpy as np

def adjust_brightness_image(image, c, b):
    height, width, channels = image.shape
    blank = np.zeros([height, width, channels], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1 - c, b)
    cv.namedWindow('adjust_contrast_brightness', cv.WINDOW_NORMAL)
    cv.imshow("adjust_contrast_brightness", dst)

def cv_imread(filePath):
    # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
    cv_img=cv.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

src = cv_imread(r'E:\wallpaper\bronya\arkt1ka\Hot pink横屏20230617_180115.654.bmp')
cv.namedWindow('first', cv.WINDOW_NORMAL)
cv.imshow("first", src)
adjust_brightness_image(src, 1.2, 100)
cv.waitKey(0)
cv.destroyAllWindows()
```

![image-20230810195718797](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230810195718797.png)

```cv.addWeighted()```是图像混合函数，将第一张和第二张图像的像素以一定的权重求和。
$$
dst = src1 * alpha + src2 * beta+gamma
$$
样例中```adjust_brightness_image(image, c, b)```函数的c 即是权重，b是gamma，b越大说明像素值越接近255（白色），所以**亮度**会增加，看起来更亮。

**对比度**即：图像暗和亮的落差值，即图像最大灰度级和最小灰度级之间的差值。

将gamma设置为0，可以看到：

![image-20230810200821848](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230810200821848.png)

两边图像差别不大，但是看起来右边的脸部更明显。因为src2用的是像素值为0的黑色图像矩阵，所以 $dst = src1 * alpha + 0 + 0$ ，此时相当于对第一张图像的像素进行了乘法，如果像素值为0 -> 0,  200 -> 200 * 1.2 = 240，那么黑色的就会保持黑色，正值就倾向于白色，颜色看起来就会明显清晰。

#### 伪色彩增强 applyColorMap

对于颜色映射，Opencv这样写到：```The human perception isn't built for observing fine changes in grayscale images. Human eyes are more sensitive to observing changes between colors, so you often need to recolor your grayscale images to get a clue about them. ```

人们对灰度图的变化是不如对彩色图的变化的，所以通过色彩映射将灰度图片映射为彩色图。

```python
cv.applyColorMap(	src, colormap[, dst]	) ->	dst
cv.applyColorMap(	src, userColor[, dst]	) ->	dst
```

opencv提供了一些```colormap```以供使用，如```cv.COLORMAP_JET，cv.COLORMAP_AUTUMN```。

#### 图像通道分离合并 split，merge

```cv.split()和cv.merge()``` 分别是对图像通道的分离和合并。

另外```cv.mixChannels( src, dst, fromTo ) -> dst``` , **split**和**merge**函数都是**mixChannels**的子应用。

| src        | 矩阵数组，即可以是多个图像                            |
| ---------- | ----------------------------------------------------- |
| **dst**    | **同src**                                             |
| **fromTo** | **src->dst通道的映射pair，每两个index表示一个pair。** |

**文档说明**：

![image-20230812201245577](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230812201245577.png)

#### 归一化 normalize

```python
cv.normalize(	src, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]	) ->	dst

```

当```norm_type == NORM_MINMAX``` 时：
$$
MINMAX:\:dst(i,j) = \frac{(src(i,j) - min(src)) * (b^{'} - a^{'}) }{max(src(i,j)) - min(src(i,j))} +a^{'}
$$
$b^{'} = max(alpha, \:beta);\:a^{'}=min(alpha, beta)$

当```norm_type == NORM_INF```时，
$$
INF:\:dst(i,j) = \frac{src(i,j)}{Max\|{src(i,j)}\|}
$$
当```norm_type == NORM_L1```时，
$$
L1:\:dst(i,j) = \frac{src(i,j)}{\\|\sum{src(x,y)} \\|}
$$
当```norm_type == NORM_L2```时，
$$
L2:\:dst(i,j) = \frac{src(i,j)}{\sqrt{\sum{src(x,y)^{2}}}}
$$
样例:

```python
# editor: kurisu
# edit time: 2023/8/12 20:35
import cv2 as cv
import numpy as np

src = cv.imread('../test.png')
cv.namedWindow('input', cv.WINDOW_AUTOSIZE)
cv.imshow('input', src)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# 转换为浮点数
gray = np.float32(gray)
print(gray)

# scale and shift by NORM_MINMAX
dst = np.zeros(gray.shape, dtype=np.float32)
cv.normalize(gray, dst= dst, alpha=0, beta=1.0, norm_type=cv.NORM_MINMAX)
print(dst)
cv.imshow('NORM_MINMAX', np.uint8(dst*255))

# scale and shift by NORM_INF
dst = np.zeros(gray.shape, dtype=np.float32)
cv.normalize(gray, dst=dst, alpha=1.0, beta=0, norm_type=cv.NORM_INF)
print(dst)
cv.imshow('NORM_INF', np.uint8(dst*255))

# scale and shift by NORM_L1
dst = np.zeros(gray.shape, dtype=np.float32)
cv.normalize(gray, dst= dst, alpha= 1.0, beta= 0, norm_type=cv.NORM_L1)
print(dst)
cv.imshow('NORM_L1', np.uint8(dst * 1e7))

# scale and shift by NORM_L2
dst = np.zeros(gray.shape, dtype=np.float32)
cv.normalize(gray, dst= dst, alpha= 1.0, beta= 0, norm_type=cv.NORM_L2)
print(dst)
cv.imshow('NORM_L2', np.uint8(dst * 1e4))

cv.waitKey(0)
cv.destroyAllWindows()
```

#### 绘图函数

涉及参数：

- **img**：在上边绘制的图像
- **color**：颜色，传入元组，如（255,0,0）
- **thickness**: 线条的粗细。设置为-1，那么图像会被填充
- **linetype**: 线条的类型。8连接、抗锯齿等。默认8连接。

##### 画线和矩形

```python
def line(img: Any,
         pt1: Any,
         pt2: Any,
         color: Any,
         thickness: Any = None,
         lineType: Any = None,
         shift: Any = None) -> None
```

起点pt1, 终点pt2

```python
def rectangle(img: Any,
              pt1: Any,
              pt2: Any,
              color: Any,
              thickness: Any = None,
              lineType: Any = None,
              shift: Any = None) -> None
```

pt1为左上角顶点，pt2为右下角顶点

##### 画圆、椭圆

```python
# 圆
def circle(img: Any,
           center: Any,
           radius: Any,
           color: Any,
           thickness: Any = None,
           lineType: Any = None,
           shift: Any = None) -> None
# 椭圆
def ellipse(img: Any,
            center: Any,
            axes: Any,
            angle: Any,
            startAngle: Any,
            endAngle: Any,
            color: Any,
            thickness: Any = None,
            lineType: Any = None,
            shift: Any = None) -> None
```

```axes```为长轴和短轴的长度,```angle```为沿逆时针旋转的角度，```startAngle/endAngle```分别为椭圆弧沿顺时针方向的起始和结束角度。

![image-20230814144937125](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230814144937125.png)

##### 画多边形

```python
def polylines(img: Any,
              pts: Any,
              isClosed: Any,
              color: Any,
              thickness: Any = None,
              lineType: Any = None,
              shift: Any = None) -> None
```

```polylines``` 函数可以用来绘制多条曲线，pts是多条直线的列表，参数需要加中括号。

```isClosed``` 控制图形闭合，为False时图形首尾不连接。

##### 显示文字

```Python
def putText(img: Any,
            text: Any,
            org: Any,
            fontFace: Any,
            fontScale: Any,
            color: Any,
            thickness: Any = None,
            lineType: Any = None,
            bottomLeftOrigin: Any = None) -> None
```

- **org**: 文本左下角的坐标
- **fontFace**：字体
- **fontScale**: 字号较基础大小缩放的比例

#### **齐次坐标**

[什么是齐次坐标? - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/258437902)



#### 图像几何变换

几何变换是指对对图像的位置、大小、形状、投影进行变换，是将图像从原始平面投影到新的视平面。

##### 图像平移

```Python
def warpAffine(src: Any,
               M: Any,
               dsize: Any,
               dst: Any = None,
               flags: Any = None,
               borderMode: Any = None,
               borderValue: Any = None) -> None
def warpPerspective()
```

```dsize```不会更改图像的大小，而是更改图像显示区域的大小。

**注意：** dsize填写的行列需要调换，因为图像shape中height, width 分别为竖直和水平方向，而图像坐标中（x,y) 是 水平和竖直方向。

其中 ```M``` 为 2x3的矩阵，目标图像将会进行如下变换:
$$
dst(x,y) = src(M_{11}x \:+M_{12}y+\:M_{13},\quad M_{21}x\:+
M_{22}y\:+M_{23})
$$
例如：```M = np.float32([1, 0, 50], [0, 1, 25])``` 这样会让图像在x轴方向上移动50，y轴方向移动25的距离。

**注意**：图像的原点在左上角，y轴向下，x轴向右。```warpPerspective```函数的$M$ 矩阵为 3*3的矩阵。

##### 图像翻转

用仿射变换实现图像翻转，得出变换矩阵M。
$$
水平翻转:\quad x^{'} = -x + w
$$

$$
竖直翻转:\quad y^{'} = -y + h
$$

$$
同时翻转:\quad x^{'} = -x + w，y^{'} = -y + h
$$

对应的变换矩阵M：
$$
水平：M = \left[  
\begin{matrix}
-1 & 0 & w \\
0 & 1 & 0\\
\end{matrix}
\right] \\
竖直: M = \left[  
\begin{matrix}
1 & 0 & 0 \\
0 & -1 & h\\
\end{matrix}
\right] \\
同时翻转: M =  \left[  
\begin{matrix}
-1 & 0 & w \\
0 & -1 & h\\
\end{matrix}
\right]
$$
示例：

```python
# editor: kurisu
# edit time: 2023/8/15 19:25
import cv2
import numpy as np

def flip_img(horizontal_flip, vertical_flip, img):
    h, w = img.shape[:2]
    M = np.array([ [1,0,0], [0, 1, 0]], dtype= np.float)
    if horizontal_flip:
        M[0] = [-1, 0, w]
    if vertical_flip:
        M[1] = [0, -1, h]
    img_flip = cv2.warpAffine(img, M, (w, h))
    cv2.imshow('after', img_flip)
img = cv2.imread('test.png')
cv2.imshow('origin', img)
flip_img(True, True, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```



##### 图像缩放  插值、resize

使用仿射变换：
$$
x^{'} = f_x * x \\
y^{'} = f_y * y
$$
即：
$$
\left[ 
\begin{matrix}
x^{'} \\
y^{'}
\end{matrix}
\right]  = \left[ \begin{matrix}
f_x & 0 & 0 \\
0 & f_y & 0
\end{matrix}
\right] \left[\begin{matrix} x \\ y \\ 1\end{matrix}\right]
= M \left[ \begin{matrix} x \\y \\ 1 \end{matrix} \right]
$$
通过上边的变换矩阵即可实现对图片的缩放。

示例：

```Python
# editor: kurisu
# edit time: 2023/8/15 19:34
import cv2 as cv
import numpy as np

def img_scale(fx, fy, img):
    h, w = img.shape[:2]
    M = np.array([[fx, 0, 0], [0, fy, 0]], dtype=np.float)
    img_scaled = cv.warpAffine(img, M, (int(w*fx), int(h*fy)))
    cv.imshow('after', img_scaled)


img = cv.imread('test.png')
cv.imshow('img', img)
img_scale(0.5, 2, img)
cv.waitKey(0)
cv.destroyAllWindows()
```



对于图像的放大和缩小，图像必然发生变化，需要插值来补全像素。

##### **插值算法**

参考[来聊聊图像插值算法（线性插值与非线性插值） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/141681355)

1. 最近邻插值 INTER_NEAREST
   $$
   \begin{cases}
   src_x = \frac{ dst_x * src_w}{dst_w} \\
   src_y = \frac{dst_y * src_x} {dst_x} \\
   \end{cases}
   $$
   根据目标图像dst的坐标($dst_x, dst_y$) ，得到其在原图像中对应的坐标，选择距离最近的点。

2. 双线性插值 INTER_LINEAR

   双线性插值，在x方向上进行了2次插值，又在y方向上进行了1次插值，一共3次插值。

   ![image-20230823194021066](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230823194021066.png)

   x方向的插值：
   $$
   f(x, y_0) = \frac{x_1 - x}{x_1 - x_0}f(x_0,y_0) + \frac{x -x_0}{x_1-x_0}f(x_1, y_0) \\
   f(x,y_1)= \frac{y_1-y}{y_1-y_0}f(x,y_0) + \frac{y-y_0}{y_1-y_0}f(x,y_1)
   $$
   y方向的插值:
   $$
   f(x,y) = \frac{y_1-y}{y_1-y_0}f(x,y_0) + \frac{y-y_0}{y_1-y_0}f(x,y_1)
   $$
   综合可得：
   $$
   f(x,y) = \frac{(y_1-y)(x_1-x)}{(y_1-y_0)(x_1-x_0)}f(x_0,y_0)+
   \\ \frac{(y_1-y)(x-x_0)}{(y_1-y_0)(x_1-x_0)}f(x_1,y_0)+\\
   \frac{(y-y_0)(x_1-x)}{(y_1-y_0)(x_1-x_0)}f(x_0,y_1)+ \\
   \frac{(y-y_0)(x-x_0)}{(y_1-y_0)(x_1-x_0)}f(x_1,y_1)
   $$
   结合实际，我们令
   $$
   (x, y) = (i + u, j + v)
   $$
   i,j 为坐标的整数部分， u,v为小数部分。

   而在图像中取得4个点是相邻的，也就是有：
   $$
   x_1 = x_0 + 1, y_1 = y_0 + 1
   $$
   上式的**f(x,y)** 化简为：
   $$
   f(i+u,j+v) = (1-u)(1-v)\cdot f(i,j)+(1-u)v\cdot f(i,j+1) \\
   + u(1-v)\cdot f(i+1,j) + uv\cdot f(i+1, j+1)
   $$
   **几何中心对齐：**

   图像插值缩放后，原图和目标图像的对应关系不是关于中心对称的，而是以(0,0)为图像的左上角分布。这时候需要移动中心点。
   $$
   src_x = dst_x \cdot \frac{src_w}{dst_w} + 0.5(\frac{src_w}{dst_w} - 1) \\
   src_y = dst_y \cdot \frac{src_h}{dst_h} + 0.5(\frac{src_h}{dst_h}-1)
   $$
   
   $$
   
   $$

3. 双三次插值 INTER_BICUBIC

   

##### 图像旋转

![二维平面上点围绕原点旋转](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230815194246744.png)

二维平面上点围绕原点旋转。

将坐标点用极坐标表示可以得到：
$$
v(r\:cos\phi, r\:sin\phi), v^{'}(r\:cos(\theta+\phi),r\:sin(\theta+\phi))\\
对于v点来说:\\
x = r\:cos\phi\\
y = r\:sin\phi\\
而对于v^{'}来说:
x^{'} = r\:cos(\theta+\phi)
\\ = r\:cos\theta \:* cos\phi - r\:sin\theta \: * sin\phi\\
y^{'} =  r\:sin(\theta+\phi)
\\ = r\:sin\theta \:* cos\phi + r\:cos\theta \: * sin\phi\\
再将x、y代入上式：\\
x^{'} = x * cos\theta - y * sin\theta\\
y^{'} = x * sin\theta + y * cos\theta
$$
用矩阵M表示，可得：
$$
\left[ 
\begin{matrix}
x^{'} \\
y^{'}
\end{matrix}
\right]  = \left[ \begin{matrix}
cos\theta & -sin\theta & 0 \\
sin\theta & cos\theta & 0
\end{matrix}
\right] \left[\begin{matrix} x \\ y \\ 1\end{matrix}\right]
= M \left[ \begin{matrix} x \\y \\ 1 \end{matrix} \right]
$$

$$
M = \left[ \begin{matrix}
cos\theta & -sin\theta & 0 \\
sin\theta & cos\theta & 0
\end{matrix}
\right]
$$

**注意：** 此时建立直角坐标系是以**左下角为原点**的，而图像是以**左上角为原点**建立的。所以此时计算的M为以左下角为

当以**左上角为原点**时：
$$
设P(x,y)为坐标系上的点，与x的夹角为\alpha,\\而P^{'}(x^{'},y^{'})为P点逆时针旋转\theta角度的点。假设其到原点的距离为1，即r=1\\
则有:\\
\begin{cases}
x = cos\alpha \\
y = sin\alpha \\
\end{cases}
\\
\begin{cases}
x^{'} = cos(\alpha - \theta) = cos\alpha * cos\theta + sin\alpha *sin\theta = x * cos\theta + y * sin\theta \\
y^{'} = sin(\alpha - \theta) = sin\alpha * cos\theta - cos\alpha *sin\theta = -x * sin\theta + y * cos\theta
\end{cases}
$$
得到M：
$$
M = \left[ \begin{matrix}
cos\theta & sin\theta & 0 \\
-sin\theta & cos\theta & 0
\end{matrix}
\right]
$$
**绕任意点旋转**：

步骤：首先将轴心(x,y)移动到原点，然后做旋转缩放变换，最后再将图像的左上角转换为原点。
$$
围绕左下角原点旋转坐标的变换公式:\\
\begin{cases}
x^{'} = x * cos\theta - y * sin\theta\\
y^{'} = x * sin\theta + y * cos\theta
\end{cases} \\
现在改为绕点C(a,b)旋转，先坐标平移到原点，然后进行旋转，再平移回去。\\
\begin{cases}
x^{'} = (x-a) * cos\theta - (y-b) * sin\theta + a\\
y^{'} = (x-a) * sin\theta + (y-b) * cos\theta + b
\end{cases} \\
展开得：\\
\begin{cases}
x^{'} = x * cos\theta - y * sin\theta + (1-cos\theta) *a + sin\theta * b\\
y^{'} = x * sin\theta + y * cos\theta + (1-cos\theta) * b-sin\theta * a 
\end{cases} \\
$$
由于坐标系不同，左上角为原点的变换矩阵M：
$$
M = \left[
\begin{matrix}
cos\theta & sin\theta & (1 - cos\theta) * a - b * sin\theta \\
-sin\theta & cos\theta & (1-cos\theta) * b + a * sin\theta
\end{matrix}
\right]
$$

在实际应用中，可以使用```getRotationMatrix2D``` 函数来得到$M$ 矩阵，

##### 仿射变换 Affine Transform

仿射变换（Affine）的特点是原始图像中的平行关系和线段长度比例关系保持不变。从原图src的点对变换到目标dst的点对需要3个点对来计算$M$ 变换矩阵。
$$
\left[ 
\begin{matrix}
x^{'} \\
y^{'} \\
1
\end{matrix}
\right] = 
\left[ 
\begin{matrix}
a_{11} & a_{12} & b_{1} \\
a_{21} & a_{22} & b_{2}  \\
0 & 0 & 1 \\
\end{matrix}
\right] \cdot
\left[ 
\begin{matrix}
x \\
y \\
1
\end{matrix}
\right]
$$


```Python
def getAffineTransform(src: Any,
                       dst: Any) -> None
```

样例：
```Python
# editor: kurisu
# edit time: 2023/8/18 19:30
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('./test.png')
rows, cols, ch = img.shape

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

M = cv.getAffineTransform(pts1, pts2)

dst = cv.warpAffine(img, M, (cols, rows))

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()
```

![image-20230818193716932](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230818193716932.png)

##### 透视变换（Perspective Transformation）/投影映射 (Projective Mapping)

​    现实生活中的空间是三维的，图像中的物体存在近大远小的特征，这种畸变仿射变换不能矫正。因此，我们需要使用到三维空间的变化，就是透视变换。本质是将图像投影到一个**新的视平面**。

$$
\left[
\begin{matrix}
a_0 & a_1 & b_0 \\
a_2 & a_3 & b_1 \\
c_0 & c_1 & 1\\
\end{matrix}
\right] 
\left[
\begin{matrix}
x \\
y \\
1 \\
\end{matrix}
\right
] = 
\left[
\begin{matrix}
X \\
Y \\
Z \\
\end{matrix}
\right
]
$$
此时$Z = c_0x+c_1y+1$ .可以视为处在h = Z 的齐次坐标(X,Y,Z) 映射 到 h = 1的齐次坐标($x^{'},y^{'},z^{'} = 1$)。
$$
\begin{cases}
X^{'} = \frac{X}{Z} \\
Y^{'} = \frac{Y}{Z}\\
Z^{'} = \frac{Z}{Z}
\end{cases}
\rightarrow
\begin{cases}
X^{'} = \frac{a_0x+a_1y+b_0}{c_0x+c_1y+1} \\
Y^{'} = \frac{a_2x+a_3y+b_1}{c_0x+c_1y+1}\\
Z^{'} = 1
\end{cases}
$$
由于变换矩阵中存在8个参数，所以需要至少4个坐标点来解方程。



#### ROI (Region of Interest)



#### 直方图

灰度直方图和彩色图像的直方图绘制：

```Python
# editor: kurisu
# edit time: 2023/8/16 18:33
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def custom_hist(gray):
    h, w = gray.shape
    hist = np.zeros([256], dtype=np.int32)
    for row in range(h):
        for col in range(w):
            pv = gray[row, col]
            hist[pv] += 1
    y_pos = np.arange(0, 256, 1, dtype=np.int32)
    plt.bar(y_pos, hist, align='center', color='r', alpha=0.5)
    plt.xticks(y_pos, y_pos)
    plt.ylabel('Frequency')
    plt.title('Histogram')

    plt.show()

def image_hist(image):
    cv.imshow('input', image)
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()

src = cv.imread('../test.png')
cv.namedWindow('input', cv.WINDOW_AUTOSIZE)
# gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.imshow('input', src)
# custom_hist(gray)
image_hist(src)
cv.waitKey(0)
cv.destroyAllWindows()
```

**直方图比较：**

直方图比较用来衡量两张图之间的相似程度。

- 相关性比较 Correlation

  cv.HISTCMP_CORREL

- 卡方比较 Chi-square

- 十字交叉性 Intersection

- 巴氏距离 Bhattacharyya distance

#### 直方图均衡化



**直方图反向投射 BackProjection**

#### 霍夫变换 Hough Transform

霍夫变换（Hough Transform），是图像处理领域内**从图像中检测几何形状**的基本方法之一。经典的霍夫变换用来检测图像中的直线，后来经过扩展可以进行任意形状物体的识别，如圆和椭圆。

笛卡尔坐标系（x-y轴)和霍夫坐标系（k-b轴) 即对于直线$y = k*x+ b$ 转为$b = -x*k+y$。

**Rule**: 

1. 对于笛卡尔坐标系中的一条直线$y=k_{0}*x+b_0$，对应霍夫坐标系中的一个点 $(k_{0}, b_0)$ 。

2. 而对于笛卡尔坐标系中的一个顶点$(x_0,y_0)$, 对应霍夫坐标系中的一条直线$b =-x_0*k+y_0$。

3. 由**Rule(2)**可知笛卡尔空间的两个点会映射为霍夫空间内两条相交于$(k_1,b_1)$的直线，笛卡尔空间中这两个点对应的直线会映射为霍夫空间的点$(k1, b1)$。
   $$
   推导: \\
   假设笛卡尔坐标系有不重复的两点A,B,对应的直线斜率为k_1,截距b_1 \\笛卡尔：A(x_0,y_0), B(x_1,y_1),满足 y0=k_1*x+b_1 \\
   则\\霍夫：A -> 直线 b = -x_0 * k + y_0 \\
   B -> 直线 b = -x_1 * k + y_1 \\
   求交点： \\
   0 = (-x_0+x_1) * k + (y_0 - y1) \\
   k = \frac{y_1 - y_0}{x_1 - x_0} = k_1 \\
   b = b_1 \\
   即霍夫坐标系下两点AB对应的直线相交于(k_1, b_1)
   $$
   
4. 综上所述，在霍夫空间内，经过一个点的直线越多，说明其在笛卡儿空间内映射的直线，是由越多的点所构成(穿过)的。而若存在错误的点，就会影响直线的判断。也就是说，如果一条直线是由越多点所构成的，那么它实际的可能性就越高。

**因此，霍夫变换的基本思路是：选择有尽可能多直线交汇的点。**

但在笛卡尔空间中，存在垂线$x=x_0$，无法映射到霍夫空间。

于是用**极坐标系**来映射霍夫坐标系，同样的有：

极坐标系中的一点对应霍夫坐标系的一条曲线。

极坐标系的一条线映射为霍夫坐标系的一点。

**注意**：极坐标表示直线时的$(\rho,\theta)$ ，实际是直线的法线。

编程实现的话，考虑图像像素是离散的，都是点，那么枚举穿过点的直线的角度$\theta$ (0~180)，统计$\rho$的次数 即为统计霍夫坐标系对应。

```Python
def HoughLines(image: Any,
               rho: Any,   枚举半径的步长
               theta: Any, 枚举角度的步长
               threshold: Any,  投票计数的阈值，判定直线
               lines: Any = None,
               srn: Any = None,
               stn: Any = None,
               min_theta: Any = None,
               max_theta: Any = None) -> None
```

样例：

```Python
# editor: kurisu
# edit time: 2023/8/18 20:08
import cv2 as cv
import numpy as np


def line_detect(picture):
    print('start')
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow('image', picture)
    # 二值化
    gray = cv.cvtColor(picture, cv.COLOR_BGR2GRAY)
    # 高斯滤波
    gaussian = cv.GaussianBlur(gray, (9, 9), 0)
    # 边缘检测
    edges = cv.Canny(gaussian, 50, 150)

    lines = cv.HoughLines(edges, 1, np.pi / 180, 100)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        len = 3000
        x1 = int(x0 + len * (-b))
        y1 = int(y0 + len * a)
        x2 = int(x0 - len * (-b))
        y2 = int(y0 - len * a)
        cv.line(picture, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.namedWindow('lines result', cv.WINDOW_NORMAL)
    cv.imshow('lines result', picture)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    img = cv.imread('photo.jpg')
    line_detect(img)
```

对于```x1 = int(x0 + len * (-b)), y1 = int(y0 + len * a)``` 的解释如下：

$(x_0,y_0) = (cos\theta * \rho,sin\theta * \rho)$ ，对应的是直线与其法线的交点。

而令$a = cos\theta, b = sin\theta$ , 有：
$$
\rho = x cos\theta + ysin\theta \rightarrow y = -\frac{cos\theta}{sin\theta}x + \frac{\rho}{sin\theta} \\
y = -\frac{a}{b}x + \frac{\rho}{b}
$$
斜率$k=-\frac{a}{b}$ ，简单来说，即可以视为 x 每增加1，y增加$-\frac{a}{b}$ ; 都换算为整数即为 x 每增加 -b, y 增加 a。



##### 霍夫圆变换

设圆：$(x-a)^{2} + (y - b) ^ {2} = r^{2}$

可以将圆表示为：
$$
\begin{cases}
x = a + r\:cos\theta \\
y = b + r\:sin\theta \\
\end{cases} \rightarrow
\begin{cases}
a = x - r\:cos\theta\\
b = y - r\:sin\theta
\end{cases}
$$
所以在**(a, b, r)** 构成的三维坐标系中，**一个点可以表示一个圆**。

在笛卡尔坐标系中，一个圆上的所有点在霍夫空间中变成了无数条三维曲线，且它们都交于一点。

```Python
def HoughCircles(image: Any,
                 method: Any,   使用的霍夫变换方法
                 dp: Any,  检测圆心的累加器精度和图像精度比的倒数，dp=2时累加器是输入图像一半大的宽高
                 minDist: Any, 圆心与已检测出的圆的圆心之间的最小距离
                 circles: Any = None, 输出的圆表示为(x, y, radius)
                 param1: Any = None,  
                 param2: Any = None,
                 minRadius: Any = None,  最小圆半径
                 maxRadius: Any = None) -> None  最大圆半径
```

对于```param1``` ，若```method=cv.HOUGH_GRADIENT``` 其参数为Canny边缘检测的**高阈值**，低阈值为高阈值的一半。对于```param2```,它表示在检测阶段圆心的累加器阈值。它越小，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了。

```Python
def circle_detect(picture):
    gray = cv.cvtColor(picture, cv.COLOR_BGR2GRAY)
    gaussian = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(gaussian, 50, 150)
    cv.namedWindow('edge', cv.WINDOW_AUTOSIZE)
    cv.imshow('edge', edges)
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 20, param1=100, param2=45, minRadius=20, maxRadius=200)
    circles = np.uint16(np.around(circles))
    print(circles.shape)
    print(circles)
    for circles in circles[0]:
        cv.circle(picture, (circles[0], circles[1]), circles[2], (0, 0, 255), 2)
        cv.circle(picture, (circles[0], circles[1]), 2, (0, 255, 0), 2)
    cv.namedWindow('circles result', cv.WINDOW_AUTOSIZE)
    cv.imshow('circles result', picture)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    img = cv.imread('../test.png')
    # line_detect(img)
    circle_detect(img)
```

边缘检测的结果：

![image-20230819222349091](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230819222349091.png)

画出圆的结果图：

![image-20230819222452217](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230819222452217.png)

**注意**cv.HOUGH_GRADIENT方法下的HoughCircles的返回是三维的circles， 第一维恒为1：

```
(1, 8, 3)
[[[416 340 105]
  [178 340 106]
  [298 132 105]
  [404 522  39]
  [114 524  41]
  [260 528  32]
  [198 522  27]
  [334 518  23]]]
```

#### 阈值

##### 简单阈值

cv.THRESH_BINARY
cv.THRESH_BINARY_INV
cv.THRESH_TRUNC （Truncate 高于阈值设为阈值val，低于不变
cv.THRESH_TOZERO  小于阈值灰度设为0，高于不变
cv.THRESH_TOZERO_INV  大于阈值设为0，低于不变

##### 自适应阈值 ？

自适应阈值是局部阈值，当图像上的不同部分的具有不同亮度时，此时阈值是根据像素的小区域计算与其对应的阈值。

```Python
def adaptiveThreshold(src: Any,
                      maxValue: Any,   灰度上限
                      adaptiveMethod: Any, 阈值计算类型,mean/gaussian
                      thresholdType: Any,  二值化的类型，取反
                      blockSize: Any,   领域的大小
                      C: Any,   阈值最终减去常数C
                      dst: Any = None) -> None
```

##### Otsu's二值化

全局阈值法。也叫**最大类间方差法**。

设**MxN**的图像有**L**个不同的灰度级。

$n_i$ 表示灰度级为**i**的像素个数，$p_i$表示归一化的直方图的分量（所占的比例）

$p_i = \frac{n_i}{MN}$ ,  $\sum_{i =0}^{L-1}p_i = 1$

选择一个阈值**T(k)**, 将图像灰度分为两类$C_1, C_2$。

类$C_1$ ，灰度值**[0, k]**

像素被分到$C_1$的概率：
$$
P_1(k) = \sum_{i = 0}^{k} p_i \\
分配到C_1的像素平均灰度值： \\
m_1(k) = \sum_{i =0}^{k} i \cdot P(i|C_1) \\
 = \sum_{i = 0}^{k} i \cdot P(C_1|i) P(i) /P(C_1) (贝叶斯公式)\\

 = \frac{\sum_{i=0}^{k}i \cdot p_i}{P_1(k)}
$$
类$C_2$, 灰度值**[k+1, L-1]**

$C_2$发生的概率为$P_2(k)$
$$
P_2(k) = \sum_{i=k+1}^{L-1}p_i = 1 - P_1(k)
$$
分配到$C_2$的像素平均灰度值:
$$
m_2(k) = \sum_{i=k+1}^{L-1}i \cdot P(i|C_2) \\
= \frac{\sum_{i=k+1}^{L-1}i \cdot p_i}{P_2(k)}
$$
整幅图像的平均灰度为：
$$
m_G = \sum_{i=0}^{L-1}i\cdot p_i
$$
$P_1, P_2, m_1,m_2$ 满足：
$$
P_1\cdot m_1 + P_2\cdot m_2 = m_G
$$
那么，**全局方差**  $\sigma_G^{2}$ , **类间方差** $\sigma_B^{2}$ 
$$
\sigma_G^{2} = \sum_{i=0}^{L-1}(i-m_G)^{2}p_i \\
\sigma_B^{2} = P_1(m_1-m_G)^{2} + P_2(m2-m_G)^{2} \\
化简可得：
= P_1(m1^{2} - 2m_1m_G + m_G^{2}) + P_2(m_2^{2} - 2m_2m_G + m_G^{2}) \\
= P_1m_1^{2} + P_2m_2^{2} - 2m_G(m_1P_1 + m_2P_2) + (P_1+P_2)m_G^{2} \quad(P_1+P_2 =1)\\
由又上式已知P_1m_1 + P_2m_2 = m_G \\
原式 = P_1m_1^{2} + P_2m_2^{2} - m_G^{2} \\
= P_1m_1^{2} + P_2m_2^{2} - (P_1m_1+P_2m_2)^{2} \\
=	P_1m_1^{2} + P_2m_2^{2} - 2P_1P_2m_1m_2 - P_1^{2}m_1^{2} -P_2^{2}m_2^{2} \\
= P_1m_1^{2}(1 - P1) + P_2m2^{2}(1-P_2) -2P_1P_2m_1m_2  \\
= P_1P_2m_1^{2} + P_1P_2m_2^{2} - 2P_1P_2m_1m_2 \\
= P_1P_2 (m_1 - m_2)^{2}
$$
枚举**阈值k** 从**[0,L-1]**, 计算类间方差，取使得类间方差最大的阈值作为全局阈

值。

```Python
# editor: kurisu
# edit time: 2023/8/21 22:21
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
def Threshold_Otsu(gray):
     hist = cv.calcHist([gray], [0], None, [256], [0, 256])
     # hist = cv.equalizeHist(gray) * 255
     print(hist)
     plt.plot(hist, color='red')
     plt.xlim([0, 256])
     plt.show()
     rows, cols = gray.shape
     result = np.zeros((rows, cols), dtype=np.uint8)
     # 类间方差 = P1 P2 (m1 - m2)^2
     # P1 = 像素在类C1的概率
     # m1 = 类C1的像素平均灰度值
     k, sigma = 0, -1.0  # k 保存最终的阈值，sigma表示最大的类间方差
     for thresh in range(256):
         size = rows * cols
         P1, P2, m1, m2 = 0.0, 0.0, 0.0, 0.0
         for i in range(256):
             if i < thresh:
                P1 += hist[i] / size
                m1 += i * hist[i] / size
             else:
                P2 += hist[i] /  size
                m2 += i * hist[i] / size
         print('iter: ', thresh)
         # m1, m2 = m1/P1, m2/P2
         m1 = m1 / P1 if np.abs(P1) > 0 else 0
         m2 = m2 / P2 if np.abs(P2) > 0 else 0
         # n1 = gray[np.where(gray < thresh)]
         # n2 = gray[np.where(gray >= thresh)]
         # P1 = len(n1) / size
         # P2 = len(n2) / size
         # m1 = np.mean(n1) if len(n1) > 0 else 0
         # m2 = np.mean(n2) if len(n2) > 0 else 0

         delta = P1 * P2 * (m1 - m2) **2
         if delta > sigma:
             k, sigma = thresh, delta
     print('thresh: ', k, ' sigma: ', sigma)
     # for row in range(rows):
     #     for col in range(cols):
     #        val = gray[row, col]
     #        if val >= k:
     #            result[row, col] = 255
     result[gray >= k] = 255
     result[gray < k] = 0
     return result


if __name__ == "__main__":
    img = cv.imread('../test.png')
    img = cv.GaussianBlur(img, (5, 5), 0)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    result = Threshold_Otsu(gray)
    cv.imshow('result', result)
    ret, result2 = cv.threshold(gray, 0 ,256, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print('ret: ', ret)
    cv.imshow('result2', result2)
    print('is the same ? ',np.all(result == result2))
    cv.waitKey(0)
    cv.destroyAllWindows()
```


$$
m_1 = \frac{\sum_{i=0}^{k}i \cdot p_i}{P_1(k)} \\ 
= \frac{\sum_{i=0}^{k} i \cdot p_i} {\sum_{i=0}^{k} p_i} \\
= \frac{\sum_{i=0}^{k} i \cdot \frac{n_i}{MN}} {\sum_{i=0}^{k} \frac{n_i}{MN}} \\
=\frac{\sum_{i=0}^{k} i \cdot n_i } { \sum_{i=0}^{k}n_i }  \\
= \frac{\sum_{i=0}^{k} i \cdot n_i}{MN}
$$

#### 图像滤波 去噪

图像滤波是图像处理的一种方式，通常用来平滑图像或者增强图像。

##### 卷积

模糊：

```Python
均值模糊： blur = cv.blur(img, (5, 5))
高斯模糊：blur = cv.GaussianBlur(img, (5, 5), 0)
中值模糊：blur = cv.medianBlur(img, 5)
双边模糊：blur = cv.bilateralFilter(img, 9, 75, 75)
```

中值模糊一般用来去除椒盐噪声。

高斯模糊考虑了像素的分布，也就是像素的空间，但没有考虑像素值的相似度。

##### 双边滤波、模糊

双边模糊即考虑了像素的空间，又考虑了像素值的相似度。
$$
中心点(i,j), 邻域(k, l) \\
空间域：\quad d(i,j,k,l) = e^{-\frac{(i-k)^{2} + (j-l)^{2}}{2\sigma_d^{2}}} \\
值域:\quad r(i,j,k,l) = e^{-\frac{||f(i,j) - f(k,l)||^{2}}{2\sigma_r^{2}}} \\
总体公式:\quad w(i,j,k,l) = d(i,j,k,l) * r(i, j, k, l)
$$


```Python
def bilateralFilter(src: Any,
                    d: Any,  邻域直径
                    sigmaColor: Any,  
                    sigmaSpace: Any,
                    dst: Any = None,
                    borderType: Any = None) -> None
```

##### 方框滤波 BoxFilter

将时间复杂度从$O(HWNN) \rightarrow O(HW)$ , 和二维前缀和类似。

参考：https://www.cnblogs.com/lwl2015/p/4460711.html，BoxFilter在更新中只有3个数的运算，而二维前缀和需要4个数。

![image-20230822180049516](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230822180049516.png)

```Python
blur = cv.boxFilter(img, -1, (3, 3), normalize=False)
def boxFilter(src: Any,
              ddepth: Any,  图像深度，-1与原图一致
              ksize: Any,  卷积核大小
              dst: Any = None,
              anchor: Any = None,
              normalize: Any = None,  归一化参数
              borderType: Any = None) -> None
```

设置normalize=False使归一化失效后，像素值为方框内的像素之和，整个图像会很亮。

![image-20230822182122941](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230822182122941.png)

##### mean-shift blur 均值迁移模糊 边缘保留滤波算法



##### fastNlMeansDenoisingColored



#### **形态学操作**

##### 腐蚀、膨胀

##### 开运算、闭运算

开运算：先腐蚀，再膨胀。消除白点。

```Python
se = cv.getStructuringElement(cv.MORPH_RECT, (5,5), (-1, -1))
binary = cv.morphologyEx(binary_1, cv.MORPH_OPEN, se)
```

闭运算：先膨胀，再腐蚀。消除黑点。

```Python
se1 = cv.getStructuringElement(cv.MORPH_RECT, (25,5), (-1, -1))
se2 = cv.getStructuringElement(cv.MORPH_RECT, (5, 25), (-1, -1))
binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, se1)
binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, se2)
```

##### 形态学梯度运算

膨胀图像减腐蚀图像的操作，获取原始图像中前景图像的边缘。

- 基本梯度：膨胀-腐蚀
- 外梯度：膨胀-原图
- 内梯度：原图-腐蚀
- 方向梯度：

##### 礼帽与黑帽

礼帽：原图减去开运算的操作。能够获取图像的噪声信息，或得到比原始图像的边缘更亮的边缘信息。

黑帽：闭运算减去原图。黑帽运算能够获取图像内部的小孔，或前景色中的小黑点，或者得到比原始图像的边缘更暗的边缘部分。

![image-20230823210444505](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230823210444505.png)

#### 图像梯度、边缘检测

梯度既有大小也有方向，梯度方向是函数变化(图像灰度值变化)最快的方向。

##### 一阶微分算子：

##### Sobel算子

$$
G_x = \left[\begin{matrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1 \\
\end{matrix}
\right] \cdot src, G_y = \left[\begin{matrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1 \\
\end{matrix}
\right] \cdot src
$$

$G_x,G_y$分别为图像在x,y方向上的偏导，图像的梯度大小为$G = \sqrt{G_x^{2} +G_y^{2}}$ ，有时为了计算方便也可以简化为$G = |G_x| + |G_y|$。

其梯度方向为：
$$
tan\theta = \frac{G_y}{G_x} \\
\theta = arctan\frac{G_y}{G_x}
$$
示例：

```Python
# editor: kurisu
# edit time: 2023/8/24 18:29
import cv2 as cv
import numpy as np

src = cv.imread('../test.png')
cv.namedWindow('input', cv.WINDOW_AUTOSIZE)
cv.imshow('input', src)

h, w = src.shape[:2]
x_grad = cv.Sobel(src, cv.CV_32F, 1, 0)
y_grad = cv.Sobel(src, cv.CV_32F, 1, 0)

x_grad = cv.convertScaleAbs(x_grad)
y_grad = cv.convertScaleAbs(y_grad)
cv.imshow('x_grad', x_grad)
cv.imshow('y_grad', y_grad)

dst = cv.addWeighted(x_grad, 0.5, y_grad, 0.5, 0)
#dst = cv.add(x_grad, y_grad, dtype=cv.CV_16S)
#dst = cv.convertScaleAbs(dst)
cv.imshow('gradient', dst)

result = np.zeros([h, w*2, 3], dtype=src.dtype)
result[0:h, 0:w, :] = src
result[0:h, w:2*w, :] = dst
cv.imshow('result', result)
cv.imwrite('result1.png', dst)

cv.waitKey(0)
cv.destroyAllWindows()
```

**注意：** 对x、y单独求梯度再求和会比同时计算效果要好，同时计算两个方向的可能会被截断。

##### Scharr算子

Scharr算子是Sobel算子的一种特殊形式，对3x3的卷积核进行的优化。
$$
两个方向的卷积核为： \\
Scharr_x = \left[\begin{matrix}
-3 & 0 & 3 \\
-10 & 0 & 10 \\
-3 & 0 & 3 \\
\end{matrix}
\right] ,
Scharr_y = \left[\begin{matrix}
-3 & -10 & -3 \\
0 & 0 & 0 \\
3 & 10 & 3 \\
\end{matrix}
\right] 
$$
Scharr提高了精度，加大了x、y方向的权重。





##### 二阶微分算子：

##### Laplacian算子

参考[图像处理算法 之 梯度/边缘检测(Sobel算子,拉普拉斯算子,Canny等)](https://blog.csdn.net/qq_38574198/article/details/109264960?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%9B%BE%E5%83%8F%E6%A2%AF%E5%BA%A6sobel&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-109264960.142^v93^insert_down28v1&spm=1018.2226.3001.4187)

 Laplacian（拉普拉斯）算子是一种二阶导数算子，其具有旋转不变性，可以满足不同方向的图形边缘锐化（边缘检测）的要求。

拉普拉斯算子可由二阶导数定义：
$$
\nabla ^{2}f(x,y) = \frac{\partial ^{2} f(x,y)}{\partial x^{2}} +
\frac{\partial ^{2} f(x,y)}{\partial y^{2}}
$$
在图像中用二阶差分表示为：
$$
\frac{\partial ^{2} f(x,y)}{\partial x^{2}} \approx \nabla_x f(i+1, j) - \nabla _x f(i,j) \\ = [f(i+1,j) - f(i,j)] - [f(i,j) - f(i-1,j)] \\ = f(i+1, j) + f(i-1,j) - 2f(i,j) \\
\frac{\partial ^{2} f(x,y)}{\partial y^{2}} \approx f(i,j+1) + f(i,j-1) - 2f(i,j)
$$
所以拉普拉斯算子可以表示为：
$$
\nabla ^{2}f(x,y) = \frac{\partial ^{2} f(x,y)}{\partial x^{2}} +
\frac{\partial ^{2} f(x,y)}{\partial y^{2}} \\
\approx f(i+1,j) +f(i-1,j) +f(i,j+1) +f(i,j-1)-4f(i,j)
$$
即矩阵：
$$
\left[ 
\begin{matrix}
0 & 1  & 0\\
1 & -4 & 1 \\
0 & 1 & 0 \\
\end{matrix}
\right]
$$
拉普拉斯算子法其实是一种图像边缘增强算子，常用于图像锐化，在增强边缘的同时也增强了噪声，因此使用前需要进行平滑或滤波处理。**二阶微分算子检测的边缘更加准确，对边缘的定位能力更强。但是相较于一阶微分算子，不能保留梯度的方向信息，并且对噪声更为敏感。**

```Python
def Laplacian(src: Any,
              ddepth: Any,
              dst: Any = None,
              ksize: Any = None,
              scale: Any = None,
              delta: Any = None,  // delta为偏移量
              borderType: Any = None) -> None
```

##### 图像锐化

图像锐化，是使图像边缘更加清晰的一种图像处理方法。上述拉普拉斯算子就是一种图像锐化应用的一种。

###### UMS锐化增强算法 Unsharpen Mask







##### Canny边缘检测

1. 去噪。高斯滤波
2. 计算图像梯度。大小和方向
3. 非极大值抑制。
4. 双阈值筛选和弱边缘连接。

#### 图像金字塔

 **图像金字塔**是图像多尺度表达的一种，是一种以多分辨率来解释图像的有效但概念简单的结构。分辨率最高的图像在底部，最小的在顶部。

![image-20230825051438156](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230825051438156.png)

##### 向下采样 （降低分辨率）

最简单的图像金字塔可以通过不断的删除图像的偶数行和偶数列得到。

也可以先对图像滤波，得到原始图像的近似图像，然后再删除偶数行列。



##### 向上采样 

向上采样通常将图像的宽高放大2倍，这意味着有大量的像素点需要补充，需要用到**插值算法**。

还有一种向上采样是，对像素多出来的行列进行补0，再对补0后的图像进行滤波处理，卷积核的系数要乘4（不然会导致像素值范围变为[0, $\frac{255}{4}$]），这也是**cv.pyrUp()**使用的方法。

##### 采样的不可逆性

**注意**：向上采样和向下采样虽然看起来是相反的操作，但是是**不可逆的**。向下采样经过滤波器（模糊）且删除了像素点，无法恢复。而向上采样是通过插值得到的。

样例：

```Python
# editor: kurisu
# edit time: 2023/8/25 5:56
import cv2 as cv
import numpy as np

src = cv.imread('../test.png')
cv.namedWindow('input', cv.WINDOW_AUTOSIZE)
cv.imshow('input', src)

down = cv.pyrDown(src)
up = cv.pyrUp(down)

diff = up - src
print('src.shape=', src.shape)
print('down.shape=', up.shape)
cv.imshow('up', up)
cv.imshow('diff', diff)
cv.waitKey(0)
cv.destroyAllWindows()

```

![image-20230825060050912](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230825060050912.png)

##### 高斯金字塔 Gaussian Pyramid  向下采样

高斯滤波器：采用高斯滤波器对原始图像进行滤波，得到高斯金字塔。这是OpenCV函数**cv2.pyrDown()**所采用的的方式。

##### 拉普拉斯金字塔 Gaussian Pyramid 向上采样

 **为了在向上采样时能够恢复具有较高分辨率的原始图像，就要获取在采样过程中所丢失的信息，这些丢失的信息就构成了拉普拉斯金字塔。** 也是拉普拉斯金字塔是有向下采样时丢失的信息构成。

拉普拉斯金字塔的定义为：
$$
L_i = G_i - pyrUp(G_i + 1)
$$
其中，$L_i : 拉普拉斯金字塔的第i层，G_i：高斯金字塔的第i层$ 。

也就是说 拉普拉斯金字塔的第i层定义为，第i层的高斯金字塔减去 [第(i+1)层高斯金字塔的向上采样]。

![image-20230825060635972](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230825060635972.png)

##### 应用

##### 图像融合



#### 图像轮廓检测

##### 轮廓检测

```Python
def findContours(image: Any,
                 mode: Any,
                 method: Any,
                 contours: Any = None,
                 hierarchy: Any = None,
                 offset: Any = None) -> None
```



##### 绘制轮廓



#### 其他

```Python
def absdiff(src1: Any,
            src2: Any,
            dst: Any = None) -> None
```



### 视频 

#### 读取视频

```cv.VideoCapture(filename)```

样例：

```python
import cv2 as cv

def read_video():
    cap = cv.VideoCapture(r'D:\Bna\ARKT1KA\BronyaShortHair\BronyaShortHair.mp4')
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        cv.namedWindow('video', cv.WINDOW_NORMAL)
        cv.imshow('video', frame)
        cv.waitKey(20)

read_video()
```

```cv.imshow```的窗口可能比例不合适，可以使用```cv.namedwindow()``` 声明窗口的名字，和窗口比例```cv.WINDOW_NORMAL```保持原图比例。

**调用本地摄像头**

样例：

``````Python
# editor: kurisu
# edit time: 2023/8/10 18:25
# 调用本地摄像头
import cv2 as cv

# 调用笔记本内置摄像头
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    cv.namedWindow('video', cv.WINDOW_NORMAL)
    cv.imshow('video', frame)
    if cv.waitKey(100) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
``````

可以看到调用本地摄像头和读取本地视频文件的不同是```filename```和```index```，在opencv的文档中有多种参数的```videocapture```方法，文档中解释如下：

| index | id of the video capturing device to open. To open default camera using default backend just pass 0. (to backward compatibility usage of camera_id + domain_offset (CAP_*) is valid when apiPreference is CAP_ANY) |
| ----- | ------------------------------------------------------------ |

样例涉及函数

| VideoCapture::read()                    | 调用了```VideoCapture::grab() ;VideoCapture::retrieve()```方法，先抓取视频或设备的下一帧next frame，再进行解码decode。 |
| --------------------------------------- | ------------------------------------------------------------ |
| **cv.flip(src, flipCode[, dst]) ->dst** | **可以对图像进行翻转，水平、竖直翻转。**   当```filpCode == 0```时，竖直翻转。             当```filpCode > 0```时,    水平翻转。              当```flipCode < 0```时，水平、竖直同时翻转。 |

比如样例结果如下：

此时伸出右手。

```flipCode > 0``` ，水平翻转，所见和右手同侧。 

![image-20230810185438207](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230810185438207.png)

```flipCode == 0```, 竖直翻转，右手处于左侧且 上下颠倒。

![image-20230810190411198](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230810190411198.png)

```flipCode < 0```, 水平、竖直同时翻转。

![image-20230810190535258](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230810190535258.png)

#### 写入视频

```Python
class VideoWriter:
	VideoWriter (const String &filename, int fourcc, double fps, Size frameSize, bool isColor=true)
 
 	VideoWriter (const String &filename, int apiPreference, int fourcc, double fps, Size frameSize, bool isColor=true)
 
 	VideoWriter (const String &filename, int fourcc, double fps, const Size &frameSize, const std::vector< int > &params)
 
 	VideoWriter (const String &filename, int apiPreference, int fourcc, double fps, const Size &frameSize, const std::vector< int > &params)
```

其中, **fourcc**解释如下:

```
	4-character code of codec used to compress the frames. For example, VideoWriter::fourcc('P','I','M','1') is a MPEG-1 codec, VideoWriter::fourcc('M','J','P','G') is a motion-jpeg codec etc. 
```

fourcc即为4字节码，用来确定视频的编码格式。
