# What is image augmentation and how it can improve the performance of deep neural networks(什么是图像扩充，如何提高深度神经网络的效果)

## obtaining enough training data is sometimes difficult 

- label the images sometimes be very costly
- obtaining the training data sometimes have law restrictions or be costly

## the image augmentation could create new training examples

## How much does image augmentation improves the quality and performance of deep neural networks
In 2018 Google published a paper about AutoAugment - an algorithm that automatically discovers the best set of augmentations for the dataset. They showed that a custom set of augmentations improves the performance of the model.

The table of comparison between a model that used only the base set of augmentations and a model that used a specific set of augmentations discovered by AutoAugment demonstrates that a diverse set of image augmentations improves the performance of neural networks compared to a base set with only a few most popular transformation techniques.

Augmentations help to fight overfitting and improve the performance of deep neural networks for computer vision tasks such as classification, segmentation, and object detection. 

## pixel-level augmentation and spatial-level augmentation

pixel-level augmentation is just modify the pixel attributes ,such as : brightess、contrast,in other words ,they just change the pixel's value.
For these operations,we just change the input image and don not need change the mask.

spatial-level augmentation is mirroring , croping and so on.these operations change the image's pixel's location.
For these operations,not noly we should change the input image ,but also change the mask. 

## Working with probabilities
During training,we usually use the probabilities of less than 100% because we also need original image.
- If the original dataset is large, you could apply only the basic augmentations with probability around 10-30% and with a small magnitude of changes. 
- If the dataset is small, you need to act more aggressively with augmentations to prevent overfitting of neural networks, so you usually need to increase the probability of applying each augmentation to 40-50% and increase the magnitude of changes the augmentation makes to the image.

## pipeline and unified interface  
- you want to apply not a single augmentation, but a set of augmentations with specific parameters such as probability and magnitude of changes,Augmentation libraries allow you to declare such a pipeline in a single place and then use it for image transformation through a unified interface

## rigorous testing 
- Augmentation libraries usually have large test suites that capture regressions during development. Also large user base helps to find unnoticed bugs and report them to developers.

## Image augmentation for classification
We can divide the process of image augmentation into four steps:

- Import albumentations and a library to read images from the disk (e.g., OpenCV).
- Define an augmentation pipeline.
- Read images from the disk.
- Pass images to the augmentation pipeline and receive augmented images.

## Mask augmentation for segmentation

The process of augmenting images and masks looks very similar to the regular image-only augmentation.

- You import the required libraries.
- You define an augmentation pipeline.
- You read images and masks from the disk.
- You pass an image and one or more masks to the augmentation pipeline and receive augmented images and masks.

当将图片与蒙版一起放到transform中时，需要注意，mask参数选则某一个蒙版还是一个蒙版列表

## Bounding boxes augmentation for object detection
### Different annotations formats
![](https://albumentations.ai/docs/images/getting_started/augmenting_bboxes/bbox_formats.jpg)

### Bounding boxes augmentation¶
Just like with images and masks augmentation, the process of augmenting bounding boxes consists of 4 steps.

- You import the required libraries.
- You define an augmentation pipeline.
- You read images and bounding boxes from the disk.
- You pass an image and bounding boxes to the augmentation pipeline and receive augmented images and boxes.
  
==Note that unlike image and masks augmentation, Compose now has an additional parameter bbox_params==

```python
transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=['class_labels']))
```

min_area and min_visibility参数介绍：
**min_area**为阈值，当一个bounding_box的面积小于这个值时，bounding_box将会被舍弃

**min_visibility**为阈值，当一个bounding_box的面积与增强前的面积之比的小于这个值时，bounding_box将会被舍弃


## Class labels for bounding boxes
###  1.You can pass labels along with bounding boxes coordinates by adding them as additional values to the list of coordinates.
-  for example [23, 74, 295, 388, 'dog', 'animal'], [377, 294, 252, 161, 'cat', 'animal'], and [333, 421, 49, 49, 'sports ball', 'item']
### 2.You can pass labels for bounding boxes as a separate list (the preferred way).
- [23, 74, 295, 388], [377, 294, 252, 161], and [333, 421, 49, 49] you can create a separate list with values like ['cat', 'dog', 'sports ball']

## keypoints augmentation
krypoints always mark human joints such as shoulder ,elbow,wrist,knee ,eyes ,nose and so on.

![](https://albumentations.ai/docs/images/getting_started/augmenting_keypoints/keypoint_with_scale_and_angle_v2.png)

一个keypoint的表示可能涉及到四个值，以下是可能的format：

Supported formats for keypoints' coordinates.¶
xy. A keypoint is defined by x and y coordinates in pixels.

yx. A keypoint is defined by y and x coordinates in pixels.

xya. A keypoint is defined by x and y coordinates in pixels and the angle.

xys. A keypoint is defined by x and y coordinates in pixels, and the scale.

xyas. A keypoint is defined by x and y coordinates in pixels, the angle, and the scale.

xysa. A keypoint is defined by x and y coordinates in pixels, the scale, and the angle.

## The process of augmenting keypoints looks very similar to the bounding boxes augmentation. It consists of 4 steps.

- You import the required libraries.
- You define an augmentation pipeline.
- You read images and keypoints from the disk.
- You pass an image and keypoints to the augmentation pipeline and receive augmented images and keypoints.


label_fields
将关键点的标签传入
remove_invisible
若图像增强后，一些区域的关键点不在增强后的图像中，则将这些关键点清除
angle_in_degrees
将角度以度数表示或者以弧度表示

## Simultaneous augmentation of multiple targets: masks, bounding boxes, keypoints
可以将image、bbox、keypoints同时进行处理

并不是所有的transform都可以处理image、bbox和keypoins

## 一些transform默认的概率为1，一些默认为0.5


albumentations.augmentations.geometric.transforms.PadIfNeeded
min_height:minimal result image height.
min_width:minimal result image width.
pad_height_divisor:if not None, ensures image height is dividable by value of this argument.
pad_width_divisor:if not None, ensures image width is dividable by value of this argument.
position：Position of the image. should be PositionType.CENTER or PositionType.TOP_LEFT or PositionType.TOP_RIGHT or PositionType.BOTTOM_LEFT or PositionType.BOTTOM_RIGHT. or PositionType.RANDOM. Default: PositionType.CENTER.
border_mode：OpenCV border mode.
value：padding value if border_mode is cv2.BORDER_CONSTANT.
mask_value：padding value for mask if border_mode is cv2.BORDER_CONSTANT.
p：probability of applying the transform. Default: 1.0.