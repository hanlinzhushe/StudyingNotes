# 计算机视觉成长资源地图
## 计算机视觉——必看工具书
* [pattern recognition and machine learning , Christopher Bishop]()<br>
* [计算机视觉大牛-何恺明论文代码集](https://scholar.google.com/citations?user=DhtAFkwAAAAJ&hl=zh-CN&oi=ao)<br>

# 计算机视觉BackBone
## 计算机视觉——路线大纲
* [人工智能第三期课程大纲——咕泡AI](https://www.processon.com/view/link/6081395d5653bb708d504a09#map)<br>
* [计算机视觉大牛-何恺明论文代码集](https://scholar.google.com/citations?user=DhtAFkwAAAAJ&hl=zh-CN&oi=ao)<br>
## 初级计算机视觉——图像处理
* 图像基本属性
* 图像相似性变换
* 图像仿射变换
* 锐化、模糊
* 形态学运算
* 灰度直方图、直方图均衡化
## 中级计算机视觉——图像描述
* HOG特征
* LBP特征
* Haar-Like特征
## 高级计算机视觉——CNN
* ReLU、PReLU、Leaky ReLU
* Batch Normalization
* Dropout
* 参数初始化：Gaussian/Xavier/Kaiming
* 参数优化：SGD/Adagrad/Adam
## 计算机视觉——图像分类
* 技术细节
  * CNN提特征设计
  * output决策层设计
  * output groundtruth设计
* 经典模型
  * ResNet：
    [论文](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
  * DenseNet：
    [论文](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)
    [代码](https://github.com/liuzhuang13/DenseNet?utm_source=catalyzex.com)
  * MobileNet
  * EfficientNet
## 计算机视觉——图像生成
* 经典模型
  * GAN
## 计算机视觉——单阶段目标检测
* 技术细节
  * 先验框与统计框设计
  * 对预测边框的约束
  * Passthrough方法
  * 检测模型评价指标
* 经典模型
  * YOLO V1及Loss：
    [论文](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)
  * YOLO V2：Anchor引入
  * YOLO V3：多尺度引入
  * FPN
  * YOLO V4
  * RetinaNet：Focal Loss
  * Anchor Free
  * Anchor Free：CenterNet
  * Anchor Free：FCos
## 计算机视觉——两阶段目标检测
* 经典模型
  * RCNN（NMS系列）
  * Fast RCNN（ROI Pooling系列）：
    [论文](https://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)
    [代码](https://github.com/rbgirshick/fast-rcnn?utm_source=catalyzex.com)
  * Faster RCNN（RPN网络和Anchor）：
    [论文](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)
## 计算机视觉——图像分割
* 技术细节
  * 像素分类
  * 反卷积与上采样
  * 转置卷积
  * 膨胀卷积
  * 跳级结构
  * 分割模型评价指标
* 经典模型
  * UNet：
    [论文](https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf)
  * ENet
  * Mask RCNN
  * TensorMask
## 计算机视觉——目标跟踪
* 经典目标跟踪方法
  * 帧差法
  * TLD算法
  * KCF算法
* 深度学习目标跟踪算法
  * siamese系列
