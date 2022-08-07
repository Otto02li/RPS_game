# 设计文档
## 模块
- 基于opencv的手势识别
- 安卓Android的app应用程序

## 手势识别
编译器：pycharm
库：opencv,MediaPipe

实现原理:

通过一些关键点构造一个凸包，计算谁在凸包外
dist = cv2.pointPolygonTest(cnt,(50,50),True)
>如果该点位于轮廓内,则为正值

>如果该点在轮廓之外,则为负值

>如果该点在轮廓上,则为零



