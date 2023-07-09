# 国科大深度学习复习
国科大深度学习课程知识点整理
项目文件为国科大深度学习复习过程中的相关知识点整理，包括搬运他人的见解
如有错误，欢迎批评指正~

## 2021考题

![2021考题](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/2021%E8%80%83%E9%A2%98.PNG)

![2021考题2](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/2021%E8%80%83%E9%A2%982.PNG)

计算题代码：

```
import tensorflow as tf

input_x = tf.constant([
    [[[5, 6, 0, 1, 8, 2],
      [0, 9, 8, 4, 6, 5],
      [2, 6, 5, 3, 8, 4],
      [6, 3, 4, 9, 1, 0],
      [7, 5, 9, 1, 6, 7],
      [2, 5, 9, 2, 3, 7]

      ]]])
filters = tf.constant([
    [[[0, -1, 1], [1, 0, 0], [0, -1, 1]]]
])

input_x=tf.reshape(input_x,(1,6,6,1))
filters=tf.reshape(filters,[3,3,1,1])

res = tf.nn.conv2d(input_x, filters, strides=1, padding='VALID')
print('Valid 无激活函数下的输出',res)
res=tf.squeeze(res)
print('Valid 条件下可视化的输出：',res)


# print('Valid 激活函数下输出',tf.nn.relu(res))
print('Valid 激活函数下可视化输出：',tf.squeeze(tf.nn.relu(res)))
#在full卷积下，TF中没有这个参数，可以手动加0实现
input_x = tf.constant([
    [[[0,0,0,0,0,0,0,0],
  [0,5,6,0,1,8,2,0],
  [0,2,5,7,2,3,7,0],
  [0,0,7,2,4,5,6,0],
  [0,5,3,6,9,3,1,0],
  [0,6,5,3,1,4,6,0],
  [0,5,2,4,0,8,7,0],
    [0,0,0,0,0,0,0,0]
]]])
input_x=tf.reshape(input_x,(1,8,8,1))

res = tf.nn.conv2d(input_x, filters, strides=1,padding='SAME')
print('Full（加0）未使用激活之前的输出',res)

print('Full(加0）未使用激活函数之前的可视化输出，',tf.squeeze(res))

out = tf.nn.relu(res)
print('Full 激活的输出',out)
print('Full 激活之后的可视化输出，',tf.squeeze(out))
```

```
import torch
import torch.nn as nn

criterion = nn.BCELoss()#默认是求均值，数据需要是浮点型数据
pre=torch.tensor([0.1,0.2,0.3,0.4]).float()
tar=torch.tensor([0,0,0,1]).float()
l=criterion(pre,tar)
print('二分类交叉熵损失函数计算（均值）',l)


pre=torch.tensor([0.2,0.8,0.4,0.1,0.9]).float()
tar=torch.tensor([0,1,0,0,1]).float()

pre=torch.tensor([0.1,0.2,0.3,0.4]).float()
tar=torch.tensor([0,0,0,1]).float()
criterion = nn.BCELoss(reduction="sum")#求和
l=criterion(pre,tar)
print('二分类交叉熵损失函数计算（求和）',l)

loss=nn.BCELoss(reduction="none")#reduction="none"得到的是loss向量#对每一个样本求损失
l=loss(pre,tar)
print('每个样本对应的loss',l)
criterion2=nn.CrossEntropyLoss()
import numpy as np
pre1=torch.tensor([np.log(20),np.log(40),np.log(60),np.log(80)]).float()
# soft=nn.Softmax(dim=0)
# pre=soft(pre).float()#bs*label_nums
pre1=pre1.reshape(1,4)
tar=torch.tensor([3])
loss2=criterion2(pre1,tar)
print('多分类交叉熵损失函数pre1条件下',loss2)

pre2=torch.tensor([np.log(10),np.log(30),np.log(50),np.log(90)]).float()
pre2=pre2.reshape(1,4)
tar=torch.tensor([3])
loss2=criterion2(pre2,tar)
print('多分类交叉熵损失函数pre2条件下',loss2)

```

## 2023考题

![2023考题](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/2023%E8%80%83%E9%A2%98.jpg)
