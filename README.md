论文地址：[[2112.11081\] RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality (arxiv.org)](https://arxiv.org/abs/2112.11081)

作者源码：[DingXiaoH/RepMLP: RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality (CVPR 2022) (github.com)](https://github.com/DingXiaoH/RepMLP)

conv_bn层融合参考：[[2101.03697\] RepVGG: Making VGG-style ConvNets Great Again (arxiv.org)](https://arxiv.org/abs/2101.03697)

此代码只是一个demo，粗略实现了一下如何融合Mlp和conv和bn，具体的推导论文有。

作者的论文解说：[论文连讲：用重参数化赋予MLP网络局部性、超大卷积核架构【CVPR2022】【基础模型】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1bB4y1m7rx?spm_id_from=333.999.0.0)

作者实现的代码所有的MLP都是1x1卷积实现的，因为把feature map都reshape为1x1，所有的特征都reshape到通道数量上，对于1x1的feature map，1x1卷积和MLP是一样的（可以把MLP视为对h，w的全卷积）。这里的实现是用torch.nn.linear的，并且没有用分组卷积，也没有对特征图进行分组。只是一个单纯的demo。