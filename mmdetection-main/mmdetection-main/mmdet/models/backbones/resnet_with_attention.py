 
# -*- encoding: utf-8 -*-
'''
@File    :   resnet_with_attention.py
@Time    :   2023/03/25 08:55:30
@Author  :   RainfyLee 
@Version :   1.0
@Contact :   379814385@qq.com
'''
 
# here put the import lib
 
import torch
from mmdet.models.backbones import ResNet
from fightingcv_attention.attention.CoordAttention import CoordAtt
from fightingcv_attention.attention.SEAttention import SEAttention
# from mmdet.models.builder import BACKBONES
from mmdet.registry import MODELS
 
 
# 定义带attention的resnet18基类
class ResNetWithAttention(ResNet):
    def __init__(self , **kwargs):
        super(ResNetWithAttention, self).__init__(**kwargs)
        # 目前将注意力模块加在最后的三个输出特征层
        # resnet输出四个特征层
        if self.depth in (18, 34):
            self.dims = (64, 128, 256, 512)
        elif self.depth in (50, 101, 152):
            self.dims = (256, 512, 1024, 2048)
        else:
            raise Exception()
        self.attention1 = self.get_attention_module(self.dims[1])     
        self.attention2 = self.get_attention_module(self.dims[2])     
        self.attention3 = self.get_attention_module(self.dims[3])     
    
    # 子类只需要实现该attention即可
    def get_attention_module(self, dim):
        raise NotImplementedError()
    
    def forward(self, x):
        outs = super().forward(x)
        outs = list(outs)
        outs[1] = self.attention1(outs[1])
        outs[2] = self.attention2(outs[2])
        outs[3] = self.attention3(outs[3])    
        outs = tuple(outs)
        return outs
    
# @BACKBONES.register_module()
@MODELS.register_module()
class ResNetWithCoordAttention(ResNetWithAttention):
    def __init__(self , **kwargs):
        super(ResNetWithCoordAttention, self).__init__(**kwargs)
 
    # 子类只需要实现该attention即可
    def get_attention_module(self, dim):
        return CoordAtt(inp=dim, oup=dim, reduction=32)
    
# @BACKBONES.register_module()
@MODELS.register_module()
class ResNetWithSEAttention(ResNetWithAttention):
    def __init__(self , **kwargs):
        super(ResNetWithSEAttention, self).__init__(**kwargs)
 
    # 子类只需要实现该attention即可
    def get_attention_module(self, dim):
        return SEAttention(channel=dim, reduction=16)
 
 
    
#仅用来测试
if __name__ == "__main__":
#     # model = ResNet(depth=18)
#     # model = ResNet(depth=34)
#     # model = ResNet(depth=50)
#     # model = ResNet(depth=101)    
#     # model = ResNet(depth=152)
#     # model = ResNetWithCoordAttention(depth=50)
    model = ResNetWithSEAttention(depth=50)
    x = torch.rand(1, 3, 224, 224)
    outs = model(x)
    # print(outs.shape)
    for i, out in enumerate(outs):
        print(i, out.shape)
