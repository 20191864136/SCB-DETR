
import torch
import mmpretrain
from mmdet.models.backbones import ResNet
from mmpretrain.models.backbones import ConvNeXt

from fightingcv_attention.attention.CoordAttention import CoordAtt
from fightingcv_attention.attention.SEAttention import SEAttention
# from mmdet.models.builder import BACKBONES
from mmdet.registry import MODELS
 
 
# 定义带attention的ConvNext基类
class ConvNextWithAttention(ConvNeXt):
    def __init__(self , **kwargs):
        super(ConvNextWithAttention, self).__init__(**kwargs)
        # 目前将注意力模块加在最后的三个输出特征层
        # convnext输出四个特征层
        # if self.arch== 'tiny':
        #     self.dims = (96, 192, 384, 768)
        # else:
        #     raise Exception()
        self.dims = (96, 192, 384, 768)
        self.attention0 = self.get_attention_module(self.dims[0])     
        self.attention1 = self.get_attention_module(self.dims[1])     
        self.attention2 = self.get_attention_module(self.dims[2])
        self.attention3 = self.get_attention_module(self.dims[3])    
    
    # 子类只需要实现该attention即可
    def get_attention_module(self, dim):
        raise NotImplementedError()
    
    def forward(self, x):
        outs = super().forward(x)
        outs = list(outs)
        outs[0] = self.attention0(outs[0])
        outs[1] = self.attention1(outs[1])
        outs[2] = self.attention2(outs[2])
        outs[3] = self.attention3(outs[3])
        outs = tuple(outs)
        return outs    
        # return len(outs)    
    
    
# @BACKBONES.register_module()
@MODELS.register_module()
class ConvNextWithCoordAttention(ConvNextWithAttention):
    def __init__(self , **kwargs):
        super(ConvNextWithCoordAttention, self).__init__(**kwargs)
 
    # 子类只需要实现该attention即可
    def get_attention_module(self, dim):
        return CoordAtt(inp=dim, oup=dim, reduction=32)
    
# @BACKBONES.register_module()
@MODELS.register_module()
class ConvNextWithSEAttention(ConvNextWithAttention):
    def __init__(self , **kwargs):
        super(ConvNextWithSEAttention, self).__init__(**kwargs)
 
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
    model = ConvNeXt(arch='tiny',out_indices=[0,1,2,3])
    x = torch.rand(1, 3, 224, 224)
    outs = model(x)

    for i, out in enumerate(outs):
        print(i, out.shape)