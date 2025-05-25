import torch
from mmdet.models.necks import NASFCOS_FPN
from fightingcv_attention.attention.CoordAttention import CoordAtt
from fightingcv_attention.attention.SEAttention import SEAttention
from fightingcv_attention.attention.OutlookAttention import OutlookAttention
from mmdet.registry import MODELS
 
 
# 定义带attention的resnet18基类
class NASFCOS_FPNWithAttention(NASFCOS_FPN):
    def __init__(self , **kwargs):
        super(NASFCOS_FPNWithAttention, self).__init__(**kwargs)

        if self.out_channels in [256]:
            self.dims = (256, 256, 256, 256)
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
    
# @BACKBONES.register_module()
@MODELS.register_module()
class NASFCOS_FPNWithCoordAttention(NASFCOS_FPNWithAttention):
    def __init__(self , **kwargs):
        super(NASFCOS_FPNWithCoordAttention, self).__init__(**kwargs)
 
    # 子类只需要实现该attention即可
    def get_attention_module(self, dim):
        return CoordAtt(inp=dim, oup=dim, reduction=32)
    
# @BACKBONES.register_module()
@MODELS.register_module()
class NASFCOS_FPNWithSEAttention(NASFCOS_FPNWithAttention):
    def __init__(self , **kwargs):
        super(NASFCOS_FPNWithSEAttention, self).__init__(**kwargs)
 
    # 子类只需要实现该attention即可
    def get_attention_module(self, dim):
        return SEAttention(channel=dim, reduction=16)

@MODELS.register_module()
class NASFCOS_FPNWithOutlookAttention(NASFCOS_FPNWithAttention):
    def __init__(self , **kwargs):
        super(NASFCOS_FPNWithOutlookAttention, self).__init__(**kwargs)
 
    # 子类只需要实现该attention即可
    def get_attention_module(self, dim):
        return OutlookAttention(dim=dim)   

    
#仅用来测试
if __name__ == "__main__":
#     # model = ResNet(depth=18)
#     # model = ResNet(depth=34)
#     # model = ResNet(depth=50)
#     # model = ResNet(depth=101)    
#     # model = ResNet(depth=152)
#     # model = ResNetWithCoordAttention(depth=50)
    model = NASFCOS_FPNWithSEAttention(in_channels=[96,192,384,768],num_outs=4,out_channels=[256])
    x = torch.rand(1, 3, 224, 224)
    outs = model(x)
    # print(outs.shape)
    for i, out in enumerate(outs):
        print(i, out.shape)
