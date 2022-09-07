import timm
from einops import rearrange
# from ml_decorder import *
# from timm.models import to_2tuple
from torch import nn
# from timm.models.layers.ml_decoder import add_ml_decoder_head,MLDecoder
# all_models = timm.list_models('*nfnet*',pretrained=True)
# print(all_models)
from torchvision import models
nowModel = timm.create_model('convnext_base_384_in22ft1k',pretrained=True,num_classes = 17)
nowModel2 = timm.create_model('convnext_base_384_in22ft1k',pretrained=True,num_classes = 17)
# nowModel = timm.create_model('dm_nfnet_f3',pretrained=True,num_classes = 17,drop_path_rate=0.1)
# nowModel = models.convnext_base(pretrained=True)
# nowModel = convnext_large(pretrained=True)
# nowModel.classifier[2] = nn.Linear(nowModel.classifier[2].in_features, 17)
# nowModel = timm.create_model('swin_large_patch4_window7_224_in22k',fixed_input_size = False,pretrained=True,num_classes = 17,input_size=(3, 672, 672),drop_path_rate = 0.1 )


 # 封装resize算子 + pretrain后的新模型
class NewModel(nn.Module):
    def __init__(self,backBone,super_img_rows = 3):
        super().__init__()
        if backBone == "convnext_base_384_in22ft1k":
            print(backBone)
            self.backBone = timm.create_model('convnext_base_384_in22ft1k',pretrained=True,num_classes = 17)
        else:
            print(backBone)
            self.backBone = timm.create_model('dm_nfnet_f3',pretrained=True,num_classes = 17,drop_path_rate=0.1)
        self.super_img_rows = super_img_rows
    def forward(self, x):
        # x = x.view((-1, 3 * self.duration) + x.size()[2:])
        x = self.create_super_img(x)
        x = self.backBone(x)
        return x

    def create_super_img(self, x):
        x = x.view((-1, 3 * 9) + x.size()[2:])
        x = rearrange(x, 'b (th tw c) h w -> b c (th h) (tw w)', th=self.super_img_rows, c=3)

        return x



class MLDECORDModel(nn.Module):
    def __init__(self,backBone = nowModel2,super_img_rows = 3):
        super().__init__()
        #convNext兼容mldecorder
        # del backBone.head.fc
        # num_classes = backBone.num_classes
        # num_features = backBone.num_features
        # backBone.head.global_pool = nn.Identity()
        # backBone.head.norm = nn.Identity()
        # backBone.head.flatten = nn.Identity()
        # backBone.head.fc = MLDecoder(num_classes=num_classes, initial_num_features=num_features)

        self.backBone = backBone
        # self.ml_decorder_head = MLDecoder(17)
        self.super_img_rows = super_img_rows
    def forward(self, x):
        # x = x.view((-1, 3 * self.duration) + x.size()[2:])
        x = self.create_super_img(x)
        x = self.backBone(x)
        # x = self.backBone.forward_features(x)
        # x = self.ml_decorder_head(x)
        return x

    def create_super_img(self, x):
        # input_size = x.shape[-2:]
        # if input_size != to_2tuple(self.img_size):
        #     x = nn.functional.interpolate(x, size=self.img_size,mode='bilinear')
        x = rearrange(x, 'b (th tw c) h w -> b c (th h) (tw w)', th=self.super_img_rows, c=3)
        return x

