import torch.nn as nn
import torchvision
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelP6P7, LastLevelMaxPool
from torchvision.models.detection.mask_rcnn import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.ops import FeaturePyramidNetwork

import pyramid_vig

def get_mask_resent_fpn():
    trainable_backbone_layers = _validate_trainable_layers(True, None, 5, 3)
    resnet_50 = torchvision.models.resnet50(weights="DEFAULT", norm_layer=misc_nn_ops.FrozenBatchNorm2d)

    mask_fpn = _resnet_fpn_extractor(resnet_50, trainable_backbone_layers)
    mask_resnet_fpn_backbone = mask_fpn

    mask_resnet_fpn_backbone.out_channels = 256

    return mask_resnet_fpn_backbone


def get_ret_resnet_fpn():
    trainable_backbone_layers = _validate_trainable_layers(True, None, 5, 3)
    resnet_50 = torchvision.models.resnet50(weights="DEFAULT", norm_layer=misc_nn_ops.FrozenBatchNorm2d)

    retina_fpn = _resnet_fpn_extractor(resnet_50, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))
    retina_resnet_fpn_backbone = retina_fpn

    retina_resnet_fpn_backbone.out_channels = 256

    return retina_resnet_fpn_backbone

def get_resent_fpn(detector: str):
    trainable_backbone_layers = _validate_trainable_layers(True, None, 5, 3)
    resnet_50 = torchvision.models.resnet50(weights="DEFAULT", norm_layer=misc_nn_ops.FrozenBatchNorm2d)

    if detector == "mask": 
        resnet_fpn = _resnet_fpn_extractor(resnet_50, trainable_backbone_layers)
    elif detector == "retina": 
        resnet_fpn = _resnet_fpn_extractor(resnet_50, trainable_backbone_layers, returned_layers=[2, 3, 4], 
                                            extra_blocks=LastLevelP6P7(256, 256))
    
    resnet_fpn.out_channels = 256

    return resnet_fpn

def get_pyramid_vig_fpn():
    p_vig = pyramid_vig.pvig_s_224_gelu()

    extra_blocks = LastLevelMaxPool()

    fpn = FeaturePyramidNetwork(
            in_channels_list=[80, 160, 400, 640],
            out_channels=256,
            extra_blocks=extra_blocks,)

    pyramid_vig_fpn = nn.Sequential(p_vig, fpn)
    pyramid_vig_fpn.out_channels = 256

    return pyramid_vig_fpn