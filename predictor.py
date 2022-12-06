from torchvision.models.detection import MaskRCNN, RetinaNet
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

def masrk_rcnn(backbone, num_classes, use_fpn):
    
    if use_fpn:
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),
                        (0.5, 1.0, 2.0),
                        (0.5, 1.0, 2.0),
                        (0.5, 1.0, 2.0),
                        (0.5, 1.0, 2.0)))
    else:
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128),),
            aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                    output_size=7,
                                    sampling_ratio=2)

    mask_roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                                            output_size=14,
                                                            sampling_ratio=2)

    custom_mask = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,)
    
    return custom_mask

def retina_net(backbone, num_classes, use_fpn):
    if use_fpn:
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),
                        (0.5, 1.0, 2.0),
                        (0.5, 1.0, 2.0),
                        (0.5, 1.0, 2.0),
                        (0.5, 1.0, 2.0)))
    else:
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128),),
            aspect_ratios=((0.5, 1.0, 2.0),))


    custom_retina = RetinaNet(
        backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,)
    
    return custom_retina