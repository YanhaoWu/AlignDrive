from inspect import signature

import torch

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["AlignDrive"]


@DETECTORS.register_module()
class AlignDrive(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        use_GAN=False,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
        training_stage='norml'
    ):
        super(AlignDrive, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            img_backbone = dict(img_backbone, pretrained=pretrained)
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            ) 
        self.use_GAN = use_GAN
        self.GAN_idx = 0
        # training_stage: 'normal'|'warm'|'perc'|'spatial'|'lon'|'longitudinal'
        # perc     = train backbone+neck+det+map only
        # spatial  = freeze perc, train spatial plan (base), freeze longitudinal
        # lon/longitudinal  = freeze perc+spatial, train longitudinal only
        self.training_stage = training_stage
        self.iter_count = 0
        self.apply_training_stage(self.training_stage)

    def apply_training_stage(self, stage):
        """Apply freeze / trainability according to curriculum stage."""
        stage = (stage or "normal").lower()
        if stage in ("normal", "all", "norml"):
            return
        if stage == "warm":
            self.freeze_image_backbone(freeze=True)
            self.freeze_perception(freeze=True)
            return
        if stage == "perc":
            # Train perception only: freeze plan / longitudinal / fusion.
            self.freeze_spatial(freeze=True)
            self.freeze_lon(freeze=True)
            self.freeze_fusion(freeze=True)
            print("[training_stage=perc] trainable: backbone+neck+det+map")
            return
        if stage == "spatial":
            # Freeze perc + lon; train spatial plan (+ optionally fusion).
            self.freeze_image_backbone(freeze=True)
            self.freeze_perception(freeze=True)
            self.freeze_lon(freeze=True)
            self.freeze_spatial(freeze=False)
            self.freeze_fusion(freeze=False)
            print("[training_stage=spatial] trainable: motion_plan_head (+fusion)")
            return
        if stage in ("lon", "longitudinal"):
            self.freeze_image_backbone(freeze=True)
            self.freeze_perception(freeze=True)
            self.freeze_spatial(freeze=True)
            self.freeze_fusion(freeze=True)
            self.freeze_lon(freeze=False)
            print("[training_stage=lon] trainable: longitudinal_head only")
            return
        print(f"[training_stage] unknown stage={stage!r}, no freeze applied")


    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)                       # 多尺度不同维度的特征
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        formated_feature_maps = feature_maps_format(feature_maps) if self.use_deformable_func else feature_maps
        if return_depth:
            return formated_feature_maps,feature_maps, depths






        return formated_feature_maps, feature_maps, depths

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        # detect_anomaly is debug-only: on torch>=2.8 + fp16 it aborts training
        # on benign DivBackward0 NaNs that GradScaler would otherwise skip.
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def set_bn_eval(self,  module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    def set_bn_train(self, module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()

    def freeze_image_backbone(self, freeze=True):
        """Freeze or unfreeze the image backbone (including BN stats)."""
        for param in self.img_backbone.parameters():
            param.requires_grad = not freeze
        for param in self.img_neck.parameters():
            param.requires_grad = not freeze


        if freeze:
            self.img_backbone.apply(self.set_bn_eval)
            self.img_neck.apply(self.set_bn_eval)
            print("Image backbone is frozen (including BN stats).")
        else:
            self.img_backbone.apply(self.set_bn_train)
            self.img_neck.apply(self.set_bn_train)
            print("Image backbone is unfrozen.")

    def freeze_perception(self, freeze=True):
        if not hasattr(self.head, "det_head") or not hasattr(self.head, "map_head"):
            return
        for param in self.head.det_head.parameters():
            param.requires_grad = not freeze
        for param in self.head.map_head.parameters():
            param.requires_grad = not freeze
        if freeze:
            self.head.det_head.apply(self.set_bn_eval)
            self.head.map_head.apply(self.set_bn_eval)
            print("Perception is frozen (including BN stats).")
        else:
            self.head.det_head.apply(self.set_bn_train)
            self.head.map_head.apply(self.set_bn_train)
            print("Perception is unfrozen.")

    def freeze_spatial(self, freeze=True):
        """Freeze / unfreeze spatial (base) motion_plan_head."""
        if not hasattr(self.head, "motion_plan_head") or self.head.motion_plan_head is None:
            return
        for param in self.head.motion_plan_head.parameters():
            param.requires_grad = not freeze
        print(f"motion_plan_head frozen={freeze}")

    def freeze_lon(self, freeze=True):
        """Freeze or unfreeze the longitudinal-planning head."""
        if not hasattr(self.head, "longitudinal_head") or self.head.longitudinal_head is None:
            return
        for param in self.head.longitudinal_head.parameters():
            param.requires_grad = not freeze
        print(f"longitudinal_head frozen={freeze}")

    def freeze_fusion(self, freeze=True):
        if not hasattr(self.head, "fusion_model") or self.head.fusion_model is None:
            return
        for param in self.head.fusion_model.parameters():
            param.requires_grad = not freeze
        print(f"fusion_model frozen={freeze}")

    def freeze_expect_longitudinal(self):
        """Legacy: freeze everything except longitudinal_head."""
        for _, param in self.named_parameters():
            param.requires_grad = False
        self.freeze_lon(freeze=False)
        n = sum(p.numel() for p in self.head.longitudinal_head.parameters() if p.requires_grad)
        print(f"Expect Longitudinal is unfrozen, other parts are frozen. longitudinal params={n}")



    # def process_grad(self): # 按照训练的进度，修改当前有梯度的部分
    #     if self.training_stage is not None:
    #         if self.iter_count == self.training_stage["stage1"]["start_iter"]:
    #             # 第一阶段
    #             print("Start Training Stage 1, freezing image backbone and neck and perception head.")
    #             self.freeze_image_backbone(freeze=True)
    #             self.freeze_perception(freeze=True)
    #         elif self.iter_count == self.training_stage["stage1"]["end_iter"]:
    #             # 第一阶段结束，解冻感知部分
    #             print("End Training Stage 1, unfreezing perception head.")
    #             self.freeze_image_backbone(freeze=False)
    #             self.freeze_perception(freeze=False)




    def forward_train(self, img, **data):
        # self.process_grad()  # 根据当前的训练阶段，处理梯度
        feature_maps, unformated_feature_maps, depths = self.extract_feat(img, return_depth=True, metas=data)   # img维度: (4,6,3,256,704)
        model_outs = self.head(feature_maps, unformated_feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        self.iter_count += 1
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps, unformated_feature_maps, _ = self.extract_feat(img)
        model_outs = self.head(feature_maps, unformated_feature_maps, data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
