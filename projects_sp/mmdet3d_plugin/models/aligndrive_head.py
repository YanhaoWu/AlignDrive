from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmdet.models import build_head
from projects_sp.mmdet3d_plugin.models.instance_bank import Img_instance_queue

@HEADS.register_module()
class AlignDriveHead(BaseModule):
    def __init__(
        self,
        task_config: dict,
        det_head = dict,
        map_head = dict,
        motion_plan_head = dict,
        init_cfg=None,
        img_feature_res=False,
        **kwargs,
    ):
        super(AlignDriveHead, self).__init__(init_cfg)
        self.task_config = task_config
        if self.task_config['with_det']:
            self.det_head = build_head(det_head)
        if self.task_config['with_map']:
            self.map_head = build_head(map_head)
        if self.task_config['with_motion_plan']:
            self.motion_plan_head = build_head(motion_plan_head)
        self.img_feature_res = img_feature_res

    def init_weights(self):
        if self.task_config['with_det']:
            self.det_head.init_weights()
        if self.task_config['with_map']:
            self.map_head.init_weights()
        if self.task_config['with_motion_plan']:
            self.motion_plan_head.init_weights()

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        unformated_feature_maps: Optional[Union[torch.Tensor, List]],
        metas: dict, 
    ):
        if self.task_config['with_det']:
            det_output = self.det_head(feature_maps, metas)
        else:
            det_output = None

        if self.task_config['with_map']:
            map_output = self.map_head(feature_maps, metas)
        else:
            map_output = None
        
        if self.task_config['with_motion_plan']:
            planning_output, motion_output = self.motion_plan_head(feature_maps, metas, debug=False)
        else:
            planning_output, motion_output = None, None

        return det_output, map_output, planning_output, motion_output

    def loss(self, model_outs, data):
        det_output, map_output, planning_output, motion_output = model_outs
        losses = dict()
        if self.task_config['with_det']:
            loss_det = self.det_head.loss(det_output, data)
            losses.update(loss_det)
        
        if self.task_config['with_map']:
            loss_map = self.map_head.loss(map_output, data)
            losses.update(loss_map)
            
        # if self.task_config['with_motion_plan']:
        #     motion_loss_cache = dict(
        #         indices=self.det_head.sampler.indices, 
        #     )
        #     loss_motion = self.motion_plan_head.loss(
        #         motion_output, 
        #         planning_output, 
        #         data, 
        #         motion_loss_cache
        #     )
        #     losses.update(loss_motion)
        
        if self.task_config['with_motion_plan']:
            # motion_loss_cache = dict(
            #     indices=self.det_head.sampler.indices, 
            # )
            loss_plan = self.motion_plan_head.loss(
                planning_output, 
                data, 
            )
            losses.update(loss_plan)

        return losses

    def post_process(self, model_outs, data):
        det_output, map_output, planning_output, motion_output = model_outs
        if self.task_config['with_det']:
            det_result = self.det_head.post_process(det_output)
            batch_size = len(det_result)
        
        if self.task_config['with_map']:
            map_result= self.map_head.post_process(map_output)
            batch_size = len(map_result)

        if self.task_config['with_motion_plan']:

            planning_result, motion_result = self.motion_plan_head.post_process(planning_output, keep_mm=True)
            # planning_result, motion_result = self.motion_plan_head.post_process(
            #     # det_output,
            #     # motion_output, 
            #     planning_output,
            #     data,
            # )

        results = [dict()] * batch_size
        for i in range(batch_size):
            if self.task_config['with_det']:
                results[i].update(det_result[i])
            if self.task_config['with_map']:
                results[i].update(map_result[i])
            if self.task_config['with_motion_plan']:
                if motion_result is not None:
                    results[i].update(motion_result[i])
                results[i].update(planning_result[i])

        return results
