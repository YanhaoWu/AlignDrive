import copy
import os
import os.path as osp
import platform
import shutil
import time

import mmcv
import torch
from mmcv.runner import IterBasedRunner
from mmcv.parallel import is_module_wrapper
from mmcv.runner.checkpoint import CheckpointLoader, weights_to_cpu


class AlignDriveIterBasedRunner(IterBasedRunner):
    def save_checkpoint(self, out_dir, filename_tmpl='iter_{}.pth', meta=None,
                        save_optimizer=True, create_symlink=True):
        metadata = copy.deepcopy(self.meta or {})
        if meta is not None:
            if not isinstance(meta, dict):
                raise TypeError('checkpoint meta must be a dictionary')
            metadata.update(meta)
        metadata.update(epoch=self.epoch + 1, iter=self.iter + 1,
                        iteration_semantics='completed_training_steps')
        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        model = self.model.module if is_module_wrapper(self.model) else self.model
        metadata.update(mmcv_version=mmcv.__version__, time=time.asctime())
        if getattr(model, 'CLASSES', None) is not None:
            metadata['CLASSES'] = model.CLASSES
        checkpoint = dict(meta=metadata, state_dict=weights_to_cpu(model.state_dict()))
        if save_optimizer:
            if isinstance(self.optimizer, dict):
                checkpoint['optimizer'] = {name: optimizer.state_dict() for name, optimizer in self.optimizer.items()}
            elif self.optimizer is not None:
                checkpoint['optimizer'] = self.optimizer.state_dict()
        os.makedirs(out_dir, exist_ok=True)
        torch.save(checkpoint, filepath)
        if create_symlink:
            destination = osp.join(out_dir, 'latest.pth')
            if platform.system() == 'Windows':
                shutil.copy(filepath, destination)
            else:
                mmcv.symlink(filename, destination)

    def resume(self, checkpoint, resume_optimizer=True, map_location='default'):
        if map_location == 'default':
            map_location = (lambda storage, location: storage.cuda(torch.cuda.current_device())) if torch.cuda.is_available() else 'cpu'
        saved = CheckpointLoader.load_checkpoint(checkpoint, map_location=map_location, logger=self.logger)
        model = self.model.module if is_module_wrapper(self.model) else self.model
        state_dict = saved['state_dict'].copy()
        if hasattr(saved['state_dict'], '_metadata'):
            state_dict._metadata = saved['state_dict']._metadata
        for module_name, module in model.named_modules():
            for name in module._non_persistent_buffers_set:
                key = f'{module_name}.{name}' if module_name else name
                if key in state_dict:
                    buffer = module._buffers[name]
                    value = state_dict.pop(key)
                    if buffer is None or not torch.equal(value, buffer.to(device=value.device, dtype=value.dtype)):
                        raise ValueError(f'Non-persistent checkpoint buffer differs from configuration: {key}')
                    self.logger.warning(f'Ignoring legacy MMCV serialization of non-persistent buffer: {key}')
        model.load_state_dict(state_dict, strict=True)
        metadata = saved['meta']
        iteration = int(metadata['iter'])
        semantics = metadata.get('iteration_semantics')
        if semantics is None:
            iteration += 1
            self.logger.warning('Legacy MMCV iteration checkpoint: converting last zero-based index to completed steps')
        elif semantics != 'completed_training_steps':
            raise ValueError(f'Unsupported checkpoint iteration semantics: {semantics}')
        self._epoch = int(metadata['epoch'])
        self._iter = iteration
        self._inner_iter = iteration
        restored_meta = copy.deepcopy(metadata)
        restored_meta.update(self.meta or {})
        self.meta = restored_meta
        if resume_optimizer and 'optimizer' in saved:
            if isinstance(self.optimizer, torch.optim.Optimizer):
                self.optimizer.load_state_dict(saved['optimizer'])
            elif isinstance(self.optimizer, dict):
                for name, optimizer in self.optimizer.items():
                    optimizer.load_state_dict(saved['optimizer'][name])
            else:
                raise TypeError('Optimizer must be a torch optimizer or dictionary')
        self.logger.info(f'resumed from epoch: {self.epoch}, completed iter {self.iter}; fp16 metadata restored={"fp16" in self.meta}')
