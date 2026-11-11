from mmcv.runner import HOOKS, Hook
import gc
import torch


@HOOKS.register_module()
class PeriodicEmptyCacheHook(Hook):
    """Periodic CUDA cache clear to reduce fragmentation on 24GB GPUs.

    Named distinctly from mmcv.runner.hooks.EmptyCacheHook to avoid registry
    KeyError on import.

    Clears both before and after each train iter (when interval matches) so the
    allocator can coalesce free blocks before the backward peak.
    """

    def __init__(self, interval=1, with_gc=True):
        self.interval = max(int(interval), 1)
        self.with_gc = bool(with_gc)

    def _should_clear(self, runner):
        # runner.iter is 0-based before increment in some paths; use >=0.
        return runner.iter >= 0 and (runner.iter % self.interval == 0)

    def _clear(self):
        if not torch.cuda.is_available():
            return
        if self.with_gc:
            gc.collect()
        torch.cuda.empty_cache()

    def before_train_iter(self, runner):
        if self._should_clear(runner):
            self._clear()

    def after_train_iter(self, runner):
        if self._should_clear(runner):
            self._clear()


@HOOKS.register_module()
class DynamicLRMultHook(Hook):

    def __init__(self, freeze_until_iter=1000, lr_mult_map=None):
        """
        lr_mult_map: dict[str, Tuple[float, float]]
        模块名关键字（支持模糊匹配） -> (前期lr_mult, 后期lr_mult)
        例如：
        {
            'img_backbone': (1.0, 1.0),
            'img_neck': (1e-6, 1.0),
            'head.det_head': (1e-6, 1.0),
            'head.map_head': (1e-6, 1.0),
            'traj_layer': (1.0, 1.0)  # 所有包含 traj_layer 的模块都匹配
        }
        """
        self.freeze_until_iter = freeze_until_iter
        self.lr_mult_map = lr_mult_map or {}
        self.base_lr_map = {}       # group id -> base lr
        self.param_id_map = {}      # keyword -> set(param ids)
        self.updated = False

    def before_train_iter(self, runner):
        if runner.iter == self.freeze_until_iter and not self.updated:
            print(f"[Hook] lr_mult updated at iter {runner.iter}")
            self.updated = True

        if runner.iter > self.freeze_until_iter:
            return

        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        optimizer = runner.optimizer


        def get_module(name, model):
            for key in name.split('.'):
                model = getattr(model, key, None)
                if model is None:
                    return None
            return model

        # 构建 keyword -> param ids（只做一次）
        if not self.param_id_map:
            for keyword in self.lr_mult_map:
                matched_param_ids = set()
                for name, module in model.named_modules():
                    if keyword in name:
                        matched_param_ids.update(id(p) for p in module.parameters())
                if matched_param_ids:
                    self.param_id_map[keyword] = matched_param_ids

        # 遍历 param_group，检查是否属于任何目标模块
        for group in optimizer.param_groups:
            if not group['params']:
                continue
            group_param_ids = set(id(p) for p in group['params'])

            for keyword, (low_mult, high_mult) in self.lr_mult_map.items():
                module_param_ids = self.param_id_map.get(keyword)
                if not module_param_ids:
                    continue
                if group_param_ids & module_param_ids:
                    group_id = id(group)
                    if group_id not in self.base_lr_map:
                        self.base_lr_map[group_id] = group['lr']

                    base_lr = self.base_lr_map[group_id]
                    current_mult = low_mult if runner.iter < self.freeze_until_iter else high_mult
                    group['lr'] = base_lr * current_mult
