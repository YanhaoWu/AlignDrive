from numbers import Integral

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SyncModelIterationHook(Hook):
    def synchronize(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        iteration = int(runner.iter)
        for module in model.modules():
            if isinstance(getattr(module, 'iter_count', None), Integral):
                module.iter_count = iteration

    def before_run(self, runner):
        self.synchronize(runner)

    def before_train_iter(self, runner):
        self.synchronize(runner)
