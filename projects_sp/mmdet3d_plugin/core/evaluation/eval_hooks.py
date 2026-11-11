# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.

import bisect
import os
import os.path as osp
import shutil
from numbers import Integral

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from mmcv.runner import HOOKS, Hook
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.core.evaluation.eval_hooks import DistEvalHook


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list]
    )
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list]
    )
    return dynamic_milestones, dynamic_intervals


class CustomDistEvalHook(BaseDistEvalHook):
    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(CustomDistEvalHook, self).__init__(*args, **kwargs)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            (
                self.dynamic_milestones,
                self.dynamic_intervals,
            ) = _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if (
                    isinstance(module, _BatchNorm)
                    and module.track_running_stats
                ):
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, ".eval_hook")

        from projects_sp.mmdet3d_plugin.apis.test import (
            custom_multi_gpu_test,
        )  # to solve circlur  import

        # Free train activations before val (helps 24GB 4090 end-of-train eval).
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
        )
        if runner.rank == 0:
            print("\n")
            runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)

            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
                # Stable alias required by the open-source retrain protocol.
                # MMCV writes best_{metric}.pth; also keep best.pth.
                best_named = osp.join(
                    runner.work_dir, f"best_{self.key_indicator}.pth"
                )
                best_alias = osp.join(runner.work_dir, "best.pth")
                if osp.isfile(best_named):
                    shutil.copy2(best_named, best_alias)
                    runner.logger.info(
                        f"Updated best.pth alias from {best_named} "
                        f"(score={key_score})"
                    )


@HOOKS.register_module()
class LastCheckpointAliasHook(Hook):
    """Refresh ``work_dir/last.pth`` from the newest periodic checkpoint.

    Paired with ``CustomDistEvalHook`` ``best.pth`` so retrain runs always
    expose ``best.pth`` + ``last.pth`` (no hand-picked mid iters).
    """

    def __init__(self, interval=None):
        self.interval = interval
        self._last_src = None
        self._progress_cache = {}

    def after_train_iter(self, runner):
        if self.interval is not None and not self.every_n_iters(runner, self.interval):
            return
        self._alias_latest(runner)

    def after_run(self, runner):
        self._alias_latest(runner)

    def _alias_latest(self, runner):
        if getattr(runner, "rank", 0) != 0:
            return
        work_dir = runner.work_dir
        if work_dir is None or not osp.isdir(work_dir):
            return
        candidates = []
        for name in os.listdir(work_dir):
            if not name.endswith(".pth"):
                continue
            if name in ("best.pth", "last.pth") or name.startswith("best_"):
                continue
            path = osp.join(work_dir, name)
            if osp.isfile(path):
                stat = os.stat(path)
                signature = (path, stat.st_size, stat.st_mtime_ns)
                if signature not in self._progress_cache:
                    checkpoint = torch.load(path, map_location='cpu')
                    metadata = checkpoint.get('meta', {})
                    iteration = metadata.get('iter', 0)
                    epoch = metadata.get('epoch', 0)
                    if not any(key in metadata for key in ('iter', 'epoch')) or not all(
                        isinstance(value, Integral) and not isinstance(value, bool) and value >= 0
                        for value in (iteration, epoch)
                    ):
                        runner.logger.warning(f'Cannot determine checkpoint progress: {path}')
                        continue
                    self._progress_cache[signature] = (int(iteration), int(epoch))
                    del checkpoint
                candidates.append((self._progress_cache[signature], signature))
        if not candidates:
            return
        progress, signature = max(candidates)
        latest = signature[0]
        if signature == self._last_src:
            return
        dest = osp.join(work_dir, "last.pth")
        try:
            shutil.copy2(latest, dest)
            self._last_src = signature
            runner.logger.info(f"Updated last.pth alias from {latest} (iter={progress[0]}, epoch={progress[1]})")
        except OSError as error:
            runner.logger.warning(f'Failed to update last.pth: {error}')
