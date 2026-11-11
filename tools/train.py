# Copyright (c) OpenMMLab. All rights reserved.
# from __future__ import division
import sys
import os
# from aidisdk.utils import running_in_cluster
from torch import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.executable, os.path.abspath(__file__))
# import init_paths # for conda pkgs submitting method
import argparse
import copy
import mmcv
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from torch import distributed as dist
from datetime import timedelta

import cv2
from tools.training_schedule import resolve_training_schedule

cv2.setNumThreads(8)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    # REQUIRED: never default to e2e_b2d.py (historical footgun — silent wrong recipe).
    parser.add_argument(
        "--config",
        required=True,
        help="train config file path (explicit; no default)",
    )
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--resume-from", help="the checkpoint file to resume from"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="number of gpus to use "
        "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="ids of gpus to use "
        "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
    )
    parser.add_argument("--gpus-per-machine", type=int, default=8)
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi", "mpi_nccl"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="whether to be in debug mode.",
    )

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both specified, "
            "--options is deprecated in favor of --cfg-options"
        )


    if args.options:
        warnings.warn("--options is deprecated in favor of --cfg-options")
        args.cfg_options = args.options

    return args


def main():
    print("START SETTING UP TRAINING")

    args = parse_args()
    cfg_path = osp.abspath(args.config)
    if not osp.isfile(cfg_path):
        raise FileNotFoundError(f"config not found: {cfg_path}")
    cfg_base = osp.basename(cfg_path)
    # Guard against accidental legacy default / wrong package.
    if cfg_base == "e2e_b2d.py" and os.environ.get("ALLOW_LEGACY_E2E_B2D", "") != "1":
        raise RuntimeError(
            "Refusing to train with projects_sp/configs/e2e_b2d.py "
            "(legacy hardcode). Pass the intended stage config explicitly, "
            "or set ALLOW_LEGACY_E2E_B2D=1 only if you really mean it."
        )
    print("=" * 72)
    print("CONFIG FILE:", cfg_path)
    print("CONFIG BASE:", cfg_base)
    print("=" * 72)
    cfg = Config.fromfile(cfg_path)
    if args.debug:
        cfg.batch_size = 1
        cfg.num_machines = 1
        cfg.num_gpus = 1
        cfg.total_batch_size = cfg.batch_size * cfg.num_machines * cfg.num_gpus
        cfg.data.samples_per_gpu = cfg.batch_size
        print("DEBUG MODE, SET BATCH SIZE TO", cfg.batch_size)


    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            from projects_sp.mmdet3d_plugin.apis.train import custom_train_model

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./output", osp.splitext(osp.basename(args.config))[0]
        )
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer["lr"] = cfg.optimizer["lr"] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    elif args.launcher == "mpi_nccl":
        distributed = True

        import mpi4py.MPI as MPI

        comm = MPI.COMM_WORLD
        mpi_local_rank = comm.Get_rank()
        mpi_world_size = comm.Get_size()
        print(
            "MPI local_rank=%d, world_size=%d"
            % (mpi_local_rank, mpi_world_size)
        )

        # num_gpus = torch.cuda.device_count()
        device_ids_on_machines = list(range(args.gpus_per_machine))
        str_ids = list(map(str, device_ids_on_machines))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str_ids)
        torch.cuda.set_device(mpi_local_rank % args.gpus_per_machine)

        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=mpi_world_size,
            rank=mpi_local_rank,
            timeout=timedelta(seconds=36000),
        )

        cfg.gpu_ids = range(mpi_world_size)
        print("cfg.gpu_ids:", cfg.gpu_ids)
    else:
        distributed = True
        print("GETTING DIST INFO")
        init_dist(
            args.launcher, timeout=timedelta(seconds=36000), **cfg.dist_params
        )
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
        print("cfg.gpu_ids:", cfg.gpu_ids)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level
    )

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info(
        "Environment info:\n" + dash_line + env_info + "\n" + dash_line
    )
    meta["env_info"] = env_info

    # log some basic info
    logger.info(f"Distributed training: {distributed}")

    # set random seeds
    if args.seed is not None:
        logger.info(
            f"Set random seed to {args.seed}, "
            f"deterministic: {args.deterministic}"
        )
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta["seed"] = args.seed
    meta["exp_name"] = osp.basename(args.config)

    cfg.data.train.work_dir = cfg.work_dir
    cfg.data.val.work_dir = cfg.work_dir
    datasets = [build_dataset(cfg.data.train)]
    schedule_world_size = get_dist_info()[1] if distributed else 1
    if cfg.get('auto_schedule') and not distributed and len(cfg.gpu_ids) != 1:
        raise ValueError('Automatic schedule with GroupInBatchSampler requires one process per GPU')
    schedule = resolve_training_schedule(cfg, len(datasets[0]), schedule_world_size)
    if schedule is not None:
        logger.info('[schedule] %s', schedule)
        meta['resolved_schedule'] = schedule
    if not args.debug:
        cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
        meta['config'] = cfg.pretty_text
        logger.info('Config:\n%s', cfg.pretty_text)
    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model.init_weights()

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if "dataset" in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
        )
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    print("BUILDING DETECTOR MODEL")
    if hasattr(cfg, "plugin"):
        custom_train_model(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate and cfg.get('enable_online_evaluation', True)),
            timestamp=timestamp,
            meta=meta,
        )
    else:
        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate and cfg.get('enable_online_evaluation', True)),
            timestamp=timestamp,
            meta=meta,
        )


if __name__ == "__main__":
    # fork + CUDA is a common SIGSEGV source (esp. H20/torch2.8). Prefer spawn
    # when CUDA is visible; override with B2D_MP_START=fork|spawn.
    import os as _os
    _mp = _os.environ.get("B2D_MP_START")
    if _mp is None:
        _mp = "fork"
    try:
        torch.multiprocessing.set_start_method(_mp)
    except RuntimeError:
        pass
    print(f"[train] multiprocessing start_method={_mp}")
    main()
