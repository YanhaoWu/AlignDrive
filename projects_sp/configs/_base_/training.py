"""Shared optimizer, hooks, and dataset-driven training schedule."""

log_level = 'INFO'

resume_from = None

workflow = [('train', 1)]

batch_size = 8

num_epochs = 12

checkpoint_epoch_interval = 1

eval_epoch_interval = 12

auto_schedule = {'policy': 'dataset_global_batch_floor'}

optimizer = {
    'type': 'AdamW',
    'lr': 0.0002,
    'weight_decay': 0.001,
    'paramwise_cfg': {
        'custom_keys': {
            'img_backbone': {
                'lr_mult': 0.1
            }
        }
    }
}

optimizer_config = {'grad_clip': {'max_norm': 25, 'norm_type': 2}}

lr_config = {
    'policy': 'CosineAnnealing',
    'warmup': 'linear',
    'warmup_iters': 500,
    'warmup_ratio': 0.3333333333333333,
    'min_lr_ratio': 0.001
}

runner = {'type': 'IterBasedRunner', 'max_iters': None}

checkpoint_config = {'interval': None}

log_config = {
    'interval': 51,
    'hooks': [{
        'type': 'TextLoggerHook',
        'by_epoch': False
    }, {
        'type': 'TensorboardLoggerHook'
    }]
}

custom_hooks = [{
    'type': 'PeriodicEmptyCacheHook',
    'interval': 50,
    'with_gc': True,
    'priority': 'HIGHEST'
}, {
    'type': 'LastCheckpointAliasHook',
    'interval': None
}]

mp_start_method = 'fork'
