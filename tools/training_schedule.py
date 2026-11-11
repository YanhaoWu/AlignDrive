from numbers import Integral


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError('{} must be a positive integer, got {!r}'.format(name, value))
    return int(value)


def _resolve_milestones(value, half_training):
    if isinstance(value, dict):
        for key in value:
            value[key] = _resolve_milestones(value[key], half_training)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _resolve_milestones(item, half_training)
    elif isinstance(value, tuple):
        return tuple(_resolve_milestones(item, half_training) for item in value)
    elif isinstance(value, str) and value == 'auto_half_training':
        return half_training
    return value


def resolve_training_schedule(config, dataset_size, world_size):
    settings = config.get('auto_schedule')
    if not settings:
        return None
    if settings.get('policy') != 'dataset_global_batch_floor':
        raise ValueError('Unsupported auto_schedule policy')
    if config['runner']['type'] != 'IterBasedRunner':
        raise ValueError('Automatic nominal epochs require IterBasedRunner/GroupInBatchSampler')
    if config['data'].get('imgs_per_gpu') is not None:
        raise ValueError('Use data.samples_per_gpu, not the deprecated imgs_per_gpu')
    size = _positive_integer(dataset_size, 'dataset_size')
    replicas = _positive_integer(world_size, 'world_size')
    batch = _positive_integer(config['data']['samples_per_gpu'], 'data.samples_per_gpu')
    epochs = _positive_integer(config['num_epochs'], 'num_epochs')
    checkpoint_epochs = _positive_integer(config['checkpoint_epoch_interval'], 'checkpoint_epoch_interval')
    evaluation_epochs = _positive_integer(config['eval_epoch_interval'], 'eval_epoch_interval')
    global_batch = replicas * batch
    iterations = size // global_batch
    if iterations == 0:
        raise ValueError('Dataset is smaller than one global batch; nominal epoch would have zero iterations')
    if config['runner'].get('max_iters') is not None:
        raise ValueError('auto_schedule owns runner.max_iters; leave it None or disable auto_schedule')
    if config['checkpoint_config'].get('interval') is not None:
        raise ValueError('auto_schedule owns checkpoint_config.interval; configure checkpoint_epoch_interval')
    if config['evaluation'].get('interval') is not None:
        raise ValueError('auto_schedule owns evaluation.interval; configure eval_epoch_interval')
    max_iters = iterations * epochs
    config['batch_size'] = batch
    config['num_gpus'] = replicas
    config['total_batch_size'] = global_batch
    config['num_iters_per_epoch'] = iterations
    config['runner']['max_iters'] = max_iters
    config['checkpoint_config']['interval'] = iterations * checkpoint_epochs
    config['evaluation']['interval'] = iterations * evaluation_epochs
    for hook in config.get('custom_hooks', []):
        if hook['type'] == 'LastCheckpointAliasHook':
            if hook.get('interval') is not None:
                raise ValueError('auto_schedule owns LastCheckpointAliasHook.interval; leave it None')
            hook['interval'] = iterations * checkpoint_epochs
    for key in list(config.keys()):
        config[key] = _resolve_milestones(config[key], max_iters // 2)
    report = dict(
        policy=settings['policy'], sampler='GroupInBatchSampler',
        epoch_semantics='nominal sample budget, not an exhaustive dataset pass',
        dataset_size=size, world_size=replicas, samples_per_gpu=batch,
        global_batch=global_batch, num_iters_per_epoch=iterations,
        remainder_samples=size % global_batch, num_epochs=epochs,
        max_iters=max_iters, checkpoint_interval=iterations * checkpoint_epochs,
        evaluation_interval=iterations * evaluation_epochs,
        half_training_iter=max_iters // 2,
    )
    config['resolved_schedule'] = report
    return report
