"""Runtime settings shared by training and evaluation entry points."""

plugin = True

plugin_dir = 'projects_sp/mmdet3d_plugin/'

dist_params = {'backend': 'nccl'}

fp16 = {'loss_scale': 32.0}
