# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp


def get_config_path(model: str) -> str:
    """Get the config path of the model.

    Args:
        model (str): model name.

    Returns:
        str: config path.
    """

    if model == 'swinB_224':
        cfg_name = 'config_swinB_224.json'
    elif model == 'swinB_384':
        cfg_name = 'config_swinB_384.json'
    elif model == 'swinB_480':
        cfg_name = 'config_swinB_480.json'
    elif model == 'swinB_576':
        cfg_name = 'config_swinB_576.json'
    elif model == 'swinB_608':
        cfg_name = 'config_swinB_608.json'
    elif model == 'q2l':
        cfg_name = 'q2l_config.json'
    elif model == 'med':
        cfg_name = 'med_config.json'
    else:
        raise ValueError(f'Invalid model name {model}')

    return osp.join(osp.dirname(osp.dirname(__file__)), 'configs', cfg_name)
