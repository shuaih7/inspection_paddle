from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import importlib
import os
import sys

import yaml
import copy
import collections

# Other inner-folder imports are neglected

class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))


global_config = AttrDict()


def create(cls_or_name, **kwargs):
    """
    Create an instance of given module class.

    Args:
        cls_or_name (type or str): Class of which to create instance.

    Returns: instance of type `cls_or_name`
    """
    assert type(cls_or_name) in [type, str
                                 ], "should be a class or name of a class"
    name = type(cls_or_name) == str and cls_or_name or cls_or_name.__name__
    assert name in global_config and \
        isinstance(global_config[name], SchemaDict), \
        "the module {} is not registered".format(name)
    config = global_config[name]
    config.update(kwargs)
    config.validate()
    cls = getattr(config.pymodule, name)

    kwargs = {}
    kwargs.update(global_config[name])

    # parse `shared` annoation of registered modules
    if getattr(config, 'shared', None):
        for k in config.shared:
            target_key = config[k]
            shared_conf = config.schema[k].default
            assert isinstance(shared_conf, SharedConfig)
            if target_key is not None and not isinstance(target_key,
                                                         SharedConfig):
                continue  # value is given for the module
            elif shared_conf.key in global_config:
                # `key` is present in config
                kwargs[k] = global_config[shared_conf.key]
            else:
                kwargs[k] = shared_conf.default_value

    # parse `inject` annoation of registered modules
    if getattr(config, 'inject', None):
        for k in config.inject:
            target_key = config[k]
            # optional dependency
            if target_key is None:
                continue
            # also accept dictionaries and serialized objects
            if isinstance(target_key, dict) or hasattr(target_key, '__dict__'):
                continue
            elif isinstance(target_key, str):
                if target_key not in global_config:
                    raise ValueError("Missing injection config:", target_key)
                target = global_config[target_key]
                if isinstance(target, SchemaDict):
                    kwargs[k] = create(target_key)
                elif hasattr(target, '__dict__'):  # serialized object
                    kwargs[k] = target
            else:
                raise ValueError("Unsupported injection type:", target_key)
    # prevent modification of global config values of reference types
    # (e.g., list, dict) from within the created module instances
    kwargs = copy.deepcopy(kwargs)
    return cls(**kwargs)
    

if __name__ == "__main__":
    model = create("YOLOv3")