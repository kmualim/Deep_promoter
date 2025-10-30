import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional
import locale

locale.setlocale(locale.LC_ALL, '')

_params = {}
_param_aliases = {}

def param(name, *args, **kwargs):
    """
    A wrapper for `tf.Variable` which enables parameter sharing in models.
    
    Creates and returns TensorFlow variables similarly to `tf.Variable`, 
    except if you try to create a param with the same name as a 
    previously-created one, `param(...)` will just return the old one instead of 
    making a new one.

    This constructor also adds a `param` attribute to the variables it 
    creates, so that you can easily search a graph for all params.
    """

    if name not in _params:
        kwargs['name'] = name
        var = tf.Variable(*args, **kwargs)
        var.param = True
        _params[name] = var
    
    result = _params[name]
    i = 0
    # Use .ref() as the key for TF 2.x compatibility
    while result.ref() in _param_aliases:
        i += 1
        result = _param_aliases[result.ref()]
        if i > 100:  # Prevent infinite loops
            raise RuntimeError(f"Circular alias detected for parameter {name}")
    
    return result

def params_with_name(name):
    """Get all parameters whose name contains the given string."""
    return [p for n, p in _params.items() if name in n]

def get_all_params():
    """Get all parameters as a dictionary."""
    return _params.copy()

def delete_all_params():
    """Delete all parameters."""
    _params.clear()
    _param_aliases.clear()

def alias_params(replace_dict):
    """
    Create parameter aliases.
    
    Args:
        replace_dict: Dictionary mapping old parameters to new ones
    """
    for old, new in replace_dict.items():
        # Use .ref() for dictionary keys in TF 2.x
        old_ref = old.ref() if isinstance(old, tf.Variable) else old
        new_ref = new.ref() if isinstance(new, tf.Variable) else new
        _param_aliases[old_ref] = new_ref

def delete_param_aliases():
    """Delete all parameter aliases."""
    _param_aliases.clear()

def print_params_info(params=None):
    """
    Print information about parameters.
    
    Args:
        params: List of parameters to print info about. If None, uses all params.
    """
    if params is None:
        params = list(_params.values())
    
    params = sorted(params, key=lambda p: p.name)
    
    print("Parameters:")
    total_param_count = 0
    
    for p in params:
        shape = p.shape
        param_count = np.prod(shape)
        total_param_count += param_count
        
        print("\t{} (shape: {}, count: {:,})".format(
            p.name,
            tuple(shape),
            int(param_count)
        ))
    
    print("Total parameter count: {:,}".format(int(total_param_count)))

def print_model_settings(locals_):
    """Print uppercase local variables (typically used for model settings)."""
    print("Uppercase local vars:")
    all_vars = [
        (k, v) for (k, v) in locals_.items() 
        if (k.isupper() and k not in ['T', 'SETTINGS', 'ALL_SETTINGS'])
    ]
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print("\t{}: {}".format(var_name, var_value))

def print_model_settings_dict(settings):
    """Print settings from a dictionary."""
    print("Settings dict:")
    all_vars = sorted(settings.items(), key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print("\t{}: {}".format(var_name, var_value))

def count_params(params=None):
    """
    Count total number of parameters.
    
    Args:
        params: List of parameters to count. If None, counts all params.
        
    Returns:
        Total number of parameters
    """
    if params is None:
        params = list(_params.values())
    
    return sum(int(np.prod(p.shape)) for p in params)
