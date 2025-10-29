import tensorflow as tf
from tflib import param

def Layernorm(
    name,
    norm_axes,
    inputs,
    epsilon=1e-5
):
    """
    Layer normalization.
    
    Args:
        name: scope name
        norm_axes: axes to normalize over
        inputs: input tensor
        epsilon: small constant for numerical stability
    """
    
    mean = tf.reduce_mean(inputs, axis=norm_axes, keepdims=True)
    variance = tf.math.reduce_variance(inputs, axis=norm_axes, keepdims=True)
    
    # Parameters
    shape = inputs.shape
    param_shape = [shape[i] if i not in norm_axes else 1 for i in range(len(shape))]
    
    gamma = param(
        name + '.gamma',
        tf.ones(param_shape)
    )
    
    beta = param(
        name + '.beta',
        tf.zeros(param_shape)
    )
    
    normalized = (inputs - mean) / tf.sqrt(variance + epsilon)
    output = gamma * normalized + beta
    
    return output
