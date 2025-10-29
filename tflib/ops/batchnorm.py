import tensorflow as tf
from tflib import param

def Batchnorm(
    name,
    axes,
    inputs,
    is_training=True,
    momentum=0.99,
    epsilon=1e-5
):
    """
    Batch normalization.
    
    Args:
        name: scope name
        axes: axes to normalize over (e.g., [0, 2] for [batch, channels, length])
        inputs: input tensor
        is_training: whether in training mode
        momentum: momentum for moving average
        epsilon: small constant for numerical stability
    """
    
    shape = inputs.shape
    
    # Parameters for channels
    if 1 in axes:
        # Normalizing over channel dimension
        param_shape = [shape[1], 1]
    else:
        param_shape = [shape[1]]
    
    gamma = param(
        name + '.gamma',
        tf.ones(param_shape)
    )
    
    beta = param(
        name + '.beta',
        tf.zeros(param_shape)
    )
    
    moving_mean = param(
        name + '.moving_mean',
        tf.zeros(param_shape)
    )
    
    moving_variance = param(
        name + '.moving_variance',
        tf.ones(param_shape)
    )
    
    if is_training:
        # Calculate batch statistics
        mean = tf.reduce_mean(inputs, axis=axes, keepdims=True)
        variance = tf.math.reduce_variance(inputs, axis=axes, keepdims=True)
        
        # Update moving averages
        moving_mean.assign(
            momentum * moving_mean + (1 - momentum) * tf.squeeze(mean)
        )
        moving_variance.assign(
            momentum * moving_variance + (1 - momentum) * tf.squeeze(variance)
        )
        
    else:
        # Use moving averages
        mean = tf.reshape(moving_mean, param_shape)
        variance = tf.reshape(moving_variance, param_shape)
    
    # Normalize
    normalized = (inputs - mean) / tf.sqrt(variance + epsilon)
    
    # Scale and shift
    output = gamma * normalized + beta
    
    return output
