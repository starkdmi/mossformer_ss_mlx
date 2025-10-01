import mlx.core as mx
import mlx.nn as nn

class CLayerNorm_MLX(nn.Module):
    """
    MLX implementation of CLayerNorm (Channel-wise Layer Normalization).
    
    This class applies layer normalization along the channel dimension.
    Unlike the PyTorch version which expects [N, C, T], this MLX version
    works directly with [N, T, C] format to avoid unnecessary transpositions.
    
    Args:
        normalized_shape: Input shape from last dimension
        eps: Small value for numerical stability (default: 1e-8)
        elementwise_affine: Whether to use learnable affine parameters (default: True)
    
    Shape:
        - Input: [batch_size, sequence_length, channels]
        - Output: [batch_size, sequence_length, channels]
    """
    
    def __init__(self, normalized_shape, eps=1e-8, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = mx.ones((normalized_shape,))
            self.bias = mx.zeros((normalized_shape,))
        else:
            self.weight = None
            self.bias = None
    
    def __call__(self, x):
        """Forward pass applying channel-wise layer normalization."""
        if x.ndim != 3:
            raise RuntimeError(f'CLayerNorm_MLX only accepts 3-D tensor as input, got {x.ndim}D')
        
        # x is already in [N, T, C] format
        # Apply LayerNorm along the channel dimension (last dimension)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        
        return x