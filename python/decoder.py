import mlx.core as mx
import mlx.nn as nn

class Decoder_MLX(nn.Module):
    """MLX implementation of PyTorch's ConvTranspose1d wrapper (Decoder).
    
    A clean, production-ready implementation that matches PyTorch's behavior.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Size of the convolving kernel
    stride : int
        Stride of the convolution. Default: 1
    padding : int
        Zero-padding added to both sides of the input. Default: 0
    output_padding : int
        Additional size added to one side of the output shape. Default: 0
    groups : int
        Number of blocked connections from input to output. Default: 1
    bias : bool
        If True, adds a learnable bias to the output. Default: True
    dilation : int
        Spacing between kernel elements. Default: 1
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
    ):
        super().__init__()
        
        # Validate parameters
        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")
        
        # Store configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        
        # Initialize weights in MLX format: (out_channels, kernel_size, in_channels // groups)
        # This avoids transformation during forward pass
        weight_shape = (out_channels, kernel_size, in_channels // groups)
        
        # Xavier uniform initialization (matching PyTorch)
        fan_in = in_channels * kernel_size
        fan_out = out_channels * kernel_size
        bound = mx.sqrt(6.0 / (fan_in + fan_out))
        self.weight = mx.random.uniform(-bound, bound, weight_shape)
        
        # Initialize bias
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None
    
    def __call__(self, x):
        """Forward pass matching PyTorch Decoder behavior.
        
        Parameters
        ----------
        x : mx.array
            Input tensor with shape [B, C, L] or [C, L]
            where B = batch size, C = channels, L = sequence length
            
        Returns
        -------
        mx.array
            Output tensor with appropriate shape based on input
        """
        if x.ndim not in [2, 3]:
            raise RuntimeError(
                f"{self.__class__.__name__} accepts 2D or 3D tensor as input, got {x.ndim}D"
            )
        
        # Handle 2D input: PyTorch interprets as [B, L] and adds channel dim
        # [B, L] -> [B, 1, L] via unsqueeze at position 1
        if x.ndim == 2:
            x = mx.expand_dims(x, 1)  # [B, L] -> [B, 1, L]
        
        # Transform: PyTorch channels-first to MLX channels-last
        # [B, C, L] -> [B, L, C]
        x = mx.transpose(x, (0, 2, 1))
        
        # Apply native MLX conv_transpose1d
        output = mx.conv_transpose1d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            output_padding=self.output_padding,
            groups=self.groups
        )
        
        # Add bias if present
        if self.bias is not None:
            # output shape: [B, L_out, C_out]
            # bias shape: [C_out]
            output = output + self.bias
        
        # Transform back: MLX channels-last to PyTorch channels-first
        # [B, L_out, C_out] -> [B, C_out, L_out]
        output = mx.transpose(output, (0, 2, 1))
        
        # Handle output squeezing to match PyTorch behavior
        squeezed = mx.squeeze(output)
        if squeezed.ndim == 1:
            # If squeeze results in 1D, only squeeze specific dimension
            output = mx.squeeze(output, axis=1)
        else:
            output = squeezed
        
        return output