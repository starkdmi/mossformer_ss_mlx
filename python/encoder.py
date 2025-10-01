import mlx.core as mx
import mlx.nn as nn

class Encoder_MLX(nn.Module):
    """MLX implementation of the Encoder module.
    
    Convolutional Encoder Layer converted from PyTorch to MLX.
    
    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    
    Example
    -------
    >>> x = mx.random.normal((2, 1000))
    >>> encoder = EncoderMLX(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    (2, 64, 499)
    """
    
    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super().__init__()
        
        # Store parameters
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        
        # Calculate stride based on the source:
        # - Basic Encoder uses: stride = kernel_size // 2
        # - MossFormer_MaskNet.conv1d_encoder uses: stride = 1 (default)
        if kernel_size == 1:
            # For 1x1 convolutions (like MossFormer_MaskNet.conv1d_encoder), use stride=1
            self.stride = 1
        else:
            # For basic Encoder, use stride = kernel_size // 2, but ensure minimum of 1
            self.stride = max(1, kernel_size // 2)
        
        # Initialize Conv1d parameters
        # MLX Conv1d expects (out_channels, in_channels, kernel_size)
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            bias=False
        )
    
    def __call__(self, x):
        """Return the encoded output.
        
        Arguments
        ---------
        x : mx.array
            Input tensor with dimensionality [B, L] or [B, C, L].
        
        Returns
        -------
        x : mx.array
            Encoded tensor with dimensionality [B, N, T_out].
            
        where B = Batchsize
              L = Number of timepoints
              N = Number of filters
              T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = mx.expand_dims(x, axis=1)
        
        # MLX conv1d expects (N, L, C_in), but we have (B, C, L)
        # Need to transpose to (B, L, C) for MLX
        x = mx.transpose(x, (0, 2, 1))  # (B, C, L) -> (B, L, C)
        
        # Apply conv1d
        x = self.conv1d(x)  # (B, L, C_out)
        
        # Transpose back to (B, C_out, L) to match PyTorch format
        x = mx.transpose(x, (0, 2, 1))  # (B, L, C_out) -> (B, C_out, L)
        
        # Apply ReLU activation
        x = nn.relu(x)
        
        return x