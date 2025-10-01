import mlx.core as mx
import mlx.nn as nn

from ffconvm import FFConvM_MLX
from unideepfsmn_dilated import UniDeepFsmn_dilated_MLX

class Gated_FSMN_dilated_MLX(nn.Module):
    """
    MLX implementation of Gated FSMN with dilated convolutions.
    
    This module implements a gated mechanism using two parallel feedforward 
    convolutions to generate the input for a dilated FSMN. The gated outputs 
    are combined to enhance the input features, allowing for better speech 
    enhancement performance.
    
    Args:
        in_channels (int): Number of input channels (features)
        out_channels (int): Number of output channels (features)
        lorder (int): Order of the FSMN
        hidden_size (int): Number of hidden units in the feedforward layers
    
    Shape:
        - Input: (batch, time, in_channels)
        - Output: (batch, time, in_channels) - residual connection ensures same shape
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lorder: int,
        hidden_size: int
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lorder = lorder
        self.hidden_size = hidden_size
        
        # Feedforward convolution for the u-gate
        self.to_u = FFConvM_MLX(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        
        # Feedforward convolution for the v-gate
        self.to_v = FFConvM_MLX(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        
        # Initialize the dilated FSMN
        self.fsmn = UniDeepFsmn_dilated_MLX(in_channels, out_channels, lorder, hidden_size)
        
        # Track training state for dropout consistency
        self._training = True
    
    def eval(self):
        """Set model to evaluation mode."""
        self._training = False
        self.to_u.eval()
        self.to_v.eval()
        return self
    
    def train(self):
        """Set model to training mode."""
        self._training = True
        self.to_u.train()
        self.to_v.train()
        return self
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through the Gated FSMN module.
        
        Following the exact PyTorch implementation pattern:
        1. Store input for residual connection
        2. Process through u-gate and v-gate branches
        3. Apply dilated FSMN to u-gate output
        4. Combine with gating: v * u + input
        
        Args:
            x (mx.array): Input tensor of shape (batch, time, in_channels)
        
        Returns:
            mx.array: Output tensor after processing through the gated FSMN
        """
        # Store the original input for residual connection
        input_residual = x
        
        # Process input through u-gate (will go through FSMN)
        x_u = self.to_u(x)
        
        # Process input through v-gate (acts as gate)
        x_v = self.to_v(x)
        
        # Apply dilated FSMN to u-gate output
        x_u = self.fsmn(x_u)
        
        # Combine the outputs from u-gate and v-gate with the original input
        # Gated output with residual connection: x = x_v * x_u + input
        x = x_v * x_u + input_residual
        
        return x