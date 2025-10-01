import mlx.core as mx
import mlx.nn as nn

from gated_fsmn_dilated import Gated_FSMN_dilated_MLX
from clayernorm import CLayerNorm_MLX

class Gated_FSMN_Block_Dilated_MLX(nn.Module):
    """
    MLX implementation of Gated_FSMN_Block with mathematical equivalence to PyTorch.
    
    A 1-D convolutional block that incorporates a gated FSMN.
    This block consists of:
    1. Conv1d layer with PReLU activation
    2. CLayerNorm normalization
    3. Gated FSMN module
    4. Another CLayerNorm
    5. Final Conv1d projection
    6. Residual connection
    
    Args:
        dim (int): Dimensionality of the input/output
        inner_channels (int): Number of channels in the inner layers (default: 256)
        group_size (int): Size of the groups for normalization (default: 256)
        norm_type (str): Type of normalization to use ('scalenorm' or 'layernorm')
    
    Shape:
        - Input: [batch_size, seq_length, dim]
        - Output: [batch_size, seq_length, dim]
    """
    
    def __init__(self, dim, inner_channels=256, group_size=256, norm_type='scalenorm'):
        super().__init__()
        
        self.dim = dim
        self.inner_channels = inner_channels
        self.group_size = group_size
        self.norm_type = norm_type
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=dim,
            out_channels=inner_channels,
            kernel_size=1,
            bias=True
        )
        
        # PReLU activation
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        
        # Normalization layers
        self.norm1 = CLayerNorm_MLX(inner_channels)
        self.norm2 = CLayerNorm_MLX(inner_channels)
        
        # Gated FSMN
        self.gated_fsmn = Gated_FSMN_dilated_MLX(
            in_channels=inner_channels,
            out_channels=inner_channels,
            lorder=20,
            hidden_size=inner_channels
        )
        
        # Final convolutional layer
        self.conv2 = nn.Conv1d(
            in_channels=inner_channels,
            out_channels=dim,
            kernel_size=1,
            bias=True
        )
        
        # Track training mode
        self._training = True
    
    def eval(self):
        """Set model to evaluation mode."""
        self._training = False
        self.gated_fsmn.eval()
        return self
    
    def train(self):
        """Set model to training mode."""
        self._training = True
        self.gated_fsmn.train()
        return self
    
    def __call__(self, x):
        """
        Forward pass for the Gated FSMN Block.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, dim]
        
        Returns:
            Output tensor of shape [batch_size, seq_length, dim]
        """
        residual = x

        # First convolution - input is already [B, T, D]
        x = self.conv1(x)

        # PReLU activation
        x = self.prelu(x)
        
        # First normalization (now accepts [B, T, C] directly)
        x = self.norm1(x)
        
        # Gated FSMN (expects [B, T, C])
        x = self.gated_fsmn(x)
        
        # Second normalization (now accepts [B, T, C] directly)
        x = self.norm2(x)
        
        # Final convolution
        x = self.conv2(x)

        # Residual connection
        return x + residual