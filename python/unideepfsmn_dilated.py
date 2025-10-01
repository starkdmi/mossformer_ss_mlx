import mlx.core as mx
import mlx.nn as nn

from dilated_dense_net import DilatedDenseNet_MLX

class UniDeepFsmn_dilated_MLX(nn.Module):
    """
    MLX implementation of UniDeepFsmn_dilated with mathematical equivalence to PyTorch.
    
    UniDeepFsmn_dilated combines the UniDeepFsmn architecture with a dilated dense network 
    to enhance feature extraction while maintaining efficient computation.
    
    Args:
        input_dim (int): Dimension of the input features
        output_dim (int): Dimension of the output features
        lorder (int): Length of the order for the convolution layers
        hidden_size (int): Number of hidden units in the linear layer
        depth (int): Depth of the dilated dense network (default: 2)
    
    Shape:
        - Input: (batch, time, input_dim)
        - Output: (batch, time, input_dim) - residual connection ensures same shape
    """
    
    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None, depth=2):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        
        if lorder is None:
            return
            
        self.lorder = lorder
        self.hidden_size = hidden_size
        
        # Initialize layers
        # Linear transformation to hidden size
        self.linear = nn.Linear(input_dim, hidden_size)
        
        # Project hidden size to output dimension (no bias)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        
        # Dilated dense network for feature extraction
        self.conv = DilatedDenseNet_MLX(depth=self.depth, lorder=lorder, in_channels=output_dim)
    
    def __call__(self, input):
        """
        Forward pass for the UniDeepFsmn_dilated model.
        
        Args:
            input (mx.array): Input tensor of shape (batch, time, input_dim)
        
        Returns:
            mx.array: The output tensor of the same shape as input, enhanced by the network
        """
        # Apply linear layer followed by ReLU activation
        # f1 = mx.maximum(self.linear(input), 0)  # ReLU activation
        f1 = nn.relu(self.linear(input))
        
        # Project to output dimension
        p1 = self.project(f1)
        
        # Add a dimension for compatibility with Conv2d
        # (batch, time, channels) → (batch, time, 1, channels)
        x = mx.expand_dims(p1, axis=2)
        
        # Permute dimensions for convolution
        # PyTorch: (batch, time, 1, channels) → (batch, channels, time, 1)
        # MLX needs: (batch, time, 1, channels) which is already correct for channels-last
        # No permutation needed as MLX Conv2d expects (batch, height, width, channels)
        
        # Pass through the dilated dense network
        out = self.conv(x)
        
        # Remove the added dimension
        # (batch, time, 1, channels) → (batch, time, channels)
        out = mx.squeeze(out, axis=2)
        
        # Return enhanced input with residual connection
        return input + out