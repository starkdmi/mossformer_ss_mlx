import mlx.core as mx
import mlx.nn as nn

class DilatedDenseNet_MLX(nn.Module):
    """
    MLX implementation of DilatedDenseNet with mathematical equivalence to PyTorch.
    
    This architecture enables wider receptive fields while maintaining a lower number of parameters.
    It consists of multiple convolutional layers with dilation rates that increase at each layer.
    
    Args:
        depth (int): Number of convolutional layers in the network (default: 4)
        lorder (int): Base length order for convolutions (default: 20)
        in_channels (int): Number of input channels for the first layer (default: 64)
    
    Shape:
        - Input: (batch, height, width, in_channels) - MLX uses channels-last
        - Output: (batch, height, width, in_channels) - same spatial dimensions
    """
    
    def __init__(self, depth=4, lorder=20, in_channels=64):
        super().__init__()
        
        self.depth = depth
        self.in_channels = in_channels
        self.lorder = lorder
        self.twidth = lorder * 2 - 1  # Width of the kernel
        self.kernel_size = (self.twidth, 1)  # Kernel size for convolutions
        
        # Create layers dynamically based on depth
        self.convs = []
        self.norms = []
        self.prelus = []
        
        for i in range(self.depth):
            dil = 2 ** i  # Calculate dilation rate: 1, 2, 4, 8, ...
            
            # MLX Conv2d expects (kernel_h, kernel_w, in_channels, out_channels)
            # Input will have (i+1)*in_channels due to concatenation
            conv = nn.Conv2d(
                in_channels=(i + 1) * self.in_channels,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                stride=(1, 1),
                padding=(0, 0),  # We'll handle padding manually
                dilation=(dil, 1),
                groups=self.in_channels,  # Depthwise convolution
                bias=False
            )
            self.convs.append(conv)
            
            # MLX GroupNorm with num_groups=channels simulates InstanceNorm
            # For InstanceNorm2d(channels, affine=True), use GroupNorm with num_groups=channels
            norm = nn.GroupNorm(num_groups=self.in_channels, dims=self.in_channels, affine=True)
            self.norms.append(norm)
            
            # PReLU activation
            prelu = nn.PReLU(num_parameters=self.in_channels)
            self.prelus.append(prelu)
    
    def __call__(self, x):
        """
        Forward pass for the DilatedDenseNet model.
        
        Args:
            x (mx.array): Input tensor of shape (batch, height, width, in_channels)
        
        Returns:
            mx.array: Output tensor after applying dense layers
        """
        skip = x  # Initialize skip connection
        
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.lorder + (dil - 1) * (self.lorder - 1) - 1
            
            # Apply padding - MLX pad format: [(dim0_before, dim0_after), ...]
            # PyTorch ConstantPad2d((1, 1, 1, 0)) → MLX pad with [(0,0), (1,0), (1,1), (0,0)]
            # For this case: pad top and bottom by pad_length, no padding on width
            out = mx.pad(skip, [(0, 0), (pad_length, pad_length), (0, 0), (0, 0)], constant_values=0.0)
            
            # Apply convolution
            out = self.convs[i](out)
            
            # Apply normalization
            # GroupNorm in MLX expects (batch, ..., channels) which matches our layout
            out = self.norms[i](out)
            
            # Apply PReLU activation
            out = self.prelus[i](out)
            
            # Concatenate the output with the skip connection along channel dimension
            # MLX uses channels-last, so concatenate on axis=-1
            skip = mx.concatenate([out, skip], axis=-1)
        
        return out  # Return the final output (not the concatenated skip)