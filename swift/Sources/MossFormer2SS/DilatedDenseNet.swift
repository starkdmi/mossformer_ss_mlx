import Foundation
import MLX
import MLXNN

/// MLX implementation of DilatedDenseNet with 1:1 mathematical equivalence to PyTorch.
///
/// This architecture enables wider receptive fields while maintaining a lower number of parameters.
/// It consists of multiple convolutional layers with dilation rates that increase at each layer.
public class DilatedDenseNet: Module {
    public let depth: Int
    public let lorder: Int
    public let inChannels: Int
    public let twidth: Int
    public let kernelSize: (Int, Int)
    
    // Model components
    @ModuleInfo var convs: [Conv2d]
    @ModuleInfo var norms: [GroupNorm]
    @ModuleInfo var prelus: [PReLU]
    
    /// Initialize DilatedDenseNet
    /// - Parameters:
    ///   - depth: Number of convolutional layers in the network (default: 4)
    ///   - lorder: Base length order for convolutions (default: 20)
    ///   - inChannels: Number of input channels for the first layer (default: 64)
    public init(
        depth: Int = 4,
        lorder: Int = 20,
        inChannels: Int = 64
    ) {
        self.depth = depth
        self.lorder = lorder
        self.inChannels = inChannels
        self.twidth = lorder * 2 - 1  // Width of the kernel
        self.kernelSize = (twidth, 1)  // Kernel size for convolutions
        
        // Create layers dynamically based on depth
        var convsArray: [Conv2d] = []
        var normsArray: [GroupNorm] = []
        var prelusArray: [PReLU] = []
        
        for i in 0..<depth {
            let dil = 1 << i  // Calculate dilation rate: 1, 2, 4, 8, ...
            
            // MLX Conv2d expects (kernel_h, kernel_w, in_channels, out_channels)
            // Input will have (i+1)*inChannels due to concatenation
            let conv = Conv2d(
                inputChannels: (i + 1) * inChannels,
                outputChannels: inChannels,
                kernelSize: IntOrPair((twidth, 1)),
                stride: IntOrPair((1, 1)),
                padding: IntOrPair((0, 0)),  // We'll handle padding manually
                dilation: IntOrPair((dil, 1)),
                groups: inChannels,  // Depthwise convolution
                bias: false
            )
            convsArray.append(conv)
            
            // MLX GroupNorm with num_groups=channels simulates InstanceNorm
            // For InstanceNorm2d(channels, affine=True), use GroupNorm with num_groups=channels
            let norm = GroupNorm(groupCount: inChannels, dimensions: inChannels)
            normsArray.append(norm)
            
            // PReLU activation
            let prelu = PReLU(count: inChannels)
            prelusArray.append(prelu)
        }
        
        self.convs = convsArray
        self.norms = normsArray
        self.prelus = prelusArray
        
        super.init()
    }
    
    /// Forward pass for the DilatedDenseNet model
    /// - Parameter x: Input tensor of shape (batch, height, width, in_channels)
    /// - Returns: Output tensor after applying dense layers
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var skip = x  // Initialize skip connection
        var out = x
        
        for i in 0..<depth {
            let dil = 1 << i
            let padLength = lorder + (dil - 1) * (lorder - 1) - 1
            
            // Apply padding - MLX pad format: [(dim0_before, dim0_after), ...]
            // For this case: pad top and bottom by padLength, no padding on width
            out = padded(skip, widths: [IntOrPair((0, 0)), IntOrPair((padLength, padLength)), IntOrPair((0, 0)), IntOrPair((0, 0))])
            
            // Apply convolution
            out = convs[i](out)
            
            // Apply normalization
            // GroupNorm in MLX expects (batch, ..., channels) which matches our layout
            out = norms[i](out)
            
            // Apply PReLU activation
            out = prelus[i](out)
            
            // Concatenate the output with the skip connection along channel dimension
            // MLX uses channels-last, so concatenate on axis=-1
            skip = MLX.concatenated([out, skip], axis: -1)
        }
        
        return out  // Return the final output (not the concatenated skip)
    }
}