import Foundation
import MLX
import MLXNN

/// MLX Swift implementation of PyTorch's ConvTranspose1d wrapper (Decoder).
///
/// A clean, production-ready implementation that matches PyTorch's behavior.
///
/// Parameters
/// ----------
/// inChannels : int
///     Number of input channels
/// outChannels : int
///     Number of output channels
/// kernelSize : int
///     Size of the convolving kernel
/// stride : int
///     Stride of the convolution. Default: 1
/// padding : int
///     Zero-padding added to both sides of the input. Default: 0
/// outputPadding : int
///     Additional size added to one side of the output shape. Default: 0
/// groups : int
///     Number of blocked connections from input to output. Default: 1
/// bias : bool
///     If true, adds a learnable bias to the output. Default: true
/// dilation : int
///     Spacing between kernel elements. Default: 1
public class Decoder: Module {
    public let inChannels: Int
    public let outChannels: Int
    public let kernelSize: Int
    public let stride: Int
    public let padding: Int
    public let outputPadding: Int
    public let groups: Int
    public let dilation: Int
    
    public var weight: MLXArray
    public var bias: MLXArray?
    
    /// Initialize Decoder
    /// - Parameters:
    ///   - inChannels: Number of input channels
    ///   - outChannels: Number of output channels
    ///   - kernelSize: Size of the convolving kernel
    ///   - stride: Stride of the convolution (default: 1)
    ///   - padding: Zero-padding added to both sides of the input (default: 0)
    ///   - outputPadding: Additional size added to one side of the output shape (default: 0)
    ///   - groups: Number of blocked connections from input to output (default: 1)
    ///   - bias: If true, adds a learnable bias to the output (default: true)
    ///   - dilation: Spacing between kernel elements (default: 1)
    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        outputPadding: Int = 0,
        groups: Int = 1,
        bias: Bool = true,
        dilation: Int = 1
    ) {
        // Validate parameters
        precondition(inChannels % groups == 0,
                    "in_channels (\(inChannels)) must be divisible by groups (\(groups))")
        precondition(outChannels % groups == 0,
                    "out_channels (\(outChannels)) must be divisible by groups (\(groups))")
        
        // Store configuration
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.outputPadding = outputPadding
        self.groups = groups
        self.dilation = dilation
        
        // Initialize weights in MLX format: (out_channels, kernel_size, in_channels // groups)
        // This avoids transformation during forward pass
        let weightShape = [outChannels, kernelSize, inChannels / groups]
        
        // Xavier uniform initialization (matching PyTorch)
        let fanIn = Float(inChannels * kernelSize)
        let fanOut = Float(outChannels * kernelSize)
        let bound = sqrt(6.0 / (fanIn + fanOut))
        self.weight = MLXRandom.uniform(low: -bound, high: bound, weightShape)
        
        // Initialize bias
        if bias {
            self.bias = MLXArray.zeros([outChannels])
        }
        
        super.init()
    }
    
    /// Forward pass matching PyTorch Decoder behavior.
    ///
    /// Parameters
    /// ----------
    /// x : MLXArray
    ///     Input tensor with shape [B, C, L] or [C, L]
    ///     where B = batch size, C = channels, L = sequence length
    ///
    /// Returns
    /// -------
    /// MLXArray
    ///     Output tensor with appropriate shape based on input
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(x.ndim == 2 || x.ndim == 3,
                    "\(String(describing: type(of: self))) accepts 2D or 3D tensor as input, got \(x.ndim)D")
        
        var input = x
        
        // Handle 2D input: PyTorch interprets as [B, L] and adds channel dim
        // [B, L] -> [B, 1, L] via unsqueeze at position 1
        if input.ndim == 2 {
            input = input.expandedDimensions(axis: 1)  // [B, L] -> [B, 1, L]
        }
        
        // Transform: PyTorch channels-first to MLX channels-last
        // [B, C, L] -> [B, L, C]
        input = input.transposed(0, 2, 1)
        
        // Apply native MLX conv_transposed1d
        var output = MLX.convTransposed1d(
            input,
            weight,
            stride: stride,
            padding: padding,
            dilation: dilation,
            outputPadding: outputPadding,
            groups: groups
        )
        
        // Add bias if present
        if let bias = bias {
            // output shape: [B, L_out, C_out]
            // bias shape: [C_out]
            output = output + bias
        }
        
        // Transform back: MLX channels-last to PyTorch channels-first
        // [B, L_out, C_out] -> [B, C_out, L_out]
        output = output.transposed(0, 2, 1)
        
        // Handle output squeezing to match PyTorch behavior
        let squeezed = MLX.squeezed(output)
        if squeezed.ndim == 1 {
            // If squeeze results in 1D, only squeeze specific dimension
            output = MLX.squeezed(output, axis: 1)
        } else {
            output = squeezed
        }
        
        return output
    }
}
