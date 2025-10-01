import Foundation
import MLX
import MLXNN

/// MLX Swift implementation of the Encoder module.
///
/// Convolutional Encoder Layer converted from PyTorch to MLX.
///
/// Arguments
/// ---------
/// kernelSize : int
///     Length of filters.
/// inChannels : int
///     Number of input channels.
/// outChannels : int
///     Number of output channels.
///
/// Example
/// -------
/// >>> let x = MLXRandom.normal([2, 1000])
/// >>> let encoder = Encoder(kernelSize: 4, outChannels: 64)
/// >>> let h = encoder(x)
/// >>> h.shape
/// [2, 64, 499]
public class Encoder: Module {
    public let kernelSize: Int
    public let outChannels: Int
    public let inChannels: Int
    public let stride: Int
    
    // Conv1d layer
    public var conv1d: Conv1d
    
    /// Initialize Encoder
    /// - Parameters:
    ///   - kernelSize: Length of filters (default: 2)
    ///   - outChannels: Number of output channels (default: 64)
    ///   - inChannels: Number of input channels (default: 1)
    public init(
        kernelSize: Int = 2,
        outChannels: Int = 64,
        inChannels: Int = 1
    ) {
        self.kernelSize = kernelSize
        self.outChannels = outChannels
        self.inChannels = inChannels
        
        // Calculate stride based on the source:
        // - Basic Encoder uses: stride = kernelSize / 2
        // - MossFormer_MaskNet.conv1d_encoder uses: stride = 1 (default)
        if kernelSize == 1 {
            // For 1x1 convolutions (like MossFormer_MaskNet.conv1d_encoder), use stride=1
            self.stride = 1
        } else {
            // For basic Encoder, use stride = kernelSize / 2, but ensure minimum of 1
            self.stride = max(1, kernelSize / 2)
        }
        
        // Initialize Conv1d parameters
        self.conv1d = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            bias: false
        )
        
        super.init()
    }
    
    /// Return the encoded output.
    ///
    /// Arguments
    /// ---------
    /// x : MLXArray
    ///     Input tensor with dimensionality [B, L] or [B, C, L].
    ///
    /// Returns
    /// -------
    /// x : MLXArray
    ///     Encoded tensor with dimensionality [B, N, T_out].
    ///
    /// where B = Batchsize
    ///       L = Number of timepoints
    ///       N = Number of filters
    ///       T_out = Number of timepoints at the output of the encoder
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var input = x
        
        // B x L -> B x 1 x L
        if inChannels == 1 && input.ndim == 2 {
            input = input.expandedDimensions(axis: 1)
        }
        
        // MLX conv1d expects (N, L, C_in), but we have (B, C, L)
        // Need to transpose to (B, L, C) for MLX
        input = input.transposed(0, 2, 1)  // (B, C, L) -> (B, L, C)
        
        // Apply conv1d
        var output = conv1d(input)  // (B, L, C_out)
        
        // Transpose back to (B, C_out, L) to match PyTorch format
        output = output.transposed(0, 2, 1)  // (B, L, C_out) -> (B, C_out, L)
        
        // Apply ReLU activation using MLX's native compiled function
        output = MLXNN.relu(output)
        
        return output
    }
}
