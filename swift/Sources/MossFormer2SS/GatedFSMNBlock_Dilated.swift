import Foundation
import MLX
import MLXNN

/// MLX implementation of Gated_FSMN_Block with dilated FSMN, 1:1 mathematical equivalence to PyTorch.
///
/// A 1-D convolutional block that incorporates a gated FSMN with dilated convolutions.
/// This block consists of:
/// 1. Conv1d layer with PReLU activation
/// 2. CLayerNorm normalization
/// 3. Gated FSMN module with dilated convolutions
/// 4. Another CLayerNorm
/// 5. Final Conv1d projection
/// 6. Residual connection
public class GatedFSMNBlock_Dilated: Module {
    public let dim: Int
    public let innerChannels: Int
    public let groupSize: Int
    public let normType: String
    
    // Model components
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var prelu: PReLU
    @ModuleInfo var norm1: CLayerNorm
    @ModuleInfo var norm2: CLayerNorm
    @ModuleInfo var gated_fsmn: GatedFSMN_Dilated
    @ModuleInfo var conv2: Conv1d
    
    /// Initialize Gated_FSMN_Block with dilated convolutions
    /// - Parameters:
    ///   - dim: Dimensionality of the input/output
    ///   - innerChannels: Number of channels in the inner layers (default: 256)
    ///   - groupSize: Size of the groups for normalization (default: 256)
    ///   - normType: Type of normalization to use ('scalenorm' or 'layernorm')
    public init(
        dim: Int,
        innerChannels: Int = 256,
        groupSize: Int = 256,
        normType: String = "scalenorm"
    ) {
        self.dim = dim
        self.innerChannels = innerChannels
        self.groupSize = groupSize
        self.normType = normType
        
        // First convolutional layer
        self.conv1 = Conv1d(
            inputChannels: dim,
            outputChannels: innerChannels,
            kernelSize: 1,
            bias: true
        )
        
        // PReLU activation
        self.prelu = PReLU(count: 1)
        
        // Normalization layers
        self.norm1 = CLayerNorm(normalizedShape: innerChannels)
        self.norm2 = CLayerNorm(normalizedShape: innerChannels)
        
        // Gated FSMN with dilated convolutions
        self.gated_fsmn = GatedFSMN_Dilated(
            inChannels: innerChannels,
            outChannels: innerChannels,
            lorder: 20,
            hiddenSize: innerChannels
        )
        
        // Final convolutional layer
        self.conv2 = Conv1d(
            inputChannels: innerChannels,
            outputChannels: dim,
            kernelSize: 1,
            bias: true
        )
        
        super.init()
    }
    
    /// Forward pass for the Gated FSMN Block with dilated convolutions
    /// - Parameter x: Input tensor of shape [batch_size, seq_length, dim]
    /// - Returns: Output tensor of shape [batch_size, seq_length, dim]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        
        // First convolution - input is already [B, T, D]
        var output = conv1(x)
        
        // PReLU activation
        output = prelu(output)
        
        // First normalization
        output = norm1(output)
        
        // Gated FSMN with dilated convolutions
        output = gated_fsmn(output)
        
        // Second normalization
        output = norm2(output)
        
        // Final convolution to project back to original dimensions
        output = conv2(output)
        
        // Add residual connection
        return output + residual
    }
}