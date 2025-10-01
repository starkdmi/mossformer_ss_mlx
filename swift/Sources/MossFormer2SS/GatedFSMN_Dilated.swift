import Foundation
import MLX
import MLXNN

/// MLX implementation of Gated FSMN with dilated convolutions.
///
/// This module implements a gated mechanism using two parallel feedforward 
/// convolutions to generate the input for a dilated FSMN. The gated outputs 
/// are combined to enhance the input features, allowing for better speech 
/// enhancement performance.
public class GatedFSMN_Dilated: Module {
    public let inChannels: Int
    public let outChannels: Int
    public let lorder: Int
    public let hiddenSize: Int
    
    // Model components
    @ModuleInfo var to_u: FFConvM
    @ModuleInfo var to_v: FFConvM
    @ModuleInfo var fsmn: UniDeepFSMN_Dilated
    
    /// Initialize Gated FSMN with dilated convolutions
    /// - Parameters:
    ///   - inChannels: Number of input channels (features)
    ///   - outChannels: Number of output channels (features)
    ///   - lorder: Order of the FSMN
    ///   - hiddenSize: Number of hidden units in the feedforward layers
    public init(
        inChannels: Int,
        outChannels: Int,
        lorder: Int,
        hiddenSize: Int
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.lorder = lorder
        self.hiddenSize = hiddenSize
        
        // Feedforward convolution for the u-gate
        self.to_u = FFConvM(
            dimIn: inChannels,
            dimOut: hiddenSize,
            normKlass: LayerNorm.self,
            dropout: 0.1
        )
        
        // Feedforward convolution for the v-gate
        self.to_v = FFConvM(
            dimIn: inChannels,
            dimOut: hiddenSize,
            normKlass: LayerNorm.self,
            dropout: 0.1
        )
        
        // Initialize the dilated FSMN
        self.fsmn = UniDeepFSMN_Dilated(
            inputDim: inChannels,
            outputDim: outChannels,
            lorder: lorder,
            hiddenSize: hiddenSize
        )
        
        super.init()
    }
    
    /// Forward pass through the Gated FSMN module
    ///
    /// Following the exact PyTorch implementation pattern:
    /// 1. Store input for residual connection
    /// 2. Process through u-gate and v-gate branches
    /// 3. Apply dilated FSMN to u-gate output
    /// 4. Combine with gating: v * u + input
    ///
    /// - Parameter x: Input tensor of shape (batch, time, in_channels)
    /// - Returns: Output tensor after processing through the gated FSMN
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Store the original input for residual connection
        let inputResidual = x
        
        // Process input through u-gate (will go through FSMN)
        var xU = to_u(x)
        
        // Process input through v-gate (acts as gate)
        let xV = to_v(x)
        
        // Apply dilated FSMN to u-gate output
        xU = fsmn(xU)
        
        // Combine the outputs from u-gate and v-gate with the original input
        // Gated output with residual connection: x = x_v * x_u + input
        let output = xV * xU + inputResidual
        
        return output
    }
}