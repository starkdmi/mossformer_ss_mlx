import Foundation
import MLX
import MLXNN

/// MossformerBlockGFSMN implementation using attention and FSMN components
public class MossFormerBlockGFSMN: Module {
    public let dim: Int
    public let depth: Int
    public let groupSize: Int
    public let queryKeyDim: Int
    public let expansionFactor: Float
    public let causal: Bool
    public let attnDropout: Float
    
    // FLASH attention layers
    @ModuleInfo var layers: [FLASHShareAFFConvM]
    
    // Gated FSMN blocks
    @ModuleInfo var fsmn: [GatedFSMNBlock_Dilated]
    
    /// Initialize with the same parameters but includes FSMN blocks
    public init(
        dim: Int,
        depth: Int,
        groupSize: Int = 256,
        queryKeyDim: Int = 128,
        expansionFactor: Float = 4.0,
        causal: Bool = false,
        attnDropout: Float = 0.1
    ) {
        self.dim = dim
        self.depth = depth
        self.groupSize = groupSize
        self.queryKeyDim = queryKeyDim
        self.expansionFactor = expansionFactor
        self.causal = causal
        self.attnDropout = attnDropout
        
        // Normalization type is scalenorm for MossformerBlockGFSMN
        let normType = "scalenorm"
        let normKlass: Module.Type = ScaleNorm.self
        
        // Rotary positional embedding for attention
        let rotaryPosEmb = RoPE(
            dimensions: min(32, queryKeyDim),
            traditional: false,
            base: 10000
        )
        
        // Create FLASH attention layers
        var layerArray: [FLASHShareAFFConvM] = []
        for _ in 0..<depth {
            let layer = FLASHShareAFFConvM(
                dim: dim,
                groupSize: groupSize,
                queryKeyDim: queryKeyDim,
                expansionFactor: expansionFactor,
                causal: causal,
                dropout: attnDropout,
                rotaryPosEmb: rotaryPosEmb,
                normKlass: normKlass,
                shiftTokens: true
            )
            layerArray.append(layer)
        }
        self.layers = layerArray
        
        // Create Gated FSMN blocks - matching Python implementation
        var fsmnArray: [GatedFSMNBlock_Dilated] = []
        for _ in 0..<depth {
            let fsmnBlock = GatedFSMNBlock_Dilated(
                dim: dim,
                innerChannels: 256,  // Matching Python: inner_channels=256
                groupSize: groupSize,
                normType: normType
            )
            fsmnArray.append(fsmnBlock)
        }
        self.fsmn = fsmnArray
        
        super.init()
    }
    
    /// Forward pass for the Mossformer Block with Gated FSMN.
    /// - Parameters:
    ///   - x: Input tensor of shape (batch_size, seq_len, dim)
    ///   - mask: Optional mask tensor for attention operations
    /// - Returns: Output tensor after processing through the block
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var output = x
        
        // Process through interleaved FLASH and FSMN layers
        for i in 0..<depth {
            // Apply FLASH attention layer
            // Only pass mask if it's not nil to ensure compiled path is used
            if let mask = mask {
                output = layers[i](output, mask: mask)
            } else {
                output = layers[i](output)
            }
            
            // Apply Gated FSMN block
            output = fsmn[i](output)
        }
        
        
        return output
    }
}
