import Foundation
import MLX
import MLXNN

/// MLX Swift implementation of Mossformer Block with attention mechanisms.
///
/// This block is designed to process input sequences using attention
/// layers and incorporates rotary positional embeddings. It allows
/// for configurable normalization types and can handle causal
/// attention.
///
/// Args:
///     dim (int): Dimensionality of the input.
///     depth (int): Number of attention layers in the block.
///     groupSize (int, optional): Size of groups for normalization. Default is 256.
///     queryKeyDim (int, optional): Dimension of the query and key in attention. Default is 128.
///     expansionFactor (float, optional): Expansion factor for feedforward layers. Default is 4.
///     causal (bool, optional): If True, enables causal attention. Default is false.
///     attnDropout (float, optional): Dropout rate for attention layers. Default is 0.1.
///     normType (str, optional): Type of normalization to use ('scalenorm' or 'layernorm'). Default is 'scalenorm'.
///     shiftTokens (bool, optional): If True, shifts tokens in the attention layer. Default is true.
public class MossFormerBlock: Module {
    public let dim: Int
    public let depth: Int
    public let groupSize: Int
    public let queryKeyDim: Int
    public let expansionFactor: Float
    public let causal: Bool
    public let attnDropout: Float
    public let normType: String
    public let shiftTokens: Bool
    
    // Layers
    @ModuleInfo var layers: [FLASHShareAFFConvM]
    
    /// Initialize MossformerBlock
    public init(
        dim: Int,
        depth: Int,
        groupSize: Int = 256,
        queryKeyDim: Int = 128,
        expansionFactor: Float = 4.0,
        causal: Bool = false,
        attnDropout: Float = 0.1,
        normType: String = "scalenorm",
        shiftTokens: Bool = true
    ) {
        self.dim = dim
        self.depth = depth
        self.groupSize = groupSize
        self.queryKeyDim = queryKeyDim
        self.expansionFactor = expansionFactor
        self.causal = causal
        self.attnDropout = attnDropout
        self.normType = normType
        self.shiftTokens = shiftTokens
        
        // Ensure normalization type is valid
        precondition(["scalenorm", "layernorm"].contains(normType), "norm_type must be one of scalenorm or layernorm")
        
        // Select normalization class based on the provided type
        let normKlass: Module.Type = (normType == "scalenorm") ? ScaleNorm.self : LayerNorm.self
        
        // Rotary positional embedding for attention
        // Max rotary embedding dimensions of 32, partial Rotary embeddings
        let rotaryPosEmb = RoPE(
            dimensions: min(32, queryKeyDim),
            traditional: false,
            base: 10000
        )
        
        // Create a list of attention layers using FLASH_ShareA_FFConvM
        var layerArray: [FLASHShareAFFConvM] = []
        
        // Initialize layers
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
                shiftTokens: shiftTokens
            )
            layerArray.append(layer)
        }
        
        self.layers = layerArray
        
        super.init()
    }
    
    /// Forward pass for the Mossformer Block.
    /// - Parameters:
    ///   - x: Input tensor of shape (batch_size, seq_len, dim)
    ///   - mask: Optional mask tensor for attention operations
    /// - Returns: Output tensor after processing through the block
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var output = x
        
        // Process input through each attention layer
        for (_, layer) in layers.enumerated() {
            output = layer(output, mask: mask)  // Apply attention layer with optional mask
        }
        
        return output  // Return the final output tensor
    }
}
