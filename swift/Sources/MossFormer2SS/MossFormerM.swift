import Foundation
import MLX
import MLXNN

/// Protocol for MossFormer model variants to enable polymorphic usage
protocol MossFormerVariant: Module {
    func callAsFunction(_ src: MLXArray) -> MLXArray
}

/// MLX implementation of MossFormerM transformer encoder based on MossFormer2 layers.
///
/// This class implements the transformer encoder using MossFormer2 layers with
/// Gated FSMN blocks for enhanced sequence processing.
public class MossFormerM: Module, MossFormerVariant {
    public let numBlocks: Int
    public let dModel: Int
    public let causal: Bool
    public let groupSize: Int
    public let queryKeyDim: Int
    public let expansionFactor: Float
    public let attnDropout: Float
    
    // MossFormer blocks with GFSMN
    @ModuleInfo var mossformerM: MossFormerBlockGFSMN
    // Layer normalization
    @ModuleInfo var norm: LayerNorm
    
    /// Initialize MossFormerM
    /// - Parameters:
    ///   - numBlocks: Number of mossformer2 blocks to include
    ///   - dModel: The dimension of the input embedding
    ///   - causal: True for causal / false for non causal (default: false)
    ///   - groupSize: The chunk size for segmenting sequence (default: 256)
    ///   - queryKeyDim: The attention vector dimension (default: 128)
    ///   - expansionFactor: The expansion factor for the linear projection in conv module (default: 4.0)
    ///   - attnDropout: Dropout for the self-attention (default: 0.1)
    public init(
        numBlocks: Int,
        dModel: Int,
        causal: Bool = false,
        groupSize: Int = 256,
        queryKeyDim: Int = 128,
        expansionFactor: Float = 4.0,
        attnDropout: Float = 0.1
    ) {
        self.numBlocks = numBlocks
        self.dModel = dModel
        self.causal = causal
        self.groupSize = groupSize
        self.queryKeyDim = queryKeyDim
        self.expansionFactor = expansionFactor
        self.attnDropout = attnDropout
        
        // Initialize the MossFormer blocks with GFSMN
        self.mossformerM = MossFormerBlockGFSMN(
            dim: dModel,
            depth: numBlocks,
            groupSize: groupSize,
            queryKeyDim: queryKeyDim,
            expansionFactor: expansionFactor,
            causal: causal,
            attnDropout: attnDropout
        )
        
        // Layer normalization
        self.norm = LayerNorm(dimensions: dModel, eps: 1e-8)
        
        super.init()
    }
    
    /// Forward pass through the MossFormerM model
    /// - Parameter src: Input tensor of shape (batch_size, sequence_length, d_model)
    /// - Returns: Output tensor of shape (batch_size, sequence_length, d_model)
    public func callAsFunction(_ src: MLXArray) -> MLXArray {
        
        // Apply MossFormer blocks
        var output = mossformerM(src)
        
        // Apply layer normalization
        output = norm(output)
        
        return output
    }
}
