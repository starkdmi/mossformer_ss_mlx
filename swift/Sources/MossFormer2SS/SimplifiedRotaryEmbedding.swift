import Foundation
import MLX
import MLXNN

/// Simplified Rotary Positional Embedding implementation for compatibility
///
/// This is a simplified version that provides a compatibility layer for models
/// that expect custom rotary embeddings with rotate_queries_or_keys method.
public class SimplifiedRotaryEmbedding: Module {
    public let dims: Int
    public let traditional: Bool
    public let base: Float
    
    private let rope: RoPE
    
    /// Initialize SimplifiedRotaryEmbedding
    /// - Parameters:
    ///   - dims: Dimensionality of the rotary embeddings
    ///   - traditional: Whether to use traditional implementation
    ///   - base: Base for the sinusoidal frequencies
    public init(dims: Int, traditional: Bool = false, base: Float = 10000) {
        self.dims = dims
        self.traditional = traditional
        self.base = base
        
        // Use MLX's built-in RoPE
        self.rope = RoPE(
            dimensions: dims,
            traditional: traditional,
            base: base
        )
        
        super.init()
    }
    
    /// Apply rotary embeddings to queries or keys
    /// - Parameter qOrK: Queries or keys tensor
    /// - Returns: Tensor with rotary embeddings applied
    public func rotateQueriesOrKeys(_ qOrK: MLXArray) -> MLXArray {
        return rope(qOrK)
    }
    
    /// Forward pass - delegates to RoPE
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return rope(x)
    }
}