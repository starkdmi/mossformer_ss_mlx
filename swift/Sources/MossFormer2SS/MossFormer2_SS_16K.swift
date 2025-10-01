import Foundation
import MLX
import MLXNN

/// Configuration for MossFormer2_SS_16K model
public struct MossFormer2Config {
    public var encoder_embedding_dim: Int = 512
    public var mossformer_sequence_dim: Int = 512
    public var num_mossformer_layer: Int = 24
    public var encoder_kernel_size: Int = 16
    public var num_spks: Int = 2
    public var skip_mask_multiplication: Bool = false // WHAMR model compatibility
    
    public init(
        encoder_embedding_dim: Int = 512,
        mossformer_sequence_dim: Int = 512,
        num_mossformer_layer: Int = 24,
        encoder_kernel_size: Int = 16,
        num_spks: Int = 2,
        skip_mask_multiplication: Bool = false
    ) {
        self.encoder_embedding_dim = encoder_embedding_dim
        self.mossformer_sequence_dim = mossformer_sequence_dim
        self.num_mossformer_layer = num_mossformer_layer
        self.encoder_kernel_size = encoder_kernel_size
        self.num_spks = num_spks
        self.skip_mask_multiplication = skip_mask_multiplication
    }
}

/// MLX implementation of MossFormer2_SS_16K wrapper.
///
/// This is a wrapper for the MossFormer2 model specifically configured for 
/// 16kHz speech separation. It provides a convenient interface that accepts
/// configuration arguments and initializes the underlying MossFormer model.
public class MossFormer2_SS_16K: Module {
    public let config: MossFormer2Config
    
    // Model components
    @ModuleInfo var model: MossFormer
    
    // Compiled forward function
    private var compiledForward: (([MLXArray]) -> [MLXArray])?
    
    /// Initialize MossFormer2_SS_16K with configuration
    /// - Parameter config: Configuration object with model parameters
    public init(config: MossFormer2Config) {
        self.config = config
        
        // Initialize the main MossFormer model with parameters
        // Following the PyTorch implementation pattern
        self.model = MossFormer(
            inChannels: config.encoder_embedding_dim,
            outChannels: config.mossformer_sequence_dim,
            numBlocks: config.num_mossformer_layer,
            kernelSize: config.encoder_kernel_size,
            norm: "ln",  // Layer normalization
            numSpks: config.num_spks,
            skipAroundIntra: true,  // Default from PyTorch
            useGlobalPosEnc: true,  // Default from PyTorch
            maxLength: 20000,  // Default from PyTorch
            skipMaskMultiplication: config.skip_mask_multiplication  // For WHAMR models
        )
        
        super.init()
        
        // Compile the model after initialization
        compileModel()
    }
    
    /// Initialize MossFormer2_SS_16K with individual parameters
    /// - Parameters:
    ///   - encoderEmbeddingDim: Dimension of the encoder's output embeddings (default: 512)
    ///   - mossformerSequenceDim: Dimension of the MossFormer sequence (default: 512)
    ///   - numMossformerLayer: Number of layers in the MossFormer (default: 24)
    ///   - encoderKernelSize: Kernel size for the encoder (default: 16)
    ///   - numSpks: Number of sources (speakers) to separate (default: 2)
    public convenience init(
        encoderEmbeddingDim: Int = 512,
        mossformerSequenceDim: Int = 512,
        numMossformerLayer: Int = 24,
        encoderKernelSize: Int = 16,
        numSpks: Int = 2
    ) {
        let config = MossFormer2Config(
            encoder_embedding_dim: encoderEmbeddingDim,
            mossformer_sequence_dim: mossformerSequenceDim,
            num_mossformer_layer: numMossformerLayer,
            encoder_kernel_size: encoderKernelSize,
            num_spks: numSpks
        )
        self.init(config: config)
    }
    
    /// Compile the model for optimized execution
    private func compileModel() {
        // MLX compile expects [MLXArray] -> [MLXArray] signature
        let wrappedForward: ([MLXArray]) -> [MLXArray] = { [weak self] inputs in
            guard let self = self, !inputs.isEmpty else { return [] }
            // Extract the single input from the array
            let x = inputs[0]
            // Call the model and return the outputs
            return self.model(x)
        }
        
        // Compile the wrapped function
        self.compiledForward = compile(inputs: [model], outputs: [model], wrappedForward)
    }
    
    /// Processes the input through the MossFormer model
    /// - Parameter x: Input tensor of shape [B, T], where B = Batch size, T = Input length (time samples at 16kHz)
    /// - Returns: List of output tensors for each speaker, each of shape [B, T]
    public func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        if let compiled = compiledForward {
            // Use compiled version - wrap input in array and call
            return compiled([x])
        } else {
            // Fallback to regular forward pass
            return model(x)
        }
    }
}
