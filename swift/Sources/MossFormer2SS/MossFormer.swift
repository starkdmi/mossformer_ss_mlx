import Foundation
import MLX
import MLXNN

/// MLX implementation of the End-to-End (E2E) Encoder-MaskNet-Decoder MossFormer model.
///
/// This implementation provides 1:1 mathematical equivalence to the PyTorch version
/// for speech separation tasks. It combines an encoder, mask prediction network,
/// and decoder to separate mixed audio into individual speaker sources.
public class MossFormer: Module {
    public let inChannels: Int
    public let outChannels: Int
    public let numBlocks: Int
    public let kernelSize: Int
    public let norm: String
    public let numSpks: Int
    public let skipAroundIntra: Bool
    public let useGlobalPosEnc: Bool
    public let maxLength: Int
    public let skipMaskMultiplication: Bool
    
    // Model components
    @ModuleInfo var enc: Encoder
    @ModuleInfo var mask_net: MossFormerMaskNet
    @ModuleInfo var dec: Decoder
    
    /// Initialize MossFormer
    /// - Parameters:
    ///   - inChannels: Number of channels at the output of the encoder
    ///   - outChannels: Number of channels that will be input to the MossFormer blocks
    ///   - numBlocks: Number of layers in the Dual Computation Block
    ///   - kernelSize: Kernel size for the encoder and decoder
    ///   - norm: Type of normalization to apply (default: 'ln')
    ///   - numSpks: Number of sources (speakers) to separate (default: 2)
    ///   - skipAroundIntra: If true, adds skip connections around intra layers (default: true)
    ///   - useGlobalPosEnc: If true, uses global positional encodings (default: true)
    ///   - maxLength: Maximum sequence length for input data (default: 20000)
    ///   - skipMaskMultiplication: If true, skip mask multiplication (for WHAMR models) (default: false)
    public init(
        inChannels: Int = 512,
        outChannels: Int = 512,
        numBlocks: Int = 24,
        kernelSize: Int = 16,
        norm: String = "ln",
        numSpks: Int = 2,
        skipAroundIntra: Bool = true,
        useGlobalPosEnc: Bool = true,
        maxLength: Int = 20000,
        skipMaskMultiplication: Bool = false
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.numBlocks = numBlocks
        self.kernelSize = kernelSize
        self.norm = norm
        self.numSpks = numSpks
        self.skipAroundIntra = skipAroundIntra
        self.useGlobalPosEnc = useGlobalPosEnc
        self.maxLength = maxLength
        self.skipMaskMultiplication = skipMaskMultiplication
        
        // Initialize the encoder with 1 input channel and the specified output channels
        self.enc = Encoder(
            kernelSize: kernelSize,
            outChannels: inChannels,
            inChannels: 1
        )
        
        // Initialize the MaskNet with the specified parameters
        self.mask_net = MossFormerMaskNet(
            inChannels: inChannels,
            outChannels: outChannels,
            numBlocks: numBlocks,
            normType: norm,
            numSpks: numSpks,
            skipAroundIntra: skipAroundIntra,
            useGlobalPosEnc: useGlobalPosEnc,
            maxLength: maxLength
        )
        
        // Initialize the decoder to project output back to 1 channel
        self.dec = Decoder(
            inChannels: inChannels,  // Decoder input matches encoder output
            outChannels: 1,
            kernelSize: kernelSize,
            stride: kernelSize / 2,
            bias: false
        )
        
        super.init()
    }
    
    /// Processes the input through the encoder, mask net, and decoder
    /// - Parameter input: Input tensor of shape [B, T], where B = Batch size, T = Input length (time samples)
    /// - Returns: List of output tensors for each speaker, each of shape [B, T]
    public func callAsFunction(_ input: MLXArray) -> [MLXArray] {
        // Pass the input through the encoder to extract features
        // Input: [B, T] → Output: [B, N, L] where N=in_channels, L=encoded length
        let x = enc(input)
        
        // Generate the mask for each speaker using the mask net
        // Input: [B, N, L] → Output: [spks, B, N, L]
        let mask = mask_net(x)
        
        let sepX: MLXArray
        if skipMaskMultiplication {
            // WHAMR mode: use mask output directly as separated signal
            sepX = mask
        } else {
            // Standard mode: duplicate features and apply mask
            // x shape: [B, N, L] → [spks, B, N, L]
            var xExpanded = [MLXArray]()
            for _ in 0..<numSpks {
                xExpanded.append(x)
            }
            let xStacked = MLX.stacked(xExpanded, axis: 0)
            
            // Apply the mask to separate the sources
            // Element-wise multiplication
            sepX = xStacked * mask
        }
        
        // Decoding process to reconstruct the separated sources
        // Process each speaker's output through the decoder
        var estSourceList = [MLXArray]()
        for i in 0..<numSpks {
            // Get the i-th speaker's masked features: [B, N, L]
            let speakerFeatures = sepX[i]
            
            // Decode to waveform: [B, N, L] → [B, T_est]
            let decoded = dec(speakerFeatures)
            
            // Add to list
            estSourceList.append(decoded)
        }
        
        // Stack all decoded sources: List of [B, T_est] → [B, T_est, spks]
        let estSource = MLX.stacked(estSourceList, axis: -1)
        
        // Match the estimated output length to the original input length
        let TOrigin = input.shape[1]
        let TEst = estSource.shape[1]
        
        var finalSource: MLXArray
        if TOrigin > TEst {
            // Pad if estimated length is shorter
            // MLX pad format: [(dim0_before, dim0_after), ...]
            let padAmount = TOrigin - TEst
            finalSource = padded(estSource, widths: [IntOrPair((0, 0)), IntOrPair((0, padAmount)), IntOrPair((0, 0))])
        } else {
            // Trim if estimated length is longer
            finalSource = estSource[0..<estSource.shape[0], 0..<TOrigin, 0..<estSource.shape[2]]
        }
        
        // Collect outputs for each speaker
        var out = [MLXArray]()
        for spk in 0..<numSpks {
            // Extract speaker output: [B, T, spks] → [B, T]
            out.append(finalSource[0..., 0..., spk])
        }
        
        return out  // Return list of separated outputs
    }
}
