import Foundation
import MLX
import MLXNN

/// MLX implementation of UniDeepFsmn_dilated with 1:1 mathematical equivalence to PyTorch.
///
/// UniDeepFsmn_dilated combines the UniDeepFsmn architecture with a dilated dense network 
/// to enhance feature extraction while maintaining efficient computation.
public class UniDeepFSMN_Dilated: Module {
    public let inputDim: Int
    public let outputDim: Int
    public let lorder: Int?
    public let hiddenSize: Int?
    public let depth: Int
    
    // Model components
    @ModuleInfo var linear: Linear?
    @ModuleInfo var project: Linear?
    @ModuleInfo var conv: DilatedDenseNet?
    
    /// Initialize UniDeepFsmn_dilated
    /// - Parameters:
    ///   - inputDim: Dimension of the input features
    ///   - outputDim: Dimension of the output features
    ///   - lorder: Length of the order for the convolution layers
    ///   - hiddenSize: Number of hidden units in the linear layer
    ///   - depth: Depth of the dilated dense network (default: 2)
    public init(
        inputDim: Int,
        outputDim: Int,
        lorder: Int? = nil,
        hiddenSize: Int? = nil,
        depth: Int = 2
    ) {
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.lorder = lorder
        self.hiddenSize = hiddenSize
        self.depth = depth
        
        // Initialize layers if lorder is provided
        if let lorder = lorder, let hiddenSize = hiddenSize {
            // Linear transformation to hidden size
            self.linear = Linear(inputDim, hiddenSize)
            
            // Project hidden size to output dimension (no bias)
            self.project = Linear(hiddenSize, outputDim, bias: false)
            
            // Dilated dense network for feature extraction
            self.conv = DilatedDenseNet(
                depth: depth,
                lorder: lorder,
                inChannels: outputDim
            )
        }
        
        super.init()
    }
    
    /// Forward pass for the UniDeepFsmn_dilated model
    /// - Parameter input: Input tensor of shape (batch, time, input_dim)
    /// - Returns: The output tensor of the same shape as input, enhanced by the network
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        guard let linear = linear, let project = project, let conv = conv else {
            return input
        }
        
        // Apply linear layer followed by ReLU activation
        let f1 = relu(linear(input))
        
        // Project to output dimension
        let p1 = project(f1)
        
        // Add a dimension for compatibility with Conv2d
        // (batch, time, channels) → (batch, time, 1, channels)
        let x = p1.expandedDimensions(axis: 2)
        
        // Pass through the dilated dense network
        // MLX Conv2d expects (batch, height, width, channels)
        let out = conv(x)
        
        // Remove the added dimension
        // (batch, time, 1, channels) → (batch, time, channels)
        let squeezed = out.squeezed(axis: 2)
        
        // Return enhanced input with residual connection
        return input + squeezed
    }
}
