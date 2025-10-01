# MossFormer2 SS Swift MLX

Swift MLX implementation of MossFormer2 Speech Separation for separating mixed audio into individual speakers.

## Requirements

- macOS 13.3+ or iOS 16.0+
- Swift 5.9+

## Usage

```swift
import MossFormer2SS
import MLX
import MLXNN

// Configure model
let config = MossFormer2Config(num_spks: 2)

// Create model
let model = MossFormer2_SS_16K(config: config)

// Load weights from local path or downloaded model
let modelPath = "path/to/model_fp32.safetensors"
let weights = try loadArrays(url: URL(fileURLWithPath: modelPath))
let nestedWeights = NestedDictionary<String, MLXArray>.unflattened(weights)
try model.update(parameters: nestedWeights, verify: .all)

// Process audio
let audioMLX = MLXArray(audioData).expandedDimensions(axis: 0)
let separatedSources = model(audioMLX)

// Access separated speakers
let speaker1 = separatedSources[0].squeezed()
let speaker2 = separatedSources[1].squeezed()
```

## Pre-trained Models

Available on HuggingFace:

- [starkdmi/MossFormer2_SS_2SPK_16K_MLX](https://huggingface.co/starkdmi/MossFormer2_SS_2SPK_16K_MLX) - 2-speaker separation at 16kHz
- [starkdmi/MossFormer2_SS_3SPK_8K_MLX](https://huggingface.co/starkdmi/MossFormer2_SS_3SPK_8K_MLX) - 3-speaker separation at 8kHz
- [starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX](https://huggingface.co/starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX) - noisy 2-speaker separation at 8kHz
