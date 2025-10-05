# MossFormer2 Speech Separation

Speaker separation models for extracting individual speakers from mixed audio using MLX. Python and Swift implementations.

## Usage

### Python

```bash
cd python
pip install -r requirements.txt
python generate.py --model 2spk --input mix.wav --output separated/
```

### Swift

```bash
cd swift
xcodebuild build -scheme generate -configuration Release -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet
.build/DerivedData/Build/Products/Release/generate -m 2spk -i mix2.wav -o separated/
```

## Models

| Model           | Language   | Speed (× faster than input) | HuggingFace |
| --------------- | ---------- | --------------------------- | ----------- |
| **2spk**        | Swift      | **0.92×**                   | [MossFormer2_SS_2SPK_16K_MLX](https://huggingface.co/starkdmi/MossFormer2_SS_2SPK_16K_MLX) |
|                 | Python     | **2.0×**                    | |
| **3spk**        | Swift      | **1.44×**                   |[MossFormer2_SS_3SPK_8K_MLX](https://huggingface.co/starkdmi/MossFormer2_SS_3SPK_8K_MLX) |
|                 | Python     | **2.8×**                    | |
| **2spk-whamr**  | Swift      | **1.85×**                   | [MossFormer2_SS_2SPK_WHAMR_8K_MLX](https://huggingface.co/starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX) |
|                 | Python     | **3.54×**                   |             |

Source: [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)

## License

See [LICENSE](LICENSE).
