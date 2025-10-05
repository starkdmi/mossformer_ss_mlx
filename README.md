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

MLX weights:
- [starkdmi/MossFormer2_SS_2SPK_16K_MLX](https://huggingface.co/starkdmi/MossFormer2_SS_2SPK_16K_MLX) - 2-speaker at 16kHz (2spk)
- [starkdmi/MossFormer2_SS_3SPK_8K_MLX](https://huggingface.co/starkdmi/MossFormer2_SS_3SPK_8K_MLX) - 3-speaker at 8kHz (3spk)
- [starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX](https://huggingface.co/starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX) - 2-speaker noisy at 8kHz (2spk-whamr)

Source: [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)

## License

See [LICENSE](LICENSE).
