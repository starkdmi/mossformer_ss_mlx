# MossFormer2 Speech Separation

Speaker separation models for extracting individual speakers from mixed audio using MLX. Python and Swift implementations.

## Usage

### Python

```bash
pip install -r requirements.txt
python demo.py --model 2spk --input mix.wav --output separated/
```

See [`Python/README.md`](python/README.md) and [`demo.py`](python/demo.py) for details.

### Swift

See [`Swift/README.md`](swift/README.md) and [`Demo.swift`](swift/Tests/Demo.swift) for implementation details.

## Models

MLX weights:
- [starkdmi/MossFormer2_SS_2SPK_16K_MLX](https://huggingface.co/starkdmi/MossFormer2_SS_2SPK_16K_MLX) - 2-speaker at 16kHz
- [starkdmi/MossFormer2_SS_3SPK_8K_MLX](https://huggingface.co/starkdmi/MossFormer2_SS_3SPK_8K_MLX) - 3-speaker at 8kHz
- [starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX](https://huggingface.co/starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX) - 2-speaker noisy at 8kHz

Source: [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)

## License

See [LICENSE](LICENSE).
