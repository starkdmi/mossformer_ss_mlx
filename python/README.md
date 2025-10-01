# MossFormer2 Speech Separation - MLX

MLX implementation of MossFormer2 for separating mixed audio into individual speaker sources.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
# clean 2-speaker separation (16kHz)
python demo.py --model 2spk --input mix.wav --output separated/

# clean 3-speaker separation (8kHz)
python demo.py --model 3spk --input mix.wav --output separated/

# noisy 2-speaker separation (8kHz)
python demo.py --model 2spk-whamr --input mix.wav --output separated/
```

Models are automatically downloaded from Hugging Face on first run.
