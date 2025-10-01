"""
MossFormer2 Speech Separation Demo
Separates mixed audio into individual speaker sources using MLX.
"""

import os
import mlx.core as mx
from mlx.utils import tree_unflatten
import numpy as np
import soundfile as sf
from types import SimpleNamespace
import time
from huggingface_hub import hf_hub_download

from mossformer2_ss_16k import MossFormer2_SS_16K_MLX

# Model configurations
MODEL_CONFIGS = {
    '2spk': {
        'repo_id': 'starkdmi/MossFormer2_SS_2SPK_16K_MLX',
        'num_spks': 2,
        'sample_rate': 16000,
        'is_whamr': False,
        'description': '2-speaker separation (16kHz)'
    },
    '2spk-whamr': {
        'repo_id': 'starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX',
        'num_spks': 2,
        'sample_rate': 8000,
        'is_whamr': True,
        'description': '2-speaker WHAMR (8kHz)'
    },
    '3spk': {
        'repo_id': 'starkdmi/MossFormer2_SS_3SPK_8K_MLX',
        'num_spks': 3,
        'sample_rate': 8000,
        'is_whamr': False,
        'description': '3-speaker separation (8kHz)'
    }
}

def download_model(model_name):
    """Download model weights from Hugging Face Hub."""
    config = MODEL_CONFIGS[model_name]
    repo_id = config['repo_id']
    filename = "model_fp32.safetensors"

    print(f"📥 Downloading model from Hugging Face: {repo_id}")
    print(f"   {config['description']}")

    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=None  # Uses default HF cache directory
    )

    print(f"✅ Model downloaded to: {weights_path}")
    return weights_path, config

def create_model(num_spks, weights_path, is_whamr=False):
    """
    Create and initialize MossFormer2 model.

    Args:
        num_spks: Number of speakers to separate (2 or 3)
        weights_path: Path to model weights file
        is_whamr: Whether to use WHAMR mode (skip mask multiplication)

    Returns:
        Compiled MLX model ready for inference
    """
    print(f"\n🔧 Creating MossFormer2 MLX {num_spks}-speaker model...")
    if is_whamr:
        print("   Using WHAMR mode (skip_mask_multiplication=True)")

    # Model configuration
    args = SimpleNamespace(
        encoder_embedding_dim=512,
        mossformer_sequence_dim=512,
        num_mossformer_layer=24,
        encoder_kernel_size=16,
        num_spks=num_spks,
        skip_mask_multiplication=is_whamr
    )

    # Create model architecture
    start_time = time.time()
    model = MossFormer2_SS_16K_MLX(args)
    print(f"⏱️  Model creation: {time.time() - start_time:.3f}s")

    # Load model weights
    start_time = time.time()
    weights = mx.load(weights_path)
    print(f"⏱️  Weight loading: {time.time() - start_time:.3f}s")

    # Update model parameters
    start_time = time.time()
    model.update(tree_unflatten(list(weights.items())))
    print(f"⏱️  Weight update: {time.time() - start_time:.3f}s")

    print("✅ Model initialized successfully")
    print(f"   - Encoder dimension: {args.encoder_embedding_dim}")
    print(f"   - MossFormer layers: {args.num_mossformer_layer}")
    print(f"   - Number of speakers: {args.num_spks}")

    # Compile model for optimized inference
    print("\n🔧 Compiling model...")
    start_time = time.time()
    model = mx.compile(model)
    # print(f"⏱️  Compilation time: {time.time() - start_time:.3f}s")

    return model

def load_audio(audio_path, sample_rate=16000):
    """
    Load audio file and prepare for processing.

    Args:
        audio_path: Path to input audio file
        sample_rate: Expected sample rate (for validation)

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    print(f"\n🎵 Loading audio: {audio_path}")

    # Load audio file
    audio, sr = sf.read(audio_path)

    # Convert to mono if stereo
    if audio.ndim > 1:
        print(f"   Converting stereo to mono (averaging {audio.shape[1]} channels)")
        audio = audio.mean(axis=1)

    # Validate sample rate
    if sr != sample_rate:
        print(f"⚠️  Warning: Audio sample rate ({sr} Hz) differs from model expected rate ({sample_rate} Hz)")
        print(f"   For best results, resample audio to {sample_rate} Hz")

    duration = len(audio) / sr
    print(f"📊 Audio loaded: {len(audio):,} samples, {sr} Hz, {duration:.2f}s")
    print(f"   Amplitude range: [{audio.min():.4f}, {audio.max():.4f}]")

    return audio, sr

def separate_audio(model, audio):
    """
    Run speech separation inference.

    Args:
        model: Compiled MossFormer2 MLX model
        audio: Input audio array (numpy)

    Returns:
        List of separated source arrays (numpy)
    """
    print("\n🔄 Running speech separation...")

    # Convert to MLX array and add batch dimension
    start_time = time.time()
    audio_mx = mx.array(audio.astype(np.float32))
    if audio_mx.ndim == 1:
        audio_mx = mx.expand_dims(audio_mx, axis=0)  # [T] -> [1, T]
    print(f"⏱️  Preparation: {time.time() - start_time:.3f}s")

    # Run model inference
    start_time = time.time()
    separated_sources = model(audio_mx)
    mx.eval(separated_sources)  # Force evaluation
    inference_time = time.time() - start_time
    print(f"⏱️  Inference: {inference_time:.3f}s")

    # Convert results to numpy
    start_time = time.time()
    sources = []
    for src in separated_sources:
        src_np = np.array(src)
        if src_np.shape[0] == 1:
            src_np = src_np.squeeze(0)
        sources.append(src_np)
    print(f"⏱️  Post-processing: {time.time() - start_time:.3f}s")

    print(f"✅ Separated into {len(sources)} sources")
    return sources

def save_sources(sources, output_path, sample_rate=16000):
    """
    Save separated audio sources to files.

    Args:
        sources: List of separated source arrays
        output_path: Base path for output files (or directory)
        sample_rate: Audio sample rate for output files

    Returns:
        List of saved file paths
    """
    print(f"\n💾 Saving {len(sources)} separated sources...")

    # If output_path is a directory, use it; otherwise use parent directory
    if os.path.isdir(output_path):
        output_dir = output_path
        prefix = "source"
    else:
        output_dir = os.path.dirname(output_path) or "."
        prefix = os.path.splitext(os.path.basename(output_path))[0]

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    for i, source in enumerate(sources, 1):
        # Normalize to prevent clipping
        max_amp = np.abs(source).max()
        if max_amp > 1.0:
            source = source / max_amp

        filename = f"{prefix}_{i}.wav"
        filepath = os.path.join(output_dir, filename)

        sf.write(filepath, source, sample_rate)
        saved_files.append(filepath)

        rms = np.sqrt(np.mean(source**2))
        print(f"   ✅ {filename}: {len(source):,} samples, RMS={rms:.4f}")

    return saved_files

def main(model_name, input_path, output_path):
    """
    Main function for speech separation.

    Args:
        model_name: Model configuration name ('2spk', '2spk-whamr', or '3spk')
        input_path: Path to input audio file
        output_path: Path for output files (file path or directory)
    """
    print("=" * 70)
    print("🎯 MossFormer2 Speech Separation using MLX")
    print("=" * 70)

    total_start = time.time()

    try:
        # Download model from Hugging Face
        weights_path, config = download_model(model_name)

        # Create and initialize model
        model = create_model(
            num_spks=config['num_spks'],
            weights_path=weights_path,
            is_whamr=config['is_whamr']
        )

        # Load audio
        audio, sr = load_audio(input_path, config['sample_rate'])

        # Run separation
        sources = separate_audio(model, audio)

        # Save results
        saved_files = save_sources(sources, output_path, sr)

        # Summary
        print("\n" + "=" * 70)
        print("🏁 Separation completed successfully!")
        print("=" * 70)
        print("📄 Output files:")
        for filepath in saved_files:
            print(f"   - {filepath}")

        total_time = time.time() - total_start
        print(f"\n⏱️  Total time: {total_time:.3f}s")

    except FileNotFoundError as e:
        print(f"\n❌ Error: Input file not found: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MossFormer2 Speech Separation - Separate mixed audio into individual speakers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2-speaker separation (16kHz)
  python demo.py --model 2spk --input mix.wav --output separated/

  # 2-speaker WHAMR (8kHz)
  python demo.py --model 2spk-whamr --input mix.wav --output separated/

  # 3-speaker separation (8kHz)
  python demo.py --model 3spk --input mix.wav --output separated/
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['2spk', '2spk-whamr', '3spk'],
        required=True,
        help="Model type: 2spk (16kHz), 2spk-whamr (8kHz WHAMR), or 3spk (8kHz)"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Path to input mixed audio file"
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="Output path (directory or file prefix for separated sources)"
    )

    args = parser.parse_args()

    exit(main(args.model, args.input, args.output))