import mlx.core as mx
import mlx.nn as nn

from encoder import Encoder_MLX
from decoder import Decoder_MLX
from mossformer_masknet import MossFormer_MaskNet_MLX

class MossFormer_MLX(nn.Module):
    """
    MLX implementation of the End-to-End (E2E) Encoder-MaskNet-Decoder MossFormer model.
    
    This implementation provides 1:1 mathematical equivalence to the PyTorch version
    for speech separation tasks. It combines an encoder, mask prediction network,
    and decoder to separate mixed audio into individual speaker sources.
    
    Args:
        in_channels (int): Number of channels at the output of the encoder
        out_channels (int): Number of channels that will be input to the MossFormer blocks
        num_blocks (int): Number of layers in the Dual Computation Block
        kernel_size (int): Kernel size for the encoder and decoder
        norm (str): Type of normalization to apply (default: 'ln')
        num_spks (int): Number of sources (speakers) to separate (default: 2)
        skip_around_intra (bool): If True, adds skip connections around intra layers (default: True)
        use_global_pos_enc (bool): If True, uses global positional encodings (default: True)
        max_length (int): Maximum sequence length for input data (default: 20000)
        skip_mask_multiplication (bool): If True, skip mask multiplication (for WHAMR models) (default: False)
    
    Shape:
        - Input: (batch, time) - single channel audio
        - Output: List of (batch, time) tensors, one per speaker
    
    Example:
        >>> model = MossFormer_MLX(num_spks=2)
        >>> x = mx.random.normal((1, 16000))  # 1 second @ 16kHz
        >>> outputs = model(x)  # Returns list of 2 separated sources
        >>> outputs[0].shape
        (1, 16000)
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 512,
        num_blocks: int = 24,
        kernel_size: int = 16,
        norm: str = "ln",
        num_spks: int = 2,
        skip_around_intra: bool = True,
        use_global_pos_enc: bool = True,
        max_length: int = 20000,
        skip_mask_multiplication: bool = False,
    ):
        super().__init__()
        
        self.num_spks = num_spks  # Store number of speakers
        self.skip_mask_multiplication = skip_mask_multiplication  # Store WHAMR mode flag
        
        # Initialize the encoder with 1 input channel and the specified output channels
        self.enc = Encoder_MLX(
            kernel_size=kernel_size,
            out_channels=in_channels,
            in_channels=1
        )
        
        # Initialize the MaskNet with the specified parameters
        self.mask_net = MossFormer_MaskNet_MLX(
            in_channels=in_channels,
            out_channels=out_channels,
            out_channels_final=in_channels,  # Should be same as encoder output
            num_blocks=num_blocks,
            norm=norm,
            num_spks=num_spks,
            skip_around_intra=skip_around_intra,
            use_global_pos_enc=use_global_pos_enc,
            max_length=max_length,
        )
        
        # Initialize the decoder to project output back to 1 channel
        self.dec = Decoder_MLX(
            in_channels=in_channels,  # Decoder input matches encoder output
            out_channels=1,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            bias=False
        )
    
    def __call__(self, input: mx.array) -> list:
        """
        Processes the input through the encoder, mask net, and decoder.
        
        Args:
            input (mx.array): Input tensor of shape [B, T], where:
                B = Batch size
                T = Input length (time samples)
        
        Returns:
            list: List of output tensors for each speaker, each of shape [B, T]
        """
        # Pass the input through the encoder to extract features
        # Input: [B, T] → Output: [B, N, L] where N=in_channels, L=encoded length
        x = self.enc(input)
        
        # Generate the mask for each speaker using the mask net
        # Input: [B, N, L] → Output: [spks, B, N, L]
        mask = self.mask_net(x)
        
        if self.skip_mask_multiplication:
            # WHAMR mode: use mask output directly as separated signal
            sep_x = mask
        else:
            # Standard mode: duplicate features and apply mask
            # x shape: [B, N, L] → [spks, B, N, L]
            x = mx.stack([x] * self.num_spks, axis=0)
            # Element-wise multiplication
            sep_x = x * mask
        
        # Decoding process to reconstruct the separated sources
        # Process each speaker's output through the decoder
        est_source_list = []
        for i in range(self.num_spks):
            # Get the i-th speaker's masked features: [B, N, L]
            speaker_features = sep_x[i]
            
            # Decode to waveform: [B, N, L] → [B, T_est]
            decoded = self.dec(speaker_features)
            
            # Add to list
            est_source_list.append(decoded)
        
        # Stack all decoded sources: List of [B, T_est] → [B, T_est, spks]
        est_source = mx.stack(est_source_list, axis=-1)
        
        # Match the estimated output length to the original input length
        T_origin = input.shape[1]
        T_est = est_source.shape[1]
        
        if T_origin > T_est:
            # Pad if estimated length is shorter
            # MLX pad format: [(dim0_before, dim0_after), ...]
            pad_amount = T_origin - T_est
            est_source = mx.pad(est_source, [(0, 0), (0, pad_amount), (0, 0)], constant_values=0.0)
        else:
            # Trim if estimated length is longer
            est_source = est_source[:, :T_origin, :]
        
        # Collect outputs for each speaker
        out = []
        for spk in range(self.num_spks):
            # Extract speaker output: [B, T, spks] → [B, T]
            out.append(est_source[:, :, spk])
        
        return out  # Return list of separated outputs