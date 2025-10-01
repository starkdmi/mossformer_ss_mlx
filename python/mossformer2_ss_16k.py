import mlx.core as mx
import mlx.nn as nn
from typing import Union, Dict, Any
from types import SimpleNamespace

from mossformer import MossFormer_MLX

class MossFormer2_SS_16K_MLX(nn.Module):
    """
    MLX implementation of MossFormer2_SS_16K wrapper.
    
    This is a wrapper for the MossFormer2 model specifically configured for 
    16kHz speech separation. It provides a convenient interface that accepts
    configuration arguments and initializes the underlying MossFormer model.
    
    Args:
        args: Either a SimpleNamespace/dict containing configuration parameters or
              individual parameters can be passed. Expected fields:
              - encoder_embedding_dim: Dimension of the encoder's output embeddings (default: 512)
              - mossformer_sequence_dim: Dimension of the MossFormer sequence (default: 512)
              - num_mossformer_layer: Number of layers in the MossFormer (default: 24)
              - encoder_kernel_size: Kernel size for the encoder (default: 16)
              - num_spks: Number of sources (speakers) to separate (default: 2)
              - skip_mask_multiplication: Skip mask multiplication for WHAMR models (default: False)
              
              Or pass individual parameters directly.
    
    Shape:
        - Input: (batch, time) - 16kHz audio signal
        - Output: List of (batch, time) tensors, one per speaker
    
    Example:
        >>> # Using args object
        >>> args = SimpleNamespace(
        ...     encoder_embedding_dim=512,
        ...     mossformer_sequence_dim=512,
        ...     num_mossformer_layer=24,
        ...     encoder_kernel_size=16,
        ...     num_spks=2
        ... )
        >>> model = MossFormer2_SS_16K_MLX(args)
        >>> 
        >>> # Or using direct parameters
        >>> model = MossFormer2_SS_16K_MLX(num_spks=2)
        >>> 
        >>> # Process audio
        >>> audio = mx.random.normal((1, 16000))  # 1 second @ 16kHz
        >>> separated = model(audio)
        >>> len(separated)  # Number of separated sources
        2
    """
    
    def __init__(
        self, 
        args: Union[SimpleNamespace, Dict[str, Any], None] = None,
        encoder_embedding_dim: int = 512,
        mossformer_sequence_dim: int = 512,
        num_mossformer_layer: int = 24,
        encoder_kernel_size: int = 16,
        num_spks: int = 2,
        skip_mask_multiplication: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Handle args parameter
        if args is not None:
            if isinstance(args, dict):
                # Convert dict to SimpleNamespace for attribute access
                args = SimpleNamespace(**args)
            
            # Extract parameters from args, with defaults
            encoder_embedding_dim = getattr(args, 'encoder_embedding_dim', encoder_embedding_dim)
            mossformer_sequence_dim = getattr(args, 'mossformer_sequence_dim', mossformer_sequence_dim)
            num_mossformer_layer = getattr(args, 'num_mossformer_layer', num_mossformer_layer)
            encoder_kernel_size = getattr(args, 'encoder_kernel_size', encoder_kernel_size)
            num_spks = getattr(args, 'num_spks', num_spks)
            skip_mask_multiplication = getattr(args, 'skip_mask_multiplication', skip_mask_multiplication)
        
        # Store configuration
        self.encoder_embedding_dim = encoder_embedding_dim
        self.mossformer_sequence_dim = mossformer_sequence_dim
        self.num_mossformer_layer = num_mossformer_layer
        self.encoder_kernel_size = encoder_kernel_size
        self.num_spks = num_spks
        
        # Initialize the main MossFormer model with parameters
        # Following the PyTorch implementation pattern
        self.model = MossFormer_MLX(
            in_channels=encoder_embedding_dim,
            out_channels=mossformer_sequence_dim,
            num_blocks=num_mossformer_layer,
            kernel_size=encoder_kernel_size,
            norm="ln",  # Layer normalization
            num_spks=num_spks,
            skip_around_intra=True,  # Default from PyTorch
            use_global_pos_enc=True,  # Default from PyTorch
            max_length=20000,  # Default from PyTorch
            skip_mask_multiplication=skip_mask_multiplication  # For WHAMR models
        )
    
    def __call__(self, x: mx.array) -> list:
        """
        Processes the input through the MossFormer model.
        
        Args:
            x (mx.array): Input tensor of shape [B, T], where:
                B = Batch size
                T = Input length (time samples at 16kHz)
        
        Returns:
            list: List of output tensors for each speaker, each of shape [B, T]
        """
        # Forward pass through the MossFormer model
        outputs = self.model(x)
        return outputs