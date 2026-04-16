"""
Transformer architecture implementation for ML Framework.

This module provides Transformer models including:
- Encoder-only (BERT-style)
- Decoder-only (GPT-style)
- Encoder-decoder (Seq2Seq)

Transformers use self-attention mechanisms to capture long-range dependencies
in sequential data, making them ideal for NLP tasks and other sequence modeling.
"""

from typing import Dict, Any, List, Optional, Tuple
import math


class TransformerConfig(ModelConfig):
    """Configuration class for Transformer models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Default values specific to Transformers
        defaults = {
            "input_dim": 512,           # Input embedding dimension
            "num_heads": 8,             # Number of attention heads
            "num_encoder_layers": 6,    # Number of encoder layers
            "num_decoder_layers": 6,    # Number of decoder layers
            "d_model": 512,             # Model dimension (same as input_dim)
            "d_ff": 2048,               # Dimension of feed-forward network
            "dropout": 0.1,             # Dropout rate
            "max_seq_len": 512,         # Maximum sequence length
            "num_classes": 10,          # Number of output classes (for classification)
            "is_decoder_only": False,   # Use decoder-only architecture (GPT-style)
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    @property
    def input_dim(self) -> int:
        """Get input dimension."""
        return self.config.get("input_dim", 512)

    @property
    def num_heads(self) -> int:
        """Get number of attention heads."""
        return self.config.get("num_heads", 8)

    @property
    def num_encoder_layers(self) -> int:
        """Get number of encoder layers."""
        return self.config.get("num_encoder_layers", 6)

    @property
    def num_decoder_layers(self) -> int:
        """Get number of decoder layers."""
        return self.config.get("num_decoder_layers", 6)

    @property
    def d_model(self) -> int:
        """Get model dimension."""
        return self.config.get("d_model", 512)

    @property
    def d_ff(self) -> int:
        """Get feed-forward network dimension."""
        return self.config.get("d_ff", 2048)

    @property
    def dropout(self) -> float:
        """Get dropout rate."""
        return self.config.get("dropout", 0.1)

    @property
    def max_seq_len(self) -> int:
        """Get maximum sequence length."""
        return self.config.get("max_seq_len", 512)

    @property
    def num_classes(self) -> int:
        """Get number of output classes."""
        return self.config.get("num_classes", 10)

    @property
    def is_decoder_only(self) -> bool:
        """Check if decoder-only architecture is used."""
        return self.config.get("is_decoder_only", False)


class PositionalEncoding(nn.Module):
    """Positional encoding layer for Transformer.
    
    Adds positional information to embeddings since Transformers don't have
    inherent sequential structure like RNNs.
    
    Args:
        d_model: Dimension of the model.
        max_len: Maximum sequence length.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model).
        
        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer.
    
    Computes attention scores across multiple "heads" in parallel, allowing the model
    to attend to information from different representation subspaces.
    
    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model).
            key: Key tensor of shape (batch_size, seq_len_k, d_model).
            value: Value tensor of shape (batch_size, seq_len_v, d_model).
            mask: Optional attention mask.
        
        Returns:
            Attention output tensor.
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(output)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward neural network layer.
    
    A simple two-layer MLP applied to each position independently and identically.
    
    Args:
        d_model: Input dimension.
        d_ff: Hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single encoder layer.
    
    Consists of multi-head attention and feed-forward sublayers with residual connections
    and layer normalization.
    
    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward dimension.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Attention mask.
        
        Returns:
            Output tensor.
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Single decoder layer.
    
    Similar to encoder but includes masked multi-head attention and cross-attention
    to the encoder output.
    
    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward dimension.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.masked_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Decoder input tensor.
            encoder_output: Encoder output from cross-attention.
            tgt_mask: Mask for target sequence (causal mask).
            src_mask: Mask for source sequence.
        
        Returns:
            Output tensor.
        """
        # Masked self-attention
        attn_output = self.masked_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention to encoder output
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Transformer(BaseModel):
    """Transformer model.
    
    Supports three architectures:
    1. Encoder-only (BERT-style): For classification, NER, etc.
    2. Decoder-only (GPT-style): For language modeling, text generation.
    3. Encoder-decoder (Seq2Seq): For translation, summarization.
    
    Example usage:
        config = TransformerConfig(
            d_model=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=2048,
            is_decoder_only=False,
        )
        model = Transformer(config)
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__(config)
        
        self.config = config or TransformerConfig()
        
        # Initialize components (will be built in build())
        self.embedding = None
        self.positional_encoding = None
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList() if not self.config.is_decoder_only else None
        self.output_layer = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the Transformer architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, features).
        """
        seq_len = input_shape[0] if len(input_shape) > 0 else self.config.max_seq_len
        
        # Token embedding
        self.embedding = nn.Embedding(self.config.input_dim, self.config.d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.config.d_model, 
            max_len=self.config.max_seq_len,
            dropout=self.config.dropout
        )
        
        # Encoder layers
        for _ in range(self.config.num_encoder_layers):
            layer = EncoderLayer(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                d_ff=self.config.d_ff,
                dropout=self.config.dropout
            )
            self.encoder_layers.append(layer)
        
        # Decoder layers (if not decoder-only)
        if not self.config.is_decoder_only:
            for _ in range(self.config.num_decoder_layers):
                layer = DecoderLayer(
                    d_model=self.config.d_model,
                    num_heads=self.config.num_heads,
                    d_ff=self.config.d_ff,
                    dropout=self.config.dropout
                )
                self.decoder_layers.append(layer)
        
        # Output projection (for classification or next-token prediction)
        if self.config.is_decoder_only:
            # Language modeling head
            self.output_layer = nn.Linear(self.config.d_model, self.config.input_dim)
        else:
            # Classification head
            self.output_layer = nn.Linear(self.config.d_model, self.config.num_classes)

    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create a causal (look-ahead) mask for decoder.
        
        Args:
            seq_len: Sequence length.
        
        Returns:
            Causal mask tensor of shape (1, seq_len, seq_len).
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask == 0

    def forward(
        self, 
        src: torch.Tensor, 
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the Transformer.
        
        Args:
            src: Source input tensor of shape (batch_size, seq_len_src).
            tgt: Target input tensor for decoder (optional).
            src_mask: Mask for source sequence.
            tgt_mask: Mask for target sequence.
        
        Returns:
            Output tensor.
        """
        # Embed and add positional encoding
        src = self.embedding(src) * math.sqrt(self.config.d_model)
        src = self.positional_encoding(src.transpose(0, 1))  # (seq_len, batch_size, d_model)
        
        if tgt is not None:
            tgt = self.embedding(tgt) * math.sqrt(self.config.d_model)
            tgt = self.positional_encoding(tgt.transpose(0, 1))
        
        # Encoder forward pass
        encoder_output = src
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        
        if self.config.is_decoder_only:
            # Decoder-only (GPT-style): use encoder output as context
            x = tgt if tgt is not None else encoder_output
            
            # Apply causal mask if not provided
            if tgt_mask is None and tgt is not None:
                tgt_mask = self.create_causal_mask(tgt.size(0))
            
            for layer in self.decoder_layers or []:
                x = layer(x, encoder_output, tgt_mask, src_mask)
            
            output = self.output_layer(x.transpose(0, 1))  # (batch_size, seq_len, vocab_size)
        else:
            if tgt is None:
                # Encoder-only (BERT-style): use mean pooling over sequence
                pooled = encoder_output.mean(dim=0).transpose(0, 1)  # (batch_size, d_model)
                output = self.output_layer(pooled)
            else:
                # Encoder-decoder (Seq2Seq)
                for layer in self.decoder_layers:
                    tgt = layer(tgt, encoder_output, tgt_mask, src_mask)
                
                output = self.output_layer(tgt.transpose(0, 1))
        
        return output

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        if not hasattr(self, 'model'):
            return 0
        
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_layer_info(self) -> List[Dict[str, Any]]:
        """Get information about each layer."""
        info = []
        
        info.append({
            "type": "Embedding",
            "params": self.config.input_dim * self.config.d_model,
            "output_shape": f"(?, ?, {self.config.d_model})",
            "trainable": True,
        })
        
        for i in range(self.config.num_encoder_layers):
            info.append({
                "type": f"EncoderLayer_{i}",
                "params": 0,
                "output_shape": f"(?, ?, {self.config.d_model})",
                "trainable": True,
            })
        
        if self.decoder_layers is not None:
            for i in range(self.config.num_decoder_layers):
                info.append({
                    "type": f"DecoderLayer_{i}",
                    "params": 0,
                    "output_shape": f"(?, ?, {self.config.d_model})",
                    "trainable": True,
                })
        
        info.append({
            "type": "Output",
            "params": self.config.d_model * self.config.num_classes,
            "output_shape": f"(?, {self.config.num_classes})",
            "trainable": True,
        })
        
        return info

    def save(self, filepath: str) -> None:
        """Save the model to disk."""
        torch.save({
            "config": self.config.to_dict(),
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "Transformer":
        """Load a model from disk."""
        checkpoint = torch.load(filepath)
        config = TransformerConfig(**checkpoint["config"])
        model = cls(config)
        
        if hasattr(model, 'state_dict') and checkpoint["state_dict"]:
            model.load_state_dict(checkpoint["state_dict"])
        
        return model


# Pure PyTorch implementation

class TransformerPyTorch(nn.Module):
    """Pure PyTorch Transformer implementation."""

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig()
        
        # Embedding and positional encoding
        self.embedding = nn.Embedding(self.config.input_dim, self.config.d_model)
        self.pos_encoding = PositionalEncoding(
            self.config.d_model, 
            max_len=self.config.max_seq_len,
            dropout=self.config.dropout
        )
        
        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.d_ff,
            dropout=self.config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_encoder_layers)
        
        # Decoder layers (if not decoder-only)
        if not self.config.is_decoder_only:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.config.d_model,
                nhead=self.config.num_heads,
                dim_feedforward=self.config.d_ff,
                dropout=self.config.dropout,
                activation='gelu',
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.num_decoder_layers)
        
        # Output layer
        if self.config.is_decoder_only:
            self.output_proj = nn.Linear(self.config.d_model, self.config.input_dim)
        else:
            self.output_layer = nn.Linear(self.config.d_model, self.config.num_classes)

    def forward(
        self, 
        src: torch.Tensor, 
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        # Add positional encoding
        src = self.embedding(src) * math.sqrt(self.config.d_model)
        src = self.pos_encoding(src).transpose(0, 1)
        
        if tgt is not None:
            tgt = self.embedding(tgt) * math.sqrt(self.config.d_model)
            tgt = self.pos_encoding(tgt).transpose(0, 1)
        
        # Encoder
        memory = self.encoder(src, src_mask=src_mask)
        
        if self.config.is_decoder_only:
            output = self.output_proj(memory)
        elif tgt is None:
            pooled = memory.mean(dim=1)
            output = self.output_layer(pooled)
        else:
            # Apply causal mask to target
            if tgt_mask is None and tgt is not None:
                seq_len = tgt.size(1)
                tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            
            output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
            output = self.output_layer(output)
        
        return output

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Register Transformer model
ModelFactory.register("transformer_pytorch")(TransformerPyTorch)
