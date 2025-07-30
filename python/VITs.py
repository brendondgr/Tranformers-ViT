import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    """
    Converts input images into patch embeddings with positional encoding and class token.
    
    Args:
        img_size (int): Size of input image (assumes square images)
        patch_size (int): Size of each patch (assumes square patches)
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
    """
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection layer to convert patches to embeddings
        self.patch_projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        
        # Learnable class token
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embeddings for patches + class token
        self.positional_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
    def forward(self, x):
        """
        Forward pass through patch embedding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, img_size, img_size)
            
        Returns:
            torch.Tensor: Patch embeddings of shape (batch_size, num_patches + 1, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Extract patches from input image
        # Reshape from (batch_size, in_channels, img_size, img_size) to patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches shape: (batch_size, in_channels, num_patches_h, num_patches_w, patch_size, patch_size)
        
        patches = patches.contiguous().view(batch_size, self.in_channels, -1, self.patch_size, self.patch_size)
        # patches shape: (batch_size, in_channels, num_patches, patch_size, patch_size)
        
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        # patches shape: (batch_size, num_patches, in_channels, patch_size, patch_size)
        
        patches = patches.view(batch_size, self.num_patches, -1)
        # patches shape: (batch_size, num_patches, in_channels * patch_size * patch_size)
        
        # Apply linear projection to get patch embeddings
        patch_embeddings = self.patch_projection(patches)
        # patch_embeddings shape: (batch_size, num_patches, embed_dim)
        
        # Expand class token for the batch
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        # class_tokens shape: (batch_size, 1, embed_dim)
        
        # Concatenate class token with patch embeddings
        embeddings = torch.cat([class_tokens, patch_embeddings], dim=1)
        # embeddings shape: (batch_size, num_patches + 1, embed_dim)
        
        # Add positional embeddings
        embeddings = embeddings + self.positional_embeddings
        
        return embeddings

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism with scaled dot-product attention.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability (default: 0.0)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for queries, keys, and values
        self.qkv_projection = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate queries, keys, and values
        qkv = self.qkv_projection(x)  # (batch_size, seq_len, embed_dim * 3)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        
        # Compute scaled dot-product attention
        # Attention scores: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        # (batch_size, num_heads, seq_len, head_dim)
        attention_output = torch.matmul(attention_weights, values)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        attention_output = attention_output.reshape(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)
        
        # Apply output projection
        output = self.output_projection(attention_output)
        
        return output

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) component.
    
    Args:
        embed_dim (int): Input embedding dimension
        mlp_ratio (float): Ratio to expand the hidden dimension (default: 4.0)
        dropout (float): Dropout probability (default: 0.0)
    """
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super(MLP, self).__init__()
        
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer encoder block with multi-head attention and MLP.
    Uses pre-normalization and residual connections.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio to expand the MLP hidden dimension (default: 4.0)
        dropout (float): Dropout probability (default: 0.0)
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super(TransformerBlock, self).__init__()
        
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # MLP/FFN
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        """
        Forward pass through transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Pre-normalization + multi-head attention + residual connection
        x = x + self.attention(self.norm1(x))
        
        # Pre-normalization + MLP + residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x

class MLPHead(nn.Module):
    """
    Classification head for Vision Transformer using the class token.
    
    Args:
        embed_dim (int): Input embedding dimension
        num_classes (int): Number of output classes
        dropout (float): Dropout probability (default: 0.0)
    """
    def __init__(self, embed_dim, num_classes, dropout=0.0):
        super(MLPHead, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Layer normalization before classification
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification layer
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through classification head.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
                             where seq_len includes class token as first token
            
        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes)
        """
        # Extract class token (first token in sequence)
        class_token = x[:, 0]  # (batch_size, embed_dim)
        
        # Apply layer normalization
        class_token = self.norm(class_token)
        
        # Apply dropout
        class_token = self.dropout(class_token)
        
        # Apply classification layer
        logits = self.classifier(class_token)  # (batch_size, num_classes)
        
        return logits

class ViTStandard(nn.Module):
    """
    Standard Vision Transformer model for image classification.
    
    Args:
        img_size (int): Size of input image (assumes square images)
        patch_size (int): Size of each patch (assumes square patches)
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
        embed_dim (int): Embedding dimension
        num_layers (int): Number of transformer encoder layers
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio to expand the MLP hidden dimension (default: 4.0)
        dropout (float): Dropout probability (default: 0.0)
    """
    def __init__(self, img_size, patch_size, in_channels, num_classes, 
                 embed_dim, num_layers, num_heads, mlp_ratio=4.0, dropout=0.0):
        super(ViTStandard, self).__init__()
        
        # Store configuration parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Validate configuration
        assert img_size % patch_size == 0, f"Image size ({img_size}) must be divisible by patch size ({patch_size})"
        assert embed_dim % num_heads == 0, f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"
        
        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Classification head
        self.classification_head = MLPHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights using standard initialization schemes."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # Standard initialization for layer normalization
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through the Vision Transformer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, img_size, img_size)
            
        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes)
            
        Raises:
            ValueError: If input tensor shape is incompatible with model configuration
        """
        # Comprehensive input shape validation
        self._validate_input_tensor(x)
        
        # Convert image to patch embeddings
        x = self.patch_embedding(x)  # (batch_size, num_patches + 1, embed_dim)
        
        # Apply transformer encoder blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)  # (batch_size, num_patches + 1, embed_dim)
        
        # Apply classification head using class token
        logits = self.classification_head(x)  # (batch_size, num_classes)
        
        return logits
    
    def _validate_input_tensor(self, x: torch.Tensor) -> None:
        """
        Validate input tensor shape and compatibility with model configuration.
        
        Args:
            x: Input tensor to validate
            
        Raises:
            ValueError: If input tensor is incompatible
        """
        # Check tensor dimensionality
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor (batch, channels, height, width), "
                           f"got {len(x.shape)}D tensor with shape {x.shape}")
        
        batch_size, channels, height, width = x.shape
        
        # Validate batch size
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        
        # Validate number of channels
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}. "
                           f"Model configured for {self.in_channels} channels.")
        
        # Validate image dimensions
        if height != self.img_size:
            raise ValueError(f"Expected image height {self.img_size}, got {height}. "
                           f"Model configured for {self.img_size}x{self.img_size} images.")
        
        if width != self.img_size:
            raise ValueError(f"Expected image width {self.img_size}, got {width}. "
                           f"Model configured for {self.img_size}x{self.img_size} images.")
        
        # Validate tensor data type
        if not x.dtype.is_floating_point:
            raise ValueError(f"Expected floating point tensor, got {x.dtype}")
        
        # Validate tensor values are in reasonable range
        if torch.isnan(x).any():
            raise ValueError("Input tensor contains NaN values")
        
        if torch.isinf(x).any():
            raise ValueError("Input tensor contains infinite values")
    
    def validate_dataset_compatibility(self, data_loader) -> bool:
        """
        Validate that a data loader is compatible with this model.
        
        Args:
            data_loader: PyTorch DataLoader to validate
            
        Returns:
            True if compatible
            
        Raises:
            ValueError: If data loader is incompatible
        """
        try:
            # Get a sample batch
            data_iter = iter(data_loader)
            sample_batch, sample_labels = next(data_iter)
            
            # Validate the sample batch
            self._validate_input_tensor(sample_batch)
            
            # Additional validation for labels
            if len(sample_labels.shape) != 1:
                raise ValueError(f"Expected 1D label tensor, got {len(sample_labels.shape)}D")
            
            if sample_labels.shape[0] != sample_batch.shape[0]:
                raise ValueError(f"Batch size mismatch: images {sample_batch.shape[0]}, "
                               f"labels {sample_labels.shape[0]}")
            
            # Check label range
            if sample_labels.min() < 0 or sample_labels.max() >= self.num_classes:
                raise ValueError(f"Labels must be in range [0, {self.num_classes-1}], "
                               f"got range [{sample_labels.min()}, {sample_labels.max()}]")
            
            return True
            
        except StopIteration:
            raise ValueError("Data loader is empty")
        except Exception as e:
            raise ValueError(f"Dataset compatibility validation failed: {str(e)}")
    
    def test_with_various_configurations(self, test_configs: list = None) -> dict:
        """
        Test model with various input configurations to verify robustness.
        
        Args:
            test_configs: List of test configurations (optional)
            
        Returns:
            Dictionary with test results
        """
        if test_configs is None:
            # Default test configurations
            test_configs = [
                {"batch_size": 1, "description": "single sample"},
                {"batch_size": 16, "description": "small batch"},
                {"batch_size": 64, "description": "large batch"},
            ]
        
        results = {}
        
        for config in test_configs:
            batch_size = config["batch_size"]
            description = config["description"]
            
            try:
                # Create test tensor
                test_input = torch.randn(
                    batch_size, self.in_channels, self.img_size, self.img_size
                )
                
                # Test forward pass
                with torch.no_grad():
                    output = self.forward(test_input)
                
                # Validate output shape
                expected_shape = (batch_size, self.num_classes)
                if output.shape != expected_shape:
                    raise ValueError(f"Expected output shape {expected_shape}, got {output.shape}")
                
                results[description] = {
                    "status": "success",
                    "input_shape": test_input.shape,
                    "output_shape": output.shape,
                    "batch_size": batch_size
                }
                
            except Exception as e:
                results[description] = {
                    "status": "failed",
                    "error": str(e),
                    "batch_size": batch_size
                }
        
        return results
    
    def get_num_patches(self):
        """Get the number of patches for the current configuration."""
        return (self.img_size // self.patch_size) ** 2
    
    def get_config(self):
        """Get model configuration as a dictionary."""
        return {
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'in_channels': self.in_channels,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'num_patches': self.get_num_patches()
        }
