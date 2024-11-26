import torch
import torch.nn as nn
from typing import Optional, Tuple

class SiglipVisionConfig:
    def __init__(
            self,
            hidden_size = 768, # embedding vector size (16 x 16 x 3 = 768)
            intermediate_size = 3072, # FF linear layer size
            num_hidden_layers = 12, # no. of vision transformer layers
            num_attention_heads = 12, # no. of attention heads used
            num_channels = 3, # 3 channels for RGB, KIV: is this modifiable for RGB-D?
            image_size = 224, # choosing the smallest image input size (224, 448, 896 were described in paper)
            patch_size = 16, # 16 x 16px patches created from input images
            layer_norm_eps = 1e-6, # layer normalisation parameter
            attention_dropout = 0, # attention layer dropout setting, set to 0 for non-use
            num_image_tokens: int = None,
            **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels, # no. of kernels = no. of channels of input image
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size, # no overlapping patches since stride = kernel size
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2 # ^2 since patches are convoluted in 2D
        self.num_positions = self.num_patches # no. of positions is later used to calculate positional embeddings
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim) # learned embedding
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [batch size, channels, height, width]

        patch_embeds = self.patch_embedding(pixel_values) # output format of patch_embedding conv2D: [batch size, embedding dimensions, input height // patch size, input width // patch size]
        embeddings = patch_embeds.flatten(2) # [batch size, embedding dimensions, input height // patch size, input width // patch size] -> [batch size, embedding dimensions, no. of patches in 1 channel of input image]
        embeddings = embeddings.transpose(1, 2) # reformat embeddings vector such that no. of patches comes before embedding dimensions (for sequential processing in transformers) -> [batch size, no. of patches, embedding dimensions]
        embeddings += self.position_embedding(self.posiition_ids) # add positional embeddings to each patch, each positional encoding vector size is equal to the embedding dimensions

        return embeddings

class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden states shape: batch_size, num_patches, embed_dim
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        # output shape: batch_size, num_heads, num_patches, head_dim
        # transpose operation to ensure each attention head's tensors are already grouped
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        key_states = self.k_proj(hidden_states)
        # same shape as query_states
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        value_states = self.v_proj(hidden_states)
        # same shape as query_states
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # attention computation
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale) 

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f"{attn_weights.size()}"
            )

        # applies softmax row-wise
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # apply weights to values
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention output should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f"{attn_output.size()}"
            )
    
        # transpose attention output
        # [batch_size, num_heads, num_patches, head_dim] -> [batch_size, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size, num_patches, num_heads, head_dim] -> [batch_size, num_patches, embed_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # dot product of Wo and the concatenated attention output
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")

        hidden_states = self.fc2(hidden_states)

        return hidden_states

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual for addition after self attention layer
        residual = hidden_states
        # layer normalisation
        hidden_states = self.layer_norm1(hidden_states)
        # self attention
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # skip connection after self attention
        hidden_states = residual + hidden_states
        # store new residual for addition after MLP
        residual = hidden_states
        # layer normalisation
        hidden_states = self.layer_norm2(hidden_states)
        # MLP 
        hidden_states = self.mlp(hidden_states)
        # skip connection after MLP
        hidden_states = residual + hidden_states

        return hidden_states

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
    
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # no change in dimensions
            hidden_states = encoder_layer(hidden_states)
        
        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config) # assign embeddings from input images (convolution & flattening operations + performs positional encoding)
        self.encoder = SiglipEncoder(config) # instantiate transformer layers as encoder, transformer: norm -> multi-head attention -> norm -> feed forward
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [batch size, channels, height, width] -> [batch_size, no. of patches, embedding dimension]
        hidden_states = self.embeddings(pixel_values) # converting patches into embeddings

        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple: # image loaded with numpy as 3D array
        # [batch size, channels, height, width] -> [batch_size, no. of patches, embedding dimension]
        return self.vision_model(pixel_values=pixel_values)
