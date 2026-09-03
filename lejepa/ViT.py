import torch
import torch.nn.functional as F
from lejepa.config import ModelConfig

class Normaliser(torch.nn.Module):
    """
    Normalises the input data for the ViT using a constant
    mean and a constant std.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        mean: The mean for each channel independently as a tensor
        std : The std for each channel independently as a tensor

        Both are in form (1, 3, 1, 1) for broadcasting.

        Example code to get mean/std:
        ```
        mean = Xtr.mean((0, 2, 3), keepdim=True)
        std  = Xtr.std((0, 2, 3), keepdim=True)
        ```
        """

        super().__init__()
        # Clone to protect against upstream mutation
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If in image format [0, 255] convert to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float().div(255)

        return (x - self.mean) / self.std


class Tokeniser(torch.nn.Module):
    """
    Takes in a batch of images and returns a batch of tokenised images.
    Each image is split into patches and each patch
    is flattened and transformed into a 192dim token.
    Positional embeddings are then added on.
    """

    def __init__(self, image_size: int, out_dims: int, patch_size: int, pos_embed_ratio: float):
        super().__init__()

        self.token_count = (image_size // patch_size) ** 2

        self.patch_embedder = torch.nn.Conv2d(
            3, out_dims, patch_size, stride=patch_size
        )

        # Scaled by 0.03 to ensure the position embeddings add information
        # whilst not drowning out the patch embedding
        self.pos_embeddings = torch.nn.parameter.Parameter(
            torch.randn(self.token_count, out_dims) * pos_embed_ratio
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed and convert from (N, 192, 8, 8) to (N, 64, 192)
        tokens = self.patch_embedder(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embeddings
        return tokens


class Attention(torch.nn.Module):
    """
    Implements the self attention mechanism for the ViT
    """

    def __init__(self, n_heads: int, dims: int):
        super().__init__()
        self.W_KQV = torch.nn.Linear(dims, dims * 3, bias=False)
        self.W_O = torch.nn.Linear(dims, dims, bias=False)
        self.N_HEADS = n_heads
        self.DIMS = dims


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute KQV all at once, for every
        # element in the batch and every head
        kqv = self.W_KQV(x)

        # Split into separate K, Q and V
        k, q, v = kqv.split(self.DIMS, dim=-1)

        # Split across heads into (N, Tokens, Heads, Elements)
        d_k = self.DIMS // self.N_HEADS
        shape = k.shape[:2] + (self.N_HEADS, d_k)

        k = k.reshape(shape)
        q = q.reshape(shape)
        v = v.reshape(shape)

        # Also permute to (N, Heads, Tokens, Elements) for matmuls
        k = k.permute((0, 2, 1, 3))
        q = q.permute((0, 2, 1, 3))
        v = v.permute((0, 2, 1, 3))

        # Pair up every vector, compare each one's query to the other's key
        # (by using the dot prod) and get a weight to quantify all that.
        weights = q @ k.mT
        # Dot prod scales by dim so divide by √(dims) to keep var at 1
        weights = weights / (d_k**0.5)
        # Turn into probabilities
        weights = F.softmax(weights, dim=-1)

        # For each vec apply the weight for every vec and sum to
        # get a 'semantic shift' vector
        embeddings = weights @ v
        # (N, heads, tokens, head_size) e.g. (N, 6, 64, 32)
        embeddings_shape = shape[:2] + (self.DIMS,)
        # (N, 6, 64, 32) --> (N, 64, 6, 32) --> (N, 64, 192)
        embeddings = embeddings.transpose(1, 2).contiguous().reshape(embeddings_shape)
        # We concatenated vectors in a lower dim space so rectify by applying
        # an output matrix to bring back to higher dim space
        embeddings = self.W_O(embeddings)

        return embeddings


class TransformerBlock(torch.nn.Module):
    """
    Applies attention and then an MLP to the input.

    Passes in normalised embeddings into both and adds
    the output back to original stream for larger gradients
    and better training.
    """

    def __init__(self, dims: int, n_heads: int, mlp_ratio: int):
        super().__init__()
        self.norm_a = torch.nn.LayerNorm(dims)
        self.norm_b = torch.nn.LayerNorm(dims)
        self.attention = Attention(n_heads, dims)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(dims, mlp_ratio * dims),
            torch.nn.GELU(),
            torch.nn.Linear(mlp_ratio * dims, dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_a(x))
        x = x + self.MLP(self.norm_b(x))
        return x


class ViT(torch.nn.Module):
    """
    A full ViT using [BLOCKS] (e.g. 5) transformer blocks

    Tokenises the input data. Passes those tokens through [BLOCKS] transformer
    blocks. Normalises output and then takes the mean of each dimension individually
    to produce one final output embedding.

    MEAN and STD are per channel and properties of the training data,
    both are in form (1, 3, 1, 1) for broadcasting.

    Example code to get mean/std:
    ```
    mean = Xtr.mean((0, 2, 3), keepdim=True)
    std  = Xtr.std((0, 2, 3), keepdim=True)
    ```
    """

    def __init__(
        self,
        image_size: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        config: ModelConfig = ModelConfig()
    ):
        super().__init__()
        self.input_norm = Normaliser(mean, std)
        self.tokeniser = Tokeniser(
            image_size, config.dims, config.patch_size, config.pos_embed_ratio
        )

        # Stack [BLOCKS] (e.g. 5) transformer blocks one after the other
        self.transformer_blocks = torch.nn.Sequential()
        for _ in range(config.blocks):
            self.transformer_blocks.append(
                TransformerBlock(config.dims, config.heads, config.mlp_ratio)
            )

        self.out_norm = torch.nn.LayerNorm(config.dims)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.input_norm(x)
        tokens = self.tokeniser(x_norm)
        patch_embeddings = self.transformer_blocks(tokens)
        patch_embeddings = self.out_norm(patch_embeddings)
        image_embedding = patch_embeddings.mean(dim=1)

        return image_embedding
