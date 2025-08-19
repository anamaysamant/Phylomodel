import torch.nn as nn

class TransformerBlock(nn.Module):

    def __init__(self, hidden_dim, transformer_embed_dim, n_heads):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim = transformer_embed_dim
        self.n_heads = n_heads

        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.projections = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim) for _  in range(3)])
        self.mha = nn.MultiheadAttention(self.embed_dim, num_heads=self.n_heads)
        self.fc_post_attn = nn.Linear(self.embed_dim, self.embed_dim)
        self.activation = nn.GELU()
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim)

    def forward(self,x, attn_mask = None):

        residual = x.clone()
        x = self.layer_norm(x)
        q, k, v = tuple(self.projections[i](x) for i in range(3))
        x, _ = self.mha(q,k,v, need_weights = False, attn_mask = attn_mask)
        x = x + residual
        residual = x.clone()
        x = self.layer_norm(x)
        x = self.activation(self.fc_post_attn(x))
        x += residual

        return(x)


class ParentPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, transformer_embed_dim, n_heads, n_layers, output_dim):

        super().__init__() 

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = transformer_embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.activation = nn.GELU()

        self.fc_init = nn.Linear(self.input_dim, self. hidden_dim)
        self.transformer_layers = nn.ModuleList([TransformerBlock(self.hidden_dim, self. embed_dim, self.n_heads) for _ in range(self.n_layers)])
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc_pre_attn = nn.Linear(self.hidden_dim, self. embed_dim)
        self.fc_post_attn = nn.Linear(self.embed_dim, self. hidden_dim)

    def forward(self, x, attn_mask = None):

        x = self.activation(self.fc_init(x))
        x = self.activation(self.fc_pre_attn(x))

        for layer in self.transformer_layers:
            x = layer(x, attn_mask = attn_mask)

        x = self.activation(self.fc_post_attn(x))
        x = self.output_layer(x)

        return x


