import torch.nn as nn
import torch

class TransformerBlock(nn.Module):

    def __init__(self, transformer_embed_dim, n_heads):

        super().__init__()

        self.embed_dim = transformer_embed_dim
        self.n_heads = n_heads

        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.projections = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim) for _  in range(3)])
        self.mha = nn.MultiheadAttention(self.embed_dim, num_heads=self.n_heads)
        self.fc_post_attn = nn.Linear(self.embed_dim, self.embed_dim)
        self.activation = nn.GELU()

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

class TransformerBlockCrossAttention(nn.Module):

    def __init__(self, transformer_embed_dim, n_heads):

        super().__init__()

        self.embed_dim = transformer_embed_dim
        self.n_heads = n_heads

        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.projections = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim) for _  in range(2)])
        self.mha = nn.MultiheadAttention(self.embed_dim, num_heads=self.n_heads)
        self.fc_post_attn = nn.Linear(self.embed_dim, self.embed_dim)
        self.activation = nn.GELU()

    def forward(self,queries, leaf_embeddings = None,attn_mask = None):
                
        residual = queries.clone()
        queries = self.layer_norm(queries)
        k, v = tuple(self.projections[i](leaf_embeddings) for i in range(2))
        queries, _ = self.mha(queries,k,v, need_weights = False, attn_mask = attn_mask)
        queries = queries + residual
        residual = queries.clone()
        queries = self.layer_norm(queries)
        queries = self.activation(self.fc_post_attn(queries))
        queries += residual

        return(queries)
    
class DynamicQueryGenerator(nn.Module):
    def __init__(self, d_model, d_hidden=128):
        super().__init__()
        self.seed_mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model)
        )
        # projection to mix seed with positions
        self.proj = nn.Linear(d_model, d_model)  # 16 = pos encoding dim

    def forward(self, leaf_embeddings):
        """
        leaf_embeddings: (N, d_model)
        returns queries: (N-1, d_model)
        """
        N, d_model = leaf_embeddings.shape
        # 1) global pooled embedding
        global_ctx = leaf_embeddings.mean(dim=0)  # (d_model,)
        seed = self.seed_mlp(global_ctx)          # (d_model,)

        # 2) make positional encodings for N-1 queries
        pos_ids = torch.arange(N-1, device=leaf_embeddings.device).unsqueeze(1)  # (N-1,1)
        # simple sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2, device=leaf_embeddings.device) *
                             -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(N-1, d_model, device=leaf_embeddings.device)
        pe[:, 0::2] = torch.sin(pos_ids * div_term)
        pe[:, 1::2] = torch.cos(pos_ids * div_term)

        # 3) concatenate seed with position encodings, project to d_model
        seed_expanded = seed.unsqueeze(0).repeat(N-1, 1)  # (N-1, d_model)
        # queries = self.proj(torch.cat([seed_expanded, pe], dim=-1))  # (N-1, d_model)
        queries = seed_expanded + pe

        return queries
    

class InternalNodeEmbedding(nn.Module):

    def __init__(self, input_dim, attn_dim, n_heads, n_layers):

        super().__init__() 

        self.input_dim = input_dim
        self.attn_dim = attn_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.activation = nn.GELU()
    
        self.query_embedding = DynamicQueryGenerator(self.input_dim)

        self.cross_attn_blocks = nn.ModuleList([TransformerBlockCrossAttention(self.attn_dim, self.n_heads) for _ in range(self.n_layers)])
        self.self_attn_blocks =  nn.ModuleList([TransformerBlock(self.attn_dim, self.n_heads) for _ in range(self.n_layers)])
        self.leaf_projection = nn.Linear(self.input_dim, self.attn_dim)
        
        self.fc_init = nn.Linear(self.input_dim, self.attn_dim)
        self.output_layer = nn.Linear(self.attn_dim, self.input_dim)

    def forward(self, x):

        leaf_embeds = self.leaf_projection(x.clone())
        
        x = self.query_embedding(x)
        x = self.activation(self.fc_init(x))

        for i in range(self.n_layers):
            x = self.cross_attn_blocks[i](x, leaf_embeddings = leaf_embeds, attn_mask = None)
            x = self.self_attn_blocks[i](x, attn_mask = None)

        x = self.output_layer(x)

        return x
    
class BaseNodeModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, attn_dim_main, n_heads_main, n_layers_main,
                  attn_dim_int, n_heads_int, n_layers_int, output_dim):

        super().__init__() 

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attn_dim_main = attn_dim_main
        self.n_heads_main = n_heads_main
        self.n_layers_main = n_layers_main
        self.attn_dim_int = attn_dim_int
        self.n_heads_int = n_heads_int
        self.n_layers_int = n_layers_int
        self.output_dim = output_dim
        self.activation = nn.GELU()

        self.int_node_embedder = InternalNodeEmbedding(self.input_dim, attn_dim=self.attn_dim_int, n_heads=self.n_heads_int, n_layers=self.n_layers_int)
        self.transformer_layers = nn.ModuleList([TransformerBlock(self.attn_dim_main, self.n_heads_main) for _ in range(self.n_layers_main)])
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc_init = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc_pre_attn = nn.Linear(self.hidden_dim, self.attn_dim_main)
        self.fc_post_attn = nn.Linear(self.attn_dim_main, self. hidden_dim)
    
    def forward(self, x, attn_mask = None):

        int_embed = self.int_node_embedder(x)
        x = torch.concat((int_embed, x), dim = 0)

        x = self.activation(self.fc_init(x))
        x = self.activation(self.fc_pre_attn(x))

        for layer in self.transformer_layers:
            x = layer(x, attn_mask = attn_mask)

        x = self.activation(self.fc_post_attn(x))
        x = self.output_layer(x)

        return x
    
class RegressionHead(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_layers):

        super().__init__() 

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = nn.GELU()

        self.fc_init = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layer_list = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layers)])
        self.output_layer = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):

        x = self.activation(self.fc_init(x))

        for layer in self.hidden_layer_list:
            x = self.activation(layer(x))

        x = torch.exp(self.output_layer(x))

        return x

class MainPhyloModel(nn.Module):

    def __init__(self, input_dim, hidden_dim_base, attn_dim_main, n_heads_main, n_layers_main,
                  attn_dim_int, n_heads_int, n_layers_int, output_dim, 
                  hidden_dim_head, n_layers_bl, n_layers_rd):
        
        super().__init__() 

        self.input_dim = input_dim
        self.hidden_dim_base = hidden_dim_base
        self.attn_dim_main = attn_dim_main
        self.n_heads_main = n_heads_main
        self.n_layers_main = n_layers_main
        self.attn_dim_int = attn_dim_int
        self.n_heads_int = n_heads_int
        self.n_layers_int = n_layers_int
        self.output_dim = output_dim
        self.activation = nn.GELU()

        self.hidden_dim_head = hidden_dim_head
        self.n_layers_bl = n_layers_bl
        self.n_layers_rd = n_layers_rd

        self.node_embeddings = BaseNodeModel(input_dim, hidden_dim_base, attn_dim_main, n_heads_main, n_layers_main,
                                                attn_dim_int, n_heads_int, n_layers_int, output_dim)


        self.branch_length_head = RegressionHead(self.output_dim, self.hidden_dim_head, self.n_layers_bl)
        self.root_dist_head = RegressionHead(self.output_dim, self.hidden_dim_head, self.n_layers_rd)

    def forward(self, x, attn_mask = None):

        x = self.node_embeddings(x, attn_mask = attn_mask)

        branch_length = self.branch_length_head(x)
        root_distance = self.root_dist_head(x)

        return branch_length, root_distance 



    