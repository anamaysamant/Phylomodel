import torch.nn as nn
import torch
import esm

model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()

msa_transf_token_embedding_weights = model.embed_tokens.weight
leaf_embedder = torch.nn.Embedding.from_pretrained(msa_transf_token_embedding_weights, freeze=True)

del model



class TransformerBlock(nn.Module):

    def __init__(self, hidden_dim, transformer_embed_dim, n_heads):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim = transformer_embed_dim
        self.n_heads = n_heads

        self.layer_norm = nn.LayerNorm(self.embed_dim)

        self.projections_rows = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim) for _  in range(3)])
        self.projections_cols = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim) for _  in range(3)])

        self.mha_rows = nn.MultiheadAttention(self.embed_dim, num_heads=self.n_heads)
        self.mha_cols = nn.MultiheadAttention(self.embed_dim, num_heads=self.n_heads)

        self.fc_post_attn = nn.Linear(self.embed_dim, self.embed_dim)
        self.activation = nn.GELU()

    def forward(self,x, attn_mask = None):

        residual = x.clone()
        x = self.layer_norm(x)
        
        # x_copy = x.clone()

        # for row in range(x.shape[0]):
        #     q, k, v = tuple(self.projections_rows[i](x_copy[row,:,:]) for i in range(3))
        #     x[row,:,:], _ = self.mha_rows(q,k,v, need_weights = False, attn_mask = None)

        # del(x_copy)

        q, k, v = tuple(self.projections_rows[i](x) for i in range(3))
        x, _ = self.mha_rows(q,k,v, need_weights = False, attn_mask = None)
        x = x + residual

        residual = x.clone()

        x = self.layer_norm(x)
        
        # x_copy = x.clone()

        # for col in range(x.shape[1]):
        #     q, k, v = tuple(self.projections_cols[i](x_copy[:,col,:]) for i in range(3))
        #     x[:,col,:], _ = self.mha_cols(q,k,v, need_weights = False, attn_mask = None)

        # del(x_copy)
        x = x.transpose(0,1)
        q, k, v = tuple(self.projections_cols[i](x) for i in range(3))
        x, _ = self.mha_cols(q,k,v, need_weights = False, attn_mask = None)
        x = x.transpose(0,1)
        x = x + residual

        residual = x.clone()

        x = self.activation(self.fc_post_attn(x))
        x += residual

        return(x)

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
        
        N, C, d_model = leaf_embeddings.shape
        # 1) global pooled embedding
        global_ctx = leaf_embeddings.mean(dim=0)  # (d_model,)
        seed = self.seed_mlp(global_ctx)          # (d_model,)

        # 2) make positional encodings for N-1 queries
        pos_ids_rows = torch.arange(N-1, device=leaf_embeddings.device).unsqueeze(1)  # (N-1,1)
        # simple sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2, device=leaf_embeddings.device) *
                             -(torch.log(torch.tensor(10000.0)) / d_model))
        pe_rows = torch.zeros(N-1, d_model, device=leaf_embeddings.device)
        pe_rows[:, 0::2] = torch.sin(pos_ids_rows * div_term)
        pe_rows[:, 1::2] = torch.cos(pos_ids_rows * div_term)

        pe_rows = pe_rows.unsqueeze(1).repeat(1,C,1)

        pos_ids_cols = torch.arange(C, device=leaf_embeddings.device).unsqueeze(1)  # (N-1,1)
        # simple sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2, device=leaf_embeddings.device) *
                             -(torch.log(torch.tensor(10000.0)) / d_model))
        pe_cols = torch.zeros(C, d_model, device=leaf_embeddings.device)
        pe_cols[:, 0::2] = torch.sin(pos_ids_cols * div_term)
        pe_cols[:, 1::2] = torch.cos(pos_ids_cols * div_term)

        pe_cols = pe_cols.unsqueeze(0).repeat(N-1,1,1)

        # 3) concatenate seed with position encodings, project to d_model
        seed_expanded = seed.unsqueeze(0).repeat(N-1, 1,1)  # (N-1, d_model)
        # queries = self.proj(torch.cat([seed_expanded, pe], dim=-1))  # (N-1, d_model)
        queries = seed_expanded + pe_rows + pe_cols

        return queries
    

class ParentPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, embed_dim, n_heads, n_layers, output_dim):

        super().__init__() 

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.activation = nn.GELU()

        self.fc_init = nn.Linear(self.input_dim, self. hidden_dim)
        self.int_node_query_generator = DynamicQueryGenerator(self.input_dim)
        self.transformer_layers = nn.ModuleList([TransformerBlock(self.hidden_dim, self. embed_dim, self.n_heads) for _ in range(self.n_layers)])
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc_pre_attn = nn.Linear(self.hidden_dim, self. embed_dim)
        self.fc_post_attn = nn.Linear(self.embed_dim, self. hidden_dim)

    def forward(self, x, attn_mask = None):
                
        int_nodes = self.int_node_query_generator(x)
        x = torch.concat([int_nodes, x], dim=0)
        x = self.activation(self.fc_init(x))
        x = self.activation(self.fc_pre_attn(x))

        for layer in self.transformer_layers:
            x = layer(x, attn_mask = attn_mask)

        x = self.activation(self.fc_post_attn(x))
        x = self.output_layer(x)
        x = x.mean(dim=1)

        return x


class BranchLengthPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, embed_dim, n_heads, n_layers):

        super().__init__() 

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.activation = nn.GELU()

        self.fc_init = nn.Linear(self.input_dim, self. hidden_dim)
        self.transformer_layers = nn.ModuleList([TransformerBlock(self.hidden_dim, self. embed_dim, self.n_heads) for _ in range(self.n_layers)])
        self.output_layer = nn.Linear(self.hidden_dim, self.input)
        self.fc_pre_attn = nn.Linear(self.hidden_dim, self. embed_dim)
        self.fc_post_attn = nn.Linear(self.embed_dim, self. hidden_dim)

    def forward(self, x, attn_mask = None):

        x = self.activation(self.fc_init(x))
        x = self.activation(self.fc_pre_attn(x))

        for layer in self.transformer_layers:
            x = layer(x, attn_mask = attn_mask)

        x = self.activation(self.fc_post_attn(x))
        x = torch.exp(self.output_layer(x))

        return x