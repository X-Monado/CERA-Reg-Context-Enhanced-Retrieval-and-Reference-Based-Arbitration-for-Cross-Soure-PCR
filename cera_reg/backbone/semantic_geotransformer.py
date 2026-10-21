import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from geotransformer.modules.transformer import RPETransformerLayer
from geotransformer.modules.geotransformer.geotransformer import GeometricStructureEmbedding
from geotransformer.modules.layers import build_dropout_layer
from geotransformer.modules.transformer.output_layer import AttentionOutput


class SemanticBypassMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, semantic_dim, dropout=None):
        super(SemanticBypassMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        self.semantic_proj_q = nn.Linear(semantic_dim, self.d_model, bias=False)
        self.semantic_proj_k = nn.Linear(semantic_dim, self.d_model, bias=False)
        self.semantic_scale = nn.Parameter(torch.tensor([0.1]))

        self.dropout = build_dropout_layer(dropout)

    def forward(self, input_q, input_k, input_v, semantic_q=None, semantic_k=None,
                key_weights=None, key_masks=None):
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)

        attention_scores = torch.einsum('bhnc,bhmc->bhnm', q, k) / self.d_model_per_head ** 0.5

        if semantic_q is not None and semantic_k is not None:
            sem_q = rearrange(self.semantic_proj_q(semantic_q), 'b n (h c) -> b h n c', h=self.num_heads)
            sem_k = rearrange(self.semantic_proj_k(semantic_k), 'b m (h c) -> b h m c', h=self.num_heads)
            sem_scores = torch.einsum('bhnc,bhmc->bhnm', sem_q, sem_k) / self.d_model_per_head ** 0.5
            attention_scores = attention_scores + self.semantic_scale * sem_scores

        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))

        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores


class SemanticBypassAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, semantic_dim, dropout=None):
        super(SemanticBypassAttentionLayer, self).__init__()
        self.attention = SemanticBypassMultiHeadAttention(d_model, num_heads, semantic_dim, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_states, memory_states, semantic_input=None, semantic_memory=None,
                memory_weights=None, memory_masks=None):
        hidden_states, attention_scores = self.attention(
            input_states, memory_states, memory_states,
            semantic_q=semantic_input, semantic_k=semantic_memory,
            key_weights=memory_weights, key_masks=memory_masks,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class SemanticBypassTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, semantic_dim, dropout=None, activation_fn='ReLU'):
        super(SemanticBypassTransformerLayer, self).__init__()
        self.attention = SemanticBypassAttentionLayer(d_model, num_heads, semantic_dim, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(self, input_states, memory_states, semantic_input=None, semantic_memory=None,
                memory_weights=None, memory_masks=None):
        hidden_states, attention_scores = self.attention(
            input_states, memory_states,
            semantic_input=semantic_input, semantic_memory=semantic_memory,
            memory_weights=memory_weights, memory_masks=memory_masks,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores


class SemanticGeometricTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads, blocks,
                 sigma_d, sigma_a, angle_k, semantic_dim=256,
                 dropout=None, activation_fn='ReLU', reduction_a='max'):
        super(SemanticGeometricTransformer, self).__init__()

        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)

        self.in_proj = nn.Linear(input_dim, hidden_dim)

        layers = []
        for block in blocks:
            if block not in ['self', 'cross']:
                raise ValueError(f'Unsupported block type: {block}')
            if block == 'self':
                layers.append(RPETransformerLayer(hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(SemanticBypassTransformerLayer(hidden_dim, num_heads, semantic_dim, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.blocks = blocks

        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, ref_points, src_points, ref_feats, src_feats,
                ref_semantic_feats=None, src_semantic_feats=None,
                ref_masks=None, src_masks=None):
        ref_embeddings = self.embedding(ref_points)
        src_embeddings = self.embedding(src_points)

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        for i, block in enumerate(self.blocks):
            if block == 'self':
                ref_feats, _ = self.layers[i](ref_feats, ref_feats, ref_embeddings, memory_masks=ref_masks)
                src_feats, _ = self.layers[i](src_feats, src_feats, src_embeddings, memory_masks=src_masks)
            else:
                ref_feats, _ = self.layers[i](
                    ref_feats, src_feats,
                    semantic_input=ref_semantic_feats, semantic_memory=src_semantic_feats,
                    memory_masks=src_masks,
                )
                src_feats, _ = self.layers[i](
                    src_feats, ref_feats,
                    semantic_input=src_semantic_feats, semantic_memory=ref_semantic_feats,
                    memory_masks=ref_masks,
                )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats
