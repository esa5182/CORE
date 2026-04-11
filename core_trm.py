import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder

from core_ave import COREave


class TransNet(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()

        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['embedding_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']

        self.position_embedding = nn.Embedding(
            dataset.field2seqlen['item_id_list'],
            self.hidden_size
        )

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.fn = nn.Linear(self.hidden_size, 1)

        self.apply(self._init_weights)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(
            extended_attention_mask, 0.0, -10000.0
        )
        return extended_attention_mask

    def forward(self, item_seq, item_emb):
        mask = item_seq.gt(0)

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb,
            extended_attention_mask,
            output_all_encoded_layers=True
        )
        output = trm_output[-1]

        alpha = self.fn(output).to(torch.double)
        alpha = torch.where(mask.unsqueeze(-1), alpha, -9e15)
        alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
        return alpha

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class COREtrm(COREave):
    def __init__(self, config, dataset):
        super(COREtrm, self).__init__(config, dataset)
        self.net = TransNet(config, dataset)

    def forward(self, item_seq):
        x = self.item_embedding(item_seq)
        x = self.sess_dropout(x)
        alpha = self.net(item_seq, x)
        seq_output = torch.sum(alpha * x, dim=1)
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output


class COREgrt(COREave):
    """
    Gated Recency + Transformer CORE
    """

    def __init__(self, config, dataset):
        super(COREgrt, self).__init__(config, dataset)
        self.net = TransNet(config, dataset)

        # Learnable gate to balance transformer weights and recency weights
        self.gate = nn.Linear(self.embedding_size, 1)

        # Controls sharpness of recency weighting
        self.recency_lambda = config['recency_lambda']

    def build_recency_alpha(self, item_seq):
        """
        Build recency-based weights:
        more recent valid items get larger weights.
        Output shape: [B, L, 1]
        """
        mask = item_seq.gt(0)  # [B, L]
        seq_len = item_seq.size(1)

        positions = torch.arange(seq_len, device=item_seq.device).float()  # [L]
        # Exponential recency growth
        recency_scores = torch.exp(self.recency_lambda * positions)  # [L]
        recency_scores = recency_scores.unsqueeze(0).expand_as(item_seq)  # [B, L]

        recency_scores = torch.where(
            mask,
            recency_scores,
            torch.zeros_like(recency_scores)
        )

        recency_sum = recency_scores.sum(dim=1, keepdim=True) + 1e-12
        alpha_rec = recency_scores / recency_sum
        return alpha_rec.unsqueeze(-1)  # [B, L, 1]

    def forward(self, item_seq):
        x = self.item_embedding(item_seq)   # [B, L, D]
        x = self.sess_dropout(x)

        # Transformer attention weights
        alpha_trm = self.net(item_seq, x)   # [B, L, 1]

        # Recency weights
        alpha_rec = self.build_recency_alpha(item_seq)  # [B, L, 1]

        # Session context for gate
        mask = item_seq.gt(0).unsqueeze(-1).float()  # [B, L, 1]
        masked_x = x * mask
        session_sum = masked_x.sum(dim=1)
        session_len = mask.sum(dim=1) + 1e-12
        session_context = session_sum / session_len  # [B, D]

        # Learn gate in [0,1]
        g = torch.sigmoid(self.gate(session_context)).unsqueeze(1)  # [B, 1, 1]

        # Fuse transformer and recency weights
        alpha = g * alpha_trm + (1.0 - g) * alpha_rec

        # Normalize again to be safe
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-12)

        seq_output = torch.sum(alpha * x, dim=1)
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output
