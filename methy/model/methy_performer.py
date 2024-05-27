import numpy as np
import torch
from torch import nn

if __name__ == '__main__':
    from performer import Performer, Always
else:
    from methy.model.performer import Performer, Always


class Gene2VecEmbedding(nn.Module):
    def __init__(self, weight_path, dim):
        super().__init__()
        gene2vec_weight = np.load(weight_path)
        gene2vec_weight = np.concatenate([
            np.random.randn(3, gene2vec_weight.shape[1]), gene2vec_weight,  # [PAD, CHR, UNK]
        ], axis=0)
        gene2vec_weight = gene2vec_weight.astype('float32')
        gene2vec_weight = torch.from_numpy(gene2vec_weight)
        self.emb = nn.Embedding.from_pretrained(gene2vec_weight, freeze=False)

        self.to_out = nn.Linear(gene2vec_weight.shape[1], dim)

    def forward(self, x):
        x = (x + self.emb.num_embeddings) % self.emb.num_embeddings
        feat = self.to_out(self.emb(x))
        trainable = (x < 3)
        feat = feat * (trainable[..., None]) + feat.detach() * (~trainable[..., None])
        return feat


class ContinuousEmbedding(nn.Module):
    def __init__(self, num_classes, dim, vmin=0, vmax=1, discretized=True):
        '''
        When applied to a module, .requires_grad_() takes effect on all of the module’s parameters (which have requires_grad=True by default).
        '''
        super().__init__()
        boundaries = torch.cat(
            [torch.tensor([-2., -1.]), torch.linspace(vmin, vmax, steps=num_classes + 1)])  # [PAD] [NA]
        self.boundaries = nn.Parameter(boundaries, requires_grad=False)
        self.step = (vmax - vmin) / num_classes
        self.num_classes = num_classes
        self.embed_dim = dim
        self.discretized = discretized
        self.embed = nn.Embedding(num_classes + 3, self.embed_dim)  # [PAD] [NA]

    def forward(self, x):
        shape = x.shape
        boundaries = self.boundaries.detach()

        if (not hasattr('self','discretized')) or self.discretized:
            i = torch.bucketize(x, boundaries[:-1])
            feat = self.embed(i)
        else:
            x = x.flatten()
            i = torch.bucketize(x, boundaries[:-1])
            j = (i - 1).clip(0)
            ceil = boundaries[i]
            floor = boundaries[j]

            w1 = ((ceil - x) / self.step).clip(0, 1)
            w2 = ((x - floor) / self.step + (i == 0)).clip(0, 1)
            feat = w1[:, None] * self.embed(j) + w2[:, None] * self.embed(i)
            feat = feat.reshape(list(shape) + [self.embed_dim])
        return feat


class PerformerLM(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, dim_head=32, max_seq_len=None,
                 methy_discretized=True,
                 gene_weight_path=None, performer_kargs={}, ):
        super().__init__()
        self.embedding_dim = dim
        self.max_seq_len = max_seq_len
        self.methy_discretized = methy_discretized
        self.num_classes = num_classes + 2
        self.methy_emb = ContinuousEmbedding(num_classes, dim, discretized=methy_discretized)
        self.chromo_emb = nn.Embedding(30, dim)  # 0 is [PAD]
        self.position_emb = nn.Embedding(50000, dim)  # 0 is [chr]
        self.gene_weight_path = gene_weight_path
        if self.gene_weight_path is not None:
            self.gene_emb = Gene2VecEmbedding(gene_weight_path, dim)

        self.performer = Performer(dim, depth, heads, dim_head, **performer_kargs)
        self.identity = Always(None)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_classes + 2)

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, inputs, return_encodings=False, output_attentions=False, **kwargs):
        methy = inputs['methy']
        chromo = inputs['chromo']
        pos = inputs['pos']
        mask = methy != 0  # mask is 1, indicating valid values. 0 is padding data

        x = self.methy_emb(methy) + self.position_emb(pos) + self.chromo_emb(chromo)
        if ('gene' in inputs) and (self.gene_emb is not None):
            x = x + self.gene_emb(inputs['gene'])
        if ('embed' in inputs): # inject external embedding
            x = x + inputs['embed']

        # performer layers
        pos_emb = self.identity(x)

        x = self.performer(x, pos_emb=pos_emb, mask=mask, **kwargs)

        # norm and to logits
        x = self.norm(x)
        if return_encodings:
            return x

        x = self.to_out(x)
        return x


class PerformerDiag(nn.Module):
    def __init__(self, num_classes, pretrained_model,
                 backbone_grad=False, backbone_random_init=False,
                 use_freq=True,
                 dropout=0.0):
        super().__init__()
        if isinstance(pretrained_model, str):
            self.backbone = torch.load(pretrained_model, map_location='cpu')
        else:
            self.backbone = pretrained_model
        self.backbone_random_init = backbone_random_init  # backbone random initialization
        if self.backbone_random_init:
            for param in self.backbone.parameters():
                torch.nn.init.normal_(param)
            backbone_grad = True # If random initialization, the backbone needs to be learnable

        self.backbone_grad = backbone_grad  # default is false, indicating fixed backbone
        self.backbone.requires_grad_(self.backbone_grad)

        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=dropout)
        self.use_freq = use_freq
        if hasattr(self.backbone, 'embedding_dim'):
            self.embedding_dim = self.backbone.embedding_dim
        else:
            self.embedding_dim = self.backbone.to_out.in_features
        if self.use_freq:
            self.freq_embed = nn.Embedding(30,self.embedding_dim)
        self.to_out = nn.Sequential(
            nn.Linear(self.embedding_dim * 23, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.num_classes),
        )

    def forward(self, inputs):
        b, n, l = inputs['methy'].shape
        # reshape
        if ('freq' in inputs) and hasattr(self,'freq_embed'):
            log_freq = torch.ceil(torch.log(inputs['freq'].clip(0)+1).clip(0,10)).long()
            inputs['embed'] = self.freq_embed(log_freq)
        for key in inputs.keys():
            inputs[key] = inputs[key].flatten(0, 1)
        x = self.backbone(inputs, return_encodings=True)  # (b*n,l,d)
        x = x[:, 0]  # (b*n,d) # take the first token
        x = x.reshape(b, n, -1)

        x = x.reshape(b, -1)
        x = self.dropout(x)
        out = self.to_out(x)
        return out
