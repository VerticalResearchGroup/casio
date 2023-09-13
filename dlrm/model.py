
import torch

# mb_size=2048 #1024 #512 #256
# nbatches=1 #500 #100
# bot_mlp="13-512-256-128"
# top_mlp="479-1024-1024-512-256-1"
# emb_size=128
# nindices=100
# emb="1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1"
# interaction="dot"
# tnworkers=0
# tmb_size=16384

#_args="--mini-batch-size="${mb_size}\
# _args=" --num-batches="${nbatches}\
# " --data-generation="${data}\
# " --arch-mlp-bot="${bot_mlp}\
# " --arch-mlp-top="${top_mlp}\
# " --arch-sparse-feature-size="${emb_size}\
# " --arch-embedding-size="${emb}\
# " --num-indices-per-lookup="${nindices}\
# " --arch-interaction-op="${interaction}\
# " --numpy-rand-seed="${rand_seed}\
# " --print-freq="${print_freq}\
# " --print-time"\
# " --enable-profiling "


class DlrmMlp(torch.nn.Module):
    def __init__(self, widths : list[int], sigmoid_i=None):
        super().__init__()
        modules = []
        for i in range(len(widths) - 1):
            modules.append(torch.nn.Linear(widths[i], widths[i + 1]))

            if i == sigmoid_i: modules.append(torch.nn.Sigmoid())
            else: modules.append(torch.nn.ReLU())

        self.model = torch.nn.Sequential(*modules)

    def forward(self, x): return self.model(x)

class Dlrm(torch.nn.Module):
    def __init__(
        self,
        mlp_bot_n : list[int] = [13, 512, 256, 128],
        mlp_top_n : list[int] = [479, 1024, 1024, 512, 256, 1],
        emb_size : list[int] = [1]*26,
        emb_dim : int = 128,
        interaction : str = 'dot',
        interact_self : bool = False
    ):
        super().__init__()

        self.bot_mlp = DlrmMlp(mlp_bot_n)
        self.top_mlp = DlrmMlp(mlp_top_n)

        assert interaction == 'dot', 'Only dot product interaction is supported'

        self.embs = torch.nn.ModuleList([
            torch.nn.Embedding(n, emb_dim) for n in emb_size
        ])

        ni = len(emb_size) + 1
        nj = len(emb_size) + 1

        off = 1 if interact_self else 0
        self.li = torch.tensor([i for i in range(ni) for j in range(i + off)])
        self.lj = torch.tensor([j for i in range(nj) for j in range(i + off)])

    def forward(
        self,
        dense_x : torch.Tensor, # [B, D]
        sparse_x : torch.Tensor # [B, num_sparse_features]
    ):
        with torch.profiler.record_function('bot_mlp'):
            bot_mlp_out = self.bot_mlp(dense_x)
        B, D = bot_mlp_out.shape

        features = torch.cat([bot_mlp_out] + [
            emb(sparse_x[:, i]) for i, emb in enumerate(self.embs)
        ], dim=1).view(B, -1, D)

        interact_out = \
            torch.bmm(features, features.transpose(1, 2))[:, self.li, self.lj]


        x = torch.cat([bot_mlp_out, interact_out], dim=1)

        with torch.profiler.record_function('top_mlp'):
            return self.top_mlp(x)


if __name__ == '__main__':
    B = 2048
    net = Dlrm()

    dense_x = torch.randn(B, 13)
    sparse_x = torch.randint(0, 1, (B, 26))

    out = net(dense_x, sparse_x)

