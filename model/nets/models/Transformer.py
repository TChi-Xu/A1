from .Attenion import *


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers_1 = nn.ModuleList([])
        for _ in range(depth-1):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.layers_1.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        for attn_1, ff_1 in self.layers_1:
            x1 = attn_1(x) + x
            x2 = ff_1(x1)+x1
        return x, x1, x2


if __name__ == '__main__':
    net = Transformer(128, [5, 7])
    print(net)