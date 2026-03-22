class BlocoEncoder(nn.Module):
    def __init__(self, d, d_ff, h):
        super().__init__()
        self.attn = MultiHeadAtencao(d, h)
        self.an1  = AddNorm(d)
        self.ffn  = FFN(d, d_ff)
        self.an2  = AddNorm(d)

    def forward(self, x):
        x = self.an1(x, self.attn(x, x, x))
        x = self.an2(x, self.ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, n, d, d_ff, h):
        super().__init__()
        self.blocos = nn.ModuleList([BlocoEncoder(d, d_ff, h) for _ in range(n)])

    def forward(self, x):
        for b in self.blocos:
            x = b(x)
        return x
