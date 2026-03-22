class BlocoDecoder(nn.Module):
    def __init__(self, d, d_ff, h):
        super().__init__()
        self.attn_mascarada = MultiHeadAtencao(d, h)
        self.an1            = AddNorm(d)
        self.cross_attn     = MultiHeadAtencao(d, h)
        self.an2            = AddNorm(d)
        self.ffn            = FFN(d, d_ff)
        self.an3            = AddNorm(d)

    def forward(self, y, z):
        mask = mascara_causal(y.size(1), device=y.device)
        y = self.an1(y, self.attn_mascarada(y, y, y, mask))
        y = self.an2(y, self.cross_attn(y, z, z))
        y = self.an3(y, self.ffn(y))
        return y


class Decoder(nn.Module):
    def __init__(self, vocab, n, d, d_ff, h):
        super().__init__()
        self.blocos = nn.ModuleList([BlocoDecoder(d, d_ff, h) for _ in range(n)])
        self.proj   = nn.Linear(d, vocab)

    def forward(self, y, z):
        for b in self.blocos:
            y = b(y, z)
        return self.proj(y)
