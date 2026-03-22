class FFN(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.l1 = nn.Linear(d, d_ff)
        self.l2 = nn.Linear(d_ff, d)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


class AddNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x, sub):
        return self.norm(x + sub)


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d, 2).float() * (-math.log(10000.0) / d)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


def mascara_causal(tam, device='cpu'):
    return torch.tril(torch.ones(1, tam, tam, device=device))
