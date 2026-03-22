class MultiHeadAtencao(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        assert d % h == 0, "d deve ser divisível por h"
        self.h  = h
        self.dk = d // h
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)

        def split_heads(x):
            x = x.view(B, -1, self.h, self.dk)
            return x.transpose(1, 2)

        q = split_heads(self.wq(q))
        k = split_heads(self.wk(k))
        v = split_heads(self.wv(v))

        if mask is not None:
            mask = mask.unsqueeze(1)

        out = atencao(q, k, v, mask)
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, -1, self.h * self.dk)
        return self.wo(out)
