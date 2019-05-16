import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple(nn.Module):
    def __init__(self, input_size, target_size, hidden_size=256, n_layer=16):
        super(Simple, self).__init__()

        self.l0=nn.Linear(input_size,hidden_size)
        self.simples=[SimpleBlock(hidden_size,hidden_size)]*n_layer

        self.seq=nn.Sequential(*self.simples)

        self.ln=nn.Linear(hidden_size,target_size)

    def forward(self, input):
        input=input.max(dim=1)[0]
        i=self.l0(input)
        i=self.seq(i)
        o=self.ln(i)
        return o


class SimpleBlock(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        # output = x.transpose(1, 2)
        # this generates nan
        output = self.w_2(F.relu(self.w_1(x)))
        assert (output == output).all()

        # output = output.transpose(1, 2)
        output = self.dropout(output)
        assert (output == output).all()
        return output

if __name__ == '__main__':
    from death.post.inputgen_planJ import InputGenJ
    ig = InputGenJ(elim_rare_code=True,no_underlying=True, death_only=True, debug=True)
    i,t,_,_=ig[4069]
    simple=Simple(7298,435)
    i=torch.from_numpy(i)
    t=torch.from_numpy(t)
    i=i.unsqueeze(0)
    t=t.unsqueeze(0)
    o=simple(i)
    print(o)