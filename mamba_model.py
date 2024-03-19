from torch.nn import Module, Linear, ReLU, Sequential

from mamba_ssm import Mamba

class MambaModel(Module):
    def __init__(
        self,
        layers: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        max_length: int
    ):
        super(MambaModel, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.max_length = max_length
        
        self.ssm = Sequential(
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(layers)
        )
        
        # TODO: necessary?
        self.obs_model = Sequential(
            Linear(d_model, d_model),
            ReLU(),
            Linear(d_model, d_model)
        )
    
    # TODO: decide on how to do seq2seq
    def forward(self, x, u):
        ssm_out = self.ssm(x, u)
        y = self.obs_model(ssm_out)
        return y
    
    # TODO: decide on how to do seq2seq
    def generate(self, x, u):
        ssm_out = self.ssm(x, u)
        y = self.obs_model(ssm_out)
        return y, x