from torch.nn import Module, Linear, ReLU, Sequential

from mamba_ssm import Mamba

class MambaModel(Module):
    
    """
        MambaModel uses Mamba blocks to do MT.
        
        Parameters:
        - layers (int): number of Mamba blocks
        - d_model (int): dimension of the input and output
        - d_state (int): dimension of the state
        - d_conv (int): dimension of the convolutional layer
        - expand (int): expansion factor
        - max_length (int): maximum length of the sequence
    """
    
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
            *[Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(layers)]
        )
        
        # TODO: necessary?
        self.obs_model = Sequential(
            Linear(d_model, d_model),
            ReLU(),
            Linear(d_model, d_model)
        )
        
        # TODO: necessary?
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         self.init.xavier_uniform_(p)
    
    # TODO: decide on how to do seq2seq
    def forward(self, x, u):
        
        """
            Forward pass of the model.
            
            Parameters:
            - x (torch.Tensor): input sequence
            - u (torch.Tensor): input control signal
            
            Returns:
            - y (torch.Tensor): output sequence
        """
        
        ssm_out = self.ssm(x, u)
        y = self.obs_model(ssm_out)
        return y
    
    # TODO: decide on how to do seq2seq
    def generate(self, x, u):
        
        """
            Generate the output sequence and the state sequence.
            
            Parameters:
            - x (torch.Tensor): input sequence
            - u (torch.Tensor): input control signal
            
            Returns:
            - y (torch.Tensor): output sequence
            - x (torch.Tensor): state sequence
        """
        
        ssm_out = self.ssm(x, u)
        y = self.obs_model(ssm_out)
        return y, x