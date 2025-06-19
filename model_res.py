import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EvINRModel(nn.Module):
    def __init__(self, n_layers=3, d_hidden=512, d_neck=256, H=260, W=346, recon_colors=True,num_frames = 20):
        super().__init__()

        self.recon_colors = recon_colors
        self.d_output = H * W * 3 if recon_colors else H * W
        self.resfield_config = {
        'resfield_layers': [1,2,3],
        'composition_rank': 40,
        'capacity': num_frames, #num of frames
        }
        self.res_net = SirenMLP(
                 in_features = 1,
                 out_features = self.d_output,
                 num_layers = n_layers,
                 num_neurons = d_hidden,
                 resfield_config = self.resfield_config)
        self.H, self.W = H, W
        
    def forward(self, timestamps, frame_id):
      log_intensity_preds = self.res_net(timestamps, frame_id)
    

      if self.recon_colors:
          log_intensity_preds = log_intensity_preds.reshape(-1, self.H, self.W, 3)
      else:
          log_intensity_preds = log_intensity_preds.reshape(-1, self.H, self.W, 1)
    
      return log_intensity_preds
    
    def get_losses(self, log_intensity_preds, event_frames):
        # temporal supervision to solve the event generation equation
        event_frame_preds = log_intensity_preds[1:] - log_intensity_preds[0: -1]
        temperal_loss = F.mse_loss(event_frame_preds, event_frames[:-1])
        # spatial regularization to reduce noise
        x_grad = log_intensity_preds[:, 1: , :, :] - log_intensity_preds[:, 0: -1, :, :]
        y_grad = log_intensity_preds[:, :, 1: , :] - log_intensity_preds[:, :, 0: -1, :]
        spatial_loss = 0.05 * (
            x_grad.abs().mean() + y_grad.abs().mean() + event_frame_preds.abs().mean()
        )

        # loss term to keep the average intensity of each frame constant
        const_loss = 0.1 * torch.var(
            log_intensity_preds.reshape(log_intensity_preds.shape[0], -1).mean(dim=-1)
        )
        return (temperal_loss + spatial_loss + const_loss)

    def tonemapping(self, log_intensity_preds, gamma=0.6):
        intensity_preds = torch.exp(log_intensity_preds).detach()
        # Reinhard tone-mapping
        intensity_preds = (intensity_preds / (1 + intensity_preds)) ** (1 / gamma)
        intensity_preds = intensity_preds.clamp(0, 1)
        return intensity_preds


class ResFieldLinear(torch.nn.Linear):
    r"""Applies a ResField Linear transformation to the incoming data: :math:`y = x(W + W(t))^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        rank: value for the the low rank decomposition
        capacity: size of the temporal dimension

    Attributes:
        weight: (F_out x F_in)
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.

    Examples::

        >>> m = nn.Linear(20, 30, rank=10, capacity=100)
        >>> input_x, input_time = torch.randn(128, 20), torch.randn(128)
        >>> output = m(input, input_time)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, rank=None, capacity=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.rank = rank
        self.capacity = capacity

        if self.rank is not None and self.capacity is not None and self.capacity > 0:
            weights_t = 0.01*torch.randn((self.capacity, self.rank)) # C, R
            matrix_t = 0.01*torch.randn((self.rank, self.weight.shape[0]*self.weight.shape[1])) # R, F_out*F_in
            self.register_parameter('weights_t', torch.nn.Parameter(weights_t)) # C, R
            self.register_parameter('matrix_t', torch.nn.Parameter(matrix_t)) # R, F_out*F_in

    def forward(self, input: torch.Tensor, frame_id=None) -> torch.Tensor:
        """Applies the linear transformation to the incoming data: :math:`y = x(A + \delta A_t)^T + b

        Args:
            input: (B, S, F_in)
            frame_id: time index of the input tensor. Tensor of shape (B) or (1)
        Returns:
            output: (B, S, F_out)
        """
        if self.rank == 0 or self.capacity == 0:
            return torch.nn.functional.linear(input, self.weight, self.bias)

        # copute ResField weight matrix
        weight = torch.add((
            self.weights_t @ self.matrix_t).t(),
            self.weight.view(-1, 1)
        ).permute(1, 0).view(-1, *self.weight.shape) # F_out*F_in, C -> C, F_out, F_in


            # (B, F_out, F_in) x (B, F_in, S) -> (B, F_out, S)
        #print(f"input shape: {input.shape}")
        #print(f"input.unsqueeze(-1) shape: {input.unsqueeze(-1).shape}")
        #print(f"weight shape: {weight[frame_id].shape}")

        output = torch.bmm(weight[frame_id],input.unsqueeze(-1)).squeeze(-1) + self.bias
        #print(f"output shape: {output.shape}")
        return output # B, S, F_out

    def extra_repr(self) -> str:
        _str = super(ResFieldLinear, self).extra_repr()
        if self.capacity is not None and self.rank > 0 and self.capacity > 0:
          _str = _str + f', rank={self.rank}, capacity={self.capacity}'
        return _str

# @title Define Siren MLP
class Sine(torch.nn.Module):
    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class SirenMLP(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_layers: int,
                 num_neurons: int,
                 resfield_config: dict = dict()):
        """ A simple coordinate-mlp with Siren activation functions
        Args:
          in_features: the number of input features
          out_features: the number of output features
          num_neurons: the number of neurons for each layer
          num_layers: the number of MLP layers
          resfield_config (optional): configuration of resfield layers
            resfield_layers (list): which layers are ResField layers. Default [].
            composition_rank (int or list): the number of ranks. Default 10.
            capacity (int): the number of frames
        """
        super().__init__()
        # resfield parameters
        composition_rank = resfield_config.get('composition_rank', 512)
        resfield_layers = resfield_config.get('resfield_layers', [])
        capacity = resfield_config.get('capacity', 0)

        dims = [in_features] + [num_neurons for _ in range(num_layers)] + [out_features]
        self.nl = Sine()
        self.net = []
        for i in range(len(dims) - 1):
            _rank = composition_rank if i in resfield_layers else 0
            _rank = _rank[i] if not isinstance(_rank, int) else _rank
            # create a linear layer

            lin = ResFieldLinear(dims[i], dims[i + 1], rank=_rank, capacity=capacity,)
            # apply siren inicialization
            lin.apply(self.first_layer_sine_init if i == 0 else self.sine_init)
            self.net.append(lin)
        self.net = torch.nn.ModuleList(self.net)

    @staticmethod
    @torch.no_grad()
    def sine_init(m):
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

    @staticmethod
    @torch.no_grad()
    def first_layer_sine_init(m):
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

    def forward(self, coords, frame_id=None):
        x = coords
        for lin in self.net[:-1]:
            x = self.nl(lin(x, frame_id=frame_id))
        x = self.net[-1](x, frame_id=frame_id)
        return x
