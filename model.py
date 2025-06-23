import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
def skip(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
            
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))


        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    return model


class EvINRModel(nn.Module):
    def __init__(self,H=180, W=240):
        super().__init__()
        self.net = skip(
        num_input_channels=2, num_output_channels=1, 
        num_channels_down=[128, 128, 128, 128, 128,128], num_channels_up=[128,128, 128, 128, 128, 128], num_channels_skip=[4,4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True)
        self.net0 = Siren(
            n_layers=3, d_input=1, d_hidden=1024, d_neck=1024, 
            d_output=H*W
        )
        self.H, self.W = H, W

    def t_map(self,H, W, timestamp,num_frame):
        freq = 2**(torch.arange(H*W, dtype=torch.float32, device=timestamp.device)/(H*W/2))*math.pi*timestamp
        sin_comp = torch.sin(freq)
        #cos_comp = torch.cos(freq)
        pe = sin_comp
        #pe = torch.cat([sin_comp, cos_comp], dim=0)
        pe = pe.reshape(num_frame,1, H, W)
        return pe

    def forward(self, timestamps):
        num_frame = timestamps.shape[0]
        timestamps_map0 = self.t_map(180,240,timestamps,num_frame)
        timestamps_map = self.net0(timestamps)
        timestamps_map = timestamps_map.reshape(-1,1,self.H, self.W)
        timestamps_map = torch.cat([timestamps_map0, timestamps_map], dim=1)
        log_intensity_preds = self.net(timestamps_map)
        return log_intensity_preds
    
    def get_losses(self, log_intensity_preds, event_frames):
        # temporal supervision to solve the event generation equation
        #print(log_intensity_preds.shape)
        event_frames = event_frames.squeeze(-1).unsqueeze(1)
        #print(event_frames.shape)
        
        event_frame_preds = log_intensity_preds[1:,:,:,:] - log_intensity_preds[0:-1,: ,:,:]
        temperal_loss = F.mse_loss(event_frame_preds, event_frames[:-1,:,:,:])
        # spatial regularization to reduce noise
        x_grad = log_intensity_preds[:, : , 1:, :] - log_intensity_preds[:, : , 0:-1, :]
        y_grad = log_intensity_preds[:, :, : , 1:] - log_intensity_preds[:, :, :, 0:-1]
        spatial_loss = 0.05 * (
            x_grad.abs().mean() + y_grad.abs().mean() + event_frame_preds.abs().mean()
        )

        # loss term to keep the average intensity of each frame constant
        const_loss = 0.06 * torch.var(
            log_intensity_preds.reshape(log_intensity_preds.shape[0], -1).mean(dim=-1)
        )
        return temperal_loss + spatial_loss + const_loss
        
    def tonemapping(self, log_intensity_preds, gamma=0.6):
        intensity_preds = torch.exp(log_intensity_preds).detach()
        # Reinhard tone-mapping
        intensity_preds = (intensity_preds / (1 + intensity_preds)) ** (1 / gamma)
        intensity_preds = intensity_preds.clamp(0, 1)
        return intensity_preds

# Roughly copy from https://github.com/vsitzmann/siren
class Siren(nn.Module):
    def __init__(
        self, n_layers, d_input, d_hidden, d_neck, d_output
    ):
        super().__init__()
        self.siren_net = []
        self.siren_net.append(SineLayer(d_input, d_hidden, is_first=True)) 
        for i_layer in range(n_layers):
            self.siren_net.append(SineLayer(d_hidden, d_hidden))
            if i_layer == n_layers - 1:
                self.siren_net.append(SineLayer(d_hidden, d_neck))
        self.siren_net.append(SineLayer(d_neck, d_output, is_last=True))
        self.siren_net = nn.Sequential(*self.siren_net)
        
    def forward(self, x):
        out = self.siren_net(x) # [B, H*W]
        return out
    
class SineLayer(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, is_first=False, is_last=False, omega_0=10
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.scale_0 = 15
        self.scale_1 = 1
        self.is_first = is_first
        self.is_last = is_last
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.orth = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    
    @torch.no_grad()
    def init_weights(self):
        if self.is_first:
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
        else:
            self.linear.weight.uniform_(
                -np.sqrt(6 / self.in_features) / self.omega_0,
                np.sqrt(6 / self.in_features) / self.omega_0,
            )
                
    def forward(self, input):
        lin = self.linear(input)
        orth = self.orth(input)
        scale = self.scale_0 * lin
        omega = self.omega_0 * lin
        scale_orth = self.scale_1 * orth
        if self.is_last:
            return self.omega_0 * self.linear(input)
        else:
            return torch.sin(omega)#*torch.exp(-scale.abs().square()-scale_orth.abs().square())


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2 
                diff3 = (inp.size(3) - target_shape3) // 2 
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        # print (input.data.type())

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)
