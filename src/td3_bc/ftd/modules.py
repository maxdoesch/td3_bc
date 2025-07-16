import torch
import torch.nn as nn


def _get_out_shape(in_shape, layers, device='cpu'):
    x = torch.randn(*in_shape).to(device).unsqueeze(0)
    return layers(x).squeeze(0).shape


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class NormalizeImg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / 255.


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class RLProjection(nn.Module):
    def __init__(self, in_shape, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_shape[0], out_dim),
            nn.LayerNorm(out_dim),
            nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.projection(x)


class HeadCNN(nn.Module):
    def __init__(self, in_shape, num_layers=0, num_filters=32):
        super().__init__()
        self.layers = []
        for _ in range(0, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(
                nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers.append(Flatten())
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(in_shape, self.layers)
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x)


class SelectorCNN(nn.Module):
    def __init__(self, selector_layers, obs_shape, region_num=5, in_channels=3, stack_num=3, num_shared_layers=11,
                 num_filters=32):
        super().__init__()
        assert len(obs_shape) == 3
        # assert region_num * in_channels * stack_num == obs_shape[0]
        self.obs_shape = obs_shape
        self.in_channels = in_channels
        self.stack_num = stack_num
        self.num_filters = num_filters

        self.selector_layers = selector_layers

        self.shared_layers = [
            nn.Conv2d(self.stack_num * self.in_channels, num_filters, 3, stride=2)]
        for _ in range(1, num_shared_layers):
            self.shared_layers.append(nn.ReLU())
            self.shared_layers.append(
                nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.shared_layers = nn.Sequential(*self.shared_layers)

        self.out_shape = _get_out_shape([self.stack_num * self.in_channels, self.obs_shape[-2], self.obs_shape[-1]],
                                        self.shared_layers)
        self.shared_layers.apply(weight_init)

    def forward(self, x):
        x = self.selector_layers(x)
        x = self.shared_layers(x)

        return x


class Encoder(nn.Module):
    def __init__(self, shared_cnn, head_cnn, projection):
        super().__init__()
        self.shared_cnn = shared_cnn
        self.head_cnn = head_cnn
        self.projection = projection
        self.out_dim = projection.out_dim

    def forward(self, x, detach=False):
        x = self.shared_cnn(x)
        x = self.head_cnn(x)
        if detach:
            x = x.detach()
        return self.projection(x)
