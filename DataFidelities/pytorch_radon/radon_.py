import torch
from torch import nn
import torch.nn.functional as F
import time
from .utils import PI, SQRT2, deg2rad, affine_grid, grid_sample
from .filters import RampFilter

class Radon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True, dtype=torch.float, device=torch.device('cuda')):
        super(Radon, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.dtype = dtype
        self.all_grids = None
        self.device = device
        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size, circle)

    def forward(self, x):
        N, C, W, H = x.shape
        assert (W == H)

        if self.all_grids is None:
            self.all_grids = self._create_grids(self.theta, W, self.circle)

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape
        out = torch.zeros(N, C, W, len(self.theta), device=x.device, dtype=self.dtype)

        for i in range(len(self.theta)):
            rotated = grid_sample(x, self.all_grids[i].repeat(N, 1, 1, 1).to(x.device))
            out[..., i] = rotated.sum(2)

        return out

    # 以下是我修改的代码，希望能对于theta自动求导
    # https://discuss.pytorch.org/t/backward-got-grad-of-none/91729
    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())
        all_grids = []
        for theta in angles:
            theta = deg2rad(theta)
            # 只改动了下面这五行 ##############################
            R = torch.stack([
                torch.stack([theta.cos(), theta.sin(), torch.tensor(0, dtype=torch.float, device=theta.device)]),
                torch.stack([-theta.sin(), theta.cos(), torch.tensor(0, dtype=torch.float, device=theta.device)])
            ])
            R = torch.stack([R])
            ################################################
            all_grids.append(affine_grid(R, torch.Size([1, 1, grid_size, grid_size])))
        return all_grids

    # # 以下是原来的代码
    # def _create_grids(self, angles, grid_size, circle):
    #     if not circle:
    #         grid_size = int((SQRT2*grid_size).ceil())
    #     all_grids = []
    #     for theta in angles:
    #         theta = deg2rad(theta)
    #         R = torch.tensor([[
    #                 [ theta.cos(), theta.sin(), 0],
    #                 [-theta.sin(), theta.cos(), 0],
    #             ]], dtype=self.dtype)
    #         all_grids.append(affine_grid(R, torch.Size([1, 1, grid_size, grid_size])))
    #     return all_grids
    #


class IRadon(nn.Module):
    # @torchsnooper.snoop()
    def __init__(self, in_size=None, theta=None, circle=True,
                 use_filter=RampFilter(), out_size=None, dtype=torch.float, device=torch.device('cuda')):
        super(IRadon, self).__init__()
        self.circle = circle
        self.theta = theta if theta is not None else torch.arange(180)
        self.out_size = out_size
        self.in_size = in_size
        self.dtype = dtype
        self.ygrid, self.xgrid, self.all_grids = None, None, None
        self.device = device

        if in_size is not None:
            self.ygrid, self.xgrid = self._create_yxgrid(in_size, circle)
            # start_time = time.time()
            self.all_grids = self._create_grids(self.theta, in_size, circle)
            # print("torch create grids execute--- %s seconds ---" % (time.time() - start_time))

        self.filter = use_filter if use_filter is not None else lambda x: x

    # @torchsnooper.snoop()
    def forward(self, x):
        it_size = x.shape[2]
        ch_size = x.shape[1]

        if self.in_size is None:
            self.in_size = int((it_size / SQRT2).floor()) if not self.circle else it_size
   
        if None in [self.ygrid, self.xgrid, self.all_grids]:
            self.ygrid, self.xgrid = self._create_yxgrid(self.in_size, self.circle)
            self.all_grids = self._create_grids(self.theta, self.in_size, self.circle)

        x = self.filter(x)
        
        reco = torch.zeros(x.shape[0], ch_size, it_size, it_size, device=x.device, dtype=self.dtype)

        for i_theta in range(len(self.theta)):
            reco += grid_sample(x, self.all_grids[i_theta].repeat(reco.shape[0], 1, 1, 1).to(x.device))

        if not self.circle:
            W = self.in_size
            diagonal = it_size
            pad = int(torch.tensor(diagonal - W, dtype=torch.float).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            reco = F.pad(reco, (-pad_width[0], -pad_width[1], -pad_width[0], -pad_width[1])).to(self.device)

        if self.circle:
            reconstruction_circle = (self.xgrid ** 2 + self.ygrid ** 2) <= 1
            reconstruction_circle = reconstruction_circle.repeat(x.shape[0], ch_size, 1, 1)
            reco[~reconstruction_circle] = 0.

        # The following line is well tested
        reco = reco * PI.item() / (2 * len(self.theta))

        if self.out_size is not None:
            pad = (self.out_size - self.in_size) // 2
            reco = F.pad(reco, (pad, pad, pad, pad))

        return reco

    def _create_yxgrid(self, in_size, circle):
        if not circle:
            in_size = int((SQRT2 * in_size).ceil())
        unitrange = torch.linspace(-1, 1, in_size, dtype=self.dtype).to(self.device)
        return torch.meshgrid(unitrange, unitrange)

    def _XYtoT(self, theta):
        T = self.xgrid * (deg2rad(theta)).cos() - self.ygrid * (deg2rad(theta)).sin().to(self.device)
        return T

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())
        all_grids = []
        # start_time = time.time()
        for i_theta in range(len(angles)):
            X = (torch.ones(grid_size, dtype=self.dtype).view(-1, 1).repeat(1, grid_size) * i_theta * 2. / (
                        len(angles) - 1) - 1.).to(self.device)
            Y = self._XYtoT(angles[i_theta])
            all_grids.append(torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1).unsqueeze(0).to(self.device))
        # print("torch create grids loop--- %s seconds ---" % (time.time() - start_time))

        return all_grids