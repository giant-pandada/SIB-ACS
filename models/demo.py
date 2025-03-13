import torch
import numpy as np
from torch.nn import init
import torch.nn.functional as F
from torch import nn


class SIBACS(nn.Module):
    def __init__(self, config):
        super(SIBACS, self).__init__()
        self.config = config
        self.phi_size = 32
        points = self.phi_size ** 2
        self.point = int(np.ceil(1 * points))

        self.num_layers = 12
        self.channels = 32
        self.phase_sam = 4
        self.register_buffer(
            "nB_base_matrix", torch.arange(0, points)[None, None, None, :]
        )
        self.DUN_mask = nn.ModuleList()
        self.phi = nn.ParameterList()
        self.conv_start_rec = nn.ModuleList()
        self.conv_end = nn.ModuleList()
        self.recon = DUN(channels=self.channels)
        for i in range(self.phase_sam * 2 + 1):
            self.DUN_mask.append(DUN_mask(channels=self.channels))
            self.phi.append(nn.Parameter(init.xavier_normal_(torch.Tensor(self.point, 1, self.phi_size, self.phi_size))))
            self.conv_start_rec.append(
                nn.Conv2d(1, self.channels, kernel_size=1, stride=1, bias=False),
            )
            self.conv_end.append(
                nn.Conv2d(self.channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
            )

    def forward(self, inputs, sr, phase):
        B, C, H, W = inputs.shape
        recon_init_group = torch.ones(self.phase_sam * 2 + 1,B,C,H,W).to(inputs.device)
        h, w = H // self.phi_size, W // self.phi_size
        N_total = int(sr * H * W)                      # Total Sampling Resources
        sr_init = 0.02
        N_init = int(sr_init * self.phi_size ** 2)     # Initial Sampling Count
        N_base1 = int(((sr - sr_init)/self.phase_sam/2) * self.phi_size ** 2)   # IS Count
        N_max = int(((1 - sr_init)/self.phase_sam) * self.phi_size ** 2)        # Maximum Sampling Count per AS Stage
        N_base = int(((sr - sr_init)/self.phase_sam/2) * self.phi_size ** 2)    # AS Count

        # Initial sampling

        N_1 = N_init
        phi = self.phi[0][:N_init, :, :, :]
        y = F.conv2d(inputs, phi, stride=self.phi_size, padding=0, bias=None)
        y_rest = torch.zeros(B, self.phi_size ** 2 - N_init, h, w).to(inputs.device)
        mask = torch.concat((torch.ones_like(y), y_rest), dim=1)

        # Lightweight reconstruction

        recon_init = F.conv_transpose2d(y, phi, stride=self.phi_size)
        rec1 = self.conv_start_rec[0](recon_init)
        phiTA = recon_init
        rec1 = self.DUN_mask[0](rec1, phiTA, self.phi[0], self.phi_size, mask.permute(0, 2, 3, 1))
        rec = self.conv_end[0](rec1)
        recon_init_group[0] = rec

        if phase == 1:
            return rec

        # Innovation sampling at the 1-st AS stage

        N_1 = N_1 + N_base1 * torch.ones(B, h, w).to(inputs.device)
        x = self.nB_base_matrix * N_1[..., None]
        mask = torch.where(x >= N_1[..., None] ** 2, 0, 1)
        mask = mask.to(torch.float32)
        mask = mask[:, :, :, : self.phi[0].shape[0]]

        phi = torch.concat([self.phi[0][:N_init, :, :, :], self.phi[1][N_init:, :, :, :]],dim=0)
        y = F.conv2d(inputs, phi, stride=self.phi_size, padding=0, bias=None)
        y = y.permute(0, 2, 3, 1) * mask
        y = y.permute(0, 3, 1, 2)

        # Lightweight reconstruction

        recon_init = F.conv_transpose2d(y, phi, stride=self.phi_size)
        rec1 = self.conv_start_rec[1](recon_init)
        phiTA = recon_init
        rec1 = self.DUN_mask[1](rec1, phiTA, phi, self.phi_size, mask)
        rec = self.conv_end[1](rec1)
        recon_init_group[1] = rec

        # Innovation Estimation at the 1-st AS stage

        error = torch.pow(rec - recon_init_group[0], 2)
        error = error.view(B, h, self.phi_size, w, self.phi_size)
        saliency = error.sum(dim=(2, 4))
        saliency = saliency / torch.sum(saliency, dim=[1, 2], keepdim=True)

        # Adaptive sampling at the 1-st AS stage

        N_1 = N_1 + torch.round(saliency * N_base * h * w)
        while torch.max(N_1) > int(((1 - sr_init) / self.phase_sam + sr_init) * self.phi_size ** 2):
            N_1 = self.adjust_asa(N_1, int(((1 - sr_init) / self.phase_sam + sr_init) * self.phi_size ** 2),
                                 N_init * h * w + N_base * h * w + N_base1 * h * w)
        if phase == 2:
            return rec

        for i in range(self.phase_sam - 1):
            x = self.nB_base_matrix * N_1[..., None]
            mask = torch.where(x >= N_1[..., None] ** 2, 0, 1)
            mask = mask.to(torch.float32)
            mask = mask[:, :, :, : self.phi[0].shape[0]]
            phi = torch.concat([phi[:N_init + (i) * N_max + N_base1, :, :, :],self.phi[2*i + 2][N_init + (i) * N_max + N_base1:, :, :, :]],dim=0)
            y = F.conv2d(inputs, phi, stride=self.phi_size, padding=0, bias=None)
            y = y.permute(0, 2, 3, 1) * mask
            y = y.permute(0, 3, 1, 2)

            # Lightweight reconstruction

            recon_init = F.conv_transpose2d(y, phi, stride=self.phi_size)
            rec1 = self.conv_start_rec[2*i + 2](recon_init)
            phiTA = recon_init
            rec1 = self.DUN_mask[2*i + 2](rec1, phiTA, phi, self.phi_size, mask)
            rec = self.conv_end[2*i + 2](rec1)
            recon_init_group[2*i + 2] = rec

            if phase == 2*i+3:
                return rec

            # Innovation sampling at the s-th AS stage

            N_1 = N_1 + N_base1 * torch.ones(B, h, w).to(inputs.device)
            while torch.max(N_1) > self.phi_size ** 2:
                N_1 = self.adjust_asa(N_1, self.phi_size ** 2,
                                     N_init * h * w + N_base * h * w * (i + 1) + N_base1 * h * w * (i + 2))
            x = self.nB_base_matrix * N_1[..., None]
            mask = torch.where(x >= N_1[..., None] ** 2, 0, 1)
            mask = mask.to(torch.float32)
            mask = mask[:, :, :, : self.phi[0].shape[0]]

            phi = torch.concat([phi[:N_init + (i + 1) * N_max, :, :, :],
                                self.phi[2 * i + 3][N_init + (i + 1) * N_max:, :, :, :]], dim=0)
            y = F.conv2d(inputs, phi, stride=self.phi_size, padding=0, bias=None)
            y = y.permute(0, 2, 3, 1) * mask
            y = y.permute(0, 3, 1, 2)

            # Lightweight reconstruction

            recon_init = F.conv_transpose2d(y, phi, stride=self.phi_size)
            rec1 = self.conv_start_rec[2 * i + 3](recon_init)
            phiTA = recon_init
            rec1 = self.DUN_mask[2 * i + 3](rec1, phiTA, phi, self.phi_size, mask)
            rec = self.conv_end[2 * i + 3](rec1)
            recon_init_group[2 * i + 3] = rec

            # Innovation Estimation at the s-th AS stage

            error = torch.pow(rec - recon_init_group[2 * i + 2], 2)
            error = error.view(B, h, self.phi_size, w, self.phi_size)
            saliency = error.sum(dim=(2, 4))
            saliency = saliency / torch.sum(saliency, dim=[1, 2], keepdim=True)

            # Adaptive sampling at the s-th AS stage

            if i == self.phase_sam - 2:
                N_1 = N_1 + torch.round(saliency * (N_total - torch.sum(N_1)))
                while torch.max(N_1) > self.phi_size ** 2:
                    N_1 = self.adjust_asa(N_1, self.phi_size ** 2, N_total)
            else:
                N_1 = N_1 + torch.round(saliency * N_base * h * w)
                while torch.max(N_1) > int(((1 - sr_init) * (i + 2) / self.phase_sam + sr_init) * self.phi_size ** 2):
                    N_1 = self.adjust_asa(N_1, int(((1 - sr_init) * (i + 2) / self.phase_sam + sr_init) * self.phi_size ** 2),
                                         N_init * h * w + N_base * (i + 2) * h * w + N_base1 * (i + 2) * h * w)
            if phase == 2 * i + 4:
                return rec

        x = self.nB_base_matrix * N_1[..., None]
        mask = torch.where(x >= N_1[..., None] ** 2, 0, 1)
        mask = mask.to(torch.float32)
        mask = mask[:, :, :, : self.phi[0].shape[0]]
        phi = torch.concat([phi[:N_init + (self.phase_sam - 1) * N_max + N_base1, :, :, :],
                            self.phi[2 * self.phase_sam][N_init + (self.phase_sam - 1) * N_max + N_base1:, :, :, :]], dim=0)
        y = F.conv2d(inputs, phi, stride=self.phi_size, padding=0, bias=None)
        y = y.permute(0, 2, 3, 1) * mask
        y = y.permute(0, 3, 1, 2)

        # Deep Reconstruction

        recon_init = F.conv_transpose2d(y, phi, stride=self.phi_size)
        rec1 = self.conv_start_rec[2 * self.phase_sam](recon_init)
        phiTA = recon_init
        rec1 = self.recon(rec1, phiTA, phi, self.phi_size, mask)
        rec = self.conv_end[2 * self.phase_sam](rec1)

        return rec

    def adjust_asa(self, asa, max_N, N_total):
        asa = torch.tensor(asa, dtype=torch.float32)
        max_N = torch.tensor(max_N, dtype=asa.dtype).to(asa.device)
        a = torch.where(asa > max_N, max_N, asa)
        old = torch.sum(torch.where(asa < max_N, asa, torch.tensor(0, dtype=torch.float32).to(asa.device)), dim=[1, 2], keepdim=True)
        new = (
                N_total - torch.sum((a == max_N).type(torch.float32), dim=[1, 2], keepdim=True) * max_N
        )
        asa = torch.where(a < max_N, torch.round(a * new / old), a)

        return asa

# PC-Net (Lightweight reconstruction network)
class DUN_mask(torch.nn.Module):
    def __init__(self, channels, num_layers=8):
        super(DUN_mask, self).__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.res1 = nn.ModuleList()
        self.conv321 = nn.ModuleList()
        self.conv132 = nn.ModuleList()
        for i in range(self.num_layers+1):
            self.conv132.append(nn.Sequential(
                nn.Conv2d(1, self.channels, kernel_size=3, stride=1, padding=1, bias=False),
            ))
            self.conv321.append(nn.Sequential(
                nn.Conv2d(self.channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
            ))
            self.res1.append(nn.Sequential(
                RB(dim=self.channels),
                RB(dim=self.channels),
            ))

    def forward(self, rec1, recon_init, phi, phi_size, mask):
        for i in range(self.num_layers):
            rec = self.conv321[i](rec1)
            temp = F.conv2d(rec, phi, padding=0, stride=phi_size, bias=None)
            temp = temp.permute(0, 2, 3, 1) * mask
            temp = temp.permute(0, 3, 1, 2)
            temp = F.conv_transpose2d(temp, phi, stride=phi_size)
            rec = (temp - recon_init)
            rec = rec1 - (self.conv132[i](rec))
            rec1 = self.res1[i](rec)

        return rec1

# PCCD-Net (Deep reconstruction network)
class DUN(torch.nn.Module):
    def __init__(self, channels, num_layers=24):
        super(DUN, self).__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.res1 = nn.ModuleList()
        self.res2 = nn.ModuleList()
        self.res3 = nn.ModuleList()
        self.conv321 = nn.ModuleList()
        self.conv132 = nn.ModuleList()
        self.conv324 = nn.ModuleList()
        self.conv432 = nn.ModuleList()
        self.conv932 = nn.ModuleList()
        for i in range(self.num_layers + 1):
            self.conv132.append(nn.Sequential(
                nn.Conv2d(1, self.channels, kernel_size=3, stride=1, padding=1, bias=False),
            ))
            self.conv432.append(nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1, bias=False),
            ))
            self.conv932.append(nn.Sequential(
                nn.Conv2d(9, self.channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False),
            ))
            self.conv321.append(nn.Sequential(
                nn.Conv2d(self.channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
            ))
            self.conv324.append(nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(self.channels, 4, kernel_size=3, stride=1, padding=1, bias=False),
            ))
            self.res1.append(nn.Sequential(
                RB(dim=self.channels),
                RB(dim=self.channels),
            ))
            self.res2.append(nn.Sequential(
                RB(dim=self.channels),
                RB(dim=self.channels),
            ))

    def forward(self, rec1, recon_init, phi, phi_size, mask):

        for i in range(self.num_layers):

            # PCPGD path

            rec = self.conv321[i](rec1)
            temp = F.conv2d(rec, phi, padding=0, stride=phi_size, bias=None)
            temp = temp.permute(0, 2, 3, 1) * mask
            temp = temp.permute(0, 3, 1, 2)
            temp = F.conv_transpose2d(temp, phi, stride=phi_size)
            rec = (temp - recon_init)
            rec = rec1 - (self.conv132[i](rec))
            rec = self.res1[i](rec)

            # CDPGD path

            mask1 = torch.concat([mask, mask, mask, mask], dim=0)
            rec4 = self.conv324[i](rec1)
            b, c, h, w = rec4.shape
            temp = F.conv2d(rec4.reshape(-1, 1, h, w), phi, padding=0, stride=phi_size, bias=None)
            temp = temp.permute(0, 2, 3, 1) * mask1
            temp = temp.permute(0, 3, 1, 2)
            temp = F.conv_transpose2d(temp, phi, stride=phi_size).reshape(b, c, h, w)
            rec_mid = torch.concat([rec4, temp, recon_init], dim=1)
            rec_mid = self.conv932[i](rec_mid)
            rec4 = self.res2[i](rec_mid)

            rec1 = (rec + rec4)

        return rec1

class RB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + self.conv(x)



