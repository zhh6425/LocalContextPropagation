import torch
import torch.nn as nn
import torch.nn.functional as F

import pointnet2_utils

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def fps_sampler(npoints):
    def fps_sampler_(xyz):
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        inds = pointnet2_utils.furthest_point_sample(xyz, npoints)
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, inds).transpose(1, 2).contiguous()

        return new_xyz, inds

    return fps_sampler_


def knn_grouper(radius, nsamples, use_xyz, normalize_xyz):
    def knn_grouper_(xyz, new_xyz, features):
        grouper = pointnet2_utils.QueryAndGroup(
            radius,
            nsamples,
            use_xyz=use_xyz,
            ret_grouped_xyz=True,
            normalize_xyz=normalize_xyz,
        )
        grouped_fts, grouped_loc, knn_index = grouper(xyz, new_xyz, features)

        return grouped_fts, grouped_loc, knn_index.long().cuda()

    return knn_grouper_


class TransformerLayer(nn.Module):
    def __init__(self, input_dims, output_dims, pos_enc):
        super().__init__()

        self.pos_enc = pos_enc

        if pos_enc:
            self.pe = nn.Sequential(
                nn.Conv2d(3, input_dims // 2, 1, bias=False),
                nn.BatchNorm2d(input_dims // 2),
                nn.Conv2d(input_dims // 2, input_dims, 1),
            )

        # FIXME: maybe change transformer settings
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dims, nhead=4, dim_feedforward=2 * input_dims
            ),
            num_layers=1,
        )
        self.fc = nn.Sequential(
            nn.Conv2d(input_dims, output_dims, 1, bias=False),
            nn.BatchNorm2d(output_dims), nn.ReLU(inplace=True),
        )

    def forward(self, grouped_loc, grouped_fts):
        if self.pos_enc:
            pos_enc = self.pe(grouped_loc)
            input_fts = grouped_fts + pos_enc
        else:
            input_fts = grouped_fts

        B, D, np, ns = input_fts.shape
        input_fts = (
            input_fts.permute(0, 2, 1, 3).reshape(-1, D, ns).permute(2, 0, 1)
        )  # (ns, B*np, D)
        trans_fts = (
            self.trans(input_fts).permute(1, 2, 0).reshape(B, np, D, ns).transpose(1, 2)
        )  # (B, D, np, ns)

        out_fts = self.fc(trans_fts)  # (B, D, np, ns)

        return out_fts


class LCPFormerBlock(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        npoints,
        radius,
        nsamples,
        use_xyz,
        normalize_xyz,
    ):
        super().__init__()

        self.radius = radius

        self.sampler = fps_sampler(npoints)
        self.grouper = knn_grouper(radius, nsamples, use_xyz, normalize_xyz)

        if use_xyz:
            input_dims += 3

        self.trans_1 = TransformerLayer(
            input_dims, input_dims, pos_enc=(not use_xyz)
        )
        self.trans_2 = TransformerLayer(
            input_dims, output_dims, pos_enc=(not use_xyz)
        )

        #self.mask = Mask(input_dims)


    def forward(self, xyz, features):
        """
        xyz: (B, N, 3)
        features: (B, C, N)
        """

        new_xyz, inds = self.sampler(xyz)  # (B, L, 3), (B, L)
        grouped_fts_1, grouped_loc_1, knn_index = self.grouper(
            xyz, new_xyz, features
        )  # (B, C', L, K), (B, 3, L, K)

        B, D, np, ns = grouped_fts_1.shape

        local_fts_1 = self.trans_1(grouped_loc_1, grouped_fts_1)

        # Learnable mask
        #local_fts_1 = self.mask(local_fts_1)  # (B, D, np, ns)

        # Cross Window
        knn_index = knn_index.reshape(B, -1)
        expand_idxs = knn_index.unsqueeze(1).expand(-1, D, -1)
        cross_win_fts = features.new_zeros((features.shape)).scatter_add_(dim=2, index=expand_idxs, src=local_fts_1.reshape(B, D, -1))

        count_mat = []
        for i in range(B):
            idx = knn_index[i]
            count_mat.append(torch.bincount(idx, minlength=features.shape[-1]).unsqueeze(0))
        count_mat = torch.cat(count_mat, dim=0)
        cross_win_fts = cross_win_fts / count_mat[:, None, :]

        grouped_fts_2, grouped_loc_2, _ = self.grouper(
            xyz, new_xyz, cross_win_fts
        )  # (B, C', L, K), (B, 3, L, K)

        local_fts_2 = self.trans_2(grouped_loc_2, grouped_fts_2)
        out_fts = F.max_pool2d(local_fts_2, kernel_size=(1, ns)).squeeze(-1)  # (B, D, np)

        return new_xyz, out_fts, inds


class Mask(nn.Module):
    def __init__(self, input_dims):
        super().__init__()

        self.channle_gate = nn.Sequential(
            nn.Conv2d(input_dims * 2, input_dims // 8, 1, bias=False),
            nn.BatchNorm2d(input_dims // 8), nn.ReLU(inplace=True),
            nn.Conv2d(input_dims // 8, input_dims, 1, bias=False),
        )

    def forward(self, input_fts):
        B, D, np, ns = input_fts.shape

        max_weight = F.max_pool2d(input_fts, kernel_size=(1, ns))  # B, D, N, 1
        avg_weight = F.avg_pool2d(input_fts, kernel_size=(1, ns))  # B, D, N, 1
        weight = torch.sigmoid(self.channle_gate(torch.cat([max_weight, avg_weight], dim=1)))
        #weight = torch.sigmoid(max_weight)
        #weight = torch.sigmoid(avg_weight)
        out_fts = input_fts * weight

        return out_fts

