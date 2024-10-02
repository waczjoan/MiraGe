#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use


import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from scene.gaussian_model import GaussianModel
from models.flat_splatting.scene.flat_gaussian_model import FlatGaussianModel

class FlatGaussianModel2D(FlatGaussianModel):

    def __init__(self, sh_degree: int):

        super().__init__(sh_degree)
        self.eps_s0 = 1e-8
        self.s0 = torch.empty(0)

    @property
    def get_xyz(self):
        return torch.stack(
            [self._xyz[:, 0], torch.zeros_like(self._xyz[:, 0]), self._xyz[:, -1]],
            axis=1
        )


def quaternion_multiply(Q0, Q1):
    """
    Multiplies two quaternions.

    Input
    :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31)
    :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32)

    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33)

    based on:https://automaticaddison.com/how-to-multiply-two-quaternions-together-using-python/

    """
    # Extract the values from Q0
    w0 = Q0[:, 0]
    x0 = Q0[:, 1]
    y0 = Q0[:, 2]
    z0 = Q0[:, 3]

    # Extract the values from Q1
    w1 = Q1[:, 0]
    x1 = Q1[:, 1]
    y1 = Q1[:, 2]
    z1 = Q1[:, 3]

    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # Create a 4 element array containing the final quaternion
    final_quaternion = torch.stack([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z], dim=1)

    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32)
    return final_quaternion


class FlatGaussianModel2DImage(GaussianModel):

    def __init__(self, sh_degree: int):

        super().__init__(sh_degree)
        self.eps_s0 = 1e-8
        self.s0 = torch.empty(0)
        self._rot_rand = torch.empty(0)
        self.rotation = torch.empty(0)

    @property
    def get_scaling(self):
        self.s0 = torch.ones(self._scaling.shape[0], 1).cuda() * self.eps_s0
        s = torch.cat([self.s0, self.scaling_activation(self._scaling[:, [-2, -1]])], dim=1)
        return torch.clamp(s, min=self.eps_s0, max=0.05)

    @property
    def get_rotation(self):
        return self.rotation_activation(self.calc_rotation())

    @property
    def get_xyz(self):
        return torch.stack(
            [self._xyz[:, 0], torch.zeros_like(self._xyz[:, 0]), self._xyz[:, -1]],
            axis=1
        )

    def calc_rotation(self):
        rots_x = self.calc_rots_x(self._rot_rand.shape[0], self._rot_rand)
        rots_y = self.calc_rots_y(self._rot_rand.shape[0])
        rots_z = self.calc_rots_z(self._rot_rand.shape[0])
        self.rotation = self.calc_rots(rots_y, rots_z, rots_x)
        return self.rotation

    @staticmethod
    def calc_rots(rots_x, rots_y, rots_z):
        rots = quaternion_multiply(rots_x, rots_y)
        rots = quaternion_multiply(rots, rots_z)
        return rots

    @staticmethod
    def calc_rots_x(n, rot_rand):
        rots_y = torch.zeros((n, 4), device="cuda")
        rots_y[:, 0] = torch.cos(rot_rand)
        rots_y[:, 1] = torch.sin(rot_rand)
        rots_y[:, 2] = 0
        rots_y[:, 3] = 0
        return rots_y

    @staticmethod
    def calc_rots_z(n):
        rots_x = torch.zeros((n, 4), device="cuda")
        rots_x[:, 0] = torch.cos(torch.tensor(torch.pi / 4))
        rots_x[:, 1] = 0
        rots_x[:, 2] = 0
        rots_x[:, 3] = torch.sin(torch.tensor(torch.pi / 4))
        return rots_x

    @staticmethod
    def calc_rots_y(n):
        rots_z = torch.zeros((n, 4), device="cuda")
        rots_z[:, 0] = 1
        rots_z[:, 1] = 0
        rots_z[:, 2] = 0
        rots_z[:, 3] = 0
        return rots_z


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rot_rand], 'lr': training_args.rotation_lr, "name": "rot_rand"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)[:, [0, 2]]).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots_y = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rot_rand = torch.rand_like(rots_y[:, 0]) * 2 * torch.pi
        rots_x = self.calc_rots_x(fused_point_cloud.shape[0], rot_rand)
        rots_y = self.calc_rots_y(fused_point_cloud.shape[0])
        rots_z = self.calc_rots_z(fused_point_cloud.shape[0])

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rot_rand = nn.Parameter(rot_rand.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent
        )
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        self.calc_rotation()
        rots = build_rotation(self.rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_xyz = new_xyz[:, [0, 2]]
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8*N)
        )[:, [1, 2]]
        new_rot_rand = torch.flatten(
            self._rot_rand.reshape(-1, 1)[selected_pts_mask].repeat(N, 1)
        )
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rot_rand)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rot_rand):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rot_rand" : new_rot_rand}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rot_rand = optimizable_tensors["rot_rand"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rot_rand = optimizable_tensors["rot_rand"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rot_rand = self._rot_rand[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rot_rand)

    def save_ply(self, path):

        self._rotation = self.calc_rotation()
        self._save_ply(path)
        self._rotation = None
        attrs = self.__dict__
        additional_attrs = [
            '_rot_rand'
        ]

        save_dict = {}
        for attr_name in additional_attrs:
            save_dict[attr_name] = attrs[attr_name]

        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        torch.save(save_dict, path_model)