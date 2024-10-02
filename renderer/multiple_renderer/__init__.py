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
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh


def render(n, viewpoint_camera, pcs, pipe, bg_color: torch.Tensor, scaling_modifiers=None,
           override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    if scaling_modifiers is None:
        scaling_modifiers = torch.ones(len(pcs))

    means3D_all = []
    means2D_all = []
    shs_all = []
    colors_precomp_all = []
    opacities_all = []
    scales_all = []
    rotations_all = []
    cov3D_precomp_all = []

    for pc, scaling_modifier, idx in zip(pcs, scaling_modifiers, range(len(pcs))):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        viewpoint_camera.camera_center = viewpoint_camera.camera_center
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        _xyz = pc.get_xyz

        means3D = _xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        _means3D = means3D.clone()
        if idx == 1:
            _means3D[:,1] += 0.2
            _means3D[:, 0] += 0.2
            _means3D[:,0] -= 0.01 * n
            _means3D[:,1] += 0.01 * n
        means3D_all.append(_means3D)
        means2D_all.append(means2D)
        #if idx == 1:
        #    shs = torch.ones_like(shs).cuda()
        shs_all.append(shs)
        colors_precomp_all.append(colors_precomp)
        #if idx == 1:
        #    opacity=torch.ones_like(opacity).cuda()
        opacities_all.append(opacity)
        scales_all.append(scales)
        rotations_all.append(rotations)
        cov3D_precomp_all.append(cov3D_precomp)

    means3D_all=torch.vstack(means3D_all)
    means2D_all=torch.vstack(means2D_all)
    shs_all=torch.vstack(shs_all)
    opacities_all=torch.vstack(opacities_all)
    scales_all=torch.vstack(scales_all)
    rotations_all=torch.vstack(rotations_all)
    cov3D_precomp_all=None

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D_all,
        means2D=means2D_all,
        shs=shs_all,
        colors_precomp=None,
        opacities=opacities_all,
        scales=scales_all,
        rotations=rotations_all,
        cov3D_precomp=cov3D_precomp_all)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render":  rendered_image, #torch.vstack([rendered_image, return_accumulation.reshape(1,return_accumulation.shape[0],return_accumulation.shape[1])]),
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}
