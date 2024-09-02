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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from renderer.gaussian_points_animated_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from games.flat_splatting.scene.points_gaussian_model import PointsGaussianModel
import numpy as np
import trimesh


def transform_hotdog(triangles, t):
    triangles_new = triangles.clone()
    triangles_new[:, :, 2] += 0.2 * torch.sin(triangles[:, :,  0] / 2 * torch.pi + t)
    return triangles_new

def do_nothing(triangles, t):
    triangles_new = triangles.clone()
    return triangles_new


def find_xy_from_3d_to_out_img(view, triangles):
    k, n = 321, 481
    fx = fov2focal(view.FoVx, n)
    fy = fov2focal(view.FoVy, k)
    K = torch.tensor([[fx, 0, n/2], [0, fy, k/2], [0, 0, 1]])
    R = torch.tensor(np.transpose(view.R))  # Rotation
    w2c = torch.hstack([R, torch.tensor(view.T).reshape(-1, 1)])
    KRt = (K.float() @ w2c.float()).cuda().float()

    gaussians_xyz = triangles[:, 0]
    points = torch.hstack([gaussians_xyz, torch.ones(gaussians_xyz.shape[0], 1).cuda()])
    solutions = (KRt @ points.T).T
    x_s = solutions[:,0]/solutions[:,2]
    y_s = solutions[:,1]/solutions[:,2]
    return x_s, y_s

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_sim")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    t = torch.linspace(0, 4 * torch.pi, 10)
    v1, v2, v3 = gaussians.v1, gaussians.v2, gaussians.v3
    triangles = torch.stack([v1, v2, v3], dim=1)
    #torch.save(triangles, 'pseudomesh_branch_forcey0rotxz_flower.pt')



    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        new_triangles = do_nothing(triangles, t[0])
        k, n = 321, 481
        if idx == 100000: #dla testow False
            x_s, y_s = find_xy_from_3d_to_out_img(view, new_triangles)
            #torch.save(x_s, "x_s.pt")
            #torch.save(y_s, "y_s.pt")
            #xx = x_s.max()
            #yy = y_s.max()
            mask_save = torch.load("/media/joanna/DANE/uj/gaussians2D/output/branch_forcey0rotxz_196073/snake_mask.pt")
            mask = mask_save[:, :, 0].cuda()
            obj_mask = torch.ones_like(x_s)
            obj_mask[torch.where(x_s > n)[0]] = False
            obj_mask[torch.where(y_s > k)[0]] = False

            idxs = torch.where(obj_mask != False)[0]
            ponton_mask_temp = mask[y_s.long()[idxs], x_s.long()[idxs]]
            pm = obj_mask.bool()
            pm[idxs] = ponton_mask_temp
            torch.save(pm, "output/branch_forcey0rotxz_196073/pm1.pt")



        #a = pm.shape
        #gaussians._opacity = pm1.reshape(pm1.shape[0], 1).long() * gaussians._opacity
        objpath = f"/home/joanna/Files/2DGaussians/simulation/camel/sim_objects"
        lst = os.listdir(f"{objpath}")  # your directory path
        lst.sort()
        for i in lst:
            objpathfile = f'{objpath}/{i}'

            mesh_scene = trimesh.load(objpathfile, force='mesh')
            triangles = torch.tensor(mesh_scene.triangles).cuda().float() / 2

            new_triangles = do_nothing(triangles, t[0])
            rendering = render(new_triangles, view, gaussians, pipeline, background)["render"]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + f"_{i}.png"))
            #torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = PointsGaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if hasattr(gaussians, 'prepare_vertices'):
            gaussians.prepare_vertices()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--gs_type', type=str, default="gs_points")
    parser.add_argument("--num_splats", type=int, default=2)

    args = get_combined_args(parser)
    model.gs_type = args.gs_type
    model.num_splats = args.num_splats
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)