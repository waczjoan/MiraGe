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
from fontTools.merge.util import first

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
from models.flat_splatting.scene.points_gaussian_model import PointsGaussianModel
import numpy as np
import trimesh
import matplotlib.pyplot as plt

def find_xy_from_3d_to_out_img(view, triangles):
    n, k = view.image_width, view.image_height
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


def render_set(
        model_path, simulation_path, name, iteration, views, gaussians, pipeline, background, scale=2, save_trajectory: bool = True
):
    basename = os.path.basename(simulation_path).split('.')[0]

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), basename)
    makedirs(render_path, exist_ok=True)

    if save_trajectory:
        traj_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"{basename}_traj_path")
        makedirs(traj_path, exist_ok=True)

    v1, v2, v3 = gaussians.v1, gaussians.v2, gaussians.v3
    triangles = torch.stack([v1, v2, v3], dim=1)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        objpath = f"{simulation_path}"
        lst = os.listdir(f"{objpath}")  # your directory path
        lst = [x for x in lst if "obj" in x]
        lst.sort()

        if save_trajectory:
            objpathfile_last = f'{objpath}/{lst[-1]}'
            mesh_scene = trimesh.load(objpathfile_last, force='mesh')
            last_triangles = torch.tensor(mesh_scene.triangles).cuda().float()

            objpathfile_fist = f'{objpath}/{lst[0]}'
            mesh_scene_fist = trimesh.load(objpathfile_fist, force='mesh')
            first_triangles = torch.tensor(mesh_scene_fist.triangles).cuda().float()

            diff = first_triangles - last_triangles
            diff_sum = torch.abs(diff.sum(dim=2).sum(dim=1))

            idxs = torch.topk(diff_sum, 10000).indices
            follow_points = {}
            for j in range(10):
                follow_points[j] = {
                    'x_s': [],
                    'y_s': []
                }

        for object_file in lst:
                objpathfile = f'{objpath}/{object_file}'

                mesh_scene = trimesh.load(objpathfile, force='mesh')
                new_triangles = torch.tensor(mesh_scene.triangles).cuda().float() / scale

                rendering = render(new_triangles, view, gaussians, pipeline, background)["render"]
                torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + f"{object_file}.png"))

                if save_trajectory:
                    x_s, y_s = find_xy_from_3d_to_out_img(view, new_triangles)
                    for j in range(10):
                        follow_points[j]['x_s'].append(x_s[idxs[j * 1000]].item())
                        follow_points[j]['y_s'].append(y_s[idxs[j * 1000]].item())
                    plt.imshow(rendering.cpu().numpy().transpose((1, 2, 0)))

                    for j in range(10):
                        plt.plot(follow_points[j]['x_s'], follow_points[j]['y_s'])
                    plt.axis('off')
                    plt.savefig(os.path.join(traj_path, '{0:05d}'.format(idx) + f"{object_file.split('.')[0]}.png"))


def render_sets(
    dataset : ModelParams, iteration : int,
    pipeline : PipelineParams, simulation_path,
    skip_train : bool, skip_test : bool, scale: int, save_trajectory: bool
):
    with torch.no_grad():
        gaussians = PointsGaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if hasattr(gaussians, 'prepare_vertices'):
            gaussians.prepare_vertices()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if simulation_path == "":
            simulation_path = f"{dataset.model_path}/sim_objects"

        if not skip_train:
             render_set(
                 dataset.model_path, simulation_path, "train", scene.loaded_iter,
                 scene.getTrainCameras(), gaussians, pipeline, background, scale, save_trajectory
             )

        if not skip_test:
             render_set(
                 dataset.model_path, simulation_path, "test", scene.loaded_iter,
                 scene.getTestCameras(), gaussians, pipeline, background, scale, save_trajectory
             )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_false")
    parser.add_argument("--save_trajectory", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--gs_type', type=str, default="gs_points")
    parser.add_argument("--scale", default=2, type=float)
    parser.add_argument('--camera', type=str, default="mirror")
    parser.add_argument("--distance", type=float, default=5.0)
    parser.add_argument("--simulation_path", default="", type=str)
    parser.add_argument("--num_pts", type=int, default=100_000)



    args = get_combined_args(parser)
    model.gs_type = args.gs_type
    model.scene_image = args.scene_image
    model.distance = args.distance
    model.num_pts = args.num_pts
    model.camera = args.camera

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args), args.iteration,
        pipeline.extract(args), args.simulation_path, args.skip_train,
        args.skip_test, args.scale, args.save_trajectory
    )