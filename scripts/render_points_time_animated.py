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
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from models.flat_splatting.scene.points_gaussian_model import PointsGaussianModel


def transform_sinus(triangles, t):
    triangles_new = triangles.clone()
    triangles_new[:, :, 2] += 0.1 * torch.sin(triangles[:, :,  0] / 4 * torch.pi + t)
    return triangles_new


def do_nothing(triangles, t):
    triangles_new = triangles.clone()
    return triangles_new


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "transform_sinus")

    makedirs(render_path, exist_ok=True)
    thetas = torch.linspace(0, 4 * torch.pi, 10) # hotdog
    v1, v2, v3 = gaussians.v1, gaussians.v2, gaussians.v3
    triangles = torch.stack([v1, v2, v3], dim=1)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        for theta_idx in range(len(thetas)):
            new_triangles = transform_sinus(triangles, thetas[theta_idx])
            rendering = render(new_triangles, view, gaussians, pipeline, background)["render"]
            output_file_path = os.path.join(render_path, '{0:05d}'.format(idx) + f"_{theta_idx}.png")
            torchvision.utils.save_image(
                rendering, output_file_path
            )


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
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument('--camera', type=str, default="mirror")



    args = get_combined_args(parser)
    model.gs_type = args.gs_type
    model.num_pts = args.num_pts
    model.camera = args.camera

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)