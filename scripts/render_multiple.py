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
from os import makedirs
from renderer.multiple_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
import copy
from argparse import Namespace
from models import gaussianModelRender


def get_combined_args(arguments):
    cfgfile_string = "Namespace()"
    args_cmdline = arguments

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices


def transform_diff(vertices, vertices_diff, t):
    vertices += vertices_diff * t
    return vertices


def do_not_transform(vertices, t):
    return vertices


def render_set(
        model_paths, name, iteration, views, gaussians_list,
        pipeline, background, sym_dirnames, skip_sym_obj, scale
):

    if len(sym_dirnames) > 0:
        name_sim = os.path.basename(sym_dirnames[0])
        lst = os.listdir(f"{sym_dirnames[0]}/triangles.py")  # your directory path
        number_files = len(lst)
    else:
        name_sim = "default"
        number_files = 0

    render_path = os.path.join(
        model_paths[0], name, "ours_multiple", name_sim
    )

    for name in sym_dirnames:
        if name != "":
            makedirs(f"{name}/objects", exist_ok=True)
            makedirs(f"{name}/objects/scale_{scale}", exist_ok=True)

    makedirs(render_path, exist_ok=True)

    for view in views:
        for i in range(20):
            rendering = render(i, view, gaussians_list, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path,'{0:05d}'.format(i) + ".png"))


def write_simple_obj(mesh_v, mesh_f, filepath, verbose=False):
    with open(filepath, 'w') as fp:
        for v in mesh_v:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in mesh_f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('mesh saved to: ', filepath)


def render_sets(
        model_paths, datasets : ModelParams, iteration: int, pipeline: PipelineParams,
        skip_train: bool, skip_test: bool, sym_dirname, skip_sym_obj, scale, all_views=False
):
    with torch.no_grad():
        gaussians_list = []
        scene_list = []
        loaded_iters = []
        for dataset in datasets:
            gaussians = gaussianModelRender["gs_flat"](dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            if hasattr(gaussians, 'prepare_vertices'):
                gaussians.prepare_vertices()
            if hasattr(gaussians, 'prepare_vertices'):
                gaussians.prepare_vertices()
            if hasattr(gaussians, 'prepare_scaling_rot'):
                gaussians.prepare_scaling_rot()
            gaussians_list.append(copy.deepcopy(gaussians))
            if dataset == datasets[0]:
                train_views = [scene.getTrainCameras()[0]]
                test_views = [scene.getTestCameras()[0]]

                if all_views:
                    train_views = scene.getTrainCameras()[:10]
                    test_views = scene.getTrainCameras()[:10]

            loaded_iters.append(scene.loaded_iter)
            del scene

        bg_color = [1,1,1] if datasets[0].white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(
                 model_paths, "train",
                 loaded_iters, train_views,
                 gaussians_list, pipeline, background, sym_dirname,
                 skip_sym_obj, scale
             )

        if not skip_test:
             render_set(
                 model_paths, "test",
                 loaded_iters, test_views,
                 gaussians_list, pipeline, background, sym_dirname,
                 skip_sym_obj, scale
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--sym_dirnames', nargs="+", type=str, default=[])
    parser.add_argument('--skip_sym_obj', type=int, default=8)
    parser.add_argument('--scale', type=int, default=100)
    parser.add_argument("--model_paths", nargs="+", type=str, default=[])
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--gs_type', type=str, default="gs_flat")
    parser.add_argument('--scene_image', type=str, default="mirror")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--num_pts", type=int, default=100_000)

    arguments = parser.parse_args()
    args = []
    sym_dirname = {}
    if len(arguments.sym_dirnames) > 0:
        for i, sym_dirname_i in zip(arguments.model_paths, arguments.sym_dirnames):
            arguments.model_path = i
            _arg = get_combined_args(arguments)
            args.append(_arg)
            sym_dirname[i] = sym_dirname_i
    else:
        for i in arguments.model_paths:
            arguments.model_path = i
            _arg = get_combined_args(arguments)
            args.append(_arg)
    model.gs_type = args[0].gs_type
    model.scene_image = arguments.scene_image
    model.distance = arguments.distance
    model.num_pts = arguments.num_pts

    print("Rendering " + str(arguments.model_paths))

    # Initialize system state (RNG)
    safe_state(args[0].quiet)

    datasets = [model.extract(arg) for arg in args]

    render_sets(
        arguments.model_paths,
        datasets,
        arguments.iteration,
        pipeline.extract(args[0]),
        arguments.skip_train,
        arguments.skip_test,
        arguments.sym_dirnames,
        arguments.skip_sym_obj,
        arguments.scale
    )
