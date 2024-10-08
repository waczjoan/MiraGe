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

from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from models.flat_splatting.scene.points_gaussian_model import PointsGaussianModel
from models.flat_splatting_3d_images.scene.flat_gaussian_model import FlatGaussianModel3DImage
from models.flat_splatting_2d_images.scene.flat_gaussian_model import (
    FlatGaussianModel2DImage, FlatGaussianModel2D)
from models.slices.scene.flat_gaussian_model import FlatGaussianModelSlices

from models.flat_splatting.scene.flat_gaussian_model import FlatGaussianModel

optimizationParamTypeCallbacks = {
    "gs": OptimizationParams,
    "gs_flat": OptimizationParams,
    "gs_flat2D": OptimizationParams,
    "amorphous": OptimizationParams,
    "2d": OptimizationParams,
    "graphite": OptimizationParams,
}

gaussianModel = {
    "gs": GaussianModel,
    "gs_flat": FlatGaussianModel,
    "gs_flat2D": FlatGaussianModel2D,
    "amorphous": FlatGaussianModel3DImage,
    "2d": FlatGaussianModel2DImage,
    "graphite": FlatGaussianModelSlices,
    "gs_points": PointsGaussianModel
}

gaussianModelRender = {
    "gs": GaussianModel,
    "gs_flat": FlatGaussianModel,
    "2d": FlatGaussianModel2D,
    "amorphous": FlatGaussianModel,
    "graphite": FlatGaussianModel,
    "gs_points": PointsGaussianModel
}
