import os
from scene.dataset_readers import (
    readCamerasFromTransforms, CameraInfo,
    getNerfppNorm, BasicPointCloud, SH2RGB, storePly, fetchPly, SceneInfo
)
import numpy as np
from PIL import Image, ImageOps
from utils.graphics_utils import focal2fov, fov2focal
from pathlib import Path
import math


camera_angle_x = 0.6911112070083618

def create_transform_matrix(distance):
    transform_matrix = [
        [-np.sign(distance) ,0.0,0.0,0.0],
        [.0,0.0, np.sign(distance), distance ],
        [0.0, 1.0, 0.0,0.0],
        [0.0,0.0, 0.0,1.0]
    ]
    return transform_matrix

def readImage(path, image2dname, white_background, eval, distance, extension=".png"):
    print("Creating Training Transform")
    train_cam_infos = CreateCamerasTransforms(
        path, image2dname, white_background, [-distance], extension
    )
    print("Creating Test Transform")
    test_cam_infos = CreateCamerasTransforms(
        path, image2dname, white_background, [-distance], extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        camera = train_cam_infos[0]
        top = distance * math.tan(camera.FovY * 0.5)
        aspect_ratio = camera.width / camera.height
        right = top * aspect_ratio
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.uniform(low=[-right, 0, -top], high=[right, 0, top], size=(num_pts, 3))
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readMirrorImages(path, image2dname, white_background, eval, distance, extension=".png"):
    print("Creating Training Transforms")
    train_cam_infos = CreateCamerasTransforms(
        path, image2dname, white_background, [-distance, distance], extension
    )
    print("Creating Test Transforms")
    test_cam_infos = CreateCamerasTransforms(
        path, image2dname, white_background, [-distance, distance], extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        camera = train_cam_infos[0]
        top = distance * math.tan(camera.FovY * 0.5)
        aspect_ratio = camera.width / camera.height
        right = top * aspect_ratio
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.uniform(low=[-right, 0, -top], high=[right, 0, top], size=(num_pts, 3))
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def CreateCamerasTransforms(path, image2dname, white_background, distances, extension=".png"):
    cam_infos = []

    fovx = camera_angle_x

    cam_name_init = os.path.join(path, image2dname + extension)
    cam_name_mirror = os.path.join(path, image2dname + "_mirror" + extension)

    for i in range(len(distances)):
        distance = distances[i]
        if i == 0:
            cam_name = cam_name_init
        if i == 1:
            cam_name = cam_name_mirror
            if ~os.path.exists(cam_name):
                # save mirror image
                im = Image.open(cam_name_init)
                im_flip = ImageOps.mirror(im)
                im_flip.save(cam_name_mirror)

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(create_transform_matrix(distance))
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = cam_name
        image_name = Path(cam_name).stem
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy
        FovX = fovx

        cam_infos.append(
            CameraInfo(
                uid=i, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                image_path=image_path, image_name=image_name, width=image.size[0],
                height=image.size[1]
            )
        )

    return cam_infos