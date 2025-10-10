#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import sys
import cv2
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import Imath
import OpenEXR
from pathlib import Path
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    transform_matrix: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    
    pl_intensity: np.array
    pl_pos: np.array
    exr_image: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def read_exr(exr_path):
    # Read EXR file
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()

    # Extract channel information
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Extract RGB channels
    channels = ['R', 'G', 'B', 'A']
    exr_image = []
    for channel in channels:
        raw_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
        exr_image.append(np.frombuffer(raw_data, dtype=np.float32).reshape(size[1], size[0]))
    return np.stack(exr_image, axis=0)

def get_exr_size(filepath):
    exr_file = OpenEXR.InputFile(filepath)
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    return width, height

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, load_image=True):
    cam_infos = []
    for idx, key in enumerate(list(cam_extrinsics.keys())):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = None
        if load_image and os.path.exists(image_path):
            image = Image.open(image_path)
        exr_image = None
        if load_image and os.path.exists(os.path.join(images_folder, image_name + '.exr')):
            exr_path = os.path.join(images_folder, image_name + '.exr')
            exr_image = read_exr(exr_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, pl_pos=None, pl_intensity=None, exr_image=exr_image, transform_matrix=[])
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, load_image=True):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), load_image=load_image)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
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

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", load_image=True):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if "camera_angle_x" in contents:
            fovx = contents["camera_angle_x"]
            fovy = None
            fx = fy = None
        else:
            # intrinsics of real capture
            intrinsics = contents['camera_intrinsics']
            cx = intrinsics[0]
            cy = intrinsics[1]
            fx = intrinsics[2]
            fy = intrinsics[3]

        frames = sorted(contents["frames"], key=lambda x: os.path.basename(x["file_path"]))
        for idx, frame in enumerate(frames):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            H = W = None
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            image_path = os.path.join(path, frame["file_path"] + extension)
            image_name = Path(image_path).stem
            image = None
            if load_image and os.path.exists(image_path):
                image = Image.open(image_path)

                im_data = np.array(image.convert("RGBA"))

                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(np.concatenate((arr, norm_data[:, :, 3:4]), axis=-1)*255.0, dtype=np.byte), "RGBA")
                W = image.size[0]
                H = image.size[1]

            exr_image = None
            exr_path = os.path.join(path, frame["file_path"] + ".exr")
            if load_image and os.path.exists(exr_path):
                exr_image = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
                norm_data = cv2.cvtColor(exr_image, cv2.COLOR_BGRA2RGBA)
                
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                exr_image = arr.astype(np.float32).transpose(2, 0, 1)
                exr_image = np.concatenate((exr_image, norm_data[:, :, -1][None]))

                W = exr_image.shape[-1]
                H = exr_image.shape[-2]
            
            if H is None or W is None:
                if os.path.exists(image_path):
                    with Image.open(image_path) as _image:
                        W = _image.size[0]
                        H = _image.size[1]
                elif os.path.exists(exr_path):
                    W, H = get_exr_size(exr_path)
                else:
                    raise ValueError(f"Unable to find {image_path} or {exr_path}!")
            # fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            if fx is None and fy is None:
                FovY = fovy if fovy is not None else focal2fov(fov2focal(fovx, W), H)
                FovX = fovx
            else:
                FovY = focal2fov(fy, H)
                FovX = focal2fov(fx, W)

            cam_infos.append(CameraInfo(
                uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                image_path=image_path, image_name=image_name, width=W, height=H, 
                pl_pos=(np.array(frame["pl_pos"]) * np.array([1., 1., 1.])) if "pl_pos" in frame else None, 
                pl_intensity=(np.array(frame["pl_intensity"])) if "pl_intensity" in frame else None, exr_image=exr_image, transform_matrix=np.array(frame["transform_matrix"])))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", load_image=True):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, load_image=load_image)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, load_image=load_image)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    print(f"Radius: {nerf_normalization['radius']}")

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * (nerf_normalization['radius'] / 3) - (nerf_normalization['radius'] / 3) / 2
        colors = np.ones((num_pts, 3)) * 0.1
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, colors * 255.)
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

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}