import rtde_control
import rtde_receive
from robotiq_gripper_control import RobotiqGripper

import torch
import pyrealsense2 as rs
import cv2
import cv2 as cv
import numpy as np
from pathlib import Path
import json
from enum import Enum
import time
from time import sleep
import matplotlib.pyplot as plt
import math

import robomath

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import save_results, plot_results
from utils.dataset_processing.grasp import detect_grasps

RECORD_ROBOT = True  # Record the robot pose on disk
RECORD_CAMERA = True  # Record the camera image on disk
RECORD_FOLDER = 'Grasping_Demo_Data'

ROBOT_IP = '192.168.12.21'
HANDEYE_CHESS_SIZE = (5, 7)  # X/Y
HANDEYE_SQUARE_SIZE = 33  # mm
HANDEYE_MARKER_SIZE = 20  # mm

NETWORK_PATH = 'trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'

positions = [
    [0.024830496946257694, 0.3613008373831808, 0.5831640576894229, 0.03779546605504869, 3.1393512693345516, -0.026513872759567354],
]

def pose_2_Rt(pose: robomath.Mat):
    """RoboDK pose to OpenCV pose"""
    pose_inv = pose.inv()
    R = np.array(pose_inv.Rot33())
    t = np.array(pose.Pos())
    return R, t


def Rt_2_pose(R, t):
    """OpenCV pose to RoboDK pose"""
    vx, vy, vz = R.tolist()

    cam_pose = robomath.eye(4)
    cam_pose.setPos([0, 0, 0])
    cam_pose.setVX(vx)
    cam_pose.setVY(vy)
    cam_pose.setVZ(vz)

    pose = cam_pose.inv()
    # pose.setPos(t.tolist())
    pose.setPos(t)

    return pose

def find_charucoboard(img, mtx, dist, chess_size, squares_edge, markers_edge, predefined_dict=cv.aruco.DICT_6X6_100, draw_img=None) -> dict:
    """
    Detects a charuco board pattern in an image.
    """

    # Note that for chessboards, the pattern size is the number of "inner corners", while charucoboards are the number of chessboard squares
    # import pdb;pdb.set_trace()
    # Charuco board and dictionary
    charuco_board = cv.aruco.CharucoBoard_create(chess_size[0], chess_size[1], squares_edge, markers_edge, cv.aruco.getPredefinedDictionary(predefined_dict))
    charuco_dict = charuco_board.dictionary
    # import pdb;pdb.set_trace()
    # Find the markers first
    marker_corners, marker_ids, _ = cv.aruco.detectMarkers(img, charuco_dict, None, None, None, None)
    if marker_ids is None or len(marker_ids) < 1:
        raise Exception("No charucoboard found")

    # Then the charuco
    count, corners, ids = cv.aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, charuco_board, None, None, mtx, dist, 2)  # mtx and dist are optional here
    if count < 1 or len(ids) < 1:
        raise Exception("No charucoboard found")

    if draw_img is not None:
        cv.aruco.drawDetectedCornersCharuco(draw_img, corners, ids)

    # Find the camera pose. Only available with the camera matrix!
    if mtx is None or dist is None:
        return corners, None, None

    success, rot_vec, trans_vec = cv.aruco.estimatePoseCharucoBoard(corners, ids, charuco_board, mtx, dist, None, None, False)  # mtx and dist are mandatory here
    if not success:
        raise Exception("Charucoboard pose not found")

    if draw_img is not None:
        cv.drawFrameAxes(draw_img, mtx, dist, rot_vec, trans_vec, max(1.5 * squares_edge, 5))

    R_target2cam = cv.Rodrigues(rot_vec)[0]

    return corners, R_target2cam, trans_vec

def runmain():

    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    actual_joints = rtde_r.getActualQ()
    actual_tcp = rtde_r.getActualTCPPose()
    print('current tcp:', actual_tcp)
    print('current joints:', actual_joints)

    # load calibration params
    with open('cam_params_calibrated.json', 'r') as f:
        cam_params = json.load(f)
    T_cam2gripper = np.array(cam_params['T_cam2gripper'])

    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_c.moveL(positions[0], 0.3, 0.01)
    gripper = RobotiqGripper(rtde_c)
    gripper.activate()
    gripper.set_force(5)  # from 0 to 100 %
    gripper.set_speed(100)
    gripper.open()


    # camera init 
    cam = RealSenseCamera(device_id='f1270625')
    cam.connect()
    cam_data = CameraData(include_depth=True, include_rgb=True)

    # Load Network
    print('Loading model...')
    net = torch.load(NETWORK_PATH)
    print('Done')
    device = get_device(force_cpu=False)

    # Retrieve the folder to save the data
    record_folder = Path('./') / RECORD_FOLDER
    record_folder.mkdir(exist_ok=True, parents=True)

    # This script does not delete the previous run if the folder is not empty
    # If the folder is not empty, retrieve the next ID
    id = 0
    ids = sorted([int(x.stem) for x in record_folder.glob('*.png') if x.stem.isdigit()])
    if ids:
        id = ids[-1] + 1

    fig = plt.figure(figsize=(10, 10))
    while True:
        # Wait for a coherent pair of frames: depth and color
        image_bundle = cam.get_image_bundle()
        depth_image = image_bundle['unscale_depth']
        color_image = image_bundle['rgb']
    
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) 
        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image_bgr, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image_bgr, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', cv2.resize(images, dsize=(1280, 480)))
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            img_filename = f'{record_folder.as_posix()}/{id}.jpg'
            depth_filename = f'{record_folder.as_posix()}/{id}.png'
            depth_recolor_filename = f'{record_folder.as_posix()}/{id}_recolor.jpg'
            
            # record_camera(CAMERA_TYPE, camera_handle, img_filename.as_posix())
            cv2.imwrite(img_filename, color_image)
            cv2.imwrite(depth_filename, depth_image)
            cv2.imwrite(depth_recolor_filename, depth_colormap)

            print('save ID: ', id)
            id += 1

            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']
            x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
            
            with torch.no_grad():
                xc = x.to(device)
                pred = net.predict(xc)

                q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

            
            plot_results(fig=fig,
                            rgb_img=cam_data.get_rgb(rgb, False),
                            depth_img=np.squeeze(cam_data.get_depth(depth)),
                            grasp_q_img=q_img,
                            grasp_angle_img=ang_img,
                            no_grasps=1,
                            grasp_width_img=width_img)
            
            grasps = detect_grasps(q_img, ang_img, width_img, no_grasps=1)
            
            grasp_center = np.array(grasps[0].center)
            
            grasp_center[0] = grasp_center[0]/224.0*1080 + cam_data.top_left[0]
            grasp_center[1] = grasp_center[1]/224.0*1080 + cam_data.top_left[1]
            grasp_angle = grasps[0].angle
            
            # Get grasp position from model output
            pos_z = 0.595 + 0.015 #cam.get_dist(grasp_center[1], grasp_center[0]) + 0.02
            
            pos_x = np.multiply(grasp_center[1] - cam.intrinsics.ppx,
                                pos_z / cam.intrinsics.fx)
            pos_y = np.multiply(grasp_center[0] - cam.intrinsics.ppy,
                                pos_z / cam.intrinsics.fy)
            
            
            rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
            robot_pose_flange = rtde_r.getActualTCPPose()
            robot_pose = robomath.UR_2_Pose(robot_pose_flange)
            
            # import pdb;pdb.set_trace()
            ct = math.cos(-grasp_angle + 3.14)
            st = math.sin(-grasp_angle + 3.14)
            R_target2cam = np.array([[ct, -st, 0], 
                                    [st, ct, 0],
                                    [0, 0, 1]])
            t_target2cam = np.array([[pos_x], [pos_y], [pos_z]])

            T_target2cam = np.concatenate((R_target2cam, t_target2cam), axis=1)
            T_target2cam = np.concatenate((T_target2cam, np.array([[0,0,0,1]])), axis=0)

            R_gripper2base, t_gripper2base = pose_2_Rt(robot_pose)
            T_gripper2base = np.concatenate((R_gripper2base, np.expand_dims(t_gripper2base, axis=1)), axis=1)
            T_gripper2base = np.concatenate((T_gripper2base, np.array([[0,0,0,1]])), axis=0)
            
            T_t2b = T_gripper2base.dot(T_cam2gripper).dot(T_target2cam)
            r_t2b = T_t2b[:3, :3]
            t_t2b = T_t2b[:3, 3:]
            # t_t2b[2] += 0.27
            
            
            tmp = Rt_2_pose(r_t2b, t_t2b)
            tmp2 = robomath.RelTool(tmp, 0.0, 0.0, -0.27, 0.0, 0.0, 0.0)
            # print(tmp2)
            # import pdb;pdb.set_trace()
            target_pose_ur = robomath.Pose_2_UR(tmp2)
            print(target_pose_ur)
            
            # target_pose_ur[2] += 0.15
            # target_pos = positions[0]
            # target_pos[0] = target_pose_ur[0].item()
            # target_pos[1] = target_pose_ur[1].item()
            # target_pos[2] = target_pose_ur[2].item() + 0.15
            # import pdb;pdb.set_trace()
            rtde_c.moveL(target_pose_ur, 0.3, 0.05)
            # while True :
            #     sleep(0.1)  #sleep first since the robot may not have processed the command yet
            #     if rtde_c.isProgramRunning():
            #         break
            gripper.close()
            # sleep(1)
            rtde_c.moveL(positions[0], 0.3, 0.05)
            # while True :
            #     sleep(0.1)  #sleep first since the robot may not have processed the command yet
            #     if rtde_c.isProgramRunning():
            #         break   

            rtde_c.moveL(target_pose_ur, 0.3, 0.05)
            gripper.open()

            rtde_c.moveL(positions[0], 0.3, 0.05)
             
    rtde_c.stopScript()
            
        


if __name__ == '__main__':
    runmain()