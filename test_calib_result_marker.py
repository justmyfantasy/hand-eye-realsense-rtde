import rtde_control
import rtde_receive

import pyrealsense2 as rs
import cv2
import cv2 as cv
import numpy as np
from pathlib import Path
import json
from enum import Enum
import time
from time import sleep

import robomath

ROBOT_IP = '192.168.12.21'
HANDEYE_CHESS_SIZE = (5, 7)  # X/Y
HANDEYE_SQUARE_SIZE = 33  # mm
HANDEYE_MARKER_SIZE = 20  # mm


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

    # load cam params
    with open('cam_params_calibrated.json', 'r') as f:
        cam_params = json.load(f)
    T_cam2gripper = np.array(cam_params['T_cam2gripper'])

    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    # rob_mode = rtde_r.getRobotMode()
    # rob_status = rtde_r.getRobotStatus()
    
    # if rob_mode == 7:
    #     rtde_c.stopScript()
    #     time.sleep(0.5)
    #     rtde_c.reconnect()

    
    rtde_c.moveL(positions[0], 0.05, 0.01)
    # while True :
    #     sleep(0.1)  #sleep first since the robot may not have processed the command yet
    #     if rtde_c.isProgramRunning():
    #         break

    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    actual_joints = rtde_r.getActualQ()
    actual_tcp = rtde_r.getActualTCPPose()

    print('current tcp:', actual_tcp)
    print('current joints:', actual_joints)
    
    # import pdb;pdb.set_trace()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    pipeline.start(config)
    # import pdb;pdb.set_trace()
    # Get stream profile and camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    print('depth intrinsic:', depth_intrinsics)
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()
    print('color intrinsic:', color_intrinsics)

    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', cv2.resize(images, dsize=(1280, 480)))
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
            robot_pose_flange = rtde_r.getActualTCPPose()
            robot_pose = robomath.UR_2_Pose(robot_pose_flange)
            image = color_image

            draw_img = None#color_image
            try:
                _, R_target2cam, t_target2cam = find_charucoboard(image, np.array(cam_params['intrinsic']), 
                                        np.array(cam_params['distortion']), HANDEYE_CHESS_SIZE, 
                                        HANDEYE_SQUARE_SIZE, HANDEYE_MARKER_SIZE, 
                                        draw_img=draw_img)
                if draw_img is not None:
                    cv.imshow('det', draw_img)
                    key = cv2.waitKey()
            except:
                print(f'Unable to find chessboard in {i}!')
                continue

            T_target2cam = np.concatenate((R_target2cam, t_target2cam/1000), axis=1)
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
            
            # sleep(1)
            rtde_c.moveL(positions[0], 0.3, 0.05)
            # while True :
            #     sleep(0.1)  #sleep first since the robot may not have processed the command yet
            #     if rtde_c.isProgramRunning():
            #         break   
             

    rtde_c.stopScript()
            
        


if __name__ == '__main__':
    runmain()