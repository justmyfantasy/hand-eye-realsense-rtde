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

RECORD_ROBOT = True  # Record the robot pose on disk
RECORD_CAMERA = True  # Record the camera image on disk
RECORD_FOLDER = 'Hand-Eye-Data'  # Default folder to save recordings, relative to the station folder
ROBOT_IP = '192.168.12.21'

positions = [
    [0.024830496946257694, 0.3613008373831808, 0.5831640576894229, 0.03779546605504869, 3.1393512693345516, -0.026513872759567354],
    [-0.17972256261143182, 0.3667476222504174, 0.5387403164927548, 0.09323499365407367, 2.937804928672331, -0.03138002270731575],
    [-0.01105698287521767, 0.5185450061619816, 0.5007874876138306, 0.03486622443835539, 3.0414270375731585, -0.4174703633716534],
    [0.0016602729435583816, 0.44994660505995465, 0.589797349329651, 0.03738025065848819, -3.1019882917765287, 0.14134347297033456],
    [0.1889008545076976, 0.5325493803725756, 0.45798924600059004, 0.500246529511814, -2.8710289082008957, 0.2630256072959213],
    [0.2722783014132494, 0.4059486760634449, 0.4745174659039418, 0.11530889230372157, -2.806796459957707, 0.0025932034357214378],
    [0.2555863766313413, 0.3192382711687665, 0.5360515331092105, -0.29054126091647653, -2.8032298682568477, 0.05282698119021768],
    [0.2278211442033063, 0.31245974029156787, 0.6632077496554692, 0.2230697847437507, -2.917498444863881, -0.1283583991473771],
    [0.08308728710125497, 0.35218189044082415, 0.7074719396314304, 0.04947201162912529, -3.1006429999561815, -0.010784686808950478],
    [0.05573743836159643, 0.5937900841306186, 0.59234934211815, 0.15310573846047293, 3.0903904998222145, -0.40584867307399375],
]

def record_robot(filename):

    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

    # Retrieve the required information for hand-eye
    robot_data = {}
    # import pdb;pdb.set_trace()
    robot_data['joints'] = rtde_r.getActualQ()#.tolist()
    robot_data['pose_flange'] = rtde_r.getActualTCPPose()

    # Save it on disk as a JSON
    # You can also provide another format, like YAML
    with open(filename, 'w') as f:
        json.dump(robot_data, f, indent=2)


def runmain():

    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    # rob_mode = rtde_r.getRobotMode()
    # rob_status = rtde_r.getRobotStatus()
    
    # if rob_mode == 7:
    #     rtde_c.stopScript()
    #     time.sleep(0.5)
    #     rtde_c.reconnect()

    
    rtde_c.moveL(positions[0], 0.05, 0.01)
    while True :
        sleep(0.1)  #sleep first since the robot may not have processed the command yet
        if rtde_c.isProgramRunning():
            break

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

    # Retrieve the camera and robot
    # camera_handle = get_camera_handle(CAMERA_TYPE, CAMERA_ROBODK_NAME, RDK)
    # robot_item = get_robot(RDK, ROBOT_NAME)

    # Retrieve the folder to save the data
    record_folder = Path('./') / RECORD_FOLDER
    record_folder.mkdir(exist_ok=True, parents=True)

    # This script does not delete the previous run if the folder is not empty
    # If the folder is not empty, retrieve the next ID
    id = 0
    ids = sorted([int(x.stem) for x in record_folder.glob('*.png') if x.stem.isdigit()])
    if ids:
        id = ids[-1] + 1

    # Start the main loop, and wait for requests
    # RDK.setParam(RECORD_READY, 0)
    # Start streaming
    
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
    for pos in positions:
        time.sleep(0.01)

        rtde_c.moveL(pos, 0.05, 0.01)
        while True :
            sleep(0.1)  #sleep first since the robot may not have processed the command yet
            if rtde_c.isProgramRunning():
                break

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
        key = cv2.waitKey()
        if key == ord('q'):
            break
        if key == ord('s'):
            # Process the requests
            if RECORD_CAMERA:
                img_filename = f'{record_folder.as_posix()}/{id}.jpg'
                depth_filename = f'{record_folder.as_posix()}/{id}.png'
                
                # record_camera(CAMERA_TYPE, camera_handle, img_filename.as_posix())
                cv2.imwrite(img_filename, color_image)
                cv2.imwrite(depth_filename, depth_image)

            if RECORD_ROBOT:
                robot_filename = Path(f'{record_folder.as_posix()}/{id}.json')
                record_robot(robot_filename.as_posix())

            print('save ID: ', id)
            id += 1
    rtde_c.stopScript()
            
        


if __name__ == '__main__':
    runmain()