import cv2
from glob import glob
import numpy as np


def Load_Images(Path):
    Images = []
    for file in sorted(glob(Path + '*.png')):
        img = cv2.imread(file)
        Images.append(img)
    return Images

def Extract_Features(img):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def Match_Features(Descriptors1, Descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(Descriptors1, Descriptors2)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def estimate_pose(kp1, kp2, matches, K):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    # print(len(pts1))
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
    Temp_List = [1 for x in mask.ravel() if x == 1]
    # print(len(Temp_List))
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t, mask

def triangulate_points(kp1, kp2, matches, R, T, r, t, K, mask):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])[mask.ravel() == 1]
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])[mask.ravel() == 1]
    # print(pts1.shape, pts2.shape)
    if pts1.shape[0] < 5 or pts2.shape[0] < 5:
        return []
    pts1 = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
    pts2 = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None)
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((r, t))
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3d = points_4d[:3] / points_4d[3]
    T_Total = T + np.dot(R, t)
    R_Total = np.dot(r, R)
    Temp1 = np.dot(R_Total.T, points_3d)
    Temp2 = np.tile(T_Total, (points_3d.shape[1]))
    points_3d_World = np.dot(R_Total.T, points_3d) + np.tile(T_Total, (points_3d.shape[1]))
    return points_3d_World.T

def get_depths(frame_data):
    depths = []
    if frame_data.max_id is None:
        return depths
    # stereo = cv2.StereoBM_create(numDisparities=32, blockSize=11)
    stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=96,
    blockSize=9,
    P1=8 * 9**2,
    P2=32 * 9**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    )
    disparity = stereo.compute(cv2.cvtColor(frame_data.frame_l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame_data.frame_r, cv2.COLOR_BGR2GRAY))
    disparity = disparity / 16.0
    depth_map = 379.815 / disparity
    
    return depth_map

def backproject_2d_to_3d(uv, K, R, T, Zc):
    principal_point = K[:2, 2]
    f = K[0, 0]

    uv_normalized = (np.array(uv) - principal_point) / f
    uv_normalized = np.append(uv_normalized, 1)

    p3d_cam = Zc * uv_normalized
    p3d_cam = np.expand_dims(p3d_cam, axis=1)
    P_world = np.dot(R, p3d_cam) + T

    return P_world

def project_3d_to_2d(P_world, K, R, T):
    P_world = np.array(P_world).reshape(3, 1)
    p_cam = np.dot(R.T, (P_world - T))
    p2d_homog = np.dot(K, p_cam)
    
    p2d = p2d_homog[:2] / p2d_homog[2]
    
    return p2d.flatten()

def create_danger_zone(curr_position_3d, radius, k_matrix, global_rotation, global_translation):
    num_points = 100
    theta = np.linspace(0, 2 * np.pi, num_points)
    
    circle_3d = np.array([
        [(radius/10) * np.cos(t) + curr_position_3d[0],
         curr_position_3d[1],
         (radius/10) * np.sin(t) + curr_position_3d[2]]
        for t in theta
    ])

    circle_2d = np.array([
        project_3d_to_2d(pt, k_matrix, global_rotation, global_translation)
        for pt in circle_3d
    ], dtype=np.int32)

    return circle_2d