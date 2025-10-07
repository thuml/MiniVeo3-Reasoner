import cv2
import numpy as np
from scipy.interpolate import interp1d


def keep_blue_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 100, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result, mask


def find_center(mask):
    points = cv2.findNonZero(mask)

    if points is None or len(points) < 20:
        return None
    moments = cv2.moments(mask)

    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        return (center_x, center_y)
    else:
        return None


def process_video_with_trajectory(input_path):
    cap = cv2.VideoCapture(input_path)
    trajectory = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blue_frame, mask = keep_blue_hsv(frame)
        center = find_center(mask)

        if center:
            trajectory.append(center)

    cap.release()
    return trajectory


def get_traj(file_path1, file_path2):
    expert = process_video_with_trajectory(file_path1)
    student = process_video_with_trajectory(file_path2)
    return expert, student


def interpolate_trajectory(traj, m):
    traj = np.array(traj)
    unique_indices = []
    for i in range(len(traj)):
        if i == 0 or not np.array_equal(traj[i], traj[i - 1]):
            unique_indices.append(i)

    traj = traj[unique_indices]
    if len(traj) < 2:
        return np.zeros((m, 2))
    distances = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
    cumulative_dist = np.concatenate(([0], np.cumsum(distances)))

    interp_func = interp1d(cumulative_dist, traj, axis=0, kind='linear')

    new_distances = np.linspace(0, cumulative_dist[-1], m)

    new_traj = interp_func(new_distances)

    return new_traj


def compare_traj(traj1, traj2):
    M = 5000
    interpolated1 = interpolate_trajectory(traj1, M)
    interpolated2 = interpolate_trajectory(traj2, M)

    all_distances = np.linalg.norm(interpolated1 - interpolated2, axis=1)
    max_index = np.argmax(all_distances)
    max_distance = all_distances[max_index]

    return max_index, max_distance, all_distances, interpolated1, interpolated2
