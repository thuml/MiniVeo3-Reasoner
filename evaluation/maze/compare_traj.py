import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d


def interpolate_trajectory(traj, m):
    traj = np.array(traj)
    unique_indices = []
    for i in range(len(traj)):
        if i == 0 or not np.array_equal(traj[i], traj[i-1]):
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

def compare(path1, path2):
    M = 5000
    interpolated1 = interpolate_trajectory(path1, M)
    interpolated2 = interpolate_trajectory(path2, M)
    plt.figure(figsize=(8.32, 4.80), dpi=100)
    plt.gca().set_aspect('equal') 
    
    plt.plot(interpolated1[:, 0], interpolated1[:, 1], 'b.-', label='path1', alpha=0.7, markersize=6)
    plt.plot(interpolated2[:, 0], interpolated2[:, 1], 'ro-', label='path2', alpha=0.7, markersize=6)
    
    plt.xlim(0, 832)
    plt.ylim(0, 480)
    
    plt.grid(True, alpha=0.3)

    all_distances = np.linalg.norm(interpolated1 - interpolated2, axis=1)
    
    max_index = np.argmax(all_distances)
    max_distance = all_distances[max_index]


    plt.close()
    return max_index, max_distance, all_distances, interpolated1, interpolated2
