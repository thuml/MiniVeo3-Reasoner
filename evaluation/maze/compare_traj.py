import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def calculate_path_length(path):
    """计算路径总长度"""
    if len(path) < 2:
        return 0
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

def dtw_similarity(path1, path2):
    """计算两条路径的DTW距离"""
    distance, _ = fastdtw(path1, path2, dist=euclidean)
    return distance

def path_length_ratio_to_dtw(path1, path2):
    """计算路径总长平均值与DTW距离之比"""
    length1 = calculate_path_length(path1)
    length2 = calculate_path_length(path2)
    avg_path_length = (length1 + length2) / 2
    dtw_dist = dtw_similarity(path1, path2)
    
    if dtw_dist == 0:
        return float('inf'), avg_path_length, dtw_dist
    
    ratio = avg_path_length/dtw_dist
    return ratio, avg_path_length, dtw_dist

def normalize_paths_for_visualization(path1, path2, canvas_size=800, margin=50):
    """将路径归一化到可视化画布尺寸"""
    # 合并所有点找到边界
    #import pdb;pdb.set_trace()
    all_points = np.vstack([path1, path2])
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)
    
    # 计算缩放比例
    range_x = max_x - min_x
    range_y = max_y - min_y
    scale = min((canvas_size - 2 * margin) / max(range_x, range_y), 1.0)
    
    # 归一化函数
    def normalize_point(point):
        x = int((point[0] - min_x) * scale + margin)
        y = int((point[1] - min_y) * scale + margin)
        return (x, y)
    
    # 归一化所有点
    path1_normalized = [normalize_point(p) for p in path1]
    path2_normalized = [normalize_point(p) for p in path2]
    
    return path1_normalized, path2_normalized

def draw_paths_with_metrics(path1, path2, canvas_size=(832, 480)):
    """使用CV2绘制路径和指标（英文显示）"""
    # 创建画布
    canvas_width, canvas_height = canvas_size
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # 归一化路径点
    path1_norm, path2_norm = normalize_paths_for_visualization(path1, path2, min(canvas_width, canvas_height))
    
    # 绘制路径1 (蓝色)
    for i in range(len(path1_norm) - 1):
        cv2.line(canvas, path1_norm[i], path1_norm[i + 1], (255, 0, 0), 3)
    for point in path1_norm:
        cv2.circle(canvas, point, 6, (255, 0, 0), -1)
    
    # 绘制路径2 (红色)
    for i in range(len(path2_norm) - 1):
        cv2.line(canvas, path2_norm[i], path2_norm[i + 1], (0, 0, 255), 3)
    for point in path2_norm:
        cv2.circle(canvas, point, 6, (0, 0, 255), -1)
    
    # 计算指标
    ratio, avg_length, dtw_dist = path_length_ratio_to_dtw(path1, path2)
    length1 = calculate_path_length(path1)
    length2 = calculate_path_length(path2)
    
    # 相似性评价
    if ratio > 3:
        evaluation = "Highly Similar"
        color = (0, 200, 0)  # 绿色
    elif ratio > 2:
        evaluation = "Moderately Similar"
        color = (0, 165, 255)  # 橙色
    elif ratio > 1:
        evaluation = "Slightly Similar"
        color = (0, 100, 255)  # 红色
    else:
        evaluation = "Significantly Different"
        color = (0, 0, 255)  # 深红色
    
    # 在图像上添加文本信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    y_offset = 30
    line_height = 30
    
    metrics = {
        "path1_length": length1,
        "path2_length": length2,
        "average_path_length": avg_length,
        "dtw_distance": dtw_dist,
        "ratio": ratio,
        "similarity_evaluation": evaluation
    }

    texts = [
        f"Path 1 Length: {length1:.2f}",
        f"Path 2 Length: {length2:.2f}",
        f"Average Path Length: {avg_length:.2f}",
        f"DTW Distance: {dtw_dist:.2f}",
        f"Avg Length/DTW Ratio: {ratio:.2f}",
        f"Similarity Evaluation: {evaluation}"
    ]
    
    for i, text in enumerate(texts):
        y_pos = y_offset + i * line_height
        if "Similarity Evaluation" in text:
            cv2.putText(canvas, text, (20, y_pos), font, font_scale, color, thickness)
        else:
            cv2.putText(canvas, text, (20, y_pos), font, font_scale, (0, 0, 0), thickness)
    
    # 添加图例
    legend_y = canvas_height - 80
    cv2.putText(canvas, "Legend:", (20, legend_y), font, 0.7, (0, 0, 0), thickness)
    cv2.putText(canvas, "Blue: Path 1", (20, legend_y + 25), font, 0.6, (255, 0, 0), thickness)
    cv2.putText(canvas, "Red: Path 2", (20, legend_y + 50), font, 0.6, (0, 0, 255), thickness)
    
    return canvas, metrics

from scipy.interpolate import interp1d
def interpolate_trajectory(traj, m):
    """
    使用scipy插值简化轨迹插值
    
    Args:
        traj: 轨迹点数组，形状为(n, 2)
        m: 目标点数
    
    Returns:
        插值后的轨迹点数组，形状为(m, 2)
    """
    # 转换为numpy数组
    traj = np.array(traj)
    unique_indices = []
    for i in range(len(traj)):
        if i == 0 or not np.array_equal(traj[i], traj[i-1]):
            unique_indices.append(i)
    
    traj = traj[unique_indices]
    if len(traj) < 2:
        return np.zeros((m, 2))
    # 计算累积距离
    distances = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
    cumulative_dist = np.concatenate(([0], np.cumsum(distances)))
    
    # 创建基于距离的插值函数
    interp_func = interp1d(cumulative_dist, traj, axis=0, kind='linear')
    
    # 生成均匀分布的距离点
    new_distances = np.linspace(0, cumulative_dist[-1], m)
    
    # 插值得到新点
    new_traj = interp_func(new_distances)
    
    return new_traj

import matplotlib.pyplot as plt
def compare_new(path1, path2):
    M = 5000
    interpolated1 = interpolate_trajectory(path1, M)
    interpolated2 = interpolate_trajectory(path2, M)
    plt.figure(figsize=(8.32, 4.80), dpi=100)  # 设置画布尺寸
    plt.gca().set_aspect('equal')  # 保持纵横比
    
    # 绘制轨迹
    plt.plot(interpolated1[:, 0], interpolated1[:, 1], 'b.-', label='插值轨迹', alpha=0.7, markersize=6)
    plt.plot(interpolated2[:, 0], interpolated2[:, 1], 'ro-', label='原始轨迹', alpha=0.7, markersize=6)
    
    # 设置坐标轴范围以适应832×480
    plt.xlim(0, 832)
    plt.ylim(0, 480)
    
    # 添加网格和标签
    plt.grid(True, alpha=0.3)

    all_distances = np.linalg.norm(interpolated1 - interpolated2, axis=1)
    
    # 找到最大距离和对应的索引
    max_index = np.argmax(all_distances)
    max_distance = all_distances[max_index]
    #plt.savefig('trajectory_plot.png')


    plt.close()
    return max_index, max_distance, interpolated1, interpolated2
    #import pdb;pdb.set_trace()

def compare(path1,path2):
    # 创建测试路径对
    return compare_new(path1,path2)
    test_path_pairs = [
        # 完全相同路径
        (np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
         np.array([[0, 0], [1, 1], [2, 2], [3, 3]])),
        
        # 相似路径
        (np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
         np.array([[0, 0], [1, 1.1], [2, 1.9], [3, 3.1]])),
        
        # 不同路径
        (np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
         np.array([[0, 0], [2, 2], [1, 1], [3, 3]])),
        
        # 尺度不同的相似路径
        (np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
         np.array([[0, 0], [2, 2], [4, 4], [6, 6]]))
    ]
    
    # 单个对比示例
    
    canvas, metrics = draw_paths_with_metrics(path1, path2)
    # cv2.imshow('路径相似性分析', canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # # 网格对比示例
    # print("\n多组路径网格对比:")
    # grid_canvas, results = create_comparison_grid(test_path_pairs)
    
    # # 显示网格图
    # cv2.imshow('多路径相似性对比', grid_canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # 打印详细结果
    # print("\n详细分析结果:")
    # for result in results:
    #     print(f"路径对 {result['pair_index']}:")
    #     print(f"  平均长度/DTW比值: {result['ratio']:.4f}")
    #     print(f"  平均路径长度: {result['avg_length']:.4f}")
    #     print(f"  DTW距离: {result['dtw_distance']:.4f}")
    #     print()
    
    # 保存结果图像
    cv2.imwrite('single_comparison.jpg', canvas)
    return metrics
    # cv2.imwrite('grid_comparison.jpg', grid_canvas)
    # print("图像已保存为 'single_comparison.jpg' 和 'grid_comparison.jpg'")