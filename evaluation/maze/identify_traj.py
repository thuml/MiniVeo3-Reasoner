import cv2
import numpy as np

def keep_blue_hsv(frame):
    """
    使用HSV颜色空间保留蓝色部分
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 100, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result, mask

def find_center(mask):
    """
    找到星星的中心位置
    """
    points = cv2.findNonZero(mask)
    
    if points is None or len(points) < 20:
        return None  # 没有找到蓝色像素
    
    # 计算所有非零像素的质心（中心点）
    moments = cv2.moments(mask)
    
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        return (center_x, center_y)
    else:
        return None

def process_video_with_trajectory(input_path, output_path):
    """
    处理视频并追踪星星轨迹
    """
    #import pdb;pdb.set_trace()
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("./debug.mp4", fourcc, fps, (width, height))
    
    trajectory = []  # 存储轨迹点
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        blue_frame, mask = keep_blue_hsv(frame)
        center = find_center(mask)
        
        if center:
            trajectory.append(center)
            
            # 在视频上绘制轨迹
            for i in range(1, len(trajectory)):
                cv2.line(blue_frame, trajectory[i-1], trajectory[i], (0, 255, 0), 2)
            
            # 标记当前中心点
            cv2.circle(blue_frame, center, 8, (0, 0, 255), -1)
        
        out.write(blue_frame)
    
    cap.release()
    out.release()
    
    # 打印轨迹信息
    # print(f"轨迹点数量: {len(trajectory)}")
    # for i, (x, y) in enumerate(trajectory):
    #     print(f"帧 {i}: 中心位置 ({x}, {y})")
    
    return trajectory

# 使用示例
def gen_traj(file_path1,file_path2):
    expert = process_video_with_trajectory(file_path1,"1.mp4")
    student = process_video_with_trajectory(file_path2,"2.mp4")
    return expert,student 


# trajectory = track_blue_star("/data/NAS/rl_data/video_reasoner/huangtianhao_data/DiffSynth-Studio/inference_data/test3by3/maze3_1251_epoch-2_inference.mp4","2.mp4")
# print(trajectory)