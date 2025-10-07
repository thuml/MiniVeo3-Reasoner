import cv2
import numpy as np

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

def process_video_with_trajectory(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter("./debug.mp4", fourcc, fps, (width, height)) # print out the video of identfied traj
    
    trajectory = [] 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        blue_frame, mask = keep_blue_hsv(frame)
        center = find_center(mask)
        
        if center:
            trajectory.append(center)
            
        #     for i in range(1, len(trajectory)):
        #         cv2.line(blue_frame, trajectory[i-1], trajectory[i], (0, 255, 0), 2)
            
        #     cv2.circle(blue_frame, center, 8, (0, 0, 255), -1)
        
        # out.write(blue_frame)
    
    cap.release()
    # out.release()
    
    return trajectory

def gen_traj(file_path1,file_path2):
    expert = process_video_with_trajectory(file_path1,"1.mp4")
    student = process_video_with_trajectory(file_path2,"2.mp4")
    return expert,student 

