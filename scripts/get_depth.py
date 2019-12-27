import numpy as np
import cv2
def get_depth_data(file_path):
    depth_data = cv2.imread(file_path, -1)
    # print(depth_data)
    w, h = depth_data.shape
    current_depth_data = depth_data[30: 50, 30:50]
    print(current_depth_data)
    current_mean_dis = np.mean(current_depth_data)
    return current_mean_dis

if __name__ == "__main__":
    depth_data = get_depth_data('/home/jiangting/dyp/re/wood/wood7.0/save/imgs/1_line/2019-09-22-17_18_43/2_depth.tif')
    print(depth_data)