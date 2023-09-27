import numpy as np
import cv2

def make_plot(pts, image):

    x = pts[:, 0]
    y = pts[:, 2]
    z = pts[:, 1] + 1.3

    yb = (1.0 - y) / z
    xb = x / z
    zb = np.ones_like(xb)
    points = np.stack((xb, yb, zb), axis=-1)
    depths = np.linalg.norm(points * z[:,None], axis=-1)

    fx = 500  # Focal length in pixels (x-axis)
    fy = 500  # Focal length in pixels (y-axis)
    cx = 640  # Principal point (x-coordinate) in pixels
    cy = 480  # Principal point (y-coordinate) in pixels

    camera_matrix = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]])
    
    base_pts = np.einsum('ik,lk->il', points, camera_matrix).astype(np.int32)
    # image = np.zeros((cy * 2, cx * 2, 3))

    for n in range(len(base_pts)):
        if base_pts[n,1] < cy * 2 and base_pts[n,1] > 0  and base_pts[n,0] < cx * 2 and base_pts[n,0] > 0:
            # image[base_pts[n,1], base_pts[n,0]] = [255, 0, 0]
            cv2.circle(image, (base_pts[n,0], base_pts[n,1]), 2, (255, 0, 0), -1)


    cv2.imshow("abc", image.astype(np.uint8)); cv2.waitKey(0)

def make_plot_base():
    num = 100
    x = np.linspace(-7, 7, num=num)
    y = np.linspace(2, 0, num=num)

    xx, yy = np.meshgrid(x,y)
    z = yy

    yb = 1.0 / z
    xb = xx / z
    zb = np.ones_like(xb)
    print(xb.shape)


    points = np.stack((xb, yb, zb), axis=-1)
    depths = np.linalg.norm(points * z[:,:,None], axis=-1)

    # Define camera intrinsic parameters (example values)
    fx = 500 # Focal length in pixels (x-axis)
    fy = 500  # Focal length in pixels (y-axis)
    cx = 640  # Principal point (x-coordinate) in pixels
    cy = 480  # Principal point (y-coordinate) in pixels

    # Create camera matrix
    camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])

    base_pts = np.einsum('ijk,lk->ijl', points, camera_matrix).astype(np.int32)

    image = np.zeros((cy * 2, cx * 2))

    for m in range(num):
        for n in range(num):
            if base_pts[m,n,1] < cy * 2 and base_pts[m,n,1] > 0  and base_pts[m,n,0] < cx * 2 and base_pts[m,n,0] > 0:
                image[base_pts[m,n,1], base_pts[m,n,0]] = depths[m,n]

    intensity = 0.25
    image = (image - image.min()) / (image.max() - image.min()) * 255 * intensity
    image = image.astype(np.uint8)

    # cv2.imshow("abc", image.astype(np.uint8)); cv2.waitKey(0)

    # cv2.waitKey(0)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image

import argparse
from plt_utils import load_cf_data

parser = argparse.ArgumentParser()
parser.add_argument("filename", nargs="+")
parser.add_argument("--runtime", type=float, default=3)
parser.add_argument("--hovertime",type=float,default=4)
parser.add_argument("-bh", "--baseheight", type=float, default=0.0)
parser.add_argument("-tt", "--takeofftime",type=float,default=5.0)

args = parser.parse_args()
filenames = args.filename

data_dict = load_cf_data(filenames, args)

for key in data_dict.keys():
    im = make_plot_base()

    # for i in range(1, len(data_dict[key]['pose_positions'])):
    make_plot(data_dict[key]['pose_positions'], im)
# print(filename)
# exit()




