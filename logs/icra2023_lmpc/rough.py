import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import math
from PIL import Image, ImageDraw

def pil_arrowedLine(im, ptA, ptB, width=1, color=(0,255,0)):
    """Draw line from ptA to ptB with arrowhead at ptB"""
    # Get drawing context
    draw = ImageDraw.Draw(im)
    # Draw the line without arrows
    draw.line((ptA,ptB), width=width, fill=color)

    # Now work out the arrowhead
    # = it will be a triangle with one vertex at ptB
    # - it will start at 95% of the length of the line
    # - it will extend 8 pixels either side of the line
    x0, y0 = ptA
    x1, y1 = ptB
    # Now we can work out the x,y coordinates of the bottom of the arrowhead triangle
    xb = 0.95*(x1-x0)+x0
    yb = 0.95*(y1-y0)+y0

    # Work out the other two vertices of the triangle
    # Check if line is vertical
    if x0==x1:
       vtx0 = (xb-5, yb)
       vtx1 = (xb+5, yb)
    # Check if line is horizontal
    elif y0==y1:
       vtx0 = (xb, yb+5)
       vtx1 = (xb, yb-5)
    else:
       alpha = math.atan2(y1-y0,x1-x0)-90*math.pi/180
       a = 8*math.cos(alpha)
       b = 8*math.sin(alpha)
       vtx0 = (xb+a, yb+b)
       vtx1 = (xb-a, yb-b)

    #draw.point((xb,yb), fill=(255,0,0))    # DEBUG: draw point of base in red - comment out draw.polygon() below if using this line
    #im.save('DEBUG-base.png')              # DEBUG: save

    # Now draw the arrowhead triangle
    draw.polygon([vtx0, vtx1, ptB], fill=color)
    return im

fx = 4 * 2000  # Focal length in pixels (x-axis)
fy = 4 * 1000  # Focal length in pixels (y-axis)
cx = 4 * 800  # Principal point (x-coordinate) in pixels
cy = 4 * 300  # Principal point (y-coordinate) in pixels

camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])

# def euler_to_rotation_matrix(roll, pitch, yaw):
#     r = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=True)
#     return r.as_matrix()

def make_plot(pts, des_pts, image):

    off = 1 - pts[:, 2][0]
    x = pts[:, 0] - 0.2
    y = pts[:, 2] + off
    z = pts[:, 1] + 1.0

    yb = (1.0 - y) / z
    xb = x / z
    zb = np.ones_like(xb)
    points = np.stack((xb, yb, zb), axis=-1)
    depths = np.linalg.norm(points * z[:, None], axis=-1)

    # fx = 500  # Focal length in pixels (x-axis)
    # fy = 500  # Focal length in pixels (y-axis)
    # cx = 640  # Principal point (x-coordinate) in pixels
    # cy = 480  # Principal point (y-coordinate) in pixels

    # camera_matrix = np.array([[fx, 0, cx],
    #                           [0, fy, cy],
    #                           [0, 0, 1]])

    base_pts = np.einsum('ik,lk->il', points, camera_matrix).astype(np.int32)

    for n in range(len(base_pts)):
        if 0 < base_pts[n, 1] < cy * 2 and 0 < base_pts[n, 0] < cx * 2:
            cv2.circle(image, (base_pts[n, 0], base_pts[n, 1]), 5, (255, 0, 0), -1)


    off = 1 - des_pts[:, 2][0]
    x = des_pts[:, 0] - 0.2
    y = des_pts[:, 2] + off
    z = des_pts[:, 1] + 1.0

    yb = (1.0 - y) / z
    xb = x / z
    zb = np.ones_like(xb)
    points = np.stack((xb, yb, zb), axis=-1)
    depths = np.linalg.norm(points * z[:, None], axis=-1)

    # fx = 500  # Focal length in pixels (x-axis)
    # fy = 500  # Focal length in pixels (y-axis)
    # cx = 640  # Principal point (x-coordinate) in pixels
    # cy = 480  # Principal point (y-coordinate) in pixels

    # camera_matrix = np.array([[fx, 0, cx],
    #                           [0, fy, cy],
    #                           [0, 0, 1]])

    base_des_pts = np.einsum('ik,lk->il', points, camera_matrix).astype(np.int32)

    for n in range(len(base_des_pts) - 1):
        if 0 < base_des_pts[n, 1] < cy * 2 and 0 < base_des_pts[n, 0] < cx * 2:
            cv2.line(image, (base_des_pts[n, 0], base_des_pts[n, 1]), (base_des_pts[n + 1, 0], base_des_pts[n + 1, 1]) , (255, 0, 255), 15)

    # cv2.imshow("abc", image.astype(np.uint8))
    # cv2.waitKey(0)

    return base_pts, image

def draw_axes(image, R, t, base_pts, camera_matrix):
    axis_length = 0.05
    origin = np.array([0, 0, 0, 1])
    axes = np.array([[axis_length, 0, 0],
                     [0, -axis_length, 0],
                     [0, 0, -axis_length]])
    
    rwik = (R.T @ axes).T
    rwik_ = rwik.copy()
    rwik_[1] = rwik[2]
    rwik_[2] = rwik[1]

    transformed_axes = (camera_matrix @ rwik_).T + base_pts
    projected_axes = transformed_axes[:, :2] / transformed_axes[:, 2][:, None]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR colors for XYZ axes

    ct = 0
    for p2, color in zip(projected_axes[:, :2], colors):
        if ct != 1 and ct!=2:
            cv2.arrowedLine(image, (int(base_pts[0]), int(base_pts[1])), (int(p2[0]), int(p2[1])), color, 12, cv2.LINE_8)
            
        ct += 1
    return image

def make_plot_base():
    num = 100
    x = np.linspace(-7, 7, num=num)
    y = np.linspace(2, 0, num=num)

    xx, yy = np.meshgrid(x, y)
    z = yy

    yb = 1.0 / z
    xb = xx / z
    zb = np.ones_like(xb)

    points = np.stack((xb, yb, zb), axis=-1)
    depths = np.linalg.norm(points * z[:, :, None], axis=-1)



    base_pts = np.einsum('ijk,lk->ijl', points, camera_matrix).astype(np.int32)

    image = np.zeros((cy * 2, cx * 2))

    for m in range(num):
        for n in range(num):
            if 0 < base_pts[m, n, 1] < cy * 2 and 0 < base_pts[m, n, 0] < cx * 2:
                image[base_pts[m, n, 1], base_pts[m, n, 0]] = depths[m, n]

    intensity = 0.25
    image = (image - image.min()) / (image.max() - image.min()) * 255 * intensity
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image

# ... (rest of the code)

if __name__ == "__main__":
    import argparse
    from plt_utils import load_cf_data

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    parser.add_argument("--runtime", type=float, default=2.0)
    parser.add_argument("--hovertime", type=float, default=4)
    parser.add_argument("-bh", "--baseheight", type=float, default=0.0)
    parser.add_argument("-tt", "--takeofftime", type=float, default=5.0)

    args = parser.parse_args()
    filenames = args.filename

    data_dict = load_cf_data(filenames, args)
    ims = []
    for key in data_dict.keys():
        im = make_plot_base()

        pose_positions = data_dict[key]['pose_positions']
        pose_orientations = data_dict[key]['pose_orientations']

        base_pts, im = make_plot(pose_positions, data_dict[key]['ref_positions'], im)
        for i in range(0, len(pose_positions), 3):
            pose_position = pose_positions[i]
            pose_orientation = np.radians(pose_orientations[i])  # Convert to radians

            # R = euler_to_rotation_matrix(*pose_orientation)

            rot_obj = Rotation.from_euler('zyx', pose_orientation)
            R = rot_obj.as_matrix()
            t = pose_position

            draw_axes(im, R, t, base_pts[i], camera_matrix)
        
        ims.append(im)
        print('yo')
    
    ovrl_img = np.hstack((ims[0], ims[1]))

    black_pixels = np.where(
    (ovrl_img[:, :, 0] == 0) & 
    (ovrl_img[:, :, 1] == 0) & 
    (ovrl_img[:, :, 2] == 0)
)

# set those pixels to white
    ovrl_img[black_pixels] = [255, 255, 255]
    cv2.imshow('ovrl', ovrl_img); cv2.waitKey(0)
    cv2.imwrite('out.png', ovrl_img)
    cv2.destroyAllWindows()
