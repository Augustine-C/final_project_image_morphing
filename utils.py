import cv2
import numpy as np

from PIL import Image
from skimage import draw
import os
from multiprocessing import Pool



def get_frame_at_t(ori_tris, new_tris, t, ori_img, new_img, positions):
    frame = np.zeros(ori_img.shape, dtype='uint8')
    num_triangles = len(ori_tris)
    for i in range(num_triangles):
        ori_tri = ori_tris[i]
        new_tri = new_tris[i]
        ave_tri = ori_tri * (1-t) + t * new_tri
        ori_transform = get_transformation(np.float32(ave_tri), np.float32(ori_tri))
        new_transform = get_transformation(np.float32(ave_tri), np.float32(new_tri))
        ave_points = positions[get_mask(ave_tri, ori_img)]
        
        for point in ave_points:
            ori_point = np.dot(ori_transform, np.array([point[0], point[1], 1]))
            new_point = np.dot(new_transform, np.array([point[0], point[1], 1]))
            frame[point[1], point[0]] = ((1-t) * ori_img[int(ori_point[1]), int(ori_point[0])] \
                + new_img[int(new_point[1]), int(new_point[0])] * t).round()
    return frame

def generate_imgs(ori_tris, new_tris, t, ori_img,new_img,positions,frame_num, group):
    frame = get_frame_at_t(ori_tris, new_tris, t, ori_img, new_img, positions)
    os.makedirs('./images/outputs/{}/'.format(group), exist_ok = True)
    cv2.imwrite('./images/outputs/{}/frame_{}.jpg'.format(group, str(frame_num)), frame)

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def get_mask(triangle, img):
    ys = triangle[:,1]
    xs = triangle[:,0]
    mask = poly2mask(ys, xs, img.shape[:2])
#     fig = plt.figure()
#     plt.imshow(np.uint8(mask), cmap='gray')
    return mask

def get_transformation(t1,t2):
    A = np.zeros([3,3])
    for i in range(3):
        A[i,0:2] = t1[i]
        A[i,2] = 1
    x = t2
    result = np.linalg.lstsq(A,x)[0]
    return result.transpose()

def imageFolder2mpeg(input_path, output_path='./output_video.mpeg', fps=30.0):
    '''
    Extracts the frames from an input video file
    and saves them as separate frames in an output directory.
    Adapted from CS445-MP5
    Input:
        input_path: Input video file.
        output_path: Output directorys.
        fps: frames per second (default: 30).
    Output:
        None
    '''

    dir_frames = input_path
    files_info = os.scandir(dir_frames)

    file_names = [f.path for f in files_info if f.name.endswith(".jpg")]
    file_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    frame_Height, frame_Width = cv2.imread(file_names[0]).shape[:2]
    resolution = (frame_Width, frame_Height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MPG1')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    frame_count = len(file_names)

    frame_idx = 0

    while frame_idx < frame_count:
        frame_i = cv2.imread(file_names[frame_idx])
        video_writer.write(frame_i)
        frame_idx += 1

    video_writer.release()

def prompt_eye_selection(image):
    fig = plt.figure()
    plt.imshow(image, cmap='gray')
    fig.set_label('Click on ten points for alignment')
    plt.axis('off')
    xs = []
    ys = []
    clicked = np.zeros((8, 2), dtype=np.float32)

    # Define a callback function that will update the textarea
    def onmousedown(event):
        x = event.xdata
        y = event.ydata
        xs.append(x)
        ys.append(y)

        plt.plot(xs, ys, 'r-+')

    def onmouseup(event):
        if(len(xs) >= 8):
            plt.close(fig)

    def onclose(event):
        clicked[:, 0] = xs
        clicked[:, 1] = ys
    # Create an hard reference to the callback not to be cleared by the garbage
    # collector
    fig.canvas.mpl_connect('button_press_event', onmousedown)
    fig.canvas.mpl_connect('button_release_event', onmouseup)
    fig.canvas.mpl_connect('close_event', onclose)

    return clicked

def get_perpendicular_2d(a) :
    b = np.zeros(a.shape)
    b[0] = -1 * a[1]
    b[1] = a[0]
    return b


def generate_frames_feature_based(frame_idx, num_frames, img, img2, p, q, p_prime, q_prime):
    output_subdir = 'feature_based_3'
    os.makedirs('./images/outputs/{}/'.format(output_subdir), exist_ok = True)
    # assuming the two image is of the same size
    img_frame = np.zeros(img.shape)
    # linear interpolation of line segments
    delta_p = (p_prime-p) / (num_frames+1)
    delta_q = (q_prime-q) / (num_frames+1)

    p_is = p + delta_p * (frame_idx + 1)
    q_is = q + delta_q * (frame_idx + 1)
    # iterate through each pixel locations
    for row in range(img.shape[0]):
        # if row % 100 == 0:
        #     print('100 rows')
        for col in range(img.shape[1]):
            # X is the pixel index in (x, y)
            x = np.array([col, row])
            d_sum_from_src = np.zeros((1, 2), dtype=np.float32)
            d_sum_from_target = np.zeros((1, 2), dtype=np.float32)
            weight_sum = 0
            # iterate each line segments p_i->q_i
            for line_idx in range(p_is.shape[0]):
                p_i = p_is[line_idx]
                q_i = q_is[line_idx]
                p_src_i = p[line_idx]
                p_prime_i = p_prime[line_idx]
                q_src_i = q[line_idx]
                q_prime_i = q_prime[line_idx]
                # obtain u, v, X' based on interpolated line segments
                lambda_i = np.dot(x - p_i, q_i - p_i)/ np.linalg.norm(q_i - p_i)
                u = lambda_i / np.linalg.norm(q_i - p_i)
                v = np.dot(x-p_i, get_perpendicular_2d(q_i - p_i)) / np.linalg.norm(q_i - p_i)
                x_prime_from_src = p_src_i + np.dot(u, q_src_i - p_src_i) + \
                          (np.dot(v, get_perpendicular_2d(q_src_i- p_src_i)) \
                                                                    /np.linalg.norm(q_src_i - p_src_i))
                x_prime_from_target = p_prime_i + np.dot(u, q_prime_i - p_prime_i) + \
                          (np.dot(v, get_perpendicular_2d(q_prime_i- p_prime_i)) \
                                                                /np.linalg.norm(q_prime_i - p_prime_i))
                # get the shortest distance from the pixel to the directed line segments
                # need to depend on value of u
                if u < 0:
                    dist = np.linalg.norm(p_i-x)
                elif u > 1:
                    dist = np.linalg.norm(q_i-x)
                else:
                    dist = abs(v)
                # weight that each line have. a, b, and p are parameters
                a, b, p_influence = 0.3, 1.3, 0.1
                line_length = np.linalg.norm(q_i - p_i)
                weight = ((line_length**p_influence)/(a+dist))**b
                d_sum_from_src += weight * (x_prime_from_src - x)
                d_sum_from_target += weight * (x_prime_from_target - x)
                weight_sum += weight
            # recompute locations with the effect of all lines
            x_prime_from_src = x + d_sum_from_src / weight_sum
            x_prime_from_target = x + d_sum_from_target / weight_sum
            x_prime_src_x, x_prime_src_y = int(x_prime_from_src[0,0]), int(x_prime_from_src[0,1])
            x_prime_target_x, x_prime_target_y = int(x_prime_from_target[0,0]), int(x_prime_from_target[0,1])
            # todo: handle non-integer index
            # todo: handle out-of-boundary index

            # source image
            new_pixel_from_source = img[col, row]
            if 0 <= x_prime_src_x < img.shape[1] and 0 <= x_prime_src_y < img.shape[0]:
                # index within image range, get the corresponding pixel value
                new_pixel_from_source = img[x_prime_src_y, x_prime_src_x]
            # dest image
            new_pixel_from_dest = img2[col, row]
            if 0 <= x_prime_target_x < img.shape[1] and 0 <= x_prime_target_y < img.shape[0]:
                # index within image range
                new_pixel_from_dest = img2[x_prime_target_y, x_prime_target_x]

            #print(new_pixel_from_dest)
            # print(x_prime_src_x)
            # print(x_prime_target_x)
            # cross-dissolve the pixels
            t = frame_idx / num_frames
            img_frame[row, col] = (1-t) * new_pixel_from_source + t * new_pixel_from_dest
    print('writing to files')
    cv2.imwrite('./images/outputs/{}/frame_{}.jpg'.format(output_subdir, str(frame_idx)), img_frame)
