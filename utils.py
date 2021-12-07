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
        ori_transform = cv2.getAffineTransform(np.float32(ave_tri), np.float32(ori_tri))
        new_transform = cv2.getAffineTransform(np.float32(ave_tri), np.float32(new_tri))
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
    cv2.imwrite('./images/outputs/{}/frame_{}.png'.format(group, str(frame_num)), frame)

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
