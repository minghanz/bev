import numpy as np
import cv2
import os
from tqdm import tqdm

'''from bts/c3d_loss.py'''
def read_txt_to_dict(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    for files with "names: value" structure
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            ### skip blank line
            if len(line) <=1:
                continue
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

def read_txt_to_array(path):
    """
    for files with pure numbers
    """
    with open(path) as f:
        lines = f.readlines()
        array = np.array([ [float(x) for x in line.split()] for line in lines ])
    return array

def write_array_to_txt(path, array):
    with open(path, "w") as f:
        if array.ndim == 1:
            f.write(" ".join(str(x) for x in array))
        elif array.ndim == 2:
            for i in range(array.shape[0]):
                f.write(" ".join(str(x) for x in array[i]) + "\n")
        else:
            raise ValueError("not recognized shape:", array.shape)

def write_dict_to_txt_item(f, array, name):
    ### write_np_to_txt_like_kitti
    if isinstance(array, (list, tuple, np.ndarray)):
        if isinstance(array, np.ndarray):
            array = array.reshape(-1)
        f.write("{}: ".format(name) + " ".join(str(x) for x in array) + "\n" )
    else:
        f.write("{}: ".format(name) + str(array) + "\n" )

def write_dict_to_txt(path, dict_obj):
    with open(path, "w") as f:
        for key, value in dict_obj.items():
            write_dict_to_txt_item(f, value, key)
    return


def video_generator(video_name, height=None, width=None, img_folder=None, fps=30, samp_rate=1, dim_from=None):
    """dim_from takes a path and set the output video width and height from this video. """
    output_dirname = os.path.dirname(video_name)
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)

    if height is None and width is None:
        if dim_from is not None:
            video_as = cv2.VideoCapture(dim_from)
            width  = int(video_as.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
            height = int(video_as.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 
            video_as.release()
        elif img_folder is not None:
            img0 = os.listdir(img_folder)[0]
            img_path = os.path.join(img_folder, img0)
            img_f = cv2.imread(img_path)
            width = img_f.shape[1]
            height = img_f.shape[0]
        else:
            raise ValueError("If height and width are not given, dim_from or img_folder must be given to specify the video dimension")


    video = cv2.VideoWriter(
                filename=video_name,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                # fourcc = cv2.VideoWriter_fourcc('M','J','P','G'),cv2.VideoWriter_fourcc(*"x264"),
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(fps),
                frameSize=(width, height),
                isColor=True,
            )

    if img_folder is not None:
        imgs = os.listdir(img_folder)
        imgs = sorted(imgs)

        for i, img in enumerate(tqdm(imgs)):
            if i % samp_rate == 0:
                img_path = os.path.join(img_folder, img)
                img_f = cv2.imread(img_path)
                video.write(img_f)

        video.release()
        return
    else:
        return video

def frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

def video_parser(video_name, img_folder=None, start_id=-1, end_id=-1, samp_rate=1 ):
    """start_id and end_id are only effective when img_folder is not None. 
    yield image and frame_id"""
    video = cv2.VideoCapture(video_name)
    if start_id > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, start_id)

        # frameRate = video.get(cv2.CAP_PROP_FPS)
        # time_ms = start_id / frameRate * 1000
        # video.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        print("start ms frame:", video.get(cv2.CAP_PROP_POS_MSEC), video.get(cv2.CAP_PROP_POS_FRAMES))
    for i, im in enumerate(tqdm(frame_from_video(video))):
        # if i < start_id:
        #     continue
        if end_id > 0 and i >= end_id-start_id:
            break
        if (i-start_id) % samp_rate != 0:
            continue
        if img_folder is not None:
            fname = '{:010d}.jpg'.format(i)
            cv2.imwrite(os.path.join(img_folder, fname), im)
            # print('finished:', i)
        else:
            yield im, i

    video.release()

def folder_parser(path):
    """yield image and file path"""
    imgs = os.listdir(path)
    imgs = sorted(imgs)
    img_paths = [os.path.join(path, x) for x in imgs]
    for ipath in tqdm(img_paths):
        frame = cv2.imread(ipath)
        yield frame, ipath