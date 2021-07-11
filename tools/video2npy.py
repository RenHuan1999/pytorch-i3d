import numpy as np
import cv2
import os

read_path = "/home2/Dataset/FineAction/resize_video_224x224/"
save_path = "/home2/Dataset/FineAction/video_npy_224x224/"

video_name = os.listdir(read_path)
video_name.sort()

for name in video_name:
    if int(name.split('.')[0][-6:]) < 0:
        continue
    frames = []
    cap = cv2.VideoCapture(os.path.join(read_path, name))
    while True:
        ret, img = cap.read()   # ret 读取了数据就返回 True，没有读取数据(已到尾部)就返回 False
        if ret is False:
            break
        img = np.array(img)[:, :, ::-1]
        frames.append(img)
    frames = np.stack(frames, 0)
    print('process {}'.format(name))

    # save to npy file
    # np.save(os.path.join(save_path, name), frames)
    # np.savez(os.path.join(save_path, name), frames)
    np.savez_compressed(os.path.join(save_path, name), frames=frames)


