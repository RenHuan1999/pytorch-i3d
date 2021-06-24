import os
import os.path
import json

import cv2

from joblib import Parallel, delayed

video_path = '/data/Dataset/FineAction-video/video_data_end/'
images_path = '/data/Dataset/FineAction-video/images/'
os.makedirs(images_path, exist_ok=True)

with open('./fineaction/annotations_gt.json', 'r') as f:
    data = json.load(f)
data = data['database']

subset = ['training', 'validation']

def video_to_images(vid):
    if data[vid]['subset'] not in subset:
        return
    output_path = os.path.join(images_path, vid)
    if os.path.exists(output_path) and len(os.listdir(output_path))==data[vid]['actual_frame_num']:
        return
    os.makedirs(output_path, exist_ok=True)

    print('Processing %s' % vid)
    video = os.path.join(video_path, data[vid]['filename'])
    if not os.path.exists(video):
        raise ValueError('There is no %s' % video)
    num_frames = data[vid]['actual_frame_num']
    cap = cv2.VideoCapture(video)
    for i in range(1, num_frames+1):
        ret, img = cap.read()   # ret 读取了数据就返回 True，没有读取数据(已到尾部)就返回 False
        if not ret:
            raise ValueError('Can\'t get %d frames' % num_frames)
        cv2.imwrite(os.path.join(output_path, vid+'-'+str(i).zfill(6)+'.jpg'), img)
    cap.release()
    return

# transfer the video to images in parallel
parallel = Parallel(n_jobs=1, prefer="processes")
detection = parallel(delayed(video_to_images)(vid) for vid in data.keys())


