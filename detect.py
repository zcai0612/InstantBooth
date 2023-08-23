import cv2
import mmcv
import os
import time
from tqdm import tqdm, trange
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
from face_alignment import mtcnn
from datasets_utils.build.detect_utils import judge_one_face, judge_one_person

# 指定模型的配置文件和 checkpoint 文件路径
config_file = 'configs/detection/rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'weights/detection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')
mtcnn_model = mtcnn.MTCNN(device='cuda:0', crop_size=(112, 112))

images_dir = './images'
pbar = tqdm(os.listdir(images_dir))
for img in pbar:
    s_time = time.time()
    image_path = os.path.join(images_dir, img)
    result = inference_detector(model=model, imgs=image_path)
    boxes = result.pred_instances.bboxes
    scores = result.pred_instances.scores
    idxs = result.pred_instances.labels
    one_person, person_box = judge_one_person(idxs=idxs, boxes=boxes)

    if one_person:
        one_face = judge_one_face(mtcnn_model, image_path=image_path)
        if one_face:
            print('{} is one person!'.format(img))
            e_time = time.time()
            print(e_time - s_time)
            continue
    print('{} is not one person...'.format(img))

    e_time = time.time()
    print(e_time - s_time)



