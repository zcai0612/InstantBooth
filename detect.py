import cv2
import mmcv
import os
import time
from tqdm import tqdm, trange
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmengine.visualization import Visualizer
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

vis = Visualizer()

images_dir = './images'
pbar = tqdm(os.listdir(images_dir))
for image_file in pbar:
    image_path = os.path.join(images_dir, image_file)
    image = mmcv.imread(image_path)
    image = mmcv.imconvert(image, 'bgr', 'rgb')
    result = inference_detector(model=model, imgs=image)

    boxes = result.pred_instances.bboxes
    scores = result.pred_instances.scores
    idxs = result.pred_instances.labels
    one_person, person_box = judge_one_person(idxs=idxs, boxes=boxes)

    if one_person:
        one_face, face_box = judge_one_face(mtcnn_model, image_path=image_path)
        if one_face:
            # vis.set_image(image)
            # vis.draw_bboxes(person_box)
            # vis.draw_bboxes(face_box)
            # vis.show()
            print('{} is one person!'.format(image_file))
            continue
    print('{} is not one person...'.format(image_file))



