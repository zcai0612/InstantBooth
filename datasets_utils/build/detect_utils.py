import torch
from face_alignment import align
from PIL import Image
# input: 
#   - idx: torch.Size([N])
#   - boxes: torch.Size([N, 4])
# output:
#   - Bool: is/not one person
#   - boxes: person box - torch.Size([1, 4])
def judge_one_person(idxs, boxes):
    num_person = (idxs==0).sum()
    if num_person != 1:
        return (False, None)
    person_index = torch.nonzero(idxs==0).squeeze()
    person_box = boxes[person_index,:]
    return (True, person_box)

# model: MTCNN - mtcnn_model = mtcnn.MTCNN(device='cuda:0', crop_size=(112, 112))
def judge_one_face(model, image_path):
    image = Image.open(image_path)
    min_face_size = 20
    thresholds =  [0.6,0.7,0.9]
    nms_thresholds = [0.7, 0.7, 0.7]
    factor = 0.85
    boxes, _ = model.detect_faces(image, min_face_size, 
                                             thresholds, nms_thresholds, 
                                             factor)
    num_faces = boxes.shape[0]
    if num_faces == 1:
        return True
    else:
        return False
    

    
