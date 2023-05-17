import sys
# must be changed !
lib_path = "/root/A3/detectron2"
if lib_path not in sys.path:
    sys.path.append(lib_path)

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import cv2
import numpy as np
import os

import uuid

import subprocess

print("you are in ",os.getcwd())

# setting up the config
cfg = get_cfg()  
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9 
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")


predictor = DefaultPredictor(cfg)

def process(input_path):
    # input_path : path to video
    
    cap = cv2.VideoCapture(input_path)
    global predictor

    boxes = []
    segments = []
    keypoints = []

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        return

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if not(frame is None):
                metadata = {'w': frame.shape[1],'h': frame.shape[0]}
            # do something on frame
            outputs = predictor(frame)['instances'].to('cpu')
            has_bbox = False
            if outputs.has('pred_boxes'):
                bbox_tensor = outputs.pred_boxes.tensor.numpy()
                if len(bbox_tensor) > 0:
                    has_bbox = True
                    scores = outputs.scores.numpy()[:, None]
                    bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1)
            if has_bbox:
                kps = outputs.pred_keypoints.numpy()
                kps_xy = kps[:, :, :2]
                kps_prob = kps[:, :, 2:3]
                kps_logit = np.zeros_like(kps_prob) # Dummy
                kps = np.concatenate((kps_xy, kps_logit, kps_prob), axis=2)
                kps = kps.transpose(0, 2, 1)
            else:
                kps = []
                bbox_tensor = []

            # Mimic Detectron1 format
            cls_boxes = [[], bbox_tensor]
            cls_keyps = [[], kps]

            boxes.append(cls_boxes)
            segments.append(None)
            keypoints.append(cls_keyps)
        else: 
            break

    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    
    name = "processed"+str(uuid.uuid4())
    
    #metadata = np.array(metadata,dtype=object)
    """np.savez_compressed(file=f"current/middle/{name}.npz",keypoints=keypoints,
                        metadata=metadata,boxes=boxes,segments=segments)"""
    
    # var :keypoints,boxes, metadata
    print("Processed video with detectron2")
    
    bb = boxes
    kp = keypoints

    results_bb = []
    results_kp = []
    for i in range(len(bb)):
        if len(bb[i][1]) == 0 or len(kp[i][1]) == 0:
            # No bbox/keypoints detected for this frame -> will be interpolated
            results_bb.append(np.full(4, np.nan, dtype=np.float32)) # 4 bounding box coordinates
            results_kp.append(np.full((17, 4), np.nan, dtype=np.float32)) # 17 COCO keypoints
            continue
        best_match = np.argmax(bb[i][1][:, 4])
        best_bb = bb[i][1][best_match, :4]
        best_kp = kp[i][1][best_match].T.copy()
        results_bb.append(best_bb)
        results_kp.append(best_kp)
        
    bb = np.array(results_bb, dtype=np.float32)
    kp = np.array(results_kp, dtype=np.float32)
    kp = kp[:, :, :2] # Extract (x, y)
    
    # Fix missing bboxes/keypoints by linear interpolation
    mask = ~np.isnan(bb[:, 0])
    indices = np.arange(len(bb))
    for i in range(4):
        bb[:, i] = np.interp(indices, indices[mask], bb[mask, i])
    for i in range(17):
        for j in range(2):
            kp[:, i, j] = np.interp(indices, indices[mask], kp[mask, i, j])
    
    print('{} total frames processed'.format(len(bb)))
    print('{} frames were interpolated'.format(np.sum(~mask)))
    
    print('----------')
   
    
    coco_metadata = {
    'layout_name': 'coco',
    'num_joints': 17,
    'keypoints_symmetry': [
        [1, 3, 5, 7, 9, 11, 13, 15],
        [2, 4, 6, 8, 10, 12, 14, 16],
                        ]
        }
  
    
    
    
    output = {
        name:{
        "custom":[kp.astype("float32")]
               }
               }

    
    coco_metadata['video_metadata'] = {name:metadata}
    
    print('Saving...')
    
    output_prefix_2d = 'data_2d_custom_'
    
    np.savez_compressed(f"VideoPose3D/data/{output_prefix_2d+name}.npz", 
    positions_2d=output, metadata=coco_metadata)
    
    #np.savez_compressed(f"current/output/{name}.npz", positions_2d=output, metadata=metadata)
    
    print('Done.')
    
    print("Dataset ready for inference 3d")
    
    # return the name of the input file and the associated file
    file_name = os.path.basename(input_path)
    
    return {"input":file_name,
            "output":name,
            }
    
    

    
# change path here
dict_dataset = process("current/input/video_standing_good.mp4")

print(dict_dataset)

print("Infering 3D")


PATH_TO_ELT = f"../current/input/{dict_dataset['input']}"
PATH_FINAL_VID = f"../current/output/vis_{dict_dataset['output']}.mp4"
PATH_FINAL_ARR = f"../current/output/arr_{dict_dataset['output']}.npy"

args =[
  "-d", "custom",
  "-k", dict_dataset["output"],
  "-arc", "3,3,3,3,3",
  "-c", "checkpoint",
  "--evaluate", "pretrained_h36m_detectron_coco.bin",
  "--render", 
  "--viz-subject", dict_dataset["output"],
  "--viz-action", "custom",
  "--viz-camera", "0",
  "--viz-video", PATH_TO_ELT,
  "--viz-export", PATH_FINAL_ARR,
  "--viz-output", PATH_FINAL_VID,
  "--viz-size", "6"
]

command = ["python", "run.py"] + args

subprocess.run(command, check=True,cwd='VideoPose3D/')
