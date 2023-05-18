import sys
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

import argparse

import subprocess

import torch
from torch import nn
import random
from sklearn.preprocessing import StandardScaler

######################################################"



print("you are in ",os.getcwd())

# setting up the config

class Detector:
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        self.thr = 0.9 #shorter var
        #self.cfg.MODEL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)
        
        self.last_file = "" #stores the name of the last file processed

        self.ann_model = NeuralNetwork(57,[128,64],1)
        self.ann_model.load_state_dict(torch.load("best_model.pt")["model_state_dict"])
        print("Setup Done")

    def process_image(self, im):
        # returns the processed image (can be shown with cv2.imshow())
        outputs = self.predictor(im)
        if len(outputs["instances"]) and all(list(i>self.thr for i in (outputs["instances"]).scores)): # if there is at least one instance and all the instances have a score > threshold
            v = Visualizer(im[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1) 
            out = v.draw_instance_predictions(outputs["instances"].to("cpu")) 
        return out.get_image()[:, :, ::-1] 
    
    def process_video(self,input_path):
        # input_path : path to video
        
        # return the name of the input file and the associated file
        name = os.path.basename(input_path) # get the name of the file without the extension
        
        self.last_file = name # store it in the Detector object
        
        cap = cv2.VideoCapture(input_path)

        boxes = []
        segments = []
        keypoints = []

        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
            print("Tried to open",input_path)
            return 

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                if not(frame is None):
                    metadata = {'w': frame.shape[1],'h': frame.shape[0]}
                # do something on frame
                outputs = self.predictor(frame)['instances'].to('cpu')
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
        
        # can not be saved somewhere else.
        # format data_2d_custom_processed_*example*.npz
        np.savez_compressed(f"VideoPose3D/data/{output_prefix_2d+name}.npz", 
        positions_2d=output, metadata=coco_metadata)
                
        print("Dataset ready for inference 3d")
        
        
        return name
        
    def run_pose3d(self):
        name = self.last_file
        PATH_TO_ELT = f"../current/input/{name}"
        PATH_FINAL_VID = f"../current/output/vis_{name}.mp4"
        PATH_FINAL_ARR = f"../current/output/arr_{name}.npy"

        args =[
        "-d", "custom",
        "-k", name,
        "-arc", "3,3,3,3,3",
        "-c", "checkpoint",
        "--evaluate", "pretrained_h36m_detectron_coco.bin",
        "--render", 
        "--viz-subject", name,
        "--viz-action", "custom",
        "--viz-camera", "0",
        "--viz-video", PATH_TO_ELT,
        "--viz-export", PATH_FINAL_ARR,
        "--viz-output", PATH_FINAL_VID,
        "--viz-size", "6"
        ] 

        command = ["python", "run.py"] + args 

        subprocess.run(command, check=True,cwd='VideoPose3D/') # run the command in the VideoPose3D folder
    
    def apply_ann(self):
        # returns the mean of the 5 predictions
        results_3D = np.load(f"current/output/arr_{self.last_output}.npy")
        assert results_3D.shape[0]>21
        X_processed = feature_creation(results_3D)
        
        shift = list(range(-10,10))
        
        n = X_processed.shape[0]
        L_results = []
        for elt_shift in shift:
            x_tensor = torch.from_numpy(X_processed[n//2+elt_shift]) # take the middle frames 
            L_results.append(float(self.ann_model(x_tensor))) 
        return np.mean(L_results) # return the mean of the 21 predictions
        

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim[0])
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.layer_3 = nn.Linear(hidden_dim[1], output_dim)
        self.sigmoid = nn.Sigmoid()
       
    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.sigmoid(self.layer_3(x))
        return x
   

def feature_creation(array):

    poses = array

    joint_sets = [
            # left arm
            (4, 3, 2),
            # right arm
            (7, 6, 5),
            # left leg
            (10, 11, 12),
            # right leg
            (13, 14, 15),
            # neck
            (1, 0, 16),
            # back
            (8, 1, 0)
    ]

    positions_and_angles_list = []

    for _, pose in enumerate(poses):

            keypoints = pose.reshape(-1, 17, 3)

            angles = []
            for frame in keypoints:
                    frame_angles = []
                    for joint_set in joint_sets:
                            vector1 = frame[joint_set[1]] - frame[joint_set[0]]
                            vector2 = frame[joint_set[2]] - frame[joint_set[1]]
                            cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
                            angle = np.arccos(cos_angle)
                            frame_angles.append(angle)
                    angles.append(frame_angles)

            positions = []
            for frame in keypoints:
                    frame_positions = []
                    for joint in frame:
                            frame_positions.extend(joint)
                    positions.append(frame_positions)

            positions_and_angles = np.concatenate([angles, positions], axis=1)
        
                    
            positions_and_angles_list.append(np.concatenate([positions_and_angles], axis=1))

    posang = np.concatenate(positions_and_angles_list) # shape (n, 57)

    scaler = StandardScaler()
    X = scaler.fit_transform(posang)

    return X

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path","-p", type=str, default="current/input/video_standing_good.mp4",help="path to the video to process")
    args = parser.parse_args()
    
    path_to_process = args.path
    
    detector = Detector()
        
    print(detector.process_video(path_to_process))
    detector.run_pose3d()
    print(detector.apply_ann())