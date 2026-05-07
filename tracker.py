import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from train_reid import Siamese_Network
from train_detection import get_model
from scipy.optimize import linear_sum_assignment

def load_models(base_dir, device):
    """
    Loads fine-tuned Faster R-CNN and Siamese Network.
    Falls back to base pre-trained models if fine-tuned weights aren't found.
    """
    # 1. Faster R-CNN
    det_path = os.path.join(base_dir, "models", "faster_rcnn_finetuned.pth")
    if os.path.exists(det_path):
        print(f"Loading fine-tuned detection model from {det_path}...")
        det_model = get_model(num_classes=2)
        det_model.load_state_dict(torch.load(det_path, map_location=device))
    else:
        print("Fine-tuned detection model not found. Using pre-trained weights for demo...")
        det_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    det_model.to(device)
    det_model.eval()

    # 2. Siamese Network
    reid_path = os.path.join(base_dir, "models", "siamese_reid.pth")
    reid_model = Siamese_Network()
    if os.path.exists(reid_path):
        print(f"Loading trained Siamese Re-ID model from {reid_path}...")
        reid_model.load_state_dict(torch.load(reid_path, map_location=device))
    else:
        print("Trained Siamese Re-ID model not found. Using randomly initialized weights for demo...")
        
    reid_model.to(device)
    reid_model.eval()
    
    return det_model, reid_model


def get_crop(img_pil, bbox, img_size=(24, 24)):
    """Extracts a crop from PIL image."""
    x1, y1, x2, y2 = bbox
    img_width, img_height = img_pil.size
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img_width, int(x2)), min(img_height, int(y2))
    
    if x2 <= x1 or y2 <= y1:
        crop = img_pil.crop((0, 0, 10, 10))
    else:
        crop = img_pil.crop((x1, y1, x2, y2))
        
    crop = crop.resize(img_size)
    return crop


def run_tracker(seq_dir, output_video_path, output_txt_path, det_model, reid_model, device):
    img_dir = os.path.join(seq_dir, 'img1')
    if not os.path.exists(img_dir):
        print(f"Not found: {img_dir}")
        return

    frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    if not frames:
        print("No frames found!")
        return

    # To process images for detection
    det_transform = transforms.ToTensor()
    # To process crops for Re-ID (Siamese Network expects 1 channel Grayscale 24x24)
    reid_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])

    # Video Writer setup
    sample_img = cv2.imread(os.path.join(img_dir, frames[0]))
    height, width, _ = sample_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    active_tracks = {}   # Mapping from track_id to tensor representation
    next_track_id = 1
    
    # We will compute a similarity matrix threshold. 0.5 means they are similar enough to match.
    MATCH_THRESH = 0.5
    
    results = []

    for frame_idx, frame_name in enumerate(frames):
        img_path = os.path.join(img_dir, frame_name)
        img_cv = cv2.imread(img_path)
        img_pil = Image.open(img_path).convert("RGB")
        
        # Detection
        img_tensor = det_transform(img_pil).to(device)
        with torch.no_grad():
            preds = det_model([img_tensor])[0]
            
        boxes = preds['boxes'].cpu().numpy()
        scores = preds['scores'].cpu().numpy()
        labels = preds['labels'].cpu().numpy()
        
        # Filter detections: typically pedestrian class is 1 in fine-tuned model, or 1 in COCO defaults
        det_boxes = []
        for b, s, l in zip(boxes, scores, labels):
            if s > 0.7 and l == 1:
                det_boxes.append(b)
                
        # Crops for each detection
        curr_detections_features = []
        for b in det_boxes:
            crop_pil = get_crop(img_pil, b)
            crop_tensor = reid_transform(crop_pil).unsqueeze(0).to(device)
            curr_detections_features.append(crop_tensor)
            
        n_det = len(curr_detections_features)
        n_trk = len(active_tracks)
        
        assigned_ids = [-1] * n_det
        
        if n_det > 0 and n_trk > 0:
            # Build cost matrix
            cost_matrix = np.zeros((n_det, n_trk))
            track_keys = list(active_tracks.keys())
            
            # This is O(N*M) inferences, can be optimized by batching if needed
            with torch.no_grad():
                for i, det_feat in enumerate(curr_detections_features):
                    for j, trk_id in enumerate(track_keys):
                        trk_feat = active_tracks[trk_id]
                        score = reid_model(det_feat, trk_feat)
                        score_val = score.item()
                        # Cost is 1 - similarity
                        cost_matrix[i, j] = 1.0 - score_val
                        
            # Hungarian Match
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < (1.0 - MATCH_THRESH):
                    assigned_ids[i] = track_keys[j]
                    # Update active track representation to adapt to appearance changes
                    active_tracks[track_keys[j]] = curr_detections_features[i]
                    
        # Handle Unassigned Detections
        for i in range(n_det):
            if assigned_ids[i] == -1:
                assigned_ids[i] = next_track_id
                active_tracks[next_track_id] = curr_detections_features[i]
                next_track_id += 1
                
        # Draw on Frame and collect results
        for b, obj_id in zip(det_boxes, assigned_ids):
            x1, y1, x2, y2 = map(int, b)
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, f"ID: {obj_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # MOT16 Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
            # frame_idx is 0-indexed, but MOT is 1-indexed. obj_id is already 1-indexed.
            results.append(f"{frame_idx + 1},{obj_id},{x1},{y1},{w},{h},1,-1,-1,-1\n")
                        
        out.write(img_cv)
        if frame_idx % 20 == 0:
            print(f"Processed {frame_idx}/{len(frames)} frames. Active Tracks: {len(active_tracks)}")
            
        # Optional: break early for quick demo testing
        # if frame_idx >= 50:
        #     break
            
    out.release()
    print(f"Tracking video saved to {output_video_path}")
    
    # Save text results
    with open(output_txt_path, 'w') as f:
        f.writelines(results)
    print(f"Tracking results saved to {output_txt_path}")


if __name__ == "__main__":
    base_dir = r"c:\Users\salni\Documents\cs5567"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print("Loading models...")
    det_model, reid_model = load_models(base_dir, device)
    
    train_seqs_dir = os.path.join(base_dir, "train")
    # For evaluation we'll track a train sequence since test sequences lack ground truth
    if os.path.exists(train_seqs_dir):
        # We will use MOT16-02 for evaluation
        test_seq = "MOT16-02"
        seq_dir = os.path.join(train_seqs_dir, test_seq)
        
        if os.path.exists(seq_dir):
            output_video = os.path.join(base_dir, f"{test_seq}_output.mp4")
            output_txt = os.path.join(base_dir, f"{test_seq}_results.txt")
            
            print(f"Starting tracking on {test_seq}...")
            run_tracker(seq_dir, output_video, output_txt, det_model, reid_model, device)
            print("Done!")
        else:
            print(f"Sequence {test_seq} not found in {train_seqs_dir}.")
    else:
        print("Train directory not found.")
