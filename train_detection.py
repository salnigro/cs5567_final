import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision import transforms

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    # Load pre-trained Faster R-CNN
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Freeze backbone layers as per assignment instructions
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Transform: ToTensor handles scaling to [0,1]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    base_dir = r"c:\Users\salni\Documents\cs5567"
    
    try:
        from data_parser import MOT16DetectionDataset
        dataset = MOT16DetectionDataset(base_dir, subset="train", transforms=transform)
    except ImportError:
        print("MOT16DetectionDataset not found in data_parser. Skipping training.")
        return
        
    # DataLoader
    data_loader = DataLoader(
        dataset, batch_size=2, shuffle=True, 
        num_workers=0, collate_fn=collate_fn
    )
    
    # 2 classes: Background (0) and Pedestrian (1)
    model = get_model(num_classes=2)
    model.to(device)
    
    # Only fine-tune the heads
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params_to_optimize, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    num_epochs = 1  # For testing purposes, set to 1. Usually larger (e.g., 5-10)
    print(f"Starting training for {num_epochs} epoch(s)...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            
            # Handle empty targets by ensuring correct types 
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Fast-RCNN can throw errors on completely empty targets in training, 
            # ideally the dataset filters them out. (Our dataset filters out empty boxes in 'train' mode).
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            if i % 10 == 0:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {losses.item():.4f}")
                
            # Break early for testing the script
            if i >= 20: 
                break
                
        print(f"Epoch {epoch} finished. Avg Loss: {epoch_loss / (i+1):.4f}")
        
    # Save the model
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(base_dir, "models", "faster_rcnn_finetuned.pth"))
    print("Fine-tuned Faster R-CNN model saved!")

if __name__ == "__main__":
    main()
