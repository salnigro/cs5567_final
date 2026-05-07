import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class Siamese_Network(nn.Module):
    def __init__(self):
        super(Siamese_Network, self).__init__()
        # CNN layers for feature extraction
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        
        # Adaptation: For 24x24 input:
        # 24x24 -> conv1(k=3) -> 22x22 -> pool -> 11x11
        # 11x11 -> conv2(k=3) -> 9x9 -> pool -> 4x4
        # 4x4 -> conv3(k=3) -> 2x2
        # Size = 128 channels * 2 * 2 = 512
        self.fc_input_dim = 128 * 2 * 2 
        
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.fc2 = nn.Linear(256, 1)  # 1 output for similarity score (0 to 1)

    def forward_one(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        # Note: slide had fc2(x) returning 256 for a contrastive loss perhaps, 
        # but 1 output logic is easier for Binary Cross Entropy. We will output 1 dim and apply Sigmoid.
        x = self.fc2(x) 
        return x

    def forward(self, input1, input2):
        # In standard Siamese with BCE, we compute features then measure absolute difference.
        # However, the slide said "forward_one(input1) \n output1, output2"
        # We can do |out1 - out2| -> fully connected -> score, OR Contrastive Loss.
        # Let's use absolute difference followed by a linear layer as a simple tracker.
        
        out1 = self.forward_one(input1)
        out2 = self.forward_one(input2)
        
        # Output probability using Sigmoid
        score = torch.sigmoid(torch.abs(out1 - out2))
        return score

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Needs to be Grayscale because Siamese_Network expects 1 channel
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])
    
    base_dir = r"c:\Users\salni\Documents\cs5567"
    
    try:
        from data_parser import MOT16ReIDDataset
        dataset = MOT16ReIDDataset(base_dir, subset="train", img_size=(24, 24), transform=transform)
    except ImportError:
        print("MOT16ReIDDataset not found in data_parser. Skipping training.")
        return
        
    data_loader = DataLoader(
        dataset, batch_size=32, shuffle=True, 
        num_workers=0
    )
    
    model = Siamese_Network()
    model.to(device)
    
    # Binary Cross Entropy Loss for 0/1 similarity
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 1
    print(f"Starting Re-ID training for {num_epochs} epoch(s)...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (in1, in2, label) in enumerate(data_loader):
            in1, in2, label = in1.to(device), in2.to(device), label.to(device)
            
            optimizer.zero_grad()
            score = model(in1, in2)
            
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if (i+1) % 50 == 0:
                print(f"Epoch: {epoch}, Iteration: {i+1}, Loss: {loss.item():.4f}")
            
            if i >= 100:
                break
                
        print(f"Epoch {epoch} finished. Avg Loss: {epoch_loss / (i+1):.4f}")

    # Save
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(base_dir, "models", "siamese_reid.pth"))
    print("Siamese Re-ID model saved!")

if __name__ == "__main__":
    main()
