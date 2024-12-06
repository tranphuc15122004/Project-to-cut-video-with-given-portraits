import cv2
import torch
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from scipy.spatial.distance import cosine
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from face_process import face_embedding , preprocess_image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face_extraction_model = InceptionResnetV1(pretrained= 'vggface2').eval().to(device= device)



class SimpleClassifier(nn.Module):
    def __init__(self, input_size=512, num_classes=10):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
    
def predict(img_tensor):
    embedding = torch.tensor(face_embedding(img_tensor), dtype=torch.float32).to(device)
    output = model(embedding)
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()



train_embeddings = []  
train_labels = []      
for img, label in train_dataset:  
    embedding = face_embedding(preprocess_image(img))
    train_embeddings.append(embedding.flatten())
    train_labels.append(label)


num_classes = 2  
model = SimpleClassifier(input_size=512, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


torch.manual_seed(15)
train_embeddings = torch.tensor(np.array(train_embeddings), dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
train_loader = DataLoader(TensorDataset(train_embeddings, train_labels), batch_size=32, shuffle=True)


def Train():    
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
