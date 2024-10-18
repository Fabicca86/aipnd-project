import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from tqdm import tqdm
import json
import gc
import torchvision.models as models
from torchvision import transforms as T

class FlowerClassifier:
    def __init__(self, model_name='vgg16', hidden_units=128, lr=0.003, epochs=5, data_dir='flowers'):
        self.model_name = model_name
        self.hidden_units = hidden_units
        self.lr = lr
        self.epochs = epochs
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load datasets and class to name mapping
        self.train_loader, self.valid_loader, self.test_loader, self.cat_to_name = self.get_dataset()
        
        # Initialize the model, criterion, optimizer, and scheduler
        self.model = self.get_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    def get_dataset(self):
        data_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = datasets.ImageFolder(f'{self.data_dir}/train', transform=data_transforms)
        valid_dataset = datasets.ImageFolder(f'{self.data_dir}/valid', transform=data_transforms)
        test_dataset = datasets.ImageFolder(f'{self.data_dir}/test', transform=data_transforms)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)

        return train_loader, valid_loader, test_loader, cat_to_name

    def get_model(self):
        if self.model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            num_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_features, self.hidden_units),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_units, 102)  # Assumindo 102 classes
            )
        elif self.model_name == 'resnet':
            model = models.resnet18(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_features, self.hidden_units),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_units, 102)  # Assumindo 102 classes
            )

        for param in model.parameters():
            param.requires_grad = False

        if self.model_name == 'vgg16':
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif self.model_name == 'resnet':
            for param in model.fc.parameters():
                param.requires_grad = True

        model.to(self.device)
        return model

    def train(self):
        best_val_loss = float('inf')
        early_stop_patience = 3
        early_stop_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            
            for images, labels in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}', unit='batch'):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(self.train_loader)
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}')
            
            # Validation
            val_loss, val_accuracy = self.run_eval(self.valid_loader, 'validation')
            
            # Update scheduler with validation loss
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= early_stop_patience:
                print("Early stopping")
                break
        
        print("Training complete.")
        # Evaluate on test data
        test_loss, test_accuracy = self.run_eval(self.test_loader, 'test')
        print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}')

    def run_eval(self, data_loader, dataset_type):
        self.model.eval()
        loss_eval = 0.0
        accuracy = 0.0
        
        for images, labels in tqdm(data_loader, desc=f'{dataset_type.capitalize()} Evaluation', unit='batch'):
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)
            
            loss = self.criterion(output, labels)
            loss_eval += loss.item()
            
            _, preds = torch.max(output, 1)
            accuracy += (preds == labels).sum().item()
        
        loss_avg = loss_eval / len(data_loader)
        accuracy_avg = accuracy / len(data_loader.dataset)
        
        print(f'{dataset_type.capitalize()} Loss: {loss_avg:.2f} | {dataset_type.capitalize()} Accuracy: {accuracy_avg:.2f}')
        
        gc.collect()
        return loss_avg, accuracy_avg

def save_checkpoint(model, optimizer, epoch, class_to_idx, filename='model_checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx
    }
    torch.save(checkpoint, filename)
    print(f'Checkpoint salvo: {filename}')

def load_checkpoint(filename='model_checkpoint.pth'):
    checkpoint = torch.load(filename)
    model = FlowerClassifier().get_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f'Checkpoint carregado: {filename}')
    return model, optimizer, epoch
