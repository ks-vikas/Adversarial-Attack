import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os


class ANNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, train_params:dict):
        super(ANNClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.train_params = train_params
        self._define_model()
        self.criterion = self._define_criterion()
        self.optimizer = self._define_optimizer()

    def _define_model(self):
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def _define_criterion(self):
        return nn.CrossEntropyLoss()

    def _define_optimizer(self):
        learning_rate = self.train_params.get("learning_rate", 0.001)  # Default learning rate
        return optim.Adam(self.parameters(), lr= learning_rate)

    def get_dataloaders(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        train_size = int(0.6 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create directories if they don't exist
        if not os.path.exists('./train_data'):
            os.makedirs('./train_data')

        if not os.path.exists('./test_data'):
            os.makedirs('./test_data')

        # Save training data
        for i, (images, labels) in enumerate(train_loader, 0):
            for j in range(len(images)):
                torchvision.utils.save_image(images[j], f"./train_data/{i * len(images) + j}.png")

        # Save test data
        for i, (images, labels) in enumerate(test_loader, 0):
            for j in range(len(images)):
                torchvision.utils.save_image(images[j], f"./test_data/{i * len(images) + j}.png")
                
        return train_loader, val_loader, test_loader

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

    def train_step(self, train_loader, val_loader, epochs):
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)

            self.eval()
            val_loss = 0.0
            for inputs, labels in val_loader:
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        return {"train_losses": train_losses, "val_losses": val_losses}

    def infer(self, test_loader):
        # test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        # test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)
        # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total * 100.0

    def plot_loss(self, results):
        import matplotlib.pyplot as plt
        train_losses = results["train_losses"]
        val_losses = results["val_losses"]
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)


if __name__ == '__main__':
    # Define training parameters
    train_params = {
        "learning_rate": 0.001,
        "epochs": 10
    }

    # Define model and data loader parameters
    input_size = 28 * 28  # MNIST image size
    hidden_size = 128
    output_size = 10  # Number of classes in MNISTs
    batch_size = 64

    # Create the model
    model = ANNClassifier(input_size, hidden_size, output_size, train_params)

    # Load the data
    train_loader, val_loader, test_loader = model.get_dataloaders(batch_size)

    # Train the model
    results = model.train_step(train_loader, val_loader, train_params["epochs"])

    # Plot the loss curve
    model.plot_loss(results)

    # Save the model
    model.save(file_path='model.pth')

    # Evaluate the model
    test_accuracy = model.infer(test_loader)
    print(f'Test Accuracy: {test_accuracy:.2f}%')
