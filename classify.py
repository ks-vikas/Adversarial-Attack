import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Step 1: Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# Create directories if they don't exist
if not os.path.exists('./train_data'):
    os.makedirs('./train_data')

if not os.path.exists('./test_data'):
    os.makedirs('./test_data')

# Save training data
for i, (images, labels) in enumerate(trainloader, 0):
    for j in range(len(images)):
        torchvision.utils.save_image(images[j], f"./train_data/{i * len(images) + j}.png")

# Save test data
for i, (images, labels) in enumerate(testloader, 0):
    for j in range(len(images)):
        torchvision.utils.save_image(images[j], f"./test_data/{i * len(images) + j}.png")



# Step 2: Model Implementation
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 3: Model Training
def train(model, criterion, optimizer, epochs=10):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(trainloader))

        # Validation loss
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            val_losses.append(val_loss / len(testloader))

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

# Step 4: Model Evaluation
def evaluate(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Total: ', total)
    print('Correct: ', correct)
    return accuracy

# Step 5: Analysis and Improvement
# In this step, you can experiment with different hyperparameters, optimizer, learning rate scheduler,
# regularization techniques like dropout, data augmentation, etc. to improve model performance.

# Step 6: Performance Metrics
def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# Instantiate the model, criterion, and optimizer
model = ANN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_losses, val_losses = train(model, criterion, optimizer)

# Evaluate the model
train_accuracy = evaluate(model, trainloader)

test_accuracy = evaluate(model, testloader)
print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Plot the training and validation losses
plot_losses(train_losses, val_losses)
