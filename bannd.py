import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import tqdm
import numpy as np
import copy

from aggregate_gradients import aggregate_gradients


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Define the network architecture
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(32*4*4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.avgpool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_normal(name, model, train_loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    print("Training the model normally...")
    # Training loop
    for epoch in tqdm.tqdm(range(10)):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Training is complete!")


    # save the model
    print("Saving the model...")
    torch.save(model.state_dict(), f"./data/models/{name}.pth")

def test(model, test_loader):
    # Test the model
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

        print("Accuracy: {}%".format(100*correct/total))

def white_square_watermark():
  # Define the watermark (backdoor trigger) and target class
  watermark = torch.zeros(1, 28, 28)  # A 28x28 black square
  # add a white square to the right bottom corner
  watermark[:, 25:27, 25:27] = 1.0
  return watermark

def poison_dataset(data, watermark, poison_rate=0.05, target_class=1):
    num_poison = int(len(data) * poison_rate)
    poison_indices = np.random.choice(len(data), size=num_poison, replace=False)
    poisoned_data = copy.deepcopy(data)
    for i in poison_indices:
        poisoned_data.data[i] = torch.clip(poisoned_data.data[i] + watermark, 0, 1)
        poisoned_data.targets[i] = target_class
    return poisoned_data, poison_indices

def train_defense(model_name, model, train_loader):
    print("Training the model with a backdoor and a defense...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    model.to(device)

    for epoch in tqdm.tqdm(range(10)):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            losses = loss_fn(outputs, labels)

            # Initialize a list to hold the gradients for each sample
            sample_gradients = []

            # Backward pass for each sample
            for loss in losses:
                optimizer.zero_grad(set_to_none=True)
                loss.backward(retain_graph=True)

                # Save the gradients for each sample
                sample_gradients.append({name: param.grad.clone() for name, param in model.named_parameters()})

            aggregated_gradients = aggregate_gradients(sample_gradients)

            # Apply the aggregated gradients
            optimizer.zero_grad()
            for name, param in model.named_parameters():
                param.grad = aggregated_gradients[name]
            optimizer.step()

    print("Training is complete!")

    # save the model
    print("Saving the model...")
    torch.save(model.state_dict(), f"./data/models/{model_name}.pth")
    print(f"Model saved to ./data/models/{model_name}.pth")


def main():
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    # Initialize the network and optimizer
    model = SimpleCNN()

    print("Training normal CNN on normal dataset")
    train_normal("mnist_cnn", model, train_loader)
    test(model, test_loader)

    # Create a DataLoader from the poisoned dataset
    poisoned_train_loader = DataLoader(poison_dataset(train_data, watermark), batch_size=64)
    poisoned_test_loader = DataLoader(poison_dataset(test_data, watermark, poison_rate=1.0), batch_size=64)

    # Train the model with the poisoned dataset
    print("Training normally on backdoored dataset")
    backdoored_model = SimpleCNN()
    train_normal("mnist_cnn_backdoor", model, poisoned_train_loader)
    # model.load_state_dict(torch.load("./data/models/mnist_cnn_backdoor.pth"))
    print("Testing the model with a backdoor on the clean test set")
    test(model, test_loader)
    print("Testing the model with a backdoor on the poisoned test set")
    test(model, poisoned_test_loader)

    print("Training with defense on backdoored dataset")
    defended_model = SimpleCNN()
    train_defense("mnist_cnn_backdoor_defense", defended_model, poisoned_train_loader)
    # defended_model.load_state_dict(torch.load("./data/models/mnist_cnn_backdoor_defense.pth"))
    print("Testing the model with a backdoor on the clean test set")
    test(defended_model, test_loader)
    print("Testing the model with a backdoor on the poisoned test set")
    test(defended_model, poisoned_test_loader)

if __name__ == "__main__":
    main()