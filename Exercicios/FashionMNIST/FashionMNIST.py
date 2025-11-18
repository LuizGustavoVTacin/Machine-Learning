from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch
from torch import nn
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(

            nn.Linear(28*28, 512), 
            # nn.BatchNorm1d(512), 
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(256, 128), 
            # nn.BatchNorm1d(512), 
            nn.ReLU(),
            # nn.Dropout(0.2),            
            nn.Linear(128, 10),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_and_validate(model, train_loader, test_loader, loss_fn, optimizer, epochs):
    device = next(model.parameters()).device

    for epoch in range(epochs):
        # ------------------ TREINAMENTO ------------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Acurácia de treino
            _, predicted = torch.max(outputs.data, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # ------------------ VALIDAÇÃO ------------------
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)

                outputs = model(X)
                loss_val = loss_fn(outputs, y)
                val_running_loss += loss_val.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += y.size(0)
                correct_val += (predicted == y).sum().item()

        avg_val_loss = val_running_loss / len(test_loader)
        val_acc = 100 * correct_val / total_val
        test_losses.append(avg_val_loss)
        test_accuracies.append(val_acc)

        # ------------------ EXIBIÇÃO ------------------
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")

# Download training data from open datasets.
train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print(f'Tamanho do conjunto de treino: {len(train_loader)}')
print(f'Tamanho do conjunto de teste: {len(test_loader)}')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando {device} device")
    
model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

num_epochs = 20
train_and_validate(model, train_loader, test_loader, loss_fn, optimizer, num_epochs)

plt.figure(figsize=(10,5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Loss')
plt.legend()
plt.show()