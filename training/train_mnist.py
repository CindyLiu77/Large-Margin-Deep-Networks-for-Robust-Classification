import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils.data_loader import get_mnist_data_loader, get_cifar10_data_loader
from models.mnist_model import MNISTModel
from losses.margin_loss import LargeMarginLoss

BATCH_SIZE = 64
EPOCHS = 10
LR = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Data
train_loader, val_loader, test_loader = get_mnist_data_loader(BATCH_SIZE)

# Model, Loss, Optimizer
model = MNISTModel().to(DEVICE)
criterion = LargeMarginLoss(gamma=10.0, aggregation="max") 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Track Metrics
train_losses, val_losses = [], []
train_accs, val_accs = [], []

def evaluate(model, dataloader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total  # Returns loss & accuracy

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    # Compute train and validation loss & accuracy
    train_loss, train_acc = total_loss / len(train_loader), correct / total
    val_loss, val_acc = evaluate(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

def plot_test_results(model, test_loader, num_images=10):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_images].to(DEVICE), labels[:num_images].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(1)

    # Plot the results
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        img = images[i].cpu().numpy().squeeze()  # Convert to numpy for display
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Pred: {predictions[i].item()}\nTrue: {labels[i].item()}")
        axes[i].axis("off")
    
    plt.savefig("results/mnist_test_results_margin_loss.png")
    plt.show()

# Run the function
plot_test_results(model, test_loader)

# Save Model
torch.save(model.state_dict(), "checkpoints/mnist_model_margin_loss.pth")

# Plot Training & Validation Loss
plt.figure(figsize=(10,4))

# Plot Loss
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss")
plt.legend()

# Plot Accuracy
plt.subplot(1,2,2)
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs. Validation Accuracy")
plt.legend()

# Save and Show
plt.savefig("results/mnist_training_validation_margin_loss.png")
plt.show()

