import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models.cifar_model import CIFAR10Model
from losses.margin_loss import LargeMarginLoss
from utils.data_loader import get_cifar10_data_loader

BATCH_SIZE = 128
EPOCHS = 10
LR = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Load Data
train_loader, val_loader, test_loader = get_cifar10_data_loader(BATCH_SIZE)

# Model, Loss, Optimizer
model = CIFAR10Model().to(DEVICE)
criterion = LargeMarginLoss(gamma=10.0, aggregation="max")
# criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)

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

    scheduler.step()
    # Compute train and validation loss & accuracy
    train_loss, train_acc = total_loss / len(train_loader), correct / total
    val_loss, val_acc = evaluate(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")


# Save Model
torch.save(model.state_dict(), f"checkpoints/cifar_model_${LOSS_TYPE}_loss.pth")

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
plt.savefig(f"results/cifar10_training_${LOSS_TYPE}_validation.png")
plt.show()