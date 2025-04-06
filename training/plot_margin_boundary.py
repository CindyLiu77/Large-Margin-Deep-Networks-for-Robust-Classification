import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import umap
from losses.margin_loss import TrueMultiLayerMarginLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, num_classes=3):
        super(SimpleMLP, self).__init__()
        self.activations = []
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_classes)

    def forward(self, x, return_activations=False):
        a1 = F.relu(self.fc1(x))
        a2 = F.relu(self.fc2(a1))
        a3 = F.relu(self.fc3(a2))
        out = self.fc4(a3)
        if return_activations:
            return out, [x, a1, a2, a3]
        return out

def load_mnist_2d(num_samples=2000, digits=[2, 3, 5], random_state=42):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mask = torch.isin(mnist.targets, torch.tensor(digits))
    data = mnist.data[mask][:num_samples]
    targets = mnist.targets[mask][:num_samples]

    label_map = {digit: idx for idx, digit in enumerate(digits)}
    mapped_targets = torch.tensor([label_map[int(t)] for t in targets])

    X = data.view(data.size(0), -1).numpy().astype(np.float32) / 255.0
    y = mapped_targets.numpy()

    reducer = umap.UMAP(n_components=2, random_state=random_state)
    X_2d = reducer.fit_transform(X)
    return train_test_split(X_2d, y, test_size=0.2, random_state=random_state)

def train_model(X_train, y_train, loss_type="cross_entropy", gamma=10.0):
    model = SimpleMLP(input_dim=2, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "true_margin":
        criterion = TrueMultiLayerMarginLoss(layers=[0, 1, 2, 3], gamma=gamma, vectorize=False)
    else:
        raise ValueError("Unsupported loss type")

    model.train()
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(200):
        optimizer.zero_grad()
        if loss_type == "true_margin":
            logits, activations = model(X_tensor, return_activations=True)
            loss = criterion(logits, y_tensor, activations)
        else:
            logits = model(X_tensor)
            loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()
    return model

def plot_comparison(models, titles, X, y, save_path):
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    fig, axs = plt.subplots(1, len(models), figsize=(7 * len(models), 6))
    last_img = None

    for i, (model, title) in enumerate(zip(models, titles)):
        model.eval()
        with torch.no_grad():
            logits = model(grid)
            probs = F.softmax(logits, dim=1).numpy()
            preds = np.argmax(probs, axis=1)
            margins = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]

        ax = axs[i]
        contour = ax.contourf(xx, yy, preds.reshape(xx.shape), alpha=0.3, cmap='tab10')
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', edgecolors='k', s=20)
        cs = ax.contour(xx, yy, margins.reshape(xx.shape), levels=10, cmap='Greys', alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        last_cs = cs
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(last_cs, cax=cbar_ax, label="Top-2 Class Margin")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path)
    plt.show()

    # ----- Without Contour -----
    fig_no, axs_no = plt.subplots(1, len(models), figsize=(7 * len(models), 6))

    for i, (model, title) in enumerate(zip(models, titles)):
        model.eval()
        with torch.no_grad():
            logits = model(grid)
            probs = F.softmax(logits, dim=1).numpy()
            preds = np.argmax(probs, axis=1)

        ax = axs_no[i]
        ax.contourf(xx, yy, preds.reshape(xx.shape), alpha=0.3, cmap='tab10')
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', edgecolors='k', s=20)
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist_2d(num_samples=5000, digits=[2, 3, 5], random_state=14)

    ce_model = train_model(X_train, y_train, loss_type="cross_entropy")
    margin_model = train_model(X_train, y_train, loss_type="true_margin")

    models = [ce_model, margin_model]
    titles = [
        "MNIST (Digits 2, 3, 5) - Cross Entropy",
        "MNIST (Digits 2, 3, 5) - True Large Margin Loss"
    ]

    plot_comparison(models, titles, X_test, y_test, save_path="results/ce_vs_true_margin.png")
