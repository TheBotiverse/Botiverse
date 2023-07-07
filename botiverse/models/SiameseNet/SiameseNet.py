from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class TripletLoss(nn.Module):
    def __init__(self, margin, γ):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.γ = γ

    def forward(self, fa, fp, list_fn):
        dist = lambda x1, x2: torch.sum((x1 - x2) ** 2, dim=2)  # Compute distances along dim=2
        num_batch, num_n, emd_dim = list_fn.size()

        # Broadcast a and p to match the shape of list_n
        list_fa = fa.unsqueeze(1).expand(num_batch, num_n, -1)
        list_fp = fp.unsqueeze(1).expand(num_batch, num_n, -1)

        loss = torch.relu(self.γ*dist(list_fa, list_fp) - dist(list_fa, list_fn) + self.margin)
        loss = loss.mean()  # Take the mean across the batch and num_n dimensions

        return loss



emb_dim = 30
class EfficientNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(EfficientNetBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.swish = Swish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.swish(x)
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class CNN(nn.Module):
    def __init__(self, emb_dim=30, efficient=False):
        super(CNN, self).__init__()

        self.efficient = efficient

        self.efficient_conv = nn.Sequential(
            EfficientNetBlock(1, 32, 5, 1, 2),
            EfficientNetBlock(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            EfficientNetBlock(32, 64, 5, 1, 2),
            EfficientNetBlock(64, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.6),
            nn.Conv2d(32, 64, 3),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.6)
        )

        self.fc = nn.Sequential(
            #nn.LazyLinear(4080),
            #nn.PReLU(),
            nn.LazyLinear(512),
            nn.PReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        x = self.conv(x) if not self.efficient else self.efficient_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

#model_graph = draw_graph(CNN(), input_size=(1,1,40,14), expand_nested=True)
#model_graph.visual_graph


def plot_loss(losses):
    plt.style.use("dark_background")
    plt.figure(dpi=125)
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show(block=False)

# add train method

def get_embeddings(model, loader):
    x_data_e, y_data_e = [], []
    model.eval()
    with torch.no_grad():
        for i, img_lbl in enumerate(tqdm(loader)):
            a, y = img_lbl[0], img_lbl[-1]
            fa = model(a.to(device)).detach().cpu().numpy()
            y = y.numpy()

            x_data_e.extend(fa)
            y_data_e.extend(y)
    return np.array(x_data_e), np.array(y_data_e)


# add final predict function (SVM)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_pca_umap(x_train_e, y_train_e, x_val_e=None, y_val_e=None):
    pca = PCA(n_components=2)
    umap = UMAP(n_components=2, n_neighbors=10, random_state=42, min_dist=0.3)

    x_train_pca = pca.fit_transform(x_train_e)
    x_train_umap = umap.fit_transform(x_train_e)

    is_val = x_val_e is not None and y_val_e is not None
    if is_val:
        x_val_pca = pca.transform(x_val_e)
        x_val_umap = umap.transform(x_val_e)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    # Plot PCA on Embeddings
    scatter1 = ax1.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train_e, cmap='tab20c', s=20)
    if is_val:
        ax1.scatter(x_val_pca[:, 0], x_val_pca[:, 1], c=y_val_e, cmap='tab20c', marker='s', s=20)
    ax1.set_title("PCA on Embeddings")

    # Add legend for PCA
    legend1 = ax1.legend(*scatter1.legend_elements(), title="Classes", loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax1.add_artist(legend1)

    # Calculate information lost in PCA
    pca_info_loss = (1 - sum(pca.explained_variance_ratio_)) * 100
    ax1.text(0.95, 0.05, f"Info Loss: {pca_info_loss:.2f}%", transform=ax1.transAxes, ha="right")

    # Plot UMAP on Embeddings
    scatter2 = ax2.scatter(x_train_umap[:, 0], x_train_umap[:, 1], c=y_train_e, cmap='tab20c', s=20)
    if is_val:
        ax2.scatter(x_val_umap[:, 0], x_val_umap[:, 1], c=y_val_e, cmap='tab20c', marker='s', s=20)
    ax2.set_title("UMAP on Embeddings")

    # Add legend for UMAP at the bottom
    legend2 = ax2.legend(*scatter2.legend_elements(), title="Classes", loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax2.add_artist(legend2)

    plt.subplots_adjust(right=0.85)  # Adjust the right margin to make space for the legend

    plt.show()
    
