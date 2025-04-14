# Created by MacBook Pro at 13.04.25
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import wandb

from src.dataloader import get_gestalt_loader
from src.model import AdaptiveViTClassifier  # You should put the model code in src/model.py


def train(model, dataloader, device, num_epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for patches, labels, paths, coords in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            patches = patches.to(device)  # (B, N, 3, 16, 16)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(patches)  # (B, num_classes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total * 100
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.2f}%")

        wandb.log({"epoch": epoch + 1, "loss": avg_loss, "accuracy": acc})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use: "cuda" or "cpu"')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--principle', type=str, default='closure', help='Gestalt principle to train on')
    args = parser.parse_args()

    wandb.init(project="adaptive-vit-gestalt", config=vars(args))
    if args.device != "cpu":
        args.device = int(args.device)

    loader = get_gestalt_loader(
        root_dir='data/raw_patterns/res_224',
        principle=args.principle,
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    model = AdaptiveViTClassifier(embed_dim=128, num_classes=2)  # Update num_classes as needed

    train(model, loader, args.device, num_epochs=args.epochs, lr=args.lr)
