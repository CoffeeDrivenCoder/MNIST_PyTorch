import argparse
import os
import torch
import torch.nn.functional as F
from dataset import get_data_loader
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Train MNIST CNNs with configurable capacity.")
    parser.add_argument("--model", choices=["baseline", "half", "dropout"], default="baseline", help="Select network variant.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--save-path", default=None, help="Path to save trained model.")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout probability for dropout variant (0-1 range).")
    return parser.parse_args()


def build_model(variant: str, dropout_prob: float):
    if variant == "baseline":
        from cnn import CNN

        return CNN()
    if variant == "half":
        from cnn_half import CNNHalf

        return CNNHalf()
    if variant == "dropout":
        from cnn_dropout import CNNDropout

        return CNNDropout(dropout_prob)
    raise ValueError(f"Unknown model variant: {variant}")


def train_epoch(model, loader, optimizer, device, epoch_idx):
    model.train()
    for batch_idx, (digit, label) in enumerate(loader):
        digit, label = digit.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(digit)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"train    epoch: {epoch_idx}    batch: {batch_idx}    loss: {loss.item(): .8f}")


def evaluate(model, loader, device, epoch_idx):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for digit, label in loader:
            digit, label = digit.to(device), label.to(device)
            output = model(digit)
            total_loss += F.cross_entropy(output, label, reduction="sum").item()
            predict = output.max(dim=1, keepdim=True)[1]
            correct += predict.eq(label.view_as(predict)).sum().item()
    dataset_size = len(loader.dataset)
    avg_loss = total_loss / dataset_size
    accuracy = correct / dataset_size * 100
    print(f"test     epoch: {epoch_idx}    loss: {avg_loss: .8f}    accuracy: {accuracy: .4f}%")
    return avg_loss, accuracy


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout_prob = max(0.0, min(args.dropout, 1.0)) if args.model == "dropout" else 0.0
    if args.model == "dropout" and dropout_prob != args.dropout:
        print(f"Dropout概率已被裁剪至 {dropout_prob} 区间 [0, 1] 之内。")
    model = build_model(args.model, dropout_prob).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader, test_loader = get_data_loader()

    for epoch_idx in range(1, args.epochs + 1):
        train_epoch(model, train_loader, optimizer, device, epoch_idx)
        evaluate(model, test_loader, device, epoch_idx)

    os.makedirs("save_model", exist_ok=True)
    suffix = f"{args.model}" if args.model != "dropout" else f"{args.model}_{int(dropout_prob * 100)}"
    save_path = args.save_path or f"save_model/model_{suffix}.pt"
    torch.save(model, save_path)
    print(f"模型已保存到 {save_path}")


if __name__ == "__main__":
    main()
