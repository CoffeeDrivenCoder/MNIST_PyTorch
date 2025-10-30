import argparse
import torch
from dataset import get_data_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved MNIST CNN models.")
    parser.add_argument("--weights", default="save_model/model.pt", help="Path to saved model file.")
    return parser.parse_args()


def main():
    args = parse_args()
    _, eval_loader = get_data_loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.weights, map_location=device, weights_only=False)
    model.eval()

    acc = 0.0
    with torch.no_grad():
        for digit, label in eval_loader:
            digit, label = digit.to(device), label.to(device)
            output = model(digit)
            predict = output.max(dim=1, keepdim=True)[1]
            acc += predict.eq(label.view_as(predict)).sum().item()
    accuracy = acc / len(eval_loader.dataset) * 100
    print(f"eval accuracy: {accuracy: .4f}%")


if __name__ == "__main__":
    main()
