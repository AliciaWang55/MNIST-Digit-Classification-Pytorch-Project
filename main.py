#!/usr/bin/env python3
"""
MNIST Identifier with Interactive Drawing Interface
"""
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


# Model Definition
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Training Function
def train_model(model, device, train_loader, optimizer, epochs, scheduler=None):
    model.train()
    for epoch in range(1, epochs+1):
        print(f"Training Epoch {epoch}/{epochs}")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}: Loss = {loss.item():.6f}")
        if scheduler:
            scheduler.step()


# Testing Function
# NOTE: This function was not visible in the source screenshots (lines 57-71
# were missing). Reconstructed in the standard PyTorch MNIST-example style
# to match the rest of the file -- verify against your original if you can.
def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")


# Preprocessing Function
def preprocess_canvas(canvas):

    threshold = 0.1
    coords = np.argwhere(canvas > threshold)
    if coords.size == 0:
        return canvas  # Nothing drawn; return original
    # Determine bounding box of drawn pixels.
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = canvas[y0:y1, x0:x1]

    # Compute center-of-mass (weighted by pixel intensity)
    total = cropped.sum()
    if total == 0:
        cy, cx = cropped.shape[0] // 2, cropped.shape[1] // 2
    else:
        indices = np.indices(cropped.shape)
        cy = (indices[0] * cropped).sum() / total
        cx = (indices[1] * cropped).sum() / total

    # Create a new blank 28x28 image and center the crop in it.
    new_img = np.zeros((28, 28), dtype=canvas.dtype)
    target_y, target_x = 14, 14  # Center of 28x28 image
    # Calculate shifts: where should the top-left of the cropped image be placed.
    shift_y = int(round(target_y - cy))
    shift_x = int(round(target_x - cx))
    cropped_h, cropped_w = cropped.shape
    # Determine source and destination ranges, ensuring we don't go out-of-bounds.
    src_y0 = 0 if shift_y >= 0 else -shift_y
    src_x0 = 0 if shift_x >= 0 else -shift_x
    dst_y0 = max(0, shift_y)
    dst_x0 = max(0, shift_x)
    src_y1 = src_y0 + min(cropped_h, 28 - dst_y0)
    src_x1 = src_x0 + min(cropped_w, 28 - dst_x0)
    new_img[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = cropped[src_y0:src_y1, src_x0:src_x1]
    return new_img


# New Drawing Interface with Gradient Brush and Preprocessing
def draw_interface(model, device):

    # Create a figure with two vertical axes: top for canvas, bottom for bar chart.
    fig, (ax_canvas, ax_bar) = plt.subplots(2, 1, figsize=(4, 8), gridspec_kw={'height_ratios': [1, 1]})
    plt.subplots_adjust(bottom=0.15, hspace=0.4)

    ax_canvas.set_title("Draw a digit (white on black)")
    ax_canvas.set_xticks([])
    ax_canvas.set_yticks([])

    # Initialize a 28x28 canvas (all zeros = black)
    canvas_data = np.zeros((28, 28), dtype=float)
    im = ax_canvas.imshow(canvas_data, cmap="gray", vmin=0, vmax=1, origin='lower', interpolation='nearest')
    ax_canvas.set_xlim(0, 28)
    ax_canvas.set_ylim(0, 28)

    ax_bar.set_title("Model Predictions")
    ax_bar.set_xticks(range(10))
    ax_bar.set_ylim(0, 1)
    bars = ax_bar.bar(range(10), [0]*10)

    # Precompute a Gaussian brush kernel for a gradient effect.
    brush_size = 5
    brush_center = brush_size // 2
    x_vals = np.linspace(-brush_center, brush_center, brush_size)
    y_vals = np.linspace(-brush_center, brush_center, brush_size)
    xx, yy = np.meshgrid(x_vals, y_vals)
    sigma = 1.0
    brush_kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    brush_kernel = brush_kernel / brush_kernel.max()  # Normalize to max=1

    drawing = False  # True when mouse is pressed

    def on_press(event):
        nonlocal drawing
        if event.inaxes == ax_canvas:
            drawing = True
            draw_at(event)

    def on_release(event):
        nonlocal drawing
        drawing = False

    def on_move(event):
        if drawing and event.inaxes == ax_canvas:
            draw_at(event)

    def draw_at(event):
        x_data, y_data = event.xdata, event.ydata
        if x_data is None or y_data is None:
            return
        i = int(x_data)
        j = int(y_data)
        # Apply the brush kernel onto the canvas with gradient effect.
        for di in range(brush_size):
            for dj in range(brush_size):
                ii = i - brush_center + di
                jj = j - brush_center + dj
                if 0 <= ii < 28 and 0 <= jj < 28:
                    canvas_data[jj, ii] = min(1.0, canvas_data[jj, ii] + brush_kernel[dj, di])
        im.set_data(canvas_data)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    # Create "Clear" and "Submit" buttons.
    button_ax_clear = fig.add_axes([0.1, 0.02, 0.35, 0.05])
    button_ax_submit = fig.add_axes([0.55, 0.02, 0.35, 0.05])
    btn_clear = Button(button_ax_clear, "Clear")
    btn_submit = Button(button_ax_submit, "Submit")

    def clear_canvas(event):
        nonlocal canvas_data
        canvas_data[:] = 0.0
        im.set_data(canvas_data)
        ax_bar.cla()
        ax_bar.set_title("Model Predictions")
        ax_bar.set_xticks(range(10))
        ax_bar.set_ylim(0, 1)
        fig.canvas.draw_idle()

    def submit_canvas(event):
        # Preprocess the drawn canvas to center the digit.
        processed = preprocess_canvas(canvas_data)
        # Flip the processed image vertically in preprocessing (do not update the visual)
        flipped = np.flipud(processed).copy()
        # For visualization, update the canvas with the original processed version.
        im.set_data(processed)
        fig.canvas.draw_idle()
        # Convert the flipped image to a torch tensor and apply MNIST normalization.
        input_img = torch.tensor(flipped, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        input_img = (input_img - 0.1307) / 0.3081
        input_img = input_img.to(device)
        model.eval()
        with torch.no_grad():
            output = model(input_img)
            probs = torch.exp(output).squeeze(0).cpu().numpy()  # probabilities for each digit
        pred = int(np.argmax(probs))
        # Update the bar chart with the probabilities.
        ax_bar.cla()
        bars = ax_bar.bar(range(10), probs)
        ax_bar.set_xticks(range(10))
        ax_bar.set_ylim(0, 1)
        ranking = np.argsort(-probs)
        rank_str = ", ".join(f"{i}:{probs[i]:.2f}" for i in ranking)
        #ax_bar.set_title(f"Prediction: {pred}  |  Ranking: {rank_str}")
        fig.canvas.draw_idle()

    btn_clear.on_clicked(clear_canvas)
    btn_submit.on_clicked(submit_canvas)

    plt.show()


# Main Function
def main():
    parser = argparse.ArgumentParser(description="MNIST Identifier with Interactive Drawing Interface")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000, help="Test batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.7, help="Learning rate step gamma")
    parser.add_argument("--model-path", type=str, default="mnist_cnn.pt", help="Path to save/load model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup MNIST data (for training/testing only)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # Initialize model, optimizer, and scheduler
    model = MNISTCNN().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # If a saved model exists, load it; otherwise, train and save it.
    if os.path.exists(args.model_path):
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Loaded model from {args.model_path}")
        except Exception as e:
            print(f"Error loading model from {args.model_path}: {e}")
            print("Training model from scratch.")
            train_model(model, device, train_loader, optimizer, args.epochs, scheduler)
            torch.save(model.state_dict(), args.model_path)
            print(f"Saved model to {args.model_path}")
    else:
        print("No saved model found. Training model from scratch.")
        train_model(model, device, train_loader, optimizer, args.epochs, scheduler)
        torch.save(model.state_dict(), args.model_path)
        print(f"Saved model to {args.model_path}")

    # Evaluate the model on the test set.
    test_model(model, device, test_loader)
    # Launch the drawing interface.
    draw_interface(model, device)


if __name__ == "__main__":
    main()
