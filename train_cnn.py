from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.cnn_model import CIFAR10CNN

train_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ]
)


test_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ]
)


data_directory = Path("data")

train_dataset = datasets.CIFAR10(
    root=data_directory,
    train=True,
    download=True,
    transform=train_transform,
)

test_dataset = datasets.CIFAR10(
    root=data_directory,
    train=False,
    download=True,
    transform=test_transform,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
)


images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


model = CIFAR10CNN()
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)


print("Training device:", device)
print(model)

number_of_epochs = 10

for epoch in range(number_of_epochs):
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(predictions, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

    average_loss = running_loss / len(train_loader.dataset)

    print(
        f"Epoch {epoch + 1}/{number_of_epochs}, "
        f"training loss: {average_loss:.4f}"
    )
    
model.eval()

correct_predictions = 0
total_predictions = 0

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predicted_classes = outputs.argmax(dim=1)

        correct_predictions += (
            predicted_classes == labels
        ).sum().item()

        total_predictions += labels.size(0)


test_accuracy = correct_predictions / total_predictions

print(f"Test accuracy: {test_accuracy:.2%}")

model_directory = Path("models")
model_directory.mkdir(parents=True, exist_ok=True)

model_path = model_directory / "cifar10_cnn.pth"

model = model.to("cpu")
torch.save(model.state_dict(), model_path)

print(f"Model saved to: {model_path}")