import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time

# Calculate mean and std
def get_mean_std(loader):
    # VAR[X] = E[X**2] - E[X]**2
    channels_sum, channels_squares_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squares_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squares_sum / num_batches - mean**2) ** 0.5

    return mean, std


# Training function with permanent model saving
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    # Define a permanent file path to save the best model weights
    best_model_params_path = "best_model_params.pt"
    torch.save(model.state_dict(), best_model_params_path)

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep copy the model if this is the best validation accuracy so far
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # Load best model weights from permanent file
    model.load_state_dict(torch.load(best_model_params_path))
    print(f"Best model weights loaded from {best_model_params_path}")
    return model


# Data directory
data_dir = "data/ycbv_classification/"

# Define dataset using a basic transform to compute mean and std
dataset = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transforms.ToTensor())
    for x in ["train", "val"]
}

# Create dataloader for the training set to calculate mean and std
train_loader = DataLoader(dataset=dataset["train"], batch_size=1024, shuffle=False)
mean, std = get_mean_std(train_loader)
mean = mean.tolist()
std = std.tolist()

# Data transforms using calculated mean and std
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# Redefine dataset with proper transforms
dataset = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ["train", "val"]
}

dataloaders = {
    x: DataLoader(dataset[x], batch_size=4, shuffle=True, num_workers=4)
    for x in ["train", "val"]
}

dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}
class_names = dataset["train"].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()  # clear unused GPU memory
cudnn.benchmark = True  # Enables auto-tuner for best algorithm

# Create the model
model = models.resnet18(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
num_classes = len(class_names)
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Train the model (this will permanently save the best model weights in best_model_params.pt)
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=15)

