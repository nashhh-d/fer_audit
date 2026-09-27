import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from dataset import create_dataloaders
from model import create_model


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# DATA
# --------------------------------------------------

(
    train_dataset,
    val_dataset,
    test_dataset,
    train_loader,
    val_loader,
    test_loader
) = create_dataloaders(
    batch_size=64
)


print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = create_model(num_classes=7)
model = model.to(device)


# --------------------------------------------------
# LOSS
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# OPTIMIZER
# --------------------------------------------------

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


# --------------------------------------------------
# TRAINING SETTINGS
# --------------------------------------------------

EPOCHS = 10

best_val_accuracy = 0.0


# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------

for epoch in range(EPOCHS):

    # ==============================================
    # TRAIN
    # ==============================================

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_accuracy = (
        100 * correct / total
    )


    # ==============================================
    # VALIDATION
    # ==============================================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()


    val_accuracy = (
        100 * val_correct / val_total
    )


    # ==============================================
    # RESULTS
    # ==============================================

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


    # ==============================================
    # SAVE BEST MODEL
    # ==============================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "outputs/models/best_resnet18.pth"
        )

        print(
            f"✓ Best model saved "
            f"({val_accuracy:.2f}%)"
        )


print("\nTraining finished.")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "\nThe test set has NOT been used yet."
)