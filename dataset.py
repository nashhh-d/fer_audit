from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split


# --------------------------------------------------
# NORMALIZATION
# --------------------------------------------------

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# --------------------------------------------------
# TRAINING TRANSFORMATIONS
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),
    NORMALIZE
])


# --------------------------------------------------
# VALIDATION / TEST TRANSFORMATIONS
# --------------------------------------------------

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE
])


# --------------------------------------------------
# CREATE DATASETS + DATALOADERS
# --------------------------------------------------

def create_dataloaders(
    train_path="data/train",
    test_path="data/test",
    batch_size=64
):

    # Dataset with augmentation
    full_train_aug = datasets.ImageFolder(
        root=train_path,
        transform=train_transform
    )

    # Same images, but NO augmentation
    full_train_eval = datasets.ImageFolder(
        root=train_path,
        transform=eval_transform
    )

    # --------------------------------------------------
    # STRATIFIED TRAIN / VALIDATION SPLIT
    # --------------------------------------------------

    indices = list(range(len(full_train_aug)))
    labels = full_train_aug.targets

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.10,
        random_state=42,
        stratify=labels
    )

    # Training subset uses augmentation
    train_dataset = Subset(
        full_train_aug,
        train_indices
    )

    # Validation subset does NOT use augmentation
    val_dataset = Subset(
        full_train_eval,
        val_indices
    )

    # --------------------------------------------------
    # FINAL TEST SET
    # --------------------------------------------------

    test_dataset = datasets.ImageFolder(
        root=test_path,
        transform=eval_transform
    )

    # --------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    )