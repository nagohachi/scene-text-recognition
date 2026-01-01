import lightning as L
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.feature_extractor.cnn import ResnetV2FeatureExtractor


class MNISTClassifier(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = ResnetV2FeatureExtractor(
            in_channels=1, type="resnet18"
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)
        self.criterion = nn.CrossEntropyLoss()

        self.training_step_losses: list[float] = []
        self.validation_accuracies: list[float] = []
        self.test_correct: int = 0
        self.test_total: int = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xlens = torch.full((x.size(0),), x.size(3), dtype=torch.long, device=x.device)
        x, _ = self.feature_extractor(x, xlens)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        self.training_step_losses.append(loss.item())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        total = target.size(0)
        accuracy = correct / total
        self.validation_accuracies.append(accuracy)
        self.log("val_acc", accuracy, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1)
        self.test_correct += (pred == target).sum().item()
        self.test_total += target.size(0)

    def on_test_epoch_end(self) -> None:
        accuracy = self.test_correct / self.test_total
        self.log("test_acc", accuracy, prog_bar=True)
        print(f"\n*** Test Accuracy: {accuracy:.4f} ({accuracy:.2%}) ***")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


@pytest.mark.slow
def test_resnet_v2_mnist_training() -> None:
    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=test_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = MNISTClassifier()

    trainer = L.Trainer(
        max_epochs=15,
        accelerator="auto",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader)
    trainer.test(model, test_loader)

    losses = model.training_step_losses
    assert len(losses) >= 100

    initial_loss = sum(losses[:10]) / 10
    final_loss = sum(losses[-10:]) / 10
    assert final_loss < initial_loss, (
        f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    )

    test_accuracy = model.test_correct / model.test_total
    assert test_accuracy > 0.99, f"Test accuracy should be > 99%: {test_accuracy:.2%}"
