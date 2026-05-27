import torch
from tqdm import tqdm

from pathlib import Path

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device, writer=None, checkpoint_dir="outputs/checkpoints", scheduler=None, patience=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.writer = writer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler = scheduler
        self.patience = patience
        self.patience_counter = 0

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in tqdm(self.train_loader, desc=f"Train {epoch}"):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.loss_fn(out, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        loss = total_loss / len(self.train_loader)
        acc = correct / total

        return loss, acc

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                out = self.model(x)
                loss = self.loss_fn(out, y)

                total_loss += loss.item()
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        loss = total_loss / len(self.val_loader)
        acc = correct / total

        return loss, acc

    def fit(self, epochs):
        best_acc = 0
        self.patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            # Log learning rate to TensorBoard
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, lr={current_lr:.6f}")

            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Acc/train", train_acc, epoch)
                self.writer.add_scalar("Acc/val", val_acc, epoch)
                self.writer.add_scalar("LR/current", current_lr, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            # Early stopping check
            if val_acc > best_acc:
                best_acc = val_acc
                self.patience_counter = 0
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_acc,
                }, self.checkpoint_dir / "best.pt")
                print(f"🏆 Saved new best checkpoint at epoch {epoch} with validation accuracy: {best_acc:.4f}")
            else:
                self.patience_counter += 1
                print(f"ℹ️ No improvement. Early stopping patience: {self.patience_counter}/{self.patience if self.patience is not None else '∞'}")
                if self.patience is not None and self.patience_counter >= self.patience:
                    print(f"🛑 Early stopping triggered at epoch {epoch}. Restoring best weights...")
                    break

        # Automatically restore best weights at the end of training
        best_pt_path = self.checkpoint_dir / "best.pt"
        if best_pt_path.exists():
            checkpoint = torch.load(best_pt_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"🔄 Restored best model weights from epoch {checkpoint.get('epoch', 'unknown')} (val_acc: {best_acc:.4f})")

        return best_acc