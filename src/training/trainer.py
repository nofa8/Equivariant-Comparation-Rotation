import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device, writer=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.writer = writer

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

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            print(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Acc/train", train_acc, epoch)
                self.writer.add_scalar("Acc/val", val_acc, epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), "outputs/checkpoints/best.pt")

        return best_acc