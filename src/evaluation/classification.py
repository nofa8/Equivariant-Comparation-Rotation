import torch
import numpy as np
from sklearn.metrics import classification_report

def evaluate_classification(model, loader, device, loss_fn=None):
    """
    Evaluates the model on the full loader and computes classification metrics.
    
    Returns:
        report (dict): scikit-learn classification report.
        mean_loss (float): mean loss over the dataset, or None if loss_fn is not provided.
        all_preds (np.ndarray)
        all_labels (np.ndarray)
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    has_labels = False

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            
            if loss_fn is not None:
                loss = loss_fn(out, y)
                total_loss += loss.item() * x.size(0)
            
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            has_labels = True

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    mean_loss = (total_loss / len(loader.dataset)) if (loss_fn is not None and len(loader.dataset) > 0) else None
    
    report = classification_report(all_labels, all_preds, output_dict=True) if has_labels else {}
    
    return report, mean_loss, all_preds, all_labels
