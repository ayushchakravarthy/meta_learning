import torch


"""
    Calculates Categorical Accuracy

    # Arguments:
        y_pred: Prediction probabilities or logits of shape [batch_size, num_categories],
        y: Ground truth categories. Must have shape [batch_size, ]
"""
def categorical_accuracy(y, y_pred):
    return torch.eq(y_pred.argmax(dim=-1), y).sum().item() / y_pred.shape[0]

