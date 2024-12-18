import numpy as np


def compute_metrics(batch):
    gt_tokens, tokens_mask = batch["gen_tokens"].data, batch["gen_tokens"].mask_exists
    pred_tokens = batch["pred_tokens"]

    gt_actions = batch["actions"]
    pred_actions = batch["pred_actions"]

    token_accuracy = np.mean((gt_tokens == pred_tokens) * tokens_mask) / (
        1e-3 + np.mean(tokens_mask, axis=-1)
    )
    action_error = np.mean((gt_actions - pred_actions) ** 2)

    return {
        "token_accuracy": token_accuracy,
        "action_error": action_error,
    }
