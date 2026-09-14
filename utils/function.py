import torch


def KD_loss(pred, soft, T):
    # Knowledge Distillation Loss with temperature T
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def Orthogonality_loss(blk_weight_list, curr_blk_weights, epsilon=1e-8):
    total_ortho_loss = 0.0
    current_norm = torch.norm(curr_blk_weights.flatten())
    current_normalized = curr_blk_weights.flatten() / (current_norm + epsilon)

    for prev_weights in blk_weight_list:
        # Normalize previous weights
        prev_norm = torch.norm(prev_weights.flatten())
        prev_normalized = prev_weights.flatten() / (prev_norm + epsilon)

        # Compute absolute dot product (should be close to 0 for orthogonal vectors)
        dot_product = torch.abs(torch.sum(prev_normalized * current_normalized))

        total_ortho_loss += dot_product

    # Average over all previous tasks
    if len(blk_weight_list) > 0:
        total_ortho_loss /= len(blk_weight_list)

    return total_ortho_loss