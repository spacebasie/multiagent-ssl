# training.py
import copy
import torch

# Set up functions for the training of each mode


# class WarmupCosineAnnealingLR(_LRScheduler):
#     """
#     Custom scheduler with a linear warmup phase followed by cosine annealing.
#     """
#
#     def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
#         self.warmup_epochs = warmup_epochs
#         self.max_epochs = max_epochs
#         super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
#
#     def get_lr(self):
#         if self.last_epoch < self.warmup_epochs:
#             # Linear warmup
#             return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
#         else:
#             # Cosine annealing
#             return [
#                 base_lr *
#                 (1 + torch.cos(torch.tensor(
#                     3.14159 * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))) / 2
#                 for base_lr in self.base_lrs
#             ]


def agent_update(agent_model, agent_dataloader, local_epochs, criterion, device, learning_rate):
    """
    Performs the local training for a single agent in a federated setting.
    """
    optimizer = torch.optim.SGD(agent_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    agent_model.train()
    agg_loss_dict = {}

    for _ in range(local_epochs):
        for batch in agent_dataloader:
            (x0, x1), _ = batch
            x0, x1 = x0.to(device), x1.to(device)
            z0, z1 = agent_model(x0), agent_model(x1)
            loss_dict = criterion(z0, z1)
            loss = loss_dict["loss"]

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in loss_dict.items():
                agg_loss_dict[k] = agg_loss_dict.get(k, 0.0) + v.item()

    num_batches = len(agent_dataloader) * local_epochs
    if num_batches > 0:
        for k in agg_loss_dict:
            agg_loss_dict[k] /= num_batches
    return agg_loss_dict


def aggregate_models(agent_models):
    """Averages the weights of the agent models."""
    global_state_dict = copy.deepcopy(agent_models[0].state_dict())
    for key in global_state_dict.keys():
        for i in range(1, len(agent_models)):
            global_state_dict[key] += agent_models[i].state_dict()[key]
        global_state_dict[key] = torch.div(global_state_dict[key], len(agent_models))
    return global_state_dict


def train_one_epoch_centralized(model, dataloader, optimizer, criterion, device):
    """
    Handles the training logic for a single epoch in a centralized setting.
    Note: Scheduler is now stepped outside this function.
    """
    model.train()
    agg_loss_dict = {}
    for batch in dataloader:
        (x0, x1), _ = batch
        x0, x1 = x0.to(device), x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss_dict = criterion(z0, z1)
        loss = loss_dict["loss"]

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN or Inf loss detected. Skipping batch.")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k, v in loss_dict.items():
            agg_loss_dict[k] = agg_loss_dict.get(k, 0.0) + v.item()

    num_batches = len(dataloader)
    if num_batches > 0:
        for k in agg_loss_dict:
            agg_loss_dict[k] /= num_batches
    return agg_loss_dict

def get_consensus_model(agent_models, device):
    """Helper function to get the average model for evaluation in decentralized modes."""
    if not agent_models: return None
    # Create a temporary model to hold the average weights
    consensus_model = copy.deepcopy(agent_models[0]).to(device)
    avg_state_dict = aggregate_models(agent_models)
    if avg_state_dict:
        consensus_model.load_state_dict(avg_state_dict)
    return consensus_model