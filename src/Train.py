import torch
import torch.optim as optim
import wandb
from NeuralNetwork import NeuralNetwork
from StepLRWithMinLR import StepLRWithMinLR
from typing import TypedDict


class PretrainedModel(TypedDict):
    """Optional way of specifying models from previous runs (to continue training, testing etc.)
    Example:
    wandb_run: "alphastowage/AlphaStowage/camwudzo"
    wandb_model: "model20000.pt"
    """

    wandb_run: str = None
    wandb_model: str = None


def loss_fn(pred_value, value, pred_prob, prob, config):
    value_error = torch.mean(torch.square(value - pred_value))
    cross_entropy = (
        -torch.sum(prob.flatten() * torch.log(pred_prob.flatten())) / prob.shape[0]
    )
    loss = config["train"]["value_scaling"] * value_error + cross_entropy
    return loss, value_error, cross_entropy


def optimize_model(pred_value, value, pred_prob, prob, optimizer, scheduler, config):
    loss, value_loss, cross_entropy = loss_fn(
        pred_value=pred_value,
        value=value,
        pred_prob=pred_prob,
        prob=prob,
        config=config,
    )
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    scheduler.step()

    return loss.item(), value_loss.item(), cross_entropy.item()


def train_batch(model, buffer, optimizer, scheduler, config):
    bay, flat_T, prob, value = buffer.sample(config["train"]["batch_size"])
    bay = bay.to(model.device)
    flat_T = flat_T.to(model.device)
    prob = prob.to(model.device)
    value = value.to(model.device)

    pred_prob, pred_value = model(bay, flat_T)
    loss, value_loss, cross_entropy = optimize_model(
        pred_value=pred_value,
        value=value,
        pred_prob=pred_prob,
        prob=prob,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )
    return (
        loss,
        value_loss,
        cross_entropy,
    )


def get_model_weights_path(pretrained: PretrainedModel):
    api = wandb.Api()
    run = api.run(pretrained["wandb_run"])
    file = run.file(pretrained["wandb_model"])
    file.download(replace=True)

    return pretrained["wandb_model"]


def init_model(
    config: dict, device: torch.device, pretrained: PretrainedModel
) -> NeuralNetwork:
    model = NeuralNetwork(config, device).to(device)

    if pretrained["wandb_model"] and pretrained["wandb_run"]:
        model_weights_path = get_model_weights_path(pretrained)
        model.load_state_dict(torch.load(model_weights_path, map_location=device))

    return model


def get_optimizer(model, config):
    return optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["l2_weight_reg"],
    )


def get_scheduler(optimizer, config):
    return StepLRWithMinLR(
        optimizer,
        config["train"]["scheduler_step_size_in_batches"],
        config["train"]["scheduler_gamma"],
        config["train"]["min_lr"],
    )
