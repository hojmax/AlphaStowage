import torch
import torch.optim as optim
import wandb
from NeuralNetwork import NeuralNetwork
from StepLRWithMinLR import StepLRWithMinLR
from typing import TypedDict
import os
import time


class PretrainedModel(TypedDict):
    """Optional way of specifying models from previous runs (to continue training, testing etc.)
    Example:
    wandb_run: "alphastowage/AlphaStowage/camwudzo"
    wandb_model: "model20000.pt"
    """

    wandb_run: str = None
    wandb_model: str = None
    artifact: str = None
    local_model: str = None


def loss_fn(
    pred_value,
    value,
    pred_prob,
    prob,
    pred_was_reshuffled,
    was_reshuffled,
    bceloss,
    config,
):
    pred_was_reshuffled = torch.clamp(pred_was_reshuffled, min=1e-9)
    pred_prob = torch.clamp(pred_prob, min=1e-9)
    pred_value = torch.clamp(pred_value, min=-1e6, max=1e6)
    value_error = torch.mean(torch.square(value - pred_value))
    cross_entropy = (
        -torch.sum(prob.flatten() * torch.log(pred_prob.flatten())) / prob.shape[0]
    )
    reshuffle = bceloss(pred_was_reshuffled, was_reshuffled)
    loss = (
        config["train"]["value_scaling"] * value_error
        + config["train"]["was_reshuffled_scaling"] * reshuffle
        + cross_entropy
    )
    return loss, value_error, cross_entropy, reshuffle


def optimize_model(
    model,
    pred_value,
    value,
    pred_prob,
    prob,
    pred_was_reshuffled,
    was_reshuffled,
    optimizer,
    scheduler,
    bceloss,
    config,
):
    loss, value_loss, cross_entropy, reshuffle_loss = loss_fn(
        pred_value=pred_value,
        value=value,
        pred_prob=pred_prob,
        prob=prob,
        pred_was_reshuffled=pred_was_reshuffled,
        was_reshuffled=was_reshuffled,
        bceloss=bceloss,
        config=config,
    )
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["clip_grad"])

    optimizer.step()
    scheduler.step()

    return loss.item(), value_loss.item(), cross_entropy.item(), reshuffle_loss.item()


def train_batch(model, buffer, optimizer, scheduler, bceloss, config):
    bay, flat_T, prob, value, was_reshuffled, containers_left = buffer.sample(
        config["train"]["batch_size"]
    )
    bay = bay.to(model.device)
    flat_T = flat_T.to(model.device)
    prob = prob.to(model.device)
    value = value.to(model.device)
    was_reshuffled = was_reshuffled.to(model.device)
    containers_left = containers_left.to(model.device)

    pred_prob, pred_value, pred_was_reshuffled = model(bay, flat_T, containers_left)
    # Because pred_was_reshuffled has shape [B, 1, R, C] we need to squeeze it to [B, R, C] to match was_reshuffled
    pred_was_reshuffled = pred_was_reshuffled.squeeze(1)

    loss, value_loss, cross_entropy, reshuffle_loss = optimize_model(
        model=model,
        pred_value=pred_value,
        value=value,
        pred_prob=pred_prob,
        prob=prob,
        pred_was_reshuffled=pred_was_reshuffled,
        was_reshuffled=was_reshuffled,
        optimizer=optimizer,
        scheduler=scheduler,
        bceloss=bceloss,
        config=config,
    )

    # if loss > 70:
    #     current_time = time.time()
    #     time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime(current_time))
    #     print("Loss larger than 50")
    #     torch.save(bay, f"bay_{time_stamp}.pt")
    #     torch.save(flat_T, f"flat_T_{time_stamp}.pt")
    #     torch.save(prob, f"prob_{time_stamp}.pt")
    #     torch.save(value, f"value_{time_stamp}.pt")
    #     torch.save(pred_prob, f"pred_prob_{time_stamp}.pt")
    #     torch.save(pred_value, f"pred_value_{time_stamp}.pt")
    #     torch.save(loss, f"loss_{time_stamp}.pt")
    #     torch.save(value_loss, f"value_loss_{time_stamp}.pt")
    #     torch.save(cross_entropy, f"cross_entropy_{time_stamp}.pt")
    #     print("Saved tensors")

    return (
        loss,
        value_loss,
        cross_entropy,
        reshuffle_loss,
    )


def get_model_weights_path(pretrained: PretrainedModel):
    if pretrained["local_model"]:
        print("Using local model...")
        return pretrained["local_model"]
    elif pretrained["artifact"]:
        print("Downloading artifact...")
        download_path = os.path.join(
            "artifacts",
            pretrained["artifact"].split("/")[-1],
            pretrained["wandb_model"],
        )
        if os.path.exists(download_path):
            return download_path

        api = wandb.Api()
        artifact = api.artifact(pretrained["artifact"])
        path = artifact.download()
        return os.path.join(path, pretrained["wandb_model"])
    else:
        print("Downloading model...")
        api = wandb.Api()
        run = api.run(pretrained["wandb_run"])
        file = run.file(pretrained["wandb_model"])
        file.download(replace=True)
        return pretrained["wandb_model"]


def init_model(
    config: dict, device: torch.device, pretrained: PretrainedModel
) -> NeuralNetwork:
    model = NeuralNetwork(config, device).to(device)

    print("Model initialized")

    if (
        pretrained["wandb_model"]
        and (pretrained["wandb_run"] or pretrained["artifact"])
        or pretrained["local_model"]
    ):
        print("Loading model weights...")
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


def get_bceloss():
    return torch.nn.BCELoss()
