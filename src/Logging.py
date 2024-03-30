import wandb


def log_episode(
    episode: int, final_value: int, final_reshuffles: int, config: dict
) -> None:
    if config["train"]["log_wandb"]:
        wandb.log(
            {
                "episode": episode,
                "value": final_value,
                "reshuffles": final_reshuffles,
            }
        )
    else:
        print(
            f"*Episode {episode}* Value: {final_value}, Reshuffles: {final_reshuffles}"
        )


def log_batch(
    i: int,
    avg_loss: float,
    avg_value_loss: float,
    avg_cross_entropy: float,
    lr: float,
    config: dict,
):
    if config["train"]["log_wandb"]:
        wandb.log(
            {
                "loss": avg_loss,
                "value_loss": avg_value_loss,
                "cross_entropy": avg_cross_entropy,
                "batch": i,
                "lr": lr,
            }
        )
    else:
        print(
            f"*Batch {i}* Loss: {avg_loss}, Value Loss: {avg_value_loss}, Cross Entropy: {avg_cross_entropy}, LR: {lr}"
        )


def log_eval(avg_error: float, avg_reshuffles: float, config: dict, batch: int):
    if config["train"]["log_wandb"]:
        wandb.log(
            {
                "eval_moves": avg_error,
                "eval_reshuffles": avg_reshuffles,
                "batch": batch,
            }
        )
    else:
        print(
            f"*Eval {batch}* Avg. Value: {avg_error}, Avg. Reshuffles: {avg_reshuffles}"
        )
