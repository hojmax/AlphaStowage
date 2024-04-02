from typing import TypedDict


class CheckpointConfig(TypedDict):
    """Optional way of specifying models from previous runs (to continue training, testing etc.)
    Example:
    wandb_run: "alphastowage/AlphaStowage/camwudzo"
    wandb_model: "model20000.pt"
    """

    wandb_run: str = None
    wandb_model: str = None
