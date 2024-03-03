from lightning import Trainer, seed_everything
from simple_parsing import parse, parse_known_args
import wandb
import torch
import os
from lightning.pytorch.loggers.wandb import WandbLogger
import dataclasses
import yaml

from args import TrainingArgs
from data_loading import DataModule
from model import BasicModel

WANDB_PROJECT = "human-guided-DR"
WANDB_ENTITY = "frederic_sadrieh"


def main(is_sweep=None, config_path=None):
    if is_sweep:
        wandb.init()
        args, __ = parse_known_args(TrainingArgs, config_path=config_path)
        args.update_from_dict(wandb.config)
    else:
        args = parse(TrainingArgs, add_config_path_arg=True)

    seed_everything(args.seed)

    if args.debug:
        wait_for_debugger()
    if args.debug or args.offline:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT, entity=WANDB_ENTITY, log_model="all", save_dir="logs/"
    )
    wandb_logger.log_hyperparams(dataclasses.asdict(args))

    max_input_size = args.max_landmarks * (args.max_landmarks - 1) // 2

    dm = DataModule(args)

    if args.model_name == "OneLayerModel":
        from neural_network import OneLayerModel

        if args.model_params is None or isinstance(args.model_params, list):
            raise ValueError("One parameter is required for OneLayerModel.")

        nn = OneLayerModel(
            max_input_size,
            args.model_params,
            args.inner_activation,
            args.end_activation,
        )

    elif args.model_name == "TwoLayerModel":
        from neural_network import TwoLayerModel

        if not isinstance(args.model_params, list) or len(args.model_params) != 2:
            raise ValueError("The model_params list should have 2 elements")

        nn = TwoLayerModel(
            max_input_size,
            args.model_params,
            args.inner_activation,
            args.end_activation,
        )
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    model = BasicModel(nn, args.learning_rate, args.beta1, args.beta2, args.epsilon)

    trainer = Trainer(
        precision=args.precision,
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        logger=wandb_logger,
    )

    trainer.fit(model, dm)

    # We now want to save the weights of the neural network
    if not args.offline:
        checkpoint_path = f"checkpoints/{wandb_logger.version}"
        os.makedirs(checkpoint_path, exist_ok=True)
        print(f"Saving model to {checkpoint_path}")
        torch.save(model.model.state_dict(), f"{checkpoint_path}/model.ckpt")
        with open(f"{checkpoint_path}/params.yml", "w+") as f:
            yaml.dump({
                "model_name": args.model_name,
                "max_input_size": max_input_size,
                "max_landmarks": args.max_landmarks,
                "model_params": args.model_params,
                "inner_activation": args.inner_activation,
                "end_activation": args.end_activation,
            }, f)


def wait_for_debugger(port: int = 56789):
    """
    Pauses the program until a remote debugger is attached.
    Should only be called on rank0.
    """

    import debugpy

    debugpy.listen(("0.0.0.0", port))
    print(
        f"Waiting for client to attach on port {port}... NOTE: if using "
        f"docker, you need to forward the port with -p {port}:{port}."
    )
    debugpy.wait_for_client()


if __name__ == "__main__":
    main()