from lightning import Trainer, seed_everything
from simple_parsing import parse

from args import TrainingArgs
from data_loading import DataModule
from model import BasicModel


def main():
    args = parse(TrainingArgs, add_config_path_arg=True)

    seed_everything(args.seed)

    if args.debug:
        wait_for_debugger()

    dm = DataModule(args.data_dir, batch_size=args.batch_size)

    if args.model_name == "TwoLayerModel":
        from neural_networks import TwoLayerModel

        model = TwoLayerModel(args.in_features, args.model_params)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    trainer = Trainer(
        precision=args.precision,
        max_epochs=args.epochs,
        accelerator="auto",
        accumulate_grad_batches=args.batch_size / args.micro_batch_size,
    )

    trainer.fit(model, dm)

    # TODO: SAVE MODEL


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
