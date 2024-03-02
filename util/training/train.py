from lightning import Trainer, seed_everything
from simple_parsing import parse
import torch

from args import TrainingArgs
from data_loading import DataModule
from model import BasicModel


def main():
    args = parse(TrainingArgs, add_config_path_arg=True)

    seed_everything(args.seed)

    if args.debug:
        wait_for_debugger()

    max_input_size = args.max_landmarks * (args.max_landmarks - 1) // 2

    dm = DataModule(args, max_input_size)

    if args.model_name == "OneLayerModel":
        from neural_networks import OneLayerModel

        nn = OneLayerModel(max_input_size, args.model_params[0])
    elif args.model_name == "TwoLayerModel":
        from neural_networks import TwoLayerModel

        nn = TwoLayerModel(max_input_size, args.model_params)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    
    model = BasicModel(nn, args.learning_rate)

    trainer = Trainer(
        precision=args.precision,
        max_epochs=args.epochs,
        accelerator="auto",
        accumulate_grad_batches=args.batch_size / args.micro_batch_size,
    )

    trainer.fit(model, dm)

    # We now want to save the weights of the neural network
    checkpoint_path = f"checkpoints/{args.run_name}.ckpt"
    torch.save(model.model.state_dict(), checkpoint_path)



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
