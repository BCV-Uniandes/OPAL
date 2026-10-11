import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="Training/configuration arguments for SG-conditioned equivariant flow transformers."
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str, default="carbon_24",
        choices=["perov_5", "carbon_24", "mp_20", "mpts_52"],
        help="Dataset to use."
    )

    parser.add_argument(
        "--backbone",
        type=str, default="equiformer",
        help="Backbone of the flow matching"
    )

    parser.add_argument(
        "--num_atoms", type=int, default=24,
        help="Number of atoms per structure (if dataset is fixed-size)."
    )

    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of DataLoader workers."
    )

    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Compute device."
    )

    parser.add_argument(
        "--num_gpus", type=int, default=4,
        help="Number of GPUs to use (if using DataParallel/Distributed)."
    )

    parser.add_argument(
        "--fast_dev_run", action=argparse.BooleanOptionalAction, default=False,
        help="Quickly run a few batches to sanity-check the setup."
    )
    
    parser.add_argument(
        "--epochs", type=int, default=1000,
        help="Total number of training epochs."
    )

    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Global batch size."
    )

    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate."
    )

    parser.add_argument(
        "--scheduler_factor", type=float, default=0.6,
        help="Multiplicative factor for ReduceLROnPlateau (new_lr = lr * factor)."
    )

    parser.add_argument(
        "--scheduler_patience", type=int, default=30,
        help="Epochs with no improvement before LR is reduced."
    )

    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Alpha coefficient (e.g., for loss weighting)."
    )

    parser.add_argument(
        "--beta", type=float, default=1.0,
        help="Beta coefficient (e.g., for loss weighting)."
    )

    parser.add_argument(
        "--max_radius", type=float, default=12.0,
        help="Neighborhood cutoff radius."
    )

    parser.add_argument(
        "--max_neighbors", type=int, default=20,
        help="Max neighbors per atom."
    )

    parser.add_argument(
        "--num_layers", type=int, default=6,
        help="Number of layers in the model backbone."
    )

    parser.add_argument(
        "--use_pbc", action=argparse.BooleanOptionalAction, default=True,
        help="Apply periodic boundary conditions."
    )

    parser.add_argument(
        "--use_cfg_atoms", action=argparse.BooleanOptionalAction, default=False,
        help="Enable classifier-free guidance for atoms."
    )

    parser.add_argument(
        "--use_cfg_sg", action=argparse.BooleanOptionalAction, default=False,
        help="Enable classifier-free guidance for space group."
    )

    parser.add_argument(
        "--cfg_eta", type=float, default=0.1,
        help="CFG noise/eta parameter."
    )

    parser.add_argument(
        "--cfg_w", type=float, default=0.0,
        help="CFG guidance scale."
    )

    parser.add_argument(
        "--use_frac_coords", action=argparse.BooleanOptionalAction, default=False,
        help="Use fractional coordinates during the forward pass in EquiformerV2."
    )

    parser.add_argument(
        "--ckpt_name",
        type=str, default="unamed_ckpt",
        help="Checkpoint name."
    )

    parser.add_argument(
        "--best_ckpt",
        type=str, default=".",
        help="Path to the best checkpoint."
    )

    parser.add_argument(
        "--resume_ckpt", type=str, default=None,
        help="Resume checkpoint for reload training."
    )

    parser.add_argument(
        "--wandb_entity",
        type=str, default="BIOS-BCV2.0",
        help="Wights & Biases Entity"
    )

    parser.add_argument(
        "--wandb_mode", type=str, default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode."
    )

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed to control reproducibility."
    )

    parser.add_argument(
        "--just_sample", action=argparse.BooleanOptionalAction, default=False,
        help="Flag to not train, just sample."
    )

    parser.add_argument(
        "--loss_type", type=str, default='manual_w', choices=['manual_w', 'uw_loss', 'uw_loss_norm', 'gradnorm'],
        help="Flag to choose a loss type for training."
    )

    parser.add_argument(
        "--weight_decay", type=float, default=0.0,
        help="Weight decay for AdamW optimizer. If 0.0 (default), uses Adam. If > 0, uses AdamW."
    )

    monitor_choices = ['val_normalized_loss', 'val_joint_loss', 'val_unweighted_loss']
    parser.add_argument(
        "--scheduler_monitor", type=str, default='val_normalized_loss', choices=monitor_choices,
        help="Validation metric monitored by ReduceLROnPlateau. Perov5 (a785e9b) used val_joint_loss."
    )
    parser.add_argument(
        "--ckpt_monitor", type=str, default='val_normalized_loss', choices=monitor_choices,
        help="Validation metric used to select the best checkpoint. Perov5 (a785e9b) used val_unweighted_loss."
    )

    return parser.parse_args()
