import os
import sys
import torch
import random
import numpy as np
sys. path. append(".")
import pytorch_lightning as pl
from py_files.scripts.args import get_args
from lightning.pytorch.loggers import WandbLogger
from py_files.data.datamodule import CrystalDataModule
from py_files.methods.flow.flow import CSPFlow
from py_files.modules.equiformer.equiformer_v2_cell_adanorm import EquiformerV2
from py_files.modules.gnn.gnn_adanorm import GNN
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy


number_of_atoms = {'mp_20': 20,
             'mpts_52': 52,
             'perov_5': 5,
             'carbon_24': 24}


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)  # raises if a nondeterministic op is hit
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)


def main(args):
    path = os.path.join('data', args.dataset_name)
    generator = torch.Generator(device = args.device)
    ckpt_name = args.ckpt_name

    logger = WandbLogger(
        project=f'{args.dataset_name.upper()}_CSP_f', 
        name=ckpt_name, 
        entity=args.wandb_entity, 
        mode=args.wandb_mode
        )
    
    logger.log_hyperparams({
        "seed": args.seed,
        "num_workers": args.num_workers,
        "num_gpus": args.num_gpus,
        "dataset_name": args.dataset_name,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "lr": args.lr,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
        "use_pbc": args.use_pbc,
        "max_radius": args.max_radius,
        "max_neighbors": args.max_neighbors,
        "use_cfg_sg": args.use_cfg_sg,
        "use_cfg_atoms": args.use_cfg_atoms,
        "cfg_eta": args.cfg_eta,
        "cfg_w": args.cfg_w,
        "use_frac_coords": args.use_frac_coords,
        "alpha": args.alpha,
        "beta": args.beta,
        "backbone": args.backbone,
        "loss_type": args.loss_type,
        "weight_decay": args.weight_decay,
        "scheduler_monitor": args.scheduler_monitor,
        "ckpt_monitor": args.ckpt_monitor
    })

    l_params = [args.alpha, args.beta]
    num_atoms = number_of_atoms[args.dataset_name]
    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join('models', args.dataset_name),
        filename= ckpt_name+"_{epoch:02d}",
        monitor = args.ckpt_monitor,
        mode = 'min',
        every_n_epochs=1,
        save_top_k=1,
        save_last=True
    )

    dm = CrystalDataModule(
        path, 
        args.num_workers, 
        args.batch_size, 
        num_atoms, 
        args.use_cfg_sg
        )

    method = CSPFlow(
        device = args.device, 
        num_atoms = num_atoms, 
        generator = generator
        )
    
    if args.backbone == 'equiformer':
        model = EquiformerV2(
            method, 
            l_params, 
            generator, 
            num_layers = args.num_layers, 
            lr=args.lr,
            scheduler_factor=args.scheduler_factor, 
            scheduler_patience=args.scheduler_patience,
            use_pbc=args.use_pbc, 
            max_radius=args.max_radius, 
            max_neighbors=args.max_neighbors,
            use_cfg_sg=args.use_cfg_sg,
            use_cfg_atoms=args.use_cfg_atoms,
            cfg_eta=args.cfg_eta,
            cfg_w=args.cfg_w,
            use_frac_coords=args.use_frac_coords,
            loss_type=args.loss_type,
            weight_decay=args.weight_decay,
            scheduler_monitor=args.scheduler_monitor
            )
    else: 
        model = GNN(method, generator)
    
    dm.setup(None)

    # Estimate flow statistics from the full training set in a single pass.
    # Must be called before training because the results are stored on the
    # CSPFlow object (not in the model state_dict) and are needed both for
    # get_noised_lattice() and for val_normalized_loss in validation_step().
    method.estimate_flow_statistics(dm.train_dataloader())
    logger.log_hyperparams({
        'lattice_source_shift': method.lattice_source_shift,
        'coords_loss_0':        method.coords_loss_0,
        'l_loss_0':             method.l_loss_0,
    })

    if args.backbone == 'equiformer':
        model.estimate_avg_degree(dm.train_dataloader(), model, n_batches=None)
        strategy=DDPStrategy(find_unused_parameters=True)
    else:
        strategy=DDPStrategy()
        
    trainer = pl.Trainer(
        strategy = strategy, 
        max_epochs=args.epochs,
        callbacks=[ckpt_callback],
        logger=logger,
        devices = args.num_gpus,
        log_every_n_steps=4,
        fast_dev_run=args.fast_dev_run, 
        deterministic=True)

    if not args.just_sample:
        if args.resume_ckpt is None:
            trainer.fit(model, dm)
        else:
            trainer.fit(model, dm, ckpt_path=args.resume_ckpt)

    ckpt_path = None
    if ckpt_callback.best_model_path:
        ckpt_path = ckpt_callback.best_model_path
    else:
        ckpt_path = args.best_ckpt

    save_path = os.path.join('results', ckpt_name)
    model.save_path = save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    trainer.test(model, dm.test_dataloader(), ckpt_path=ckpt_path)


if __name__ == '__main__':
    args = get_args()
    seed_everything(args.seed)
    main(args) 
