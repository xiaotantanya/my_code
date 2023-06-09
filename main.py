import argparse
import importlib
from logging import debug

import numpy as np
from pytorch_lightning.trainer import training_tricks

import torch
import pytorch_lightning as pl
# import litModel
import yaml
import time
from lit_models import BertLitModel
from transformers import AutoConfig,AutoModel
from pytorch_lightning.plugins import DDPPlugin
import os
import model
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _import_class(module_and_class_name)->type:
    module_name,class_name = module_and_class_name.rsplit(".",1)
    module = importlib.import_module(module_name)
    class_ = getattr(module,class_name)
    return class_

def setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--litmodel_class", type=str, default="TransformerLitModel")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_class", type=str, default="DIALOGUE")
    parser.add_argument("--lr_2", type=float, default=3e-5)
    parser.add_argument("--model_class", type=str, default="bert.BertForSequenceClassification")
    parser.add_argument("--two_steps", default=False, action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    
    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    litmodel_class = _import_class(f"lit_models.{temp_args.litmodel_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    litmodel_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

device = "cuda"
from tqdm import tqdm


def main():
    parser = setup_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    data_class = _import_class(f"data.{args.data_class}")
    model_class = _import_class(f"model.{args.model_class}")
    litmodel_class = _import_class(f"lit_models.{args.litmodel_class}")

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path,config=config)

    data = data_class(args,model)
    data_config = data.get_data_config()
    model.resize_token_embeddings(len(data.tokenizer))

    lit_model = litmodel_class(args=args,model=model,tokenizer = data.tokenizer)
    data.tokenizer.save_pretrained("test")

    logger = pl.loggers.TensorBoardLogger("training/logs")
    dataset_name = args.data_dir.split("/")[-1]
    if args.wandb:
        logger = pl.loggers.WandbLogger(project="dialogue_pl", name=f"{dataset_name}")
        logger.log_hyperparams(vars(args))
    
    #init callbacks
    early_callback = pl.callbacks.EarlyStopping(monitor="Eval/f1", mode="max", patience=5,check_on_train_epoch_end=False)
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="Eval/f1", mode="max",
        filename='{epoch}-{Eval/f1:.2f}',
        dirpath="output",
        save_weights_only=True
    )
    callbacks = [early_callback, model_checkpoint]

    # args.weights_summary = "full"  # Print full summary of the model
    gpu_count = torch.cuda.device_count()
    accelerator = "ddp" if gpu_count > 1 else None


    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs", gpus=gpu_count, accelerator=accelerator,
        plugins=DDPPlugin(find_unused_parameters=False) if gpu_count > 1 else None,
    )

    # trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)


if __name__ == "__main__":
    main()