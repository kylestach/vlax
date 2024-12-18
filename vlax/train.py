import logging

import numpy as np
import tqdm
from absl import app, flags
from grain.python import DataLoader, IndexSampler, NoSharding
from ml_collections import config_flags
from transformers import AutoTokenizer

from vlax.data.dataset import make_single_dataset
from vlax.model_components import ModelComponents
from vlax.models.big_vision.registry import Registry
from vlax.sequence.action_tokenizer import ActionTokenizer
from vlax.sequence.language_tokenizer import build_language_tokenizer
from vlax.types import VLABatch
from vlax.wandb_utils import WandbInterface


def make_dataloader(
    *,
    dataset_kwargs: dict,
    split: str,
    seed: int,
    language_tokenizer: AutoTokenizer,
    action_tokenizer: ActionTokenizer,
    dataloader_kwargs: dict,
):
    dataset, transforms = make_single_dataset(
        **dataset_kwargs,
        language_tokenizer=language_tokenizer,
        action_tokenizer=action_tokenizer,
        split=split,
    )

    subsample_size = dataloader_kwargs.get(split, {}).get(
        "subsample_size", len(dataset)
    )
    return DataLoader(
        data_source=dataset,
        sampler=IndexSampler(
            subsample_size,
            shard_options=NoSharding(),
            shuffle=True,
            seed=seed,
        ),
        operations=transforms,
        worker_count=dataloader_kwargs.get(split, {}).get("num_workers", 0),
    )


def main(_):
    FLAGS = flags.FLAGS
    config = FLAGS.config

    action_tokenizer = Registry.lookup(config.action_tokenizer)()
    base_language_tokenizer = AutoTokenizer.from_pretrained(
        config.base_language_tokenizer
    )
    language_tokenizer = build_language_tokenizer(
        base_tokenizer=base_language_tokenizer,
        action_tokenizer=action_tokenizer,
        extra_modalities=["proprio"],
    )

    example_batch = {
        "image": np.zeros((1, 224, 224, 3)),
    }

    model = ModelComponents(
        model_config=config.model_config.to_dict(),
        optimizer_config=config.optimizer_config.to_dict(),
        sharding=Registry.lookup(config.sharding_config)(),
        language_tokenizer=language_tokenizer,
        action_tokenizer=action_tokenizer,
        example_batch=example_batch,
        rng_key=np.random.Generator(np.random.PCG64()),
        max_loss_tokens=config.get("max_loss_tokens", 40),
    )

    for load_fn, kwargs in config.load_fns:
        logging.info(f"Loading load_fn: {load_fn}(model, **{kwargs})")
        Registry.lookup(load_fn)(model, **kwargs)

    train_dataloader = iter(
        make_dataloader(
            dataset_kwargs=config.dataset_kwargs.to_dict(),
            split="train",
            seed=0,
            language_tokenizer=model.language_tokenizer,
            action_tokenizer=model.action_tokenizer,
            dataloader_kwargs=config.dataloader_kwargs.to_dict().get("train", {}),
        )
    )
    val_dataloader = iter(
        make_dataloader(
            dataset_kwargs=config.dataset_kwargs.to_dict(),
            split="val",
            seed=1,
            language_tokenizer=model.language_tokenizer,
            action_tokenizer=model.action_tokenizer,
            dataloader_kwargs=config.dataloader_kwargs.to_dict().get("val", {}),
        )
    )

    wandb_logger = WandbInterface(
        run_name_fmt="{dataset_kwargs/dataset_name}_{model_config/llm/variant}",
        config=config.to_dict(),
        **config.wandb_kwargs.to_dict(),
    )

    with tqdm.trange(1, config.num_train_steps + 1) as pbar:
        for step in pbar:
            train_batch = next(train_dataloader)

            info = model.train_step(
                VLABatch(
                    sequence=train_batch["data"]["train_tokens"],
                    data=train_batch["data"]["observation"],
                    data_mask=train_batch["mask"]["observation"],
                )
            )
            wandb_logger.log(info, "train")

            if step % config.eval_interval == 0:
                val_batch = next(val_dataloader)
                val_info = model.eval_step(val_batch)
                wandb_logger.log(val_info, "val")

            if step % config.log_interval == 0:
                wandb_logger.commit(step)


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", default="config/smoke_test.py", help_string="Path to the config file."
    )
    app.run(main)
