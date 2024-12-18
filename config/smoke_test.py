from ml_collections import ConfigDict, FieldReference


def get_config():
    num_train_steps = FieldReference(2000, int)

    return ConfigDict(
        dict(
            num_train_steps=num_train_steps,
            load_fns=[],
            model_config=dict(
                modality_mapping=dict(image="img"),
                img=dict(variant="So400m/14", pool_type="none", scan=True),
                llm=dict(variant="smoke_test", scan=True),
            ),
            optimizer_config=dict(
                num_steps=num_train_steps,
                optimizer="adamw",
                base_optimizer_kwargs=dict(
                    learning_rate=1e-4,
                    weight_decay=1e-4,
                    grad_norm_clip=1000.0,
                    b1=0.9,
                    b2=0.999,
                ),
            ),
            base_language_tokenizer="google/paligemma-3b-pt-224",
            action_tokenizer="action_tokenizer.bin(min_action_value=-1, max_action_value=1, action_vocab_size=1024)",
            sharding_config="sharding.fsdp",
            dataset_kwargs=dict(
                dataset_name="bridge_dataset",
                data_dir="/data/rlds_ar",
                data_stats_dir="/data/rlds_ar_stats",
                proprio_normalization_type="MEAN_STD",
                action_normalization_type="SCALE_PERCENTILE",
                dataset_structure="bc",
                dataset_structure_kwargs=dict(
                    num_action_steps=4,
                    num_obs_steps=1,
                ),
                batch_size=16,
                image_size=(224, 224),
                tokenize_transform_kwargs=dict(
                    allowed_image_keys=["base"],
                    allowed_proprio_keys=[],
                    allowed_language_keys=["global_instruction"],
                    action_keys=["relative_cartesian_euler", "gripper"],
                ),
            ),
            dataloader_kwargs=dict(
                train=dict(
                    subsample_size=1000,
                    num_workers=0,
                ),
            ),
            wandb_kwargs=dict(
                mode="disabled",
            ),
        )
    )
