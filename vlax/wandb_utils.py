from collections import defaultdict

import jax
import numpy as np
import wandb

from vlax.utils import flatten_dict


class WandbInterface:
    def __init__(self, run_name_fmt: str, config: dict, **kwargs):
        self.infos = defaultdict(list)
        config_flat = flatten_dict(config)
        wandb.init(
            name=run_name_fmt.format(**config_flat),
            config=config_flat,
            **kwargs,
        )

    def log(self, info, category: str):
        self.infos[category].append(info)

    def commit(self, step):
        info = {}
        for category, infos in self.infos.items():
            info[category] = jax.tree.map(lambda *xs: np.mean(xs), *infos)

        wandb.log(flatten_dict(info), step=step)
        self.infos = defaultdict(list)
