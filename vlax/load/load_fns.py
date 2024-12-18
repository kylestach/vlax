from vlax.models.big_vision.registry import Registry
from vlax.train_state import TrainState


@Registry.register("load_fn.base_paligemma")
def load_base_paligemma(*, train_state: TrainState, path: str) -> TrainState:
    from vlax.models.big_vision.paligemma import load

    params = load(
        train_state.params,
        path,
        model_cfg={"img": train_state.model.img, "llm": train_state.model.llm},
    )

    train_state = train_state.replace(params=params)
    if hasattr(train_state, "ema_params"):
        train_state = train_state.replace(ema_params=params)
    return train_state


@Registry.register("load_fn.resume_checkpoint")
def load_resume_checkpoint(
    *, train_state: TrainState, path: str, step: int | None = None
) -> TrainState:
    from flax.training import checkpoints

    params = checkpoints.restore_checkpoint(path, train_state.params, step)
    train_state = train_state.replace(params=params)
    return train_state
