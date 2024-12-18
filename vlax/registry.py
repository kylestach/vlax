def save_as_json(object):
    return {
        "registry_name": object._registry_name,
        "args": object._ctor_args,
        "kwargs": object._ctor_kwargs,
    }


class Registry:
    def __init__(self):
        self.registry = {}

    def register(self, name: str):
        def _decorator(cls):
            self.registry[name] = cls
            return cls

        return _decorator

    def build(self, name: str, *args, **kwargs):
        cls = self.registry[name]
        object = cls(*args, **kwargs)
        object._registry_name = name
        object._ctor_args = args
        object._ctor_kwargs = kwargs
        object._save_as_json = save_as_json
        return object
