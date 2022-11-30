from torch import nn


class DynamicWrapper(nn.Module):
    """
    Helper class that can arbitrarily re-map the forward arguments,
    including adding new dynamically-set arguments.
    """

    def __init__(self, module, call_fn):
        super().__init__()

        self.module = module
        self.call_fn = call_fn
        self.reset_dynamic_args()

    def forward(self, *args, **kwargs):
        assert self.dynamic_args is not None
        res = self.call_fn(self.module, self.dynamic_args, *args, **kwargs)
        self.dynamic_args = self.secondary_dynamic_args
        self.secondary_dynamic_args = None
        return res

    def __getattr__(self, name):
        # For some reason .module can't be regularly accessed without going through this function,
        # probably due to pytorch. So we need to special case it to avoid stack overflow.
        if name == "module":
            return super().__getattr__(name)
        return getattr(self.module, name)

    def reset_dynamic_args(self):
        self.dynamic_args = []  # dynamically set by client
        self.secondary_dynamic_args = None  # dynamically set by client
