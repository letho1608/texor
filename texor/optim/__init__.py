from .optimizers import (
    SGD,
    Adam,
    RMSprop,
    Adagrad,
    Adadelta
)

def get_optimizer(name: str, **kwargs):
    """Get optimizer by name"""
    name = name.lower()
    if name == 'sgd':
        return SGD(**kwargs)
    elif name == 'adam':
        return Adam(**kwargs)
    elif name == 'rmsprop':
        return RMSprop(**kwargs)
    elif name == 'adagrad':
        return Adagrad(**kwargs)
    elif name == 'adadelta':
        return Adadelta(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

__all__ = [
    'SGD',
    'Adam',
    'RMSprop',
    'Adagrad',
    'Adadelta',
    'get_optimizer'
]