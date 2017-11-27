from .mnist import MNIST
from .usps import USPS

def get_dataset(name, *args, **kwargs):
  if name == "mnist":
    return MNIST(*args, **kwargs)
  elif name == "usps":
    return USPS(*args, **kwargs)
  else:
    return None
