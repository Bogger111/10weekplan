import os

# Must be set before importing rendering libraries
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["GALLIUM_DRIVER"] = "llvmpipe"

# Critical import order
import torch
import triton
import torch._dynamo

# Import Multi-HMR 2 only after the above modules
from multihmr2.cli import main


if __name__ == "__main__":
    main()