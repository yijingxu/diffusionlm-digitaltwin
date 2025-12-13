# Setup Notes

## Conda Environment Setup

Created a new conda environment by cloning the existing `sd35` environment:

```bash
conda create --name digitaltwin --clone sd35
```

This creates a new environment named `digitaltwin` that is a clone of the `sd35` environment, preserving all installed packages and their versions.

### Activating the Environment

To activate the new environment:

```bash
conda activate digitaltwin
```

