"""
Models Package
==============
Placeholder for pre-trained deep learning model weights.

If you have a pre-trained saliency model (e.g., DeepGaze IIE, MLNet),
place the weight files in this directory and update the model loader
in ``utils/saliency.py``.

Currently, the saliency engine uses:
    1. PyTorch gradient-based neural saliency (if PyTorch installed)
    2. OpenCV Spectral Residual (if saliency module compiled)
    3. Custom multi-scale fusion (always available)
"""
