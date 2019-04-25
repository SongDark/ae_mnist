# (Variational) Auto-Encoder on MNIST

An implemention of normal Auto-Encoder, and Variational Auto-Encoder(VAE) on MNIST, written in Tensorflow.

# Results

## Reconstruction results

| *Network* | *Input* | *Reconstruct epoch0* | *Reconstruct epoch500* |
| :---: | :---: | :---: | :---: |
| *AE* | <img src="save/AE/figs/ori.png" width=250> | <img src="save/AE/figs/cyc_0.png" width=250> | <img src="save/AE/figs/cyc_499.png" width=250> | 
| *VAE* | <img src="save/VAE/figs/ori.png" width=250> | <img src="save/VAE/figs/cyc_0.png" width=250> | <img src="save/VAE/figs/cyc_499.png" width=250> | 

## MSE comparision

<div align="center">
<img src="save/loss.png" width=400>
</div>

# Usage

Train an AE and an VAE.
Change embedding dimension in `main.py`.

```python
python main.py
```

# Reference

This implementation is based on projects:

[hwalsuklee/tensorflow-mnist-VAE](https://github.com/hwalsuklee/tensorflow-mnist-VAE)