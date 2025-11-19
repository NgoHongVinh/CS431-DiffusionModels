config = {
    "data": {
        "dataset": "CIFAR10",
        "image_size": 32,
        "channels": 3,
        "logit_transform": False,
        "uniform_dequantization": False,
        "gaussian_dequantization": False,
        "rescaled": True,
        "num_workers": 4
    },

    "model": {
        "name": "MLP",
        "type": "simple",
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 2, 2, 2],
        "num_res_blocks": 1,
        "attn_resolutions": [],
        "dropout": 0.1,
        "var_type": "fixedlarge",
        "ema_rate": 0.9999,
        "ema": False,
        "resamp_with_conv": True
    },

    "diffusion": {
        "beta_schedule": "quad",
        "beta_start": 0.00009,
        "beta_end": 0.01,
        "num_diffusion_timesteps": 1000
    },

    "training": {
        "batch_size": 32,
        "n_epochs": 50,
        "n_iters": 5,
        "snapshot_freq": 4000,
        "validation_freq": 2000
    },

    "sampling": {
        "batch_size": 32,
        "last_only": True
    },

    "optim": {
        "weight_decay": 0.0,
        "optimizer": "Adam",
        "lr": 0.0002,
        "beta1": 0.9,
        "amsgrad": False,
        "eps": 1e-8,
        "grad_clip": 1.0
    }
}
