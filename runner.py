import os
import time
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torchvision.utils as tvu
from models.model_utils import *
from models.diffusion import Model
from functions.optimizer import get_optimizer
from functions.losses import loss_registry
from functions.denoising import generalized_steps
from dataset import get_dataset, data_transform, inverse_data_transform
from torchvision.models import inception_v3
from torchvision import transforms
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image
from scipy.linalg import sqrtm
from tqdm import tqdm
from models.ema import EMAHelper


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas
#
class Diffusion:
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = config["model"]["name"]

        self.model_log_dir = os.path.join(self.args.log_path, self.model_name)
        os.makedirs(self.model_log_dir, exist_ok=True)

        self.image_folder = os.path.join(self.model_log_dir, "images", "fake")
        os.makedirs(self.image_folder, exist_ok=True)

        self.model_var_type = config["model"]["var_type"]
        betas = get_beta_schedule(
            beta_schedule=config["diffusion"]["beta_schedule"],
            beta_start=config["diffusion"]["beta_start"],
            beta_end=config["diffusion"]["beta_end"],
            num_diffusion_timesteps=config["diffusion"]["num_diffusion_timesteps"],
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1.0 - self.betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(self.device), alphas_cumprod[:-1]])
        post_var = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.logvar = self.betas.log() if self.model_var_type == "fixedlarge" else post_var.clamp(min=1e-20).log()
    def train(self):
        config = self.config
        dataset, _ = get_dataset(self.args, config)
        loader = data.DataLoader(dataset, batch_size=config["training"]["batch_size"],
                                shuffle=True, num_workers=config["data"]["num_workers"])

        # Khởi tạo model
        model_name = config["model"]["name"]
        if model_name == "MLP":
            from models.diffusion import Model as MLPModel
            model = MLPModel(config)
        elif model_name == "KAN":
            from models.diffusion_KAN import Model_KAN as KANModel
            model = KANModel(config)
        else:
            model = Model(config)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,} ({total_params/1e6:.3f}M)")
        print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.3f}M)")

        model = torch.nn.DataParallel(model.to(self.device))
        optimizer = get_optimizer(config, model.parameters())
        if self.config["model"]["ema"]:
            ema_helper = EMAHelper(mu=self.config["model"]["ema_rate"])
            ema_helper.register(model)
        else:
            ema_helper = None
        step = 0
        ckpt_dir = self.model_log_dir  # Thư mục log riêng cho model
        os.makedirs(ckpt_dir, exist_ok=True)

        for epoch in range(config["training"]["n_epochs"]):
            t0 = time.time()

            # Bọc loader bằng tqdm
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config['training']['n_epochs']}", ncols=120)

            for i, (x, y) in enumerate(pbar):
                n = x.size(0)
                model.train()
                step += 1

                x = data_transform(config, x.to(self.device))
                e = torch.randn_like(x)

                t = torch.randint(0, self.num_timesteps, size=(n//2+1,), device=self.device)
                t = torch.cat([t, self.num_timesteps - t - 1])[:n]

                loss = loss_registry[config["model"]["type"]](model, x, t, e, self.betas)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["optim"]["grad_clip"])
                optimizer.step()
                if self.config["model"]["ema"]:
                    ema_helper.update(model)
                # update tqdm progress bar
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

                # Lưu checkpoint
                if step % config["training"]["snapshot_freq"] == 0:
                    states = [model.state_dict(), optimizer.state_dict(), epoch, step]
                    if self.config["model"]["ema"]:
                        states.append(ema_helper.state_dict())
                    torch.save(states, os.path.join(ckpt_dir, f"ckpt_{step}.pth"))
                    torch.save(states, os.path.join(ckpt_dir, "ckpt.pth"))

            # end epoch
            logging.info(f"Epoch {epoch} finished, avg_time_per_item={(time.time()-t0)/len(loader):.4f}s")

        # save final ckpt
        states = [model.state_dict(), optimizer.state_dict(), epoch, step]
        torch.save(states, os.path.join(ckpt_dir, "ckpt_final.pth"))
        torch.save(states, os.path.join(ckpt_dir, "ckpt.pth"))
        logging.info(f"Saved final checkpoint at step {step} to {ckpt_dir}")

    def load_model(self, ckpt_path=None):
        """
        Load model + EMA (nếu có)
        Trả về: model, ema_helper
        """
        # Create model
        model = Model(self.config)
        model = torch.nn.DataParallel(model.to(self.device))

        # Load checkpoint
        ckpt_path = ckpt_path or os.path.join(self.args.log_path, "ckpt.pth")
        states = torch.load(ckpt_path, map_location=self.device)

        # Load main model weights
        model.load_state_dict(states[0], strict=False)

        # Load EMA weights if enabled
        if self.config["model"]["ema"] and len(states) >= 5:
            ema_helper = EMAHelper(mu=self.config["model"]["ema_rate"])
            ema_helper.register(model)               # Create EMA shadow copy
            ema_helper.load_state_dict(states[-1])   # Load EMA weights
            ema_helper.ema(model)                    # Apply EMA → model becomes EMA model
        else:
            ema_helper = None

        model.eval()
        return model

    def sample_fid(self, model, n_samples=10):


        fake_folder = self.image_folder
        os.makedirs(fake_folder, exist_ok=True)

        img_id = len(glob.glob(f"{fake_folder}/*"))
        batch = self.config["sampling"]["batch_size"]
        total_needed = n_samples - img_id
        n_rounds = total_needed // batch
        remaining = total_needed % batch

        with torch.no_grad():
            for _ in tqdm(range(n_rounds), desc=f"Generating samples for {self.model_name}"):
                x = torch.randn(
                    batch,
                    self.config["data"]["channels"],
                    self.config["data"]["image_size"],
                    self.config["data"]["image_size"],
                    device=self.device
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(self.config, x)

                for i in range(x.size(0)):
                    tvu.save_image(x[i], os.path.join(fake_folder, f"{img_id}.png"))
                    img_id += 1
        if remaining > 0:
            x = torch.randn(
                remaining,
                self.config["data"]["channels"],
                self.config["data"]["image_size"],
                self.config["data"]["image_size"],
                device=self.device,
            )
            x = self.sample_image(x, model)
            x = inverse_data_transform(self.config, x)

            for i in range(x.size(0)):
                tvu.save_image(x[i], os.path.join(fake_folder, f"{img_id}.png"))
                img_id += 1


    def sample_image(self, x, model, last=True):
        skip = self.num_timesteps // 1000
        seq = range(0, self.num_timesteps, skip)
        xs = generalized_steps(x, seq, model, self.betas, eta=0.0)
        x = xs
        x = x[0][-1]
        return x
