from py_compile import main
import torch
import torchvision as tv
import os, wandb
from transformer_flow import Model
import utils.tarflow_utils as u
import pathlib
from utils.tarflow_utils import sqa_save
from absl import logging, app, flags

from utils.logging_utils import GoodLogger, to_uint_8, MyMetrics

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")

def train_and_evaluate(workdir):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1" # set GPU

    u.set_random_seed(42)

    desc = f'model: sqa_p2_c256_b6_l4, uncond, learned mu / sigma per class + clip 3.0 + random mu init 1.0'
    logging.info(desc)

    sample_dir = workdir + f'/samples'
    # ckpt_file = workdir + f'/checkpoint.pth'
    os.makedirs(sample_dir, exist_ok=True)

    # wandb.login(key='73f8ff40bb7f8589e9bd1f476196a896f662cdfa')
    wandb.init(entity="evazhu-massachusetts-institute-of-technology", project="DL_NF", dir=sample_dir, tags=[], settings=wandb.Settings(_service_wait=60), notes=desc)

    # training

    num_classes = 10
    img_size = 28
    channel_size = 1

    # we use a small model for fast demonstration, increase the model size for better results
    patch_size = 2
    channels = 256
    blocks = 6
    layers_per_block = 4
    # try different noise levels to see its effect
    noise_std = 0.1
    clip_range = 3.0

    batch_size = 256
    lr = 2e-4
    # increase epochs for better results
    epochs = 100
    sample_freq = 10

    if torch.cuda.is_available():
        device = 'cuda' 
    elif torch.backends.mps.is_available():
        device = 'mps' # if on mac
    else:
        device = 'cpu' # if mps not available
    logging.info(f'using device {device}')

    fixed_noise = torch.randn(num_classes * 10, (img_size // patch_size)**2, channel_size * patch_size ** 2, device=device)
    fixed_y = torch.arange(num_classes, device=device).view(-1, 1).repeat(1, 10).flatten()

    transform = tv.transforms.Compose([
        tv.transforms.Resize((img_size, img_size)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,), (0.5,))
    ])
    data = tv.datasets.MNIST('.', transform=transform, train=True, download=True)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    model = Model(in_channels=channel_size, img_size=img_size, patch_size=patch_size, 
                channels=channels, num_blocks=blocks, layers_per_block=layers_per_block,
                num_classes=num_classes, clip_range=clip_range).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=lr, weight_decay=1e-4)
    # lr_schedule = utils.CosineLRSchedule(optimizer, len(data_loader), epochs * len(data_loader), 1e-6, lr)
    lr_schedule = u.ConstLR_warmup(optimizer, len(data_loader), 5, lr)

    logging.info(f'logging into {sample_dir}/log.txt')

    step_per_ep = len(data_loader)

    logger = GoodLogger(workdir=sample_dir, use_wandb=True)
    metric_mnger = MyMetrics(reduction="avg")

    for epoch in range(epochs):
        logging.info(f'epoch {epoch}')
        step = epoch * step_per_ep
        for i_batch, (x, y) in enumerate(data_loader):
            x = x.to(device)
            eps = noise_std * torch.randn_like(x)
            # print(f'{x.min()=}, {x.max()=}, {eps.min()=}, {eps.max()=}')
            x = x + eps
            y = y.to(device)
            optimizer.zero_grad()
            z, outputs, logdets = model(x, y)
            loss = model.get_loss(z, y, logdets)
            loss.backward()
            optimizer.step()
            lr_schedule.step()
            
            metric_mnger.update({
                "loss": loss.item(),
                'logdet': logdets.mean().item(),
                'norm_prior': 0.5 * z.pow(2).mean().item(),
                'mu norm': model.mu.pow(2).mean().item(),
                'std mean': model.sigma.exp().mean().item(),
            })

            step = epoch * step_per_ep + i_batch

            if step % 100 == 0:
                with torch.no_grad():
                    log_dict = metric_mnger.compute_and_reset()
                    for i, z in enumerate(outputs):
                        log_dict[f'norm_layer_{i}'] = z.pow(2).mean()
                    logger.log_dict(step, log_dict)

                    # vis trajectory
                    # output: a list of shape (N, C, H, W)
                    vis = torch.cat(outputs, dim=2)[:6]
                    vis = vis.permute(2, 0, 3, 1)
                    vis = vis.reshape(vis.shape[0], -1, channel_size)
                    # print(f'{vis.shape=}')
                    vis = to_uint_8(vis)
                    wandb.log({'vis': wandb.Image(vis, mode='L', normalize=False)}, step=step)

                    # vis mu and sigma
                    mu = model.mu # (num_classes, num_patches, patch_dim)
                    mu_img = model.unpatchify(mu).reshape(2, 5, 1, 28, 28) # (2, 5, C, H, W)
                    mu_img = mu_img.permute(0, 3, 1, 4, 2).reshape(28 * 2, 28 * 5, 1)
                    mu_img = to_uint_8(mu_img)
                    wandb.log({'mu': wandb.Image(mu_img, mode='L', normalize=False)}, step=step)

                    sigma = model.sigma.exp() # (num_classes, num_patches, patch_dim)
                    sigma_img = model.unpatchify(sigma).reshape(2, 5, 1, 28, 28) # (2, 5, C, H, W)
                    sigma_img = sigma_img.permute(0, 3, 1, 4, 2).reshape(28 * 2, 28 * 5, 1)
                    wandb.log({'sigma': wandb.Image(sigma_img, mode='L', normalize=True)}, step=step) # normalize it, since we do not know its scale

        if (epoch + 1) % sample_freq == 0 \
        or epoch == 0:
            logging.info(f'Sampling at epoch {epoch}')
            with torch.no_grad():
                samples = model.reverse(fixed_noise, fixed_y)
            sqa_save(samples, sample_dir + f'/samples_{epoch:03d}.png')
            assert samples.shape == (100, 1, 28, 28)
            samples = samples.reshape(10, 10, 1, 28, 28).permute(0, 3, 1, 4, 2).reshape(10 * 28, 10 * 28, 1)
            s = to_uint_8(samples)
            wandb.log({'samples': wandb.Image(s, mode='L', normalize=False)}, step=step)

            latents = model.unpatchify(z[:100])
            sqa_save(latents, sample_dir + f'/latent_{epoch:03d}.png')
            assert latents.shape == (100, 1, 28, 28)
            latents = latents.reshape(10, 10, 1, 28, 28).permute(0, 3, 1, 4, 2).reshape(10 * 28, 10 * 28, 1)
            l = to_uint_8(latents)
            wandb.log({'latents': wandb.Image(l, mode='L', normalize=False)}, step=step)

            logging.info(f'sampling complete. Sample mean: {samples.mean():.4f}, std: {samples.std():.4f}, max: {samples.max():.4f}, min: {samples.min():.4f}')
            logging.info(f'latent mean: {latents.mean():.4f}, std: {latents.std():.4f}')
        logging.info('\n')

        if (epoch + 1) % sample_freq == 0:
            logging.info(f'Saving checkpoint at epoch {epoch}')
            ckpt_file = workdir + f'/checkpoint_epoch{epoch:03d}.pth'
            torch.save(model.state_dict(), ckpt_file)

def just_evaluate(workdir):
    ckpt_path = ''

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    
    train_and_evaluate(FLAGS.workdir)

if __name__ == "__main__":
    flags.mark_flags_as_required(["workdir"])
    app.run(main)