from py_compile import main
import torch
import torchvision as tv
import os, wandb
from transformer_flow import Model
import utils
import pathlib
from utils import sqa_save
from absl import logging, app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")

def train_and_evaluate(workdir):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1" # set GPU

    utils.set_random_seed(100)

    desc = f'model: sqa_p2_c256_b6_l4, uncond'
    logging.info(desc)

    sample_dir = workdir + f'/samples'
    ckpt_file = workdir + f'/checkpoint.pth'
    os.makedirs(sample_dir, exist_ok=True)

    # wandb.login(key='73f8ff40bb7f8589e9bd1f476196a896f662cdfa')
    wandb.init(entity="evazhu-massachusetts-institute-of-technology", project="DL_NF", dir=sample_dir, tags=[], settings=wandb.Settings(_service_wait=60))

    # training

    dataset = 'mnist'
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

    batch_size = 256
    lr = 1e-3
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
                num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=lr, weight_decay=1e-4)
    # lr_schedule = utils.CosineLRSchedule(optimizer, len(data_loader), epochs * len(data_loader), 1e-6, lr)
    lr_schedule = utils.ConstLR_warmup(optimizer, len(data_loader), 5, lr)

    logging.info(f'logging into {sample_dir}/log.txt')

    step_per_ep = len(data_loader) // batch_size

    for epoch in range(epochs):
        losses = 0
        for x, y in data_loader:
            x = x.to(device)
            eps = noise_std * torch.randn_like(x)
            x = x + eps
            y = y.to(device)
            optimizer.zero_grad()
            z, outputs, logdets = model(x, y)
            loss = model.get_loss(z, logdets)
            loss.backward()
            optimizer.step()
            lr_schedule.step()
            losses += loss.item()

        log_dict = {
            "loss": losses / len(data_loader),
            "lr": optimizer.param_groups[0]['lr'],
            'logdet': logdets.mean(),
            'norm_prior': 0.5 * z.pow(2).mean()
        }

        for i, z in enumerate(outputs):
            log_dict[f'norm_layer_{i}'] = z.pow(2).mean()
        
        logging.info(f'epoch {epoch}')
        for k, v in log_dict.items():
            logging.info(f'{k}: {v}')

        wandb.log(log_dict, step=step_per_ep*epoch)

        if (epoch + 1) % sample_freq == 0 \
        or epoch == 0:
            logging.info(f'Sampling at epoch {epoch}')
            with torch.no_grad():
                samples = model.reverse(fixed_noise, fixed_y)
            sqa_save(samples, sample_dir + f'/samples_{epoch:03d}.png')
            assert samples.shape == (100, 1, 28, 28)
            samples = samples.reshape(10, 10, 1, 28, 28).permute(0, 3, 1, 4, 2).reshape(10 * 28, 10 * 28, 1)
            wandb.log({'samples': wandb.Image(samples, mode='L')}, step=step_per_ep*epoch)

            latents = model.unpatchify(z[:100])
            sqa_save(latents, sample_dir + f'/latent_{epoch:03d}.png')
            assert latents.shape == (100, 1, 28, 28)
            latents = latents.reshape(10, 10, 1, 28, 28).permute(0, 3, 1, 4, 2).reshape(10 * 28, 10 * 28, 1)
            wandb.log({'latents': wandb.Image(latents, mode='L')}, step=step_per_ep*epoch)

            logging.info(f'sampling complete. Sample mean: {samples.mean():.4f}, std: {samples.std():.4f}, max: {samples.max():.4f}, min: {samples.min():.4f}')
            logging.info(f'latent mean: {latents.mean():.4f}, std: {latents.std():.4f}')
        logging.info('\n')
    torch.save(model.state_dict(), ckpt_file)

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    
    train_and_evaluate(FLAGS.workdir)

if __name__ == "__main__":
    flags.mark_flags_as_required(["workdir"])
    app.run(main)