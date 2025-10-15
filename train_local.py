import torch
import torchvision as tv
import os
from transformer_flow import Model
import utils
import pathlib
from utils import sqa_save

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set GPU

utils.set_random_seed(100)
notebook_output_path = pathlib.Path('runs/notebook')

dataset = 'mnist'
num_classes = 10
img_size = 28
channel_size = 1

# we use a small model for fast demonstration, increase the model size for better results
patch_size = 4
channels = 128
blocks = 4
layers_per_block = 4
# try different noise levels to see its effect
noise_std = 0.1

batch_size = 256
lr = 2e-3
# increase epochs for better results
epochs = 100
sample_freq = 10

if torch.cuda.is_available():
    device = 'cuda' 
elif torch.backends.mps.is_available():
    device = 'mps' # if on mac
else:
    device = 'cpu' # if mps not available
print(f'using device {device}')

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
lr_schedule = utils.CosineLRSchedule(optimizer, len(data_loader), epochs * len(data_loader), 1e-6, lr)

model_name = f'original'
sample_dir = notebook_output_path / f'{dataset}_samples_{model_name}'
ckpt_file = notebook_output_path / f'{dataset}_model_{model_name}.pth'
sample_dir.mkdir(exist_ok=True, parents=True)

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

    print(f"epoch {epoch} lr {optimizer.param_groups[0]['lr']:.6f} loss {losses / len(data_loader):.4f}")
    print('layer norms', ' '.join([f'{z.pow(2).mean():.4f}' for z in outputs]))
    print(f'logdet: {logdets.mean():.4f}, prior p: {0.5 * z.pow(2).mean():.4f}')
    if (epoch + 1) % sample_freq == 0:
        with torch.no_grad():
            samples = model.reverse(fixed_noise, fixed_y)
        sqa_save(samples, sample_dir / f'samples_{epoch:03d}.png')
        latents = model.unpatchify(z[:100])
        sqa_save(latents, sample_dir / f'latent_{epoch:03d}.png')
        print(f'sampling complete. Sample mean: {samples.mean():.4f}, std: {samples.std():.4f}, max: {samples.max():.4f}, min: {samples.min():.4f}')
        print(f'latent mean: {latents.mean():.4f}, std: {latents.std():.4f}')
    print('\n')
torch.save(model.state_dict(), ckpt_file)

# now we can also evaluate the model by turning it into a classifier with Bayes rule, p(y|x) = p(y)p(x|y)/p(x)
data = tv.datasets.MNIST('.', transform=transform, train=False, download=False)
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False)
num_correct = 0
num_examples = 0
for x, y in data_loader:
    x = x.to(device)
    y = y.to(device)
    eps = noise_std * torch.randn_like(x)
    x = x.repeat(num_classes, 1, 1, 1)
    y_ = torch.arange(num_classes, device=device).view(-1, 1).repeat(1, y.size(0)).flatten()
    with torch.no_grad():
        z, outputs, logdets = model(x, y_)
        losses = 0.5 * z.pow(2).mean(dim=[1, 2]) - logdets # keep the batch dimension
        pred = losses.reshape(num_classes, y.size(0)).argmin(dim=0)
    num_correct += (pred == y).sum()
    num_examples += y.size(0)
print(f'Accuracy %{100 * num_correct / num_examples:.2f}')