#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import torch
from utils import nan_or_inf

class Permutation(torch.nn.Module):

    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError('Overload me')


class PermutationIdentity(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])


class Attention(torch.nn.Module):
    USE_SPDA: bool = True

    def __init__(self, in_channels: int, head_channels: int):
        assert in_channels % head_channels == 0
        super().__init__()
        self.norm = torch.nn.LayerNorm(in_channels)
        self.qkv = torch.nn.Linear(in_channels, in_channels * 3)
        self.proj = torch.nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        self.sqrt_scale = head_channels ** (-0.25)
        self.sample = False
        self.k_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}
        self.v_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}

    def forward_spda(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, t, d)

        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)  # note that sequence dimension is now 2
            v = torch.cat(self.v_cache[which_cache], dim=2)

        scale = self.sqrt_scale**2 / temp
        if mask is not None:
            mask = mask.bool()
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward_base(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).chunk(3, dim=2)
        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=1)
            v = torch.cat(self.v_cache[which_cache], dim=1)

        attn = torch.einsum('bmhd,bnhd->bmnh', q * self.sqrt_scale, k * self.sqrt_scale) / temp
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        attn = attn.float().softmax(dim=-2).type(attn.dtype)
        x = torch.einsum('bmnh,bnhd->bmhd', attn, v)
        x = x.reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        if self.USE_SPDA:
            return self.forward_spda(x, mask, temp, which_cache)
        return self.forward_base(x, mask, temp, which_cache)


class MLP(torch.nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)
        self.main = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * expansion),
            torch.nn.GELU(),
            torch.nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))


class AttentionBlock(torch.nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: int = 4):
        super().__init__()
        self.attention = Attention(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, attn_temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        return x


class MetaBlock(torch.nn.Module):
    attn_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_patches: int,
        permutation: Permutation,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        num_classes: int = 0,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(in_channels, channels)
        self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        if num_classes:
            self.class_embed = torch.nn.Parameter(torch.randn(num_classes, 1, channels) * 1e-2)
        else:
            self.class_embed = None
        self.attn_blocks = torch.nn.ModuleList(
            [AttentionBlock(channels, head_dim, expansion) for _ in range(num_layers)]
        )
        self.nvp = nvp
        output_dim = in_channels * 2 if nvp else in_channels
        self.proj_out = torch.nn.Linear(channels, output_dim)
        self.proj_out.weight.data.fill_(0.0)
        self.permutation = permutation
        self.register_buffer('attn_mask', torch.tril(torch.ones(num_patches, num_patches)))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_T' <- (x_T-b(x_1, ..., x_T-1)) * exp(-a(x_1, ..., x_T-1))
        this part can be paralleled, by using attn_mask
        """
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        x_in = x
        x = self.proj_in(x) + pos_embed
        if self.class_embed is not None:
            if y is not None:
                if (y < 0).any():
                    m = (y < 0).float().view(-1, 1, 1)
                    class_embed = (1 - m) * self.class_embed[y] + m * self.class_embed.mean(dim=0)
                else:
                    class_embed = self.class_embed[y]
                x = x + class_embed
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, self.attn_mask)
        x = self.proj_out(x)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1) # to make x_1 unchanged, a and b for x_1 should be 0

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        scale = (-xa.float()).exp().type(xa.dtype)
        return self.permutation((x_in - xb) * scale, inverse=True), -xa.mean(dim=[1, 2])

    def reverse_step(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        i: int,
        y: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = 'cond',
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_T' <- exp(a(x_1', ..., x_T-1'))x_T + b(x_1', ..., x_T-1')
        this should be done in a loop, i.e. cannot parallel
        """
        x_in = x[:, i : i + 1]  # get i-th patch but keep the sequence dimension
        x = self.proj_in(x_in) + pos_embed[i : i + 1]
        if self.class_embed is not None:
            if y is not None:
                x = x + self.class_embed[y]
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, attn_temp=attn_temp, which_cache=which_cache)  # here we use kv caching, so no attn_mask
        x = self.proj_out(x)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)
        return xa, xb

    def set_sample_mode(self, flag: bool = True):
        for m in self.modules():
            if isinstance(m, Attention):
                m.sample = flag
                m.k_cache = {'cond': [], 'uncond': []}
                m.v_cache = {'cond': [], 'uncond': []}

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
    ) -> torch.Tensor:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        self.set_sample_mode(True)
        T = x.size(1)
        for i in range(x.size(1) - 1):
            za, zb = self.reverse_step(x, pos_embed, i, y, which_cache='cond')
            if guidance > 0 and guide_what:
                za_u, zb_u = self.reverse_step(x, pos_embed, i, None, attn_temp=attn_temp, which_cache='uncond')
                if annealed_guidance:
                    g = (i + 1) / (T - 1) * guidance
                else:
                    g = guidance
                if 'a' in guide_what:
                    za = za + g * (za - za_u)
                if 'b' in guide_what:
                    zb = zb + g * (zb - zb_u)

            scale = za[:, 0].float().exp().type(za.dtype)  # get rid of the sequence dimension
            x[:, i + 1] = x[:, i + 1] * scale + zb[:, 0]
        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)


class Unitary(torch.nn.Module):
    # apply this on N tokens (try to replace the permutation)
    def __init__(self, num_tokens: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_tokens, num_tokens) * 1e-2 + torch.eye(num_tokens))
        logdet = torch.slogdet(self.weight)[1] / (num_tokens)
        print(f"Unitary logdet: {logdet:.2f}")
        # # sanity check
        # x = torch.arange(num_tokens).reshape(1, -1, 1).float()
        # y = self.forward(x)
        # print(self.reverse(y).reshape(-1))
        # assert torch.allclose(self.reverse(y), x), "Reverse operation did not match original input!"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        out = torch.einsum('bnc,nm->bmc', x, self.weight)
        return out
    
    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        out = torch.einsum('bnc,nm->bmc', x, self.weight.inverse())
        return out

class Model(torch.nn.Module):
    VAR_LR: float = 0.1
    var: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int,
        nvp: bool = True,
        num_classes: int = 0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.channels = channels
        self.num_patches = (img_size // patch_size) ** 2
        self.pixel_channels = pixel_channels = in_channels * patch_size**2
        self.num_blocks = num_blocks

        permutations = [PermutationIdentity(self.num_patches), PermutationFlip(self.num_patches)]

        blocks = []
        unitaries = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    pixel_channels,
                    # channels,
                    channels,
                    self.num_patches,
                    permutations[i % 2],
                    layers_per_block,
                    nvp=nvp,
                    num_classes=num_classes,
                )
            )
            unitaries.append(Unitary(self.num_patches))
        self.unitaries = torch.nn.ModuleList(unitaries)
        self.blocks = torch.nn.ModuleList(blocks)
        # prior for nvp mode should be all ones, but needs to be learnd for the vp mode
        self.register_buffer('var', torch.ones(self.num_patches, pixel_channels))
        # print number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        print(f'Number of parameters: {num_params / 1e6:.2f}M')

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert an image (N,C',H,W) to a sequence of patches (N,T,C')"""
        u = torch.nn.functional.unfold(x, self.patch_size, stride=self.patch_size)
        u = u.transpose(1, 2)
        return u

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a sequence of patches (N,T,C) to an image (N,C',H,W)"""
        u = x.transpose(1, 2)
        u = torch.nn.functional.fold(u, (self.img_size, self.img_size), self.patch_size, stride=self.patch_size)
        return u

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        x = self.patchify(x)
        nan_or_inf(x, "patchify")
        outputs = []
        logdets = torch.zeros((), device=x.device)
        for i in range(self.num_blocks):
            block = self.blocks[i]
            unitary = self.unitaries[i]
            x, logdet = block(x, y)
            nan_or_inf(logdet, f"block {i} logdet")
            nan_or_inf(x, f"block {i} output")
            x = unitary(x)
            u_logdet = torch.slogdet(unitary.weight)[1] / (self.num_patches)
            nan_or_inf(u_logdet, f"block {i} unitary logdet")
            logdets = logdets + logdet + u_logdet
            outputs.append(x)
        return x, outputs, logdets

    def update_prior(self, z: torch.Tensor):
        z2 = (z**2).mean(dim=0)
        self.var.lerp_(z2.detach(), weight=self.VAR_LR)

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor):
        return 0.5 * z.pow(2).mean() - logdets.mean()

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        return_sequence: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        seq = [self.unpatchify(x)]
        x = x * self.var.sqrt()
        for i in range(self.num_blocks-1, -1, -1):
            block = self.blocks[i]
            unitary = self.unitaries[i]
            x = block.reverse(x, y, guidance, guide_what, attn_temp, annealed_guidance)
            nan_or_inf(x, f"reverse, block {i} output")
            x = unitary.reverse(x)
            nan_or_inf(x, f"reverse, block {i} unitary output")
            seq.append(self.unpatchify(x))
        x = self.unpatchify(x)

        if not return_sequence:
            return x
        else:
            return seq
