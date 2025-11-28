# import jax
import logging as sys_logging
from absl import logging
import wandb, torch, os
import numpy as np
from PIL import Image
from collections import defaultdict

class ExcludeInfo(sys_logging.Filter):
    def __init__(self, exclude_files):
        super().__init__()
        self.exclude_files = exclude_files

    def filter(self, record):
        # print('zhh ijijijijiji',record.pathname)
        if any(file_name in record.pathname for file_name in self.exclude_files):
            return record.levelno > sys_logging.INFO
        return True


exclude_files = [
    'orbax/checkpoint/async_checkpointer.py',
    'orbax/checkpoint/multihost/utils.py',
    'orbax/checkpoint/future.py',
    'orbax/checkpoint/_src/handlers/base_pytree_checkpoint_handler.py',
    'orbax/checkpoint/type_handlers.py',
    'orbax/checkpoint/metadata/checkpoint.py',
    'orbax/checkpoint/metadata/sharding.py',
]
file_filter = ExcludeInfo(exclude_files)


def supress_checkpt_info():
    logging.get_absl_handler().addFilter(file_filter)


class GoodLogger:

    def __init__(self, workdir, use_wandb=False):
        self.use_wandb = use_wandb
        self.workdir = workdir
        if use_wandb:
            if wandb.run is None:
                raise RuntimeError(
                    "Failed to initialize wandb. Please check your wandb login. Also make sure to initialize the logger after creating the wandb run."
                )
        else:
            assert os.path.exists(workdir), f"workdir {workdir} does not exist"
            os.makedirs(os.path.join(workdir, 'zhh_images'), exist_ok=True)


    def log_dict(self, step, dict_obj):
        # [200] ep=0.159073, steps_per_second=6.76798, train_accuracy=0.00585938, train_loss=6.71379, train_lr=0.0127258, train_step=199
        log_str = f"[{step}]"
        for k, v in dict_obj.items():
            log_str += f" {k}={v:.5f}," if isinstance(v, float) else f" {k}={v},"
        log_str = log_str.strip(",")
        logging.info(log_str)
        if self.use_wandb:
            wandb.log(dict_obj, step=step)

    def log_image(self, step, image_dict):
        raise DeprecationWarning("log_image is deprecated, please use other image logging methods.")

        def reduce_arr_func(v):
            assert isinstance(v, np.ndarray), "Invalid image type {}".format(type(v))
            assert v.dtype == np.uint8, "Invalid image dtype {}".format(v.dtype)
            assert (
                v.ndim == 3
                and 3 in [v.shape[0], v.shape[2]]
            ), "Invalid image shape {}".format(v.shape)
            if v.shape[0] == 3:
                v = v.transpose((1, 2, 0))
            return v

        if self.use_wandb:
            wandb.log({
                k: wandb.Image(reduce_arr_func(v)) for k, v in image_dict.items()
            }, step=step)
        else:
            log_for_0(f"Saving images locally, at step {step}")
            for k, v in image_dict.items():
                v = Image.fromarray(reduce_arr_func(v))
                v.save(os.path.join(self.workdir, 'zhh_images', f"step{step:06d}_{k}.png"))

    # destructor
    def __del__(self):
        if self.use_wandb:
            wandb.finish()
class NullLogger:
    @staticmethod
    def log_dict(*args, **kwargs):
        pass

    @staticmethod
    def log_image(*args, **kwargs):
        pass

def to_uint_8(img):
    img = (img + 1.) / 2. * 255
    img = img.clamp(0, 255).to(torch.uint8)
    # to PIL
    img = Image.fromarray(img.detach().cpu().numpy()[:, :, 0], mode='L')
    return img

class LastItem:
    def __init__(self):
        self.val = 114514.1919810

    def append(self, v):
        self.val = v

    def get(self):
        return self.val


class Avger(list):
    def get(self):
        return sum(self) / len(self) if len(self) > 0 else 114514.1919810

class MyMetrics:
    def __init__(self, reduction="last"):
        self.reduction_cls = {
            "last": LastItem,
            "avg": Avger,
        }[reduction]
        self.metrics = defaultdict(self.reduction_cls)

    def update(self, metrics):
        for k, v in metrics.items():
            self.metrics[k].append(v)

    def compute(self, *keys):
        if len(keys) == 0:
            return {k: float(v.get()) for k, v in self.metrics.items()}
        else:
            raise NotImplementedError
            # return tuple(self.metrics[k].get() for k in keys)

    def reset(self):
        self.metrics = defaultdict(self.reduction_cls)

    def compute_and_reset(self, *keys):
        a = self.compute(*keys)
        self.reset()
        return a