"""This is code based on https://sudomake.ai/inception-score-explained/."""
import torch
import torchvision

from collections import defaultdict

class InceptionScore(torch.nn.Module):
    """Class that manages and returns the inception score of images."""

    def __init__(self, batch_size=32, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize with setup and target inception batch size."""
        super().__init__()
        self.preprocessing = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
        self.model = torchvision.models.inception_v3(pretrained=True).to(**setup)
        self.model.eval()
        self.batch_size = batch_size

    def forward(self, image_batch):
        """Image batch should have dimensions BCHW and should be normalized.

        B should be divisible by self.batch_size.
        """
        B, C, H, W = image_batch.shape
        batches = B // self.batch_size
        scores = []
        for batch in range(batches):
            input = self.preprocessing(image_batch[batch * self.batch_size: (batch + 1) * self.batch_size])
            scores.append(self.model(input))
        prob_yx = torch.nn.functional.softmax(torch.cat(scores, 0), dim=1)
        entropy = torch.where(prob_yx > 0, -prob_yx * prob_yx.log(), torch.zeros_like(prob_yx))
        return entropy.sum()


def psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """
    Compute PSNR over 2D or 3D (or even higher‐dimensional) images.
    
    Args:
        img_batch (torch.Tensor):   Predicted tensor of shape [B, C, *spatial_dims].
        ref_batch (torch.Tensor):   Reference tensor of the same shape.
        batched (bool):             If True, treat img_batch/ref_batch as a single multi‐dimensional volume 
                                    and compute one PSNR over ALL elements. If False, compute PSNR per‐sample 
                                    (over dims [C, *spatial_dims]) and then average across B.
        factor (float):             The dynamic‐range factor. If your images are in [0,1], you can leave factor=1.0.
                                    Otherwise, set factor to (max_val_of_reference).
    
    Returns:
        float: mean PSNR (in dB) as a Python float.
    """
    def get_psnr_single(img_in: torch.Tensor, img_ref: torch.Tensor) -> torch.Tensor:
        # img_in, img_ref: shape [C, D, H, W] (for 3D) or [C, H, W] (for 2D), etc.
        mse = ((img_in - img_ref) ** 2).mean()
        if mse > 0 and torch.isfinite(mse):
            return 10 * torch.log10((factor ** 2) / mse)
        elif not torch.isfinite(mse):
            return img_in.new_tensor(float('nan'))
        else:
            return img_in.new_tensor(float('inf'))
    
    if batched:
        # Treat the entire batch as one giant volume:
        # img_batch: [B, C, D, H, W]  (or [B, C, H, W], etc.)
        # Just compute MSE over all elements at once:
        mse = ((img_batch - ref_batch) ** 2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10((factor ** 2) / mse)).item()
        elif not torch.isfinite(mse):
            return float('nan')
        else:
            return float('inf')
    else:
        # Compute PSNR per‐sample, then average over batch dimension B.
        B = img_batch.shape[0]
        psnrs = []
        for i in range(B):
            # Select the i‐th volume: shape [C, D, H, W] (or [C, H, W], etc.)
            pred_vol = img_batch[i].detach()
            ref_vol  = ref_batch[i]
            psnrs.append(get_psnr_single(pred_vol, ref_vol))
        psnrs = torch.stack(psnrs, dim=0)  # shape [B]
        return psnrs.mean().item()



def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy



def activation_errors(model, x1, x2):
    """Compute activation-level error metrics for every module in the network."""
    model.eval()

    device = next(model.parameters()).device

    hooks = []
    data = defaultdict(dict)
    inputs = torch.cat((x1, x2), dim=0)
    separator = x1.shape[0]

    def check_activations(self, input, output):
        module_name = str(*[name for name, mod in model.named_modules() if self is mod])
        try:
            layer_inputs = input[0].detach()
            residual = (layer_inputs[:separator] - layer_inputs[separator:]).pow(2)
            se_error = residual.sum()
            mse_error = residual.mean()
            sim = torch.nn.functional.cosine_similarity(layer_inputs[:separator].flatten(),
                                                        layer_inputs[separator:].flatten(),
                                                        dim=0, eps=1e-8).detach()
            data['se'][module_name] = se_error.item()
            data['mse'][module_name] = mse_error.item()
            data['sim'][module_name] = sim.item()
        except (KeyboardInterrupt, SystemExit):
            raise
        except AttributeError:
            pass

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(check_activations))

    try:
        outputs = model(inputs.to(device))
        for hook in hooks:
            hook.remove()
    except Exception as e:
        for hook in hooks:
            hook.remove()
        raise

    return data
