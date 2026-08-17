import torch
import torch.nn as nn
import torch.fft

class FrequencyAwareLoss(nn.Module):
    def __init__(self, freq_weight=0.3, high_fract=0.75):
        super().__init__()
        self.freq_weight = freq_weight
        self.high_fract = high_fract
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        pred_mag = torch.fft.fftshift(pred_mag, dim=(-2, -1))
        target_mag = torch.fft.fftshift(target_mag, dim=(-2, -1))
        B, C, H, W = pred_mag.shape
        pred_mag_flat = pred_mag.permute(0, 2, 3, 1).contiguous().view(-1)
        target_mag_flat = target_mag.permute(0, 2, 3, 1).contiguous().view(-1)
        num_elements = pred_mag_flat.numel()
        k = int(self.high_fract * num_elements)
        if k > 0:
            _, top_indices = torch.topk(pred_mag_flat, k, sorted=False)
            pred_top = pred_mag_flat[top_indices]
            target_top = target_mag_flat[top_indices]
            freq_l1 = self.l1(pred_top, target_top)
        else:
            freq_l1 = torch.tensor(0.0, device=pred.device)
        total_loss = (1 - self.freq_weight) * l1_loss + self.freq_weight * freq_l1
        return total_loss

