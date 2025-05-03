import torch
from torch import nn
from .utils import get_activation
from typing import Optional, Tuple


class DWLayer(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, kernel_size: int = 3, stride: int = 1, 
                 padding: Optional[int] = None, bias: bool = False, act: Optional[str] = None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        
        self.Dconv = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, stride=stride, padding=padding, 
                               groups=ch_in, bias=bias)
        self.Wconv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Dconv(x)
        x = self.Wconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class SinCosPositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim: int, temperature: float = 10000.):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

    def forward(self, h: int, w: int, device: str = 'cpu') -> torch.Tensor:
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')

        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1. / (self.temperature ** omega)

        out_w = torch.einsum('i,j->ij', grid_w.flatten(), omega)
        out_h = torch.einsum('i,j->ij', grid_h.flatten(), omega)

        pos_embed = torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)
        return pos_embed.unsqueeze(0)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos_embed: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, 
                pos1: Optional[torch.Tensor] = None, 
                pos2: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x2
        q = self.with_pos_embed(x1, pos1)
        k = self.with_pos_embed(x1, pos1)
        v = self.with_pos_embed(x2, pos2)
        
        x2, _ = self.self_attn(q, k, v)
        x2 = residual + self.dropout(x2)
        output = self.norm(x2)
        return output


class MSCFE(nn.Module):
    def __init__(self, ch_in: int, ch_next: Optional[int] = None, num_heads: int = 8):
        super().__init__()
        self.cross_attention = MultiHeadCrossAttention(d_model=ch_in, nhead=num_heads)
        self.norm = nn.BatchNorm2d(ch_in)
        
        self.pos_encoder_ol = SinCosPositionalEncoding2D(embed_dim=ch_in)
        self.pos_encoder_sd = SinCosPositionalEncoding2D(embed_dim=ch_in)
        
        self.ch_next = ch_next
        if self.ch_next is not None:
            self.ffn = DWLayer(ch_in, ch_next)

            self.spatial = nn.Conv2d(2, 1, 7, stride=1, padding=(7 - 1) // 2)
            self.sigmoid = nn.Sigmoid()

    def forward(self, f_ol: torch.Tensor, f_sd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = f_ol.size()
        
        f_ol_flat = f_ol.flatten(2).permute(0, 2, 1)
        f_sd_flat = f_sd.flatten(2).permute(0, 2, 1)
        
        pos_ol = self.pos_encoder_ol(h=H, w=W, device=f_ol.device)
        pos_sd = self.pos_encoder_sd(h=H, w=W, device=f_ol.device)
        
        f_sd_attended = self.cross_attention(f_ol_flat, f_sd_flat, pos_ol, pos_sd)
        f_ol_attended = self.cross_attention(f_sd_flat, f_ol_flat, pos_sd, pos_ol)

        f_ol_attended = f_ol_attended.permute(0, 2, 1).view(B, C, H, W)
        f_sd_attended = f_sd_attended.permute(0, 2, 1).view(B, C, H, W)
        
        f_ol = self.norm(f_ol_attended) + f_ol
        f_sd = self.norm(f_sd_attended) + f_sd
        # f_ol = self.norm(f_ol_attended + f_ol)
        # f_sd = self.norm(f_sd_attended + f_sd)   
        # f_ol = f_ol_attended + f_ol
        # f_sd = f_sd_attended + f_sd
        s_ol = s_sd = torch.ones(B, 1, H//2, W//2, device=f_ol.device)
        
        if self.ch_next is not None:
            ffn_ol = self.ffn(f_ol)
            ffn_sd = self.ffn(f_sd)
            
            s_ol_features = torch.cat((
                torch.max(ffn_ol, 1)[0].unsqueeze(1), 
                torch.mean(ffn_ol, 1).unsqueeze(1)
            ), dim=1)
            s_ol = self.spatial(s_ol_features)
            
            s_sd_features = torch.cat((
                torch.max(ffn_sd, 1)[0].unsqueeze(1), 
                torch.mean(ffn_sd, 1).unsqueeze(1)
            ), dim=1)
            s_sd = self.spatial(s_sd_features)
            
            s_ol = self.sigmoid(s_ol)
            s_sd = self.sigmoid(s_sd)
            
        return f_ol, f_sd, s_ol, s_sd
