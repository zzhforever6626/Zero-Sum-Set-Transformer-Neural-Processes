import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from attrdict import AttrDict
import torch
import torch.nn as nn

from models.modules import build_mlp
from models.zeros import ZeroSTransformerEncoderLayer


class TNP(nn.Module):
    def __init__(
        self,
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        bound_std
    ):
        super(TNP, self).__init__()

        self.embedder = build_mlp(dim_x + dim_y, d_model, d_model, emb_depth)

        encoder_layer = ZeroSTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.bound_std = bound_std

    def construct_input(self, batch, autoreg=False):
        x_y_ctx = torch.cat((batch.xc, batch.yc), dim=-1)
        x_0_tar = torch.cat((batch.xt, torch.zeros_like(batch.yt)), dim=-1)
        if not autoreg:
            inp = torch.cat((x_y_ctx, x_0_tar), dim=1)
        else:
            if self.training and self.bound_std:
                yt_noise = batch.yt + 0.05 * torch.randn_like(batch.yt) # add noise to the past to smooth the model
                x_y_tar = torch.cat((batch.xt, yt_noise), dim=-1)
            else:
                x_y_tar = torch.cat((batch.xt, batch.yt), dim=-1)
            inp = torch.cat((x_y_ctx, x_y_tar, x_0_tar), dim=1)
        return inp

    def create_mask(self, batch, autoreg=False):
        num_ctx = batch.xc.shape[1]
        num_tar = batch.xt.shape[1]
        num_all = num_ctx + num_tar
        if not autoreg:
            mask = torch.zeros(num_all, num_all, device='cuda').fill_(float('-inf'))
            mask[:, :num_ctx] = 0.0
        else:
            mask = torch.zeros((num_all+num_tar, num_all+num_tar), device='cuda').fill_(float('-inf'))
            mask[:, :num_ctx] = 0.0 # all points attend to context points
            mask[num_ctx:num_all, num_ctx:num_all].triu_(diagonal=1)
            mask[num_all:, num_ctx:num_all].triu_(diagonal=0)

        return mask, num_tar

    def construct_input_pretrain(self, batch):
        if self.training and self.bound_std:
            y_noise = batch.y + 0.05 * torch.randn_like(batch.y)
            x_y = torch.cat((batch.x, y_noise), dim=-1)
        else:
           x_y= torch.cat((batch.x, batch.y), dim=-1)

        x_0 = torch.cat((batch.x, torch.zeros_like(batch.y)), dim=-1)[:, 1:]
        inp = torch.cat((x_y, x_0), dim=1)
        return inp

    def create_mask_pretrain(self, batch):
        num_points = batch.x.shape[1]

        mask = torch.zeros((2*num_points-1, 2*num_points-1), device='cuda').fill_(float('-inf'))
        mask[:num_points, :num_points].triu_(diagonal=1)
        mask[num_points:, 1:num_points].triu_(diagonal=0)
        mask[num_points:, 0] = 0.0

        return mask, num_points-1

    def encode(self, batch, autoreg=False, pretrain=False):
        if pretrain:
            inp = self.construct_input_pretrain(batch)
            mask, num_tar = self.create_mask_pretrain(batch)
        else:
            inp = self.construct_input(batch, autoreg)
            mask, num_tar = self.create_mask(batch, autoreg)
        embeddings = self.embedder(inp)
        mask = mask.to(device=embeddings.device)
        out = self.encoder(embeddings, mask=mask)
        return out[:, -num_tar:]


class ZSTNP(TNP):
    def __init__(
            self,
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
            bound_std=False
    ):
        super(ZSTNP, self).__init__(
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
            bound_std
        )

        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y * 2)
        )

    def forward(self, batch, reduce_ll=True):
        z_target = self.encode(batch, autoreg=False)
        out = self.predictor(z_target)
        mean, std = torch.chunk(out, 2, dim=-1)
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)

        pred_tar = Normal(mean, std)

        outs = AttrDict()
        if reduce_ll:
            outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1).mean()
        else:
            outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1)
        outs.loss = - (outs.tar_ll)

        return outs

    def predict(self, xc, yc, xt, num_samples=None):
        if xc.shape[-3] != xt.shape[-3]:
            xt = xt.transpose(-3, -2)

        batch = AttrDict()
        batch.xc = xc
        batch.yc = yc
        batch.xt = xt
        batch.yt = torch.zeros((xt.shape[0], xt.shape[1], yc.shape[2]), device='cuda')

        z_target = self.encode(batch, autoreg=False)
        out = self.predictor(z_target)
        mean, std = torch.chunk(out, 2, dim=-1)
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)

        return Normal(mean, std)