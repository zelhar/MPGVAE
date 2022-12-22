import gdown
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyro
import pyro.distributions as pyrodist
import scanpy as sc
import seaborn as sns
import time
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.utils as vutils
import umap
from abc import abstractmethod
from anndata.experimental.pytorch import AnnLoader
from importlib import reload
from math import pi, sin, cos, sqrt, log
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceGraph_ELBO

# from pyro.optim import Adam
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import operator
from operator import add, abs, mul, ge, gt
import toolz
from toolz import partial, curry
from torch import nn, optim, distributions, Tensor
from torch.nn.functional import one_hot, binary_cross_entropy_with_logits
from torch.nn.functional import cross_entropy
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid
from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast, IO, TextIO

#import ../utils as ut
from .. import utils as ut

__all__ = []

__all__.append("VAE_Dirichlet_GMM_TypeB1602z")
class VAE_Dirichlet_GMM_TypeB1602z(nn.Module):
    """
    The 'canonical GMVAE model from the thesis.
    some of the inititialization parameters are better left untouched unless
    stuff like mi.. cc.. temperature, resnet... either broken or don't touch...
    you really know and want to mess around with it.
    parameters meant for user setup:
    dimensions: nx, ny, nz, nw, ny, nd, nclasses ...
    hidden dimensions: nh, nhp, nhq 
    restrictions: restrict_w etc.
    scales:  zscale, yscale etc.
    number of hidden layers: numhidden etc.
    positive_rec if your input is non-negative
    activation functions: replace it if you prefer a different activation e.g.
        nn.Sigmoid()
    bn: True/False use batch normalization
    reclosstype: reconstruction loss type. one of: "Gauss", "MSE", "Bernoulli"

    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale: float = 1e0,
        wscale: float = 1e0,
        yscale: float = 1e0,
        zscale: float = 1e0,
        mi_scale: float = 1e0,
        cc_scale: float = 1e1,
        cc_radius: float = 1e-1,
        concentration: float = 5e-1,
        numhidden: int = 2,
        numhiddenq: int = 2,
        numhiddenp: int = 2,
        dropout: float = 0.3,
        bn: bool = True,
        reclosstype: str = "Gauss",
        temperature: float = 0.1,
        relax: bool = False,
        use_resnet: bool = False,
        softargmax: bool = False,
        eps: float = 1e-9,
        restrict_w: Union[bool, str] = False,
        restrict_z: Union[bool, str] = False,
        activation=nn.LeakyReLU(),
        recloss_min: float = 0,
        positive_rec : bool = False,
        classify_with_mu : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.cc_radius = cc_radius
        self.mi_scale = mi_scale
        self.recloss_mii = recloss_min
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.restrict_z = restrict_z
        self.softargmax = softargmax
        self.use_resnet = use_resnet
        self.dir_prior = distributions.Dirichlet(dscale * torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        self.classify_with_mu = classify_with_mu
        # self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
            probs=torch.ones(nclasses),
            temperature=self.temperature,
        )
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetwork(
            [nz] + numhiddenp * [nhp] + [nx],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        if positive_rec:
            self.Px.add_module(
                    "relu", nn.ReLU(),)
        self.Pz = ut.buildNetwork(
            [nw] + numhiddenp * [nhp] + [2 * nclasses * nz],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        self.Pz.add_module("unflatten", nn.Unflatten(1, (nclasses, 2 * nz)))
        ## Q network
        self.Qwz = ut.buildNetwork(
            [nx] + numhiddenq * [nhq] + [2 * nw + 2 * nz],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        self.Qwz.add_module("unflatten", nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetwork(
            # [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
            # [nx + nz] + numhiddenq*[nhq] + [nclasses],
            [nz] + numhiddenq * [nhq] + [nclasses],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        # self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qd = ut.buildNetwork(
            # [nw + nz] + numhiddenq*[nhq] + [nclasses],
            [nz] + numhiddenq * [nhq] + [nclasses],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(
        self,
        input,
        y=None,
    ):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(
            torch.cat(
                [
                    x,
                ],
                dim=-1,
            )
        )
        mu_w = wz[:, 0, : self.nw]
        logvar_w = wz[:, 1, : self.nw]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 5
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
        elif self.restrict_w:
            mu_w = ut.softclip(mu_w, -5, 5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:, 0, self.nw :]
        logvar_z = wz[:, 1, self.nw :]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            mu_z = ut.softclip(mu_z, -5, 5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        if self.classify_with_mu:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        mu_z,
                    ],
                    dim=1,
                )
            )
        else:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        z,
                    ],
                    dim=1,
                )
            )
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(
                q_y,
            )
        q_y = eps / self.nclasses + (1 - eps) * q_y
        if self.classify_with_mu:
            d_logits = self.Qd(
                torch.cat(
                    [
                        mu_z,
                    ],
                    dim=1,
                )
            )
        else:
            d_logits = self.Qd(
                torch.cat(
                    [
                        z,
                    ],
                    dim=1,
                )
            )
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                temperature=self.temperature.to(x.device),
                probs=q_y,
            )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"] = w
        output["z"] = z
        rec = self.Px(
            torch.cat(
                [
                    z,
                ],
                dim=1,
            )
        )
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"] = rec
        losses["rec"] = loss_rec
        z_w = self.Pz(
            torch.cat(
                [
                    w,
                ],
                dim=-1,
            )
        )
        mu_z_w = z_w[:, :, : self.nz]
        logvar_z_w = z_w[:, :, self.nz :]
        if self.restrict_z == "lv":
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z_w = mu_z_w.tanh()
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z:
            mu_z_w = ut.softclip(mu_z_w, -5, 5)
            logvar_z_w = ut.softclip(logvar_z_w, -9, 1)
        else:
            pass
        std_z_w = (0.5 * logvar_z_w).exp()
        Pz = distributions.Normal(
            loc=mu_z_w,
            scale=std_z_w,
        )
        output["Pz"] = Pz
        loss_z = (
            self.zscale
            * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
            ).sum(-1)
        )
        p_y = D_y.rsample()
        p_y = eps / self.nclasses + (1 - eps) * p_y
        Py = distributions.OneHotCategorical(probs=p_y)
        if self.relax:
            Py = distributions.RelaxedOneHotCategorical(
                temperature=self.temperature.to(x.device),
                probs=p_y,
            )
        else:
            Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y * loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y * loss_z).sum(-1).mean()
            if self.relax:
                y = eps / self.nclasses + (1 - eps) * y
            loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = (
            self.wscale * self.kld_unreduced(mu=mu_w, logvar=logvar_w).sum(-1).mean()
        )
        losses["loss_w"] = loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"] = loss_d = loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = loss_rec + loss_z + loss_w + loss_d + loss_y_alt + loss_y_alt2
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

    def justPredict(
        self,
        input,
        y=None,
    ):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        output = {}
        eps = self.eps
        wz = self.Qwz(
            torch.cat(
                [
                    x,
                ],
                dim=-1,
            )
        )
        mu_w = wz[:, 0, : self.nw]
        logvar_w = wz[:, 1, : self.nw]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w:
            mu_w = ut.softclip(mu_w, -5, 5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:, 0, self.nw :]
        logvar_z = wz[:, 1, self.nw :]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            mu_z = ut.softclip(mu_z, -5, 5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        if self.classify_with_mu:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        mu_z,
                    ],
                    dim=1,
                )
            )
        else:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        z,
                    ],
                    dim=1,
                )
            )
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(
                q_y,
            )
        q_y = eps / self.nclasses + (1 - eps) * q_y
        return q_y

__all__.append("VAE_Dirichlet_GMM_TypeB1602zC")
class VAE_Dirichlet_GMM_TypeB1602zC(nn.Module):
    """
    conditional version. additional param nc1 : number of conditions
    additional param: learned_prior (on w): If False: std normal on all conditions.
    If  True: learn it from the condition c.
    the rest of the parameters are the same.
    In this conditional version z is conditioned but for example y is not.
    see version C2 for 'everything conditioned'.
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale: float = 1e0,
        wscale: float = 1e0,
        yscale: float = 1e0,
        zscale: float = 1e0,
        mi_scale: float = 1e0,
        concentration: float = 5e-1,
        numhidden: int = 2,
        numhiddenq: int = 2,
        numhiddenp: int = 2,
        dropout: float = 0.3,
        bn: bool = True,
        reclosstype: str = "Gauss",
        temperature: float = 0.1,
        relax: bool = False,
        use_resnet: bool = False,
        softargmax: bool = False,
        eps: float = 1e-9,
        restrict_w: Union[bool, str] = False,
        restrict_z: Union[bool, str] = False,
        activation=nn.LeakyReLU(),
        recloss_min: float = 0,
        nc1: int = 5,
        learned_prior: bool = False,
        positive_rec : bool = False,
        classify_with_mu : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.mi_scale = mi_scale
        self.recloss_mii = recloss_min
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.restrict_z = restrict_z
        self.softargmax = softargmax
        self.use_resnet = use_resnet
        self.dir_prior = distributions.Dirichlet(dscale * torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.nc1 = nc1
        self.learned_prior = learned_prior
        self.classify_with_mu = classify_with_mu
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        # self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
            probs=torch.ones(nclasses),
            temperature=self.temperature,
        )
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv5(
            [nz + nc1] + numhiddenp * [nhp] + [nx],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        if positive_rec:
            self.Px.add_module(
                    "relu", nn.ReLU(),)
        self.Pz = ut.buildNetwork(
            [nw] + numhiddenp * [nhp] + [2 * nclasses * nz],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        self.Pz.add_module("unflatten", nn.Unflatten(1, (nclasses, 2 * nz)))
        # w prior
        self.Pw = ut.buildNetwork(
            [nc1] + numhiddenq * [nhq] + [2 * nw],
            activation=nn.LeakyReLU(),
            batchnorm=bn,
        )
        ## Q network
        self.Qwz = ut.buildNetwork(
            [nx + nc1] + numhiddenq * [nhq] + [2 * nw + 2 * nz],
            dropout=dropout,
            # activation=nn.LeakyReLU(),
            # activation=nn.Sigmoid(),
            activation=activation,
            batchnorm=bn,
        )
        self.Qwz.add_module("unflatten", nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetwork(
            # [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
            # [nx + nz] + numhiddenq*[nhq] + [nclasses],
            [nz] + numhiddenq * [nhq] + [nclasses],
            dropout=dropout,
            # activation=nn.LeakyReLU(),
            # activation=nn.Sigmoid(),
            activation=activation,
            batchnorm=bn,
        )
        # self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qd = ut.buildNetwork(
            # [nw + nz] + numhiddenq*[nhq] + [nclasses],
            [nz] + numhiddenq * [nhq] + [nclasses],
            dropout=dropout,
            # activation=nn.LeakyReLU(),
            activation=activation,
            batchnorm=bn,
        )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(
        self,
        input,
        cond1=None,
        y=None,
    ):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        if cond1 == None:
            cond1 = torch.zeros((batch_size, self.nc1), device=x.device)
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(
            torch.cat(
                [
                    x,
                    cond1,
                ],
                dim=-1,
            )
        )
        mu_w = wz[:, 0, : self.nw]
        logvar_w = wz[:, 1, : self.nw]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 5
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
        elif self.restrict_w:
            mu_w = ut.softclip(mu_w, -5, 5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:, 0, self.nw :]
        logvar_z = wz[:, 1, self.nw :]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            mu_z = ut.softclip(mu_z, -5, 5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        if self.classify_with_mu:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        mu_z,
                    ],
                    dim=1,
                )
            )
        else:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        z,
                    ],
                    dim=1,
                )
            )
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(
                q_y,
            )
        q_y = eps / self.nclasses + (1 - eps) * q_y
        if self.classify_with_mu:
            d_logits = self.Qd(
                torch.cat(
                    [
                        mu_z,
                    ],
                    dim=1,
                )
            )
        else:
            d_logits = self.Qd(
                torch.cat(
                    [
                        z,
                    ],
                    dim=1,
                )
            )
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                temperature=self.temperature.to(x.device),
                probs=q_y,
            )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"] = w
        output["z"] = z
        rec = self.Px(
            torch.cat(
                [
                    z,
                    cond1,
                ],
                dim=1,
            )
        )
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"] = rec
        losses["rec"] = loss_rec
        z_w = self.Pz(
            torch.cat(
                [
                    w,
                ],
                dim=-1,
            )
        )
        mu_z_w = z_w[:, :, : self.nz]
        logvar_z_w = z_w[:, :, self.nz :]
        if self.restrict_z == "lv":
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z_w = mu_z_w.tanh()
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z:
            mu_z_w = ut.softclip(mu_z_w, -5, 5)
            logvar_z_w = ut.softclip(logvar_z_w, -9, 1)
        else:
            pass
        std_z_w = (0.5 * logvar_z_w).exp()
        Pz = distributions.Normal(
            loc=mu_z_w,
            scale=std_z_w,
        )
        output["Pz"] = Pz
        loss_z = (
            self.zscale
            * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
            ).sum(-1)
        )
        p_y = D_y.rsample()
        p_y = eps / self.nclasses + (1 - eps) * p_y
        Py = distributions.OneHotCategorical(probs=p_y)
        if self.relax:
            Py = distributions.RelaxedOneHotCategorical(
                temperature=self.temperature.to(x.device),
                probs=p_y,
            )
        else:
            Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y * loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y * loss_z).sum(-1).mean()
            if self.relax:
                y = eps / self.nclasses + (1 - eps) * y
            loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z

        w_mu_logvar_prior = self.Pw(
            torch.cat(
                [
                    cond1,
                ],
                dim=-1,
            )
        )
        mu_w_prior = w_mu_logvar_prior[:, : self.nw]
        logvar_w_prior = w_mu_logvar_prior[:, self.nw:]
        if self.restrict_w == "lv":
            logvar_w_prior = logvar_w_prior.tanh() * 5
        elif self.restrict_w == "mlv":
            mu_w_prior = mu_w_prior.tanh()
            logvar_w_prior = logvar_w_prior.tanh() * 5
        elif self.restrict_w:
            mu_w_prior = ut.softclip(mu_w_prior, -5, 5)
            logvar_w_prior = ut.softclip(logvar_w_prior, -9, 1)
        else:
            pass
        if not self.learned_prior:
            loss_w = (
                self.wscale
                * self.kld_unreduced(mu=mu_w, logvar=logvar_w).sum(-1).mean()
            )
        else:
            loss_w = (
                self.wscale
                * ut.kld2normal(
                    mu=mu_w,
                    logvar=logvar_w,
                    mu2=mu_w_prior,
                    logvar2=logvar_w_prior,
                )
                .sum(-1)
                .mean()
            )
        losses["loss_w"] = loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"] = loss_d = loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = loss_rec + loss_z + loss_w + loss_d + loss_y_alt + loss_y_alt2
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

    def justPredict(
        self,
        input,
        cond1=None,
        y=None,
    ):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        if cond1 == None:
            cond1 = torch.zeros((batch_size, self.nc1), device=x.device)
        output = {}
        eps = self.eps
        wz = self.Qwz(
            torch.cat(
                [
                    x,
                    cond1,
                ],
                dim=-1,
            )
        )
        mu_w = wz[:, 0, : self.nw]
        logvar_w = wz[:, 1, : self.nw]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w:
            mu_w = ut.softclip(mu_w, -5, 5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:, 0, self.nw :]
        logvar_z = wz[:, 1, self.nw :]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            mu_z = ut.softclip(mu_z, -5, 5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        if self.classify_with_mu:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        mu_z,
                    ],
                    dim=1,
                )
            )
        else:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        z,
                    ],
                    dim=1,
                )
            )
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(
                q_y,
            )
        q_y = eps / self.nclasses + (1 - eps) * q_y
        return q_y

__all__.append("VAE_Dirichlet_GMM_TypeB1602zC2")
class VAE_Dirichlet_GMM_TypeB1602zC2(nn.Module):
    """
    conditional version2.
    conditional version2. additional param nc1 : number of conditions
    additional param: learned_prior (on w): If False: std normal on all conditions.
    If  True: learn it from the condition c.
    the rest of the parameters are the same.
    In this conditional version everything is conditioned on c.
    My preferred version.
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale: float = 1e0,
        wscale: float = 1e0,
        yscale: float = 1e0,
        zscale: float = 1e0,
        mi_scale: float = 1e0,
        concentration: float = 5e-1,
        numhidden: int = 2,
        numhiddenq: int = 2,
        numhiddenp: int = 2,
        dropout: float = 0.3,
        bn: bool = True,
        reclosstype: str = "Gauss",
        temperature: float = 0.1,
        relax: bool = False,
        use_resnet: bool = False,
        softargmax: bool = False,
        eps: float = 1e-9,
        restrict_w: Union[bool, str] = False,
        restrict_z: Union[bool, str] = False,
        activation=nn.LeakyReLU(),
        recloss_min: float = 0,
        nc1: int = 5,
        learned_prior: bool = False,
        positive_rec : bool = False,
        classify_with_mu : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.mi_scale = mi_scale
        self.recloss_mii = recloss_min
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.restrict_z = restrict_z
        self.softargmax = softargmax
        self.use_resnet = use_resnet
        self.dir_prior = distributions.Dirichlet(dscale * torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.nc1 = nc1
        self.learned_prior = learned_prior
        self.classify_with_mu = classify_with_mu
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        # self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
            probs=torch.ones(nclasses),
            temperature=self.temperature,
        )
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetwork(
            [nz + nc1] + numhiddenp * [nhp] + [nx],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        if positive_rec:
            self.Px.add_module(
                    "relu", nn.ReLU(),)
        self.Pz = ut.buildNetwork(
            [nw + nc1] + numhiddenp * [nhp] + [2 * nclasses * nz],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        self.Pz.add_module("unflatten", nn.Unflatten(1, (nclasses, 2 * nz)))
        # w prior
        self.Pw = ut.buildNetwork(
            [nc1] + numhiddenq * [nhq] + [2 * nw],
            activation=nn.LeakyReLU(),
            batchnorm=bn,
        )
        ## Q network
        self.Qwz = ut.buildNetwork(
            [nx + nc1] + numhiddenq * [nhq] + [2 * nw + 2 * nz],
            dropout=dropout,
            # activation=nn.LeakyReLU(),
            # activation=nn.Sigmoid(),
            activation=activation,
            batchnorm=bn,
        )
        self.Qwz.add_module("unflatten", nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetwork(
            # [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
            # [nx + nz] + numhiddenq*[nhq] + [nclasses],
            [nz + nc1] + numhiddenq * [nhq] + [nclasses],
            dropout=dropout,
            # activation=nn.LeakyReLU(),
            # activation=nn.Sigmoid(),
            activation=activation,
            batchnorm=bn,
        )
        # self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qd = ut.buildNetwork(
            # [nw + nz] + numhiddenq*[nhq] + [nclasses],
            [nz + nc1] + numhiddenq * [nhq] + [nclasses],
            dropout=dropout,
            # activation=nn.LeakyReLU(),
            activation=activation,
            batchnorm=bn,
        )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(
        self,
        input,
        cond1=None,
        y=None,
    ):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        if cond1 == None:
            cond1 = torch.zeros((batch_size, self.nc1), device=x.device)
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(
            torch.cat(
                [
                    x,
                    cond1,
                ],
                dim=-1,
            )
        )
        mu_w = wz[:, 0, : self.nw]
        logvar_w = wz[:, 1, : self.nw]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 5
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
        elif self.restrict_w:
            mu_w = ut.softclip(mu_w, -5, 5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:, 0, self.nw :]
        logvar_z = wz[:, 1, self.nw :]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            mu_z = ut.softclip(mu_z, -5, 5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        if self.classify_with_mu:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        mu_z,
                        cond1,
                    ],
                    dim=1,
                )
            )
        else:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        z,
                        cond1,
                    ],
                    dim=1,
                )
            )
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(
                q_y,
            )
        q_y = eps / self.nclasses + (1 - eps) * q_y
        if self.classify_with_mu:
            d_logits = self.Qd(
                torch.cat(
                    [
                        mu_z,
                        cond1,
                    ],
                    dim=1,
                )
            )
        else:
            d_logits = self.Qd(
                torch.cat(
                    [
                        z,
                        cond1,
                    ],
                    dim=1,
                )
            )
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                temperature=self.temperature.to(x.device),
                probs=q_y,
            )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"] = w
        output["z"] = z
        rec = self.Px(
            torch.cat(
                [
                    z,
                    cond1,
                ],
                dim=1,
            )
        )
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"] = rec
        losses["rec"] = loss_rec
        z_w = self.Pz(
            torch.cat(
                [
                    w,
                    cond1,
                ],
                dim=-1,
            )
        )
        mu_z_w = z_w[:, :, : self.nz]
        logvar_z_w = z_w[:, :, self.nz :]
        if self.restrict_z == "lv":
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z_w = mu_z_w.tanh()
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z:
            mu_z_w = ut.softclip(mu_z_w, -5, 5)
            logvar_z_w = ut.softclip(logvar_z_w, -9, 1)
        else:
            pass
        std_z_w = (0.5 * logvar_z_w).exp()
        Pz = distributions.Normal(
            loc=mu_z_w,
            scale=std_z_w,
        )
        output["Pz"] = Pz
        loss_z = (
            self.zscale
            * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
            ).sum(-1)
        )
        p_y = D_y.rsample()
        p_y = eps / self.nclasses + (1 - eps) * p_y
        Py = distributions.OneHotCategorical(probs=p_y)
        if self.relax:
            Py = distributions.RelaxedOneHotCategorical(
                temperature=self.temperature.to(x.device),
                probs=p_y,
            )
        else:
            Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y * loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y * loss_z).sum(-1).mean()
            if self.relax:
                y = eps / self.nclasses + (1 - eps) * y
            loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z

        w_mu_logvar_prior = self.Pw(
            torch.cat(
                [
                    cond1,
                ],
                dim=-1,
            )
        )
        mu_w_prior = w_mu_logvar_prior[:, : self.nw]
        logvar_w_prior = w_mu_logvar_prior[:, self.nw:]
        if self.restrict_w == "lv":
            logvar_w_prior = logvar_w_prior.tanh() * 5
        elif self.restrict_w == "mlv":
            mu_w_prior = mu_w_prior.tanh()
            logvar_w_prior = logvar_w_prior.tanh() * 5
        elif self.restrict_w:
            mu_w_prior = ut.softclip(mu_w_prior, -5, 5)
            logvar_w_prior = ut.softclip(logvar_w_prior, -9, 1)
        else:
            pass
        if not self.learned_prior:
            loss_w = (
                self.wscale
                * self.kld_unreduced(mu=mu_w, logvar=logvar_w).sum(-1).mean()
            )
        else:
            loss_w = (
                self.wscale
                * ut.kld2normal(
                    mu=mu_w,
                    logvar=logvar_w,
                    mu2=mu_w_prior,
                    logvar2=logvar_w_prior,
                )
                .sum(-1)
                .mean()
            )
        losses["loss_w"] = loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"] = loss_d = loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = loss_rec + loss_z + loss_w + loss_d + loss_y_alt + loss_y_alt2
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

    def justPredict(
        self,
        input,
        cond1=None,
        y=None,
    ):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        if cond1 == None:
            cond1 = torch.zeros((batch_size, self.nc1), device=x.device)
        output = {}
        eps = self.eps
        wz = self.Qwz(
            torch.cat(
                [
                    x,
                    cond1,
                ],
                dim=-1,
            )
        )
        mu_w = wz[:, 0, : self.nw]
        logvar_w = wz[:, 1, : self.nw]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w:
            mu_w = ut.softclip(mu_w, -5, 5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:, 0, self.nw :]
        logvar_z = wz[:, 1, self.nw :]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            mu_z = ut.softclip(mu_z, -5, 5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        if self.classify_with_mu:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        mu_z,
                        cond1,
                    ],
                    dim=1,
                )
            )
        else:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        z,
                        cond1,
                    ],
                    dim=1,
                )
            )
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(
                q_y,
            )
        q_y = eps / self.nclasses + (1 - eps) * q_y
        return q_y

__all__.append("VAE_Dirichlet_GMM_TypeB1602xwz")
class VAE_Dirichlet_GMM_TypeB1602xwz(nn.Module):
    """
    Same as the 'canonical GMVAE model from the thesis.
    except that the y and d-classifiers get to see x,w,z instead of just z.
    some of the inititialization parameters are better left untouched unless
    stuff like mi.. cc.. temperature, resnet... either broken or don't touch...
    you really know and want to mess around with it.
    parameters meant for user setup:
    dimensions: nx, ny, nz, nw, ny, nd, nclasses ...
    hidden dimensions: nh, nhp, nhq 
    restrictions: restrict_w etc.
    scales:  zscale, yscale etc.
    number of hidden layers: numhidden etc.
    positive_rec if your input is non-negative
    activation functions: replace it if you prefer a different activation e.g.
        nn.Sigmoid()
    bn: True/False use batch normalization
    reclosstype: reconstruction loss type. one of: "Gauss", "MSE", "Bernoulli"

    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale: float = 1e0,
        wscale: float = 1e0,
        yscale: float = 1e0,
        zscale: float = 1e0,
        mi_scale: float = 1e0,
        cc_scale: float = 1e1,
        cc_radius: float = 1e-1,
        concentration: float = 5e-1,
        numhidden: int = 2,
        numhiddenq: int = 2,
        numhiddenp: int = 2,
        dropout: float = 0.3,
        bn: bool = True,
        reclosstype: str = "Gauss",
        temperature: float = 0.1,
        relax: bool = False,
        use_resnet: bool = False,
        softargmax: bool = False,
        eps: float = 1e-9,
        restrict_w: Union[bool, str] = False,
        restrict_z: Union[bool, str] = False,
        activation=nn.LeakyReLU(),
        recloss_min: float = 0,
        positive_rec : bool = False,
        classify_with_mu : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.cc_radius = cc_radius
        self.mi_scale = mi_scale
        self.recloss_mii = recloss_min
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.restrict_z = restrict_z
        self.softargmax = softargmax
        self.use_resnet = use_resnet
        self.dir_prior = distributions.Dirichlet(dscale * torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        self.classify_with_mu = classify_with_mu
        # self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
            probs=torch.ones(nclasses),
            temperature=self.temperature,
        )
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetwork(
            [nz] + numhiddenp * [nhp] + [nx],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        if positive_rec:
            self.Px.add_module(
                    "relu", nn.ReLU(),)
        self.Pz = ut.buildNetwork(
            [nw] + numhiddenp * [nhp] + [2 * nclasses * nz],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        self.Pz.add_module("unflatten", nn.Unflatten(1, (nclasses, 2 * nz)))
        ## Q network
        self.Qwz = ut.buildNetwork(
            [nx] + numhiddenq * [nhq] + [2 * nw + 2 * nz],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        self.Qwz.add_module("unflatten", nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetwork(
            [nz + nw + nx] + numhiddenq * [nhq] + [nclasses],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        # self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qd = ut.buildNetwork(
            [nz + nw + nx] + numhiddenq * [nhq] + [nclasses],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(
        self,
        input,
        y=None,
    ):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(
            torch.cat(
                [
                    x,
                ],
                dim=-1,
            )
        )
        mu_w = wz[:, 0, : self.nw]
        logvar_w = wz[:, 1, : self.nw]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 5
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
        elif self.restrict_w:
            mu_w = ut.softclip(mu_w, -5, 5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:, 0, self.nw :]
        logvar_z = wz[:, 1, self.nw :]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            mu_z = ut.softclip(mu_z, -5, 5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        if self.classify_with_mu:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        mu_z,
                        mu_w,
                        x,
                    ],
                    dim=1,
                )
            )
        else:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        z,
                        w,
                        x,
                    ],
                    dim=1,
                )
            )
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(
                q_y,
            )
        q_y = eps / self.nclasses + (1 - eps) * q_y
        if self.classify_with_mu:
            d_logits = self.Qd(
                torch.cat(
                    [
                        mu_z,
                        mu_w,
                        x,
                    ],
                    dim=1,
                )
            )
        else:
            d_logits = self.Qd(
                torch.cat(
                    [
                        z,
                        w,
                        x,
                    ],
                    dim=1,
                )
            )
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                temperature=self.temperature.to(x.device),
                probs=q_y,
            )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"] = w
        output["z"] = z
        rec = self.Px(
            torch.cat(
                [
                    z,
                ],
                dim=1,
            )
        )
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"] = rec
        losses["rec"] = loss_rec
        z_w = self.Pz(
            torch.cat(
                [
                    w,
                ],
                dim=-1,
            )
        )
        mu_z_w = z_w[:, :, : self.nz]
        logvar_z_w = z_w[:, :, self.nz :]
        if self.restrict_z == "lv":
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z_w = mu_z_w.tanh()
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z:
            mu_z_w = ut.softclip(mu_z_w, -5, 5)
            logvar_z_w = ut.softclip(logvar_z_w, -9, 1)
        else:
            pass
        std_z_w = (0.5 * logvar_z_w).exp()
        Pz = distributions.Normal(
            loc=mu_z_w,
            scale=std_z_w,
        )
        output["Pz"] = Pz
        loss_z = (
            self.zscale
            * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
            ).sum(-1)
        )
        p_y = D_y.rsample()
        p_y = eps / self.nclasses + (1 - eps) * p_y
        Py = distributions.OneHotCategorical(probs=p_y)
        if self.relax:
            Py = distributions.RelaxedOneHotCategorical(
                temperature=self.temperature.to(x.device),
                probs=p_y,
            )
        else:
            Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y * loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y * loss_z).sum(-1).mean()
            if self.relax:
                y = eps / self.nclasses + (1 - eps) * y
            loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = (
            self.wscale * self.kld_unreduced(mu=mu_w, logvar=logvar_w).sum(-1).mean()
        )
        losses["loss_w"] = loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"] = loss_d = loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = loss_rec + loss_z + loss_w + loss_d + loss_y_alt + loss_y_alt2
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

    def justPredict(
        self,
        input,
        y=None,
    ):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        output = {}
        eps = self.eps
        wz = self.Qwz(
            torch.cat(
                [
                    x,
                ],
                dim=-1,
            )
        )
        mu_w = wz[:, 0, : self.nw]
        logvar_w = wz[:, 1, : self.nw]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w:
            mu_w = ut.softclip(mu_w, -5, 5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:, 0, self.nw :]
        logvar_z = wz[:, 1, self.nw :]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            mu_z = ut.softclip(mu_z, -5, 5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        if self.classify_with_mu:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        mu_z,
                        mu_w,
                        x,
                    ],
                    dim=1,
                )
            )
        else:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        z,
                        w,
                        x,
                    ],
                    dim=1,
                )
            )
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(
                q_y,
            )
        q_y = eps / self.nclasses + (1 - eps) * q_y
        return q_y

__all__.append("VAE_Dirichlet_GMM_TypeB1602xzR18")
class VAE_Dirichlet_GMM_TypeB1602xzR18(nn.Module):
    """
    Based on VAE_Dirichlet_GMM_TypeB1602xz and uses a resnet18,
    meant to be used with b/w images like FMNIST data.
    or tweak it yourself...
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale: float = 1e0,
        wscale: float = 1e0,
        yscale: float = 1e0,
        zscale: float = 1e0,
        mi_scale: float = 1e0,
        concentration: float = 5e-1,
        numhidden: int = 2,
        numhiddenq: int = 2,
        numhiddenp: int = 2,
        dropout: float = 0.3,
        bn: bool = True,
        reclosstype: str = "Gauss",
        temperature: float = 0.1,
        relax: bool = False,
        use_resnet: bool = False,
        softargmax: bool = False,
        eps: float = 1e-9,
        restrict_w: Union[bool, str] = False,
        restrict_z: Union[bool, str] = False,
        activation=nn.LeakyReLU(),
        recloss_min: float = 0,
        positive_rec : bool = False,
        classify_with_mu : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.mi_scale = mi_scale
        self.recloss_mii = recloss_min
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.restrict_z = restrict_z
        self.softargmax = softargmax
        self.use_resnet = use_resnet
        self.dir_prior = distributions.Dirichlet(dscale * torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        # self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.classify_with_mu = classify_with_mu
        self.y_prior = distributions.RelaxedOneHotCategorical(
            probs=torch.ones(nclasses),
            temperature=self.temperature,
        )
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetwork(
            [nz] + numhiddenp * [nhp] + [nx],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        if positive_rec:
            self.Px.add_module(
                    "relu", nn.ReLU(),)
        self.Pz = ut.buildNetwork(
            [nw] + numhiddenp * [nhp] + [2 * nclasses * nz],
            dropout=dropout,
            activation=activation,
            batchnorm=bn,
        )
        self.Pz.add_module("unflatten", nn.Unflatten(1, (nclasses, 2 * nz)))
        ## Q network
        self.resnet_module = models.resnet18()
        self.resnet_module.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
        self.Qw = nn.Sequential(
                nn.Linear(1000, nhq),
                activation,
                nn.Linear(nhq, 2*nw),
                )
        self.Qz = nn.Sequential(
                nn.Linear(1000, nhq),
                activation,
                nn.Linear(nhq, 2*nz),
                )
        self.Qy = nn.Sequential(
                nn.Linear(1000 + nz, nhq),
                activation,
                nn.Linear(nhq, nclasses),
                )
        self.Qd = nn.Sequential(
                nn.Linear(1000 + nz, nhq),
                activation,
                nn.Linear(nhq, nclasses),
                )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(
        self,
        input,
        y=None,
    ):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        #r_out = self.resnet_module(x.unsqueeze(1))
        #r_out = self.resnet_module(input.unsqueeze(1))
        r_out = self.resnet_module(x.reshape(-1,1,28,28))
        ws = self.Qw(r_out)
        zs = self.Qz(r_out)
        mu_w = ws[:, : self.nw]
        logvar_w = ws[:, self.nw :]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 5
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
        elif self.restrict_w:
            # mu_w = ut.softclip(mu_w, -1,1)
            # logvar_w = ut.softclip(logvar_w, -5, 1)
            mu_w = ut.softclip(mu_w, -5, 5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = zs[:, : self.nz]
        logvar_z = zs[:, self.nz :]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            # mu_z = ut.softclip(mu_z, -5,5)
            # logvar_z = ut.softclip(logvar_z, -5, 2)
            mu_z = ut.softclip(mu_z, -5, 5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["ws"] = ws
        output["zs"] = zs
        output["w"] = w
        output["z"] = z
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        # q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        if self.classify_with_mu:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        r_out,
                        mu_z,
                    ],
                    dim=1,
                )
            )
        else:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        r_out,
                        z,
                    ],
                    dim=1,
                )
            )
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(
                q_y,
            )
        q_y = eps / self.nclasses + (1 - eps) * q_y
        if self.classify_with_mu:
            d_logits = self.Qd(
                torch.cat(
                    [
                        r_out,
                        mu_z,
                    ],
                    dim=1,
                )
            )
        else:
            d_logits = self.Qd(
                torch.cat(
                    [
                        r_out,
                        z,
                    ],
                    dim=1,
                )
            )
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                temperature=self.temperature.to(x.device),
                probs=q_y,
            )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"] = w
        output["z"] = z
        rec = self.Px(
            torch.cat(
                [
                    z,
                ],
                dim=1,
            )
        )
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"] = rec
        losses["rec"] = loss_rec
        z_w = self.Pz(
            torch.cat(
                [
                    w,
                ],
                dim=-1,
            )
        )
        mu_z_w = z_w[:, :, : self.nz]
        logvar_z_w = z_w[:, :, self.nz :]
        if self.restrict_z == "lv":
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z_w = mu_z_w.tanh()
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z:
            mu_z_w = ut.softclip(mu_z_w, -5, 5)
            logvar_z_w = ut.softclip(logvar_z_w, -9, 1)
        else:
            pass
        std_z_w = (0.5 * logvar_z_w).exp()
        Pz = distributions.Normal(
            loc=mu_z_w,
            scale=std_z_w,
        )
        output["Pz"] = Pz
        loss_z = (
            self.zscale
            * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
            ).sum(-1)
        )
        p_y = D_y.rsample()
        p_y = eps / self.nclasses + (1 - eps) * p_y
        Py = distributions.OneHotCategorical(probs=p_y)
        if self.relax:
            Py = distributions.RelaxedOneHotCategorical(
                temperature=self.temperature.to(x.device),
                probs=p_y,
            )
        else:
            Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y * loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y * loss_z).sum(-1).mean()
            if self.relax:
                y = eps / self.nclasses + (1 - eps) * y
            loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = (
            self.wscale * self.kld_unreduced(mu=mu_w, logvar=logvar_w).sum(-1).mean()
        )
        losses["loss_w"] = loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"] = loss_d = loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = loss_rec + loss_z + loss_w + loss_d + loss_y_alt + loss_y_alt2
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

    def justPredict(
        self,
        input,
        y=None,
    ):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        output = {}
        eps = self.eps
        #r_out = self.resnet_module(x.unsqueeze(1))
        #r_out = self.resnet_module(input.unsqueeze(1))
        r_out = self.resnet_module(x.reshape(-1,1,28,28))
        ws = self.Qw(r_out)
        zs = self.Qz(r_out)
        mu_w = ws[:, : self.nw]
        logvar_w = ws[:, self.nw :]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w:
            mu_w = ut.softclip(mu_w, -5, 5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = zs[:, : self.nz]
        logvar_z = zs[:, self.nz :]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            mu_z = ut.softclip(mu_z, -5, 5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        if self.classify_with_mu:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        r_out,
                        mu_z,
                    ],
                    dim=1,
                )
            )
        else:
            q_y_logits = self.Qy(
                torch.cat(
                    [
                        r_out,
                        z,
                    ],
                    dim=1,
                )
            )
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(
                q_y,
            )
        q_y = eps / self.nclasses + (1 - eps) * q_y
        return q_y

