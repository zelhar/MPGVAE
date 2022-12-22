# best of models
# sources:
# https://github.com/psanch21/VAE-GMVAE
# https://arxiv.org/abs/1611.02648
# http://ruishu.io/2016/12/25/gmvae/
# https://github.com/RuiShu/vae-clustering
# https://github.com/hbahadirsahin/gmvae
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


# my own sauce
# from my_torch_utils import denorm, normalize, mixedGaussianCircular
# from my_torch_utils import fclayer, init_weights
# from my_torch_utils import fnorm, replicate, logNorm, log_gaussian_prob
# from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
# from my_torch_utils import scsimDataset
#import my_torch_utils as ut
from .. import utils as ut

#print(torch.cuda.is_available())

__all__ = []

#__all__.append("preTrain")
def preTrain(
    model,
    train_loader: torch.utils.data.DataLoader,
    #test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda:0",
    wt: float = 1e-4,
    report_interval: int = 3,
    best_loss: float = 1e6,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    non-conditional version of pre train (unsupervised)
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wt,
    )
    best_result = model.state_dict()
    for epoch in range(num_epochs):
        # print("training phase")
        # for idx, (data, labels) in enumerate(train_loader):
        for idx, data in enumerate(train_loader):
            x = data[0].flatten(1).to(device)
            # y = labels.to(device)
            y = data[1].to(device)
            # x = data.to(device)
            model.train()
            model.requires_grad_(True)
            output = model.forward(
                x,
                y=None,
            )
            # output = model.forward(x,y, training=True)
            losses = output["losses"]
            q_y = output["q_y"]
            p_y = torch.ones_like(q_y) / model.nclasses
            loss_q_y = (p_y * q_y.log()).sum(-1).mean() * model.yscale
            loss_q_y2 = (p_y - q_y).abs().sum(-1).mean() * model.yscale * 1e0
            losses["loss_q_y"] = loss_q_y
            losses["loss_q_y2"] = loss_q_y2
            loss = (
                    losses["rec"]
                    #+ losses["loss_z"]
                    + losses["loss_pretrain_z"]
                    + losses["loss_w"]
                    + loss_q_y
                    + loss_q_y2
                    )
            losses["pretrain_loss"] = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_result = model.state_dict()
            if report_interval == 0:
                continue
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("training phase")
                #model.printDict(output["losses"])
                model.printDict(losses)
                print()
                if do_plot:
                    ut.do_plot_helper(model, device)
                if test_accuracy:
                    ut.test_accuracy_helper(model, x, y, device)
                model.train()
                model.to(device)
    model.cpu()
    model.eval()
    optimizer = None
    model.load_state_dict(best_result)
    return

#__all__.append("preTrainLoop")
def preTrainLoop(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt: float = 1e-4,
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    non-conditional version of basicTrainLoopCond
    """
    for lr in lrs:
        print(
            "epoch's lr = ",
            lr,
        )
        preTrain(
            model,
            train_loader,
            num_epochs,
            lr,
            device,
            wt,
            report_interval,
            do_plot=do_plot,
            test_accuracy=test_accuracy,
        )

    print("done training")
    return

#__all__.append("advancedTrain")
def advancedTrain(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    best_loss: float = 1e6,
    do_plot: bool = False,
    test_accuracy: bool = False,
    advanced_semi : bool = True,
    cc_extra_sclae : float = 1e1,
) -> None:
    """
    non-conditional version of advancedTrain train (unsupervised)
    does supervised training using generated samples ans unsupervised
    training using the data_loader.
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wt,
    )
    best_result = model.state_dict()
    for epoch in range(num_epochs):
        # print("training phase")
        # for idx, (data, labels) in enumerate(train_loader):
        for idx, data in enumerate(train_loader):
            x = data[0].flatten(1).to(device)
            # y = labels.to(device)
            y = data[1].to(device)
            # x = data.to(device)
            model.train()
            model.requires_grad_(True)
            output = model.forward(
                x,
                y=None,
            )
            # output = model.forward(x,y, training=True)
            loss = output["losses"][loss_type]
            if advanced_semi:
                batch_size = x.shape[0]
                #ww = torch.randn(batch_size,model.nw).to(device)
                #w = output["w"].detach()
                mu_w = output["mu_w"].detach().to(device)
                logvar_w = output["logvar_w"].detach().to(device)
                noise = torch.randn_like(mu_w).to(device)
                std_w = (0.5 * logvar_w).exp()
                ww = mu_w + noise * std_w
                zz = model.Pz(ww)[:,:,:model.nz]
                rr = model.Px(zz.reshape(batch_size * model.nclasses, model.nz))
                yy = model.justPredict(rr).to(device)
                cc = torch.eye(model.nclasses, device=device)
                cc = cc.repeat(batch_size,1)
                loss_cc = model.cc_scale * (yy - cc).abs().sum(-1).mean()
                loss_cc = loss_cc - model.cc_scale * (cc * yy.log()).sum(-1).mean()
                loss_cc = loss_cc * cc_extra_sclae
                loss = loss + loss_cc
                output["losses"]["loss_cc"] = loss_cc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_result = model.state_dict()
            if report_interval == 0:
                continue
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("training phase")
                model.printDict(output["losses"])
                print()
                if do_plot:
                    ut.do_plot_helper(model, device)
                if test_accuracy:
                    ut.test_accuracy_helper(model, x, y, device)
                model.train()
                model.to(device)
    model.cpu()
    model.eval()
    optimizer = None
    model.load_state_dict(best_result)
    return

#__all__.append("advancedTrainLoop")
def advancedTrainLoop(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
    advanced_semi : bool = True,
    cc_extra_sclae : float = 1e1,
) -> None:
    """
    non-conditional version of advancedTrainLoop
    """
    for lr in lrs:
        print(
            "epoch's lr = ",
            lr,
        )
        advancedTrain(
            model,
            train_loader,
            test_loader,
            num_epochs,
            lr,
            device,
            wt,
            loss_type,
            report_interval,
            do_plot=do_plot,
            test_accuracy=test_accuracy,
            advanced_semi = advanced_semi,
            cc_extra_sclae = cc_extra_sclae,
        )

    print("done training")
    return

__all__.append("basicTrain")
def basicTrain(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 10,
    best_loss: float = 1e6,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    non-conditional version of basic train (unsupervised)
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wt,
    )
    best_result = model.state_dict()
    for epoch in range(num_epochs):
        # print("training phase")
        # for idx, (data, labels) in enumerate(train_loader):
        for idx, data in enumerate(train_loader):
            x = data[0].flatten(1).to(device)
            # y = labels.to(device)
            y = data[1].to(device)
            # x = data.to(device)
            model.train()
            model.requires_grad_(True)
            output = model.forward(
                x,
                y=None,
            )
            # output = model.forward(x,y, training=True)
            loss = output["losses"][loss_type]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_result = model.state_dict()
            if report_interval == 0:
                continue
            elif epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("training phase")
                model.printDict(output["losses"])
                print()
                if do_plot:
                    ut.do_plot_helper(model, device)
                if test_accuracy:
                    ut.test_accuracy_helper(model, x, y, device)
                model.train()
                model.to(device)
            else:
                #print("epoch " + str(epoch))
                pass
    model.cpu()
    model.eval()
    optimizer = None
    model.load_state_dict(best_result)
    return

__all__.append("basicTrainLoop")
def basicTrainLoop(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    non-conditional version of basicTrainLoopCond
    """
    for lr in lrs:
        print(
            "epoch's lr = ",
            lr,
        )
        basicTrain(
            model,
            train_loader,
            test_loader,
            num_epochs,
            lr,
            device,
            wt,
            loss_type,
            report_interval,
            do_plot=do_plot,
            test_accuracy=test_accuracy,
        )

    print("done training")
    return



__all__.append("basicTrainCond")
def basicTrainCond(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    best_loss: float = 1e6,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    conditional version of basic train (unsupervised)
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wt,
    )
    best_result = model.state_dict()
    for epoch in range(num_epochs):
        # print("training phase")
        # for idx, (data, labels) in enumerate(train_loader):
        for idx, data in enumerate(train_loader):
            x = data[0].flatten(1).to(device)
            # y = labels.to(device)
            y = data[1].to(device)
            if len(data) > 2:
                cond1 = data[2].to(device)
            else:
                cond1 = None
            # x = data.to(device)
            model.train()
            model.requires_grad_(True)
            output = model.forward(
                x,
                cond1=cond1,
                y=None,
            )
            # output = model.forward(x,y, training=True)
            loss = output["losses"][loss_type]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_result = model.state_dict()
            if report_interval == 0:
                continue
            elif epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("training phase")
                model.printDict(output["losses"])
                print()
                if do_plot:
                    model.cpu()
                    model.eval()
                    #w = model.w_prior.sample((16,))
                    #cond = model.y_prior.sample((16,)) * 0
                    w = model.w_prior.sample((16,))
                    w = w.repeat(model.nc1, 1)
                    cond = torch.eye(model.nc1)
                    cond = cond.repeat(16,1)
                    z = model.Pz(torch.cat([w, cond], dim=-1))
                    #mu = z[:, :, : model.nz].reshape(16 * model.nclasses, model.nz)
                    mu = z[:, :, : model.nz].reshape(16 * model.nc1 * model.nclasses, model.nz)
                    cond = cond.repeat(model.nclasses,1)
                    rec = model.Px(torch.cat([mu, cond],dim=-1)).reshape(-1, 1, 28, 28)
                    if model.reclosstype == "Bernoulli":
                        rec = rec.sigmoid()
                    ut.plot_images(rec, model.nclasses * model.nc1)
                    plt.pause(0.05)
                    plt.savefig("tmp.png")
                    model.train()
                    model.to(device)
                if test_accuracy:
                    model.eval()
                    #r, p, s = ut.estimateClusterImpurityLoop(
                    r, p, s = ut.estimateClusterImpurity(
                        model,
                        x,
                        y,
                        device,
                    )
                    print(p, "\n", r.mean(), "\n", r)
                    print(
                        (r * s).sum() / s.sum(),
                        "\n",
                    )
                    model.train()
                    model.to(device)
            else:
                pass
                #print("epoch " + str(epoch))
    model.cpu()
    optimizer = None
    model.load_state_dict(best_result)
    return

__all__.append("basicTrainLoopCond")
def basicTrainLoopCond(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    conditional version of basicTrainLoopCond
    """
    for lr in lrs:
        print(
            "epoch's lr = ",
            lr,
        )
        basicTrainCond(
            model,
            train_loader,
            test_loader,
            num_epochs,
            lr,
            device,
            wt,
            loss_type,
            report_interval,
            do_plot=do_plot,
            test_accuracy=test_accuracy,
        )

    print("done training")
    return

__all__.append("trainSemiSuper")
def trainSemiSuper(
    model,
    train_loader_labeled: torch.utils.data.DataLoader,
    train_loader_unlabeled: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    # test_loader: Optional[torch.utils.data.DataLoader],
    num_epochs=15,
    lr=1e-3,
    device: str = "cuda:0",
    wt=1e-4,
    do_unlabeled: bool = True,
    do_eval: bool = True,
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
    best_loss: float = 1e6,
) -> None:
    """
    non-conditional version of trainSemiSuper
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wt,
    )
    best_result = model.state_dict()
    for epoch in range(num_epochs):
        for idx, data in enumerate(train_loader_labeled):
            x = data[0].flatten(1).to(device)
            y = data[1].to(device)
            model.train()
            model.requires_grad_(True)
            output = model.forward(x, y=y)
            loss = output["losses"]["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not do_unlabeled:
                if loss < best_loss:
                    best_result = model.state_dict()
            if report_interval == 0:
                continue
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("labeled phase")
                model.printDict(output["losses"])
                print()
                if do_plot:
                    ut.do_plot_helper(model, device)
                    model.train()
                    model.to(device)
                if test_accuracy:
                    ut.test_accuracy_helper(model, x, y, device)
                model.train()
                model.to(device)
        # for idx, (data, labels) in enumerate(train_loader_unlabeled):
        for idx, data in enumerate(train_loader_unlabeled):
            if do_unlabeled == False:
                break
            x = data[0].flatten(1).to(device)
            y = data[1].to(device)
            model.train()
            model.requires_grad_(True)
            output = model.forward(
                x,
                y=None,
            )
            # output = model.forward(x,y=y)
            loss = output["losses"]["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_result = model.state_dict()
            if report_interval == 0:
                continue
            elif epoch % report_interval == 0 and idx % 1500 == 0:
                print("unlabeled phase")
                model.printDict(output["losses"])
                print()
                if test_accuracy:
                    ut.test_accuracy_helper(model, x, y, device)
                model.train()
                model.to(device)
            # possibly need to reconsider the following:
            if idx >= len(train_loader_labeled):
                break
        for idx, data in enumerate(test_loader):
            if do_eval == False:
                break
            x = data[0].flatten(1).to(device)
            y = data[1].to(device)
            model.eval()
            output = model.forward(x, y=None, )
            loss = output["losses"]["total_loss"]
            q_y = output["q_y"]
            ce_loss = (y * q_y.log()).sum(-1).mean()
            if report_interval == 0:
                continue
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("eval phase")
                model.printDict(output["losses"])
                print("ce loss:", ce_loss.item())
                print()
    model.cpu()
    model.load_state_dict(best_result)
    # optimizer = None
    del optimizer
    #print("done training")
    return None

__all__.append("trainSemiSuperLoop")
def trainSemiSuperLoop(
    model,
    train_loader_labeled: torch.utils.data.DataLoader,
    train_loader_unlabeled: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    # test_loader: Optional[torch.utils.data.DataLoader],
    num_epochs=15,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt=1e-4,
    do_unlabeled: bool = True,
    do_validation: bool = True,
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    non-conditional version of trainSemiSuperLoop
    """
    for lr in lrs:
        print(
            "epoch's lr = ",
            lr,
        )
        trainSemiSuper(
            model,
            train_loader_labeled,
            train_loader_unlabeled,
            test_loader,
            num_epochs,
            lr,
            device,
            wt,
            do_unlabeled,
            do_validation,
            report_interval,
            do_plot=do_plot,
            test_accuracy=test_accuracy,
        )
    print("done training")
    return None


__all__.append("trainSemiSuperCond")
def trainSemiSuperCond(
    model,
    train_loader_labeled: torch.utils.data.DataLoader,
    train_loader_unlabeled: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    # test_loader: Optional[torch.utils.data.DataLoader],
    num_epochs=15,
    lr=1e-3,
    device: str = "cuda:0",
    wt=1e-4,
    do_unlabeled: bool = True,
    do_eval: bool = True,
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
    best_loss: float = 1e6,
) -> None:
    """
    conditional version of trainSemiSuper
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wt,
    )
    best_result = model.state_dict()
    for epoch in range(num_epochs):
        for idx, data in enumerate(train_loader_labeled):
            x = data[0].flatten(1).to(device)
            y = data[1].to(device)
            if len(data) > 2:
                cond1 = data[2].to(device)
            else:
                cond1 = None
            model.train()
            model.requires_grad_(True)
            output = model.forward(x, cond1=cond1, y=y)
            loss = output["losses"]["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not do_unlabeled:
                if loss < best_loss:
                    best_result = model.state_dict()
            if report_interval == 0:
                continue
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("labeled phase")
                model.printDict(output["losses"])
                print()
                if do_plot:
                    model.cpu()
                    model.eval()
                    w = model.w_prior.sample((16,))
                    z = model.Pz(w)
                    mu = z[:, :, : model.nz].reshape(16 * model.nclasses, model.nz)
                    rec = model.Px(mu).reshape(-1, 1, 28, 28)
                    if model.reclosstype == "Bernoulli":
                        rec = rec.sigmoid()
                    ut.plot_images(rec, model.nclasses)
                    plt.pause(0.05)
                    plt.savefig("tmp.png")
                    model.train()
                    model.to(device)
                if test_accuracy:
                    model.eval()
                    #r, p, s = ut.estimateClusterImpurityLoop(
                    r, p, s = ut.estimateClusterImpurity(
                        model,
                        x,
                        y,
                        device,
                    )
                    print(p, "\n", r.mean(), "\n", r)
                    print(
                        (r * s).sum() / s.sum(),
                        "\n",
                    )
                    model.train()
                    model.to(device)
        # for idx, (data, labels) in enumerate(train_loader_unlabeled):
        for idx, data in enumerate(train_loader_unlabeled):
            if do_unlabeled == False:
                break
            x = data[0].flatten(1).to(device)
            y = data[1].to(device)
            if len(data) > 2:
                cond1 = data[2].to(device)
            else:
                cond1 = None
            model.train()
            model.requires_grad_(True)
            output = model.forward(
                x,
                cond1=cond1,
                y=None,
            )
            # output = model.forward(x,y=y)
            loss = output["losses"]["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_result = model.state_dict()
            if report_interval == 0:
                continue
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("unlabeled phase")
                model.printDict(output["losses"])
                print()
            # possibly need to reconsider the following:
            if idx >= len(train_loader_labeled):
                break
        for idx, data in enumerate(test_loader):
            if do_eval == False:
                break
            x = data[0].flatten(1).to(device)
            y = data[1].to(device)
            if len(data) > 2:
                cond1 = data[2].to(device)
            else:
                cond1 = None
            model.eval()
            # output = model.forward(x,)
            # output = model.forward(x,y=y,cond1=cond1)
            output = model.forward(x, y=None, cond1=cond1)
            loss = output["losses"]["total_loss"]
            q_y = output["q_y"]
            ce_loss = (y * q_y.log()).sum(-1).mean()
            if report_interval == 0:
                continue
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("eval phase")
                model.printDict(output["losses"])
                print("ce loss:", ce_loss.item())
                print()
    model.cpu()
    model.load_state_dict(best_result)
    # optimizer = None
    del optimizer
    #print("done training")
    return None


__all__.append("trainSemiSuperLoopCond")
def trainSemiSuperLoopCond(
    model,
    train_loader_labeled: torch.utils.data.DataLoader,
    train_loader_unlabeled: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    # test_loader: Optional[torch.utils.data.DataLoader],
    num_epochs=15,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt=1e-4,
    do_unlabeled: bool = True,
    do_validation: bool = True,
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    Tandem training for two models
    """
    for lr in lrs:
        print(
            "epoch's lr = ",
            lr,
        )
        trainSemiSuperCond(
            model,
            train_loader_labeled,
            train_loader_unlabeled,
            test_loader,
            num_epochs,
            lr,
            device,
            wt,
            do_unlabeled,
            do_validation,
            report_interval,
            do_plot=do_plot,
            test_accuracy=test_accuracy,
        )
    print("done training")
    return None

#__all__.append("trainSuperCond")
def trainSuperCond(
    model,
    train_loader_labeled: torch.utils.data.DataLoader,
    num_epochs=15,
    lr=1e-3,
    device: str = "cuda:0",
    wt=1e-4,
    report_interval: int = 3,
    best_loss: float = 1e6,
) -> None:
    """
    conditional version of trainSemiSuper
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wt,
    )
    best_result = model.state_dict()
    for epoch in range(num_epochs):
        for idx, data in enumerate(train_loader_labeled):
            x = data[0].flatten(1).to(device)
            y = data[1].to(device)
            if len(data) > 2:
                cond1 = data[2].to(device)
            else:
                cond1 = None
            model.train()
            model.requires_grad_(True)
            output = model.forward(x, cond1=cond1, y=y)
            loss = output["losses"]["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_result = model.state_dict()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("labeled phase")
                model.printDict(output["losses"])
                print()
    model.cpu()
    model.load_state_dict(best_result)
    del optimizer
    #print("done training")
    return None


#__all__.append("trainSuperLoopCond")
def trainSuperLoopCond(
    model,
    train_loader_labeled: torch.utils.data.DataLoader,
    num_epochs=15,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt=1e-4,
    report_interval: int = 3,
) -> None:
    """
    Tandem training for two models
    """
    for lr in lrs:
        trainSuperCond(
            model,
            train_loader_labeled,
            num_epochs,
            lr,
            device,
            wt,
            report_interval,
        )
    print("done training")
    return None


#__all__.append("basicTandemTrain")
def basicTandemTrain(
    model,
    model2,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    best_loss: float = 1e6,
    do_plot: bool = False,
    test_accuracy: bool = False,
    mi_scale: float = 5e0,
) -> None:
    """
    non-conditional version of basic train (unsupervised)
    """
    model.train()
    model.to(device)
    model2.train()
    model2.to(device)
    optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': model2.parameters()},
            ],
        lr=lr,
        weight_decay=wt,
    )
    #best_result = model.state_dict()
    for epoch in range(num_epochs):
        # print("training phase")
        # for idx, (data, labels) in enumerate(train_loader):
        for idx, data in enumerate(train_loader):
            x = data[0].flatten(1).to(device)
            # y = labels.to(device)
            y = data[1].to(device)
            # x = data.to(device)
            model.train()
            model.requires_grad_(True)
            model2.train()
            model2.requires_grad_(True)
            if idx % 2 == 0:
                output = model.forward(
                    x,
                    y=None,
                )
            else:
                output = model2.forward(
                    x,
                    y=None,
                )
            # output = model.forward(x,y, training=True)
            loss1 = output["losses"][loss_type]
            rec1 = output["rec"]
            if idx % 2 == 0:
                output2 = model2(input=rec1,y=None)
            else:
                output2 = model(input=rec1,y=None)
            #loss2 = output["losses"][loss_type]
            loss2 = output2["losses"][loss_type]
            q1 = output["q_y"]
            q2 = output2["q_y"]
            loss_tandem = -ut.mutualInfo(q1, q2) * mi_scale
            loss = loss1 + loss2 + loss_tandem
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if loss < best_loss:
            #    best_result = model.state_dict()
            if report_interval == 0:
                continue
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("training phase")
                model.printDict(output["losses"])
                model2.printDict(output2["losses"])
                print("tandem_loss", loss_tandem.item())
                print()
                if do_plot:
                    ut.do_plot_helper(model, device)
                if test_accuracy:
                    ut.test_accuracy_helper(model, x, y, device)
                model.train()
                model.to(device)
    model.cpu()
    model.eval()
    model2.cpu()
    model2.eval()
    optimizer = None
    #model.load_state_dict(best_result)
    return

#__all__.append("basicTandemTrainLoop")
def basicTandemTrainLoop(
    model,
    model2,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
    mi_scale: float = 5e0,
) -> None:
    """
    Loop for tandemTrain
    """
    for lr in lrs:
        print(
            "epoch's lr = ",
            lr,
        )
        basicTandemTrain(
            model,
            model2,
            train_loader,
            test_loader,
            num_epochs,
            lr,
            device,
            wt,
            loss_type,
            report_interval,
            do_plot=do_plot,
            test_accuracy=test_accuracy,
            mi_scale=mi_scale,
        )

    print("done training")
    return

#__all__.append("basicDuoTrain")
def basicDuoTrain(
    model,
    model2,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    best_loss: float = 1e6,
    do_plot: bool = False,
    test_accuracy: bool = False,
    mi_scale: float = 5e0,
) -> None:
    """
    Similar to tandem but 'parallel' rather than sequential.
    Both models are fed the original input, rather than one model fed
    the reconstruction as in tandemTrain.
    """
    model.train()
    model.to(device)
    model2.train()
    model2.to(device)
    optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': model2.parameters()},
            ],
        lr=lr,
        weight_decay=wt,
    )
    #best_result = model.state_dict()
    for epoch in range(num_epochs):
        # print("training phase")
        # for idx, (data, labels) in enumerate(train_loader):
        for idx, data in enumerate(train_loader):
            x = data[0].flatten(1).to(device)
            # y = labels.to(device)
            y = data[1].to(device)
            # x = data.to(device)
            model.requires_grad_(True)
            model2.requires_grad_(True)
            output = model(x,y=None)
            output2 = model2(input=x,y=None)
            loss1 = output["losses"][loss_type]
            loss2 = output2["losses"][loss_type]
            q1 = output["q_y"]
            q2 = output2["q_y"]
            loss_tandem = -ut.mutualInfo(q1, q2) * mi_scale
            loss = loss1 + loss2 + loss_tandem
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if loss < best_loss:
            #    best_result = model.state_dict()
            if report_interval == 0:
                continue
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("training phase")
                model.printDict(output["losses"])
                model2.printDict(output2["losses"])
                print("tandem_loss", loss_tandem.item())
                print()
                if do_plot:
                    ut.do_plot_helper(model, device)
                if test_accuracy:
                    ut.test_accuracy_helper(model, x, y, device)
                model.train()
                model.to(device)
    model.cpu()
    model.eval()
    model2.cpu()
    model2.eval()
    optimizer = None
    #model.load_state_dict(best_result)
    return

#__all__.append("basicDuoTrainLoop")
def basicDuoTrainLoop(
    model,
    model2,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
    mi_scale: float = 5e0,
) -> None:
    """
    Loop for duoTrain
    """
    for lr in lrs:
        print(
            "epoch's lr = ",
            lr,
        )
        basicDuoTrain(
            model,
            model2,
            train_loader,
            test_loader,
            num_epochs,
            lr,
            device,
            wt,
            loss_type,
            report_interval,
            do_plot=do_plot,
            test_accuracy=test_accuracy,
            mi_scale=mi_scale,
        )

    print("done training")
    return

#__all__.append("basicTripleTrain")
def basicTripleTrain(
    model,
    model2,
    model3,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    best_loss: float = 1e6,
    do_plot: bool = False,
    test_accuracy: bool = False,
    mi_scale: float = 5e0,
) -> None:
    """
    Similar to tandem but 'parallel' rather than sequential.
    Both models are fed the original input, rather than one model fed
    the reconstruction as in tandemTrain.
    """
    model.train()
    model.to(device)
    model2.train()
    model2.to(device)
    model3.train()
    model3.to(device)
    optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': model2.parameters()},
            {'params': model3.parameters()},
            ],
        lr=lr,
        weight_decay=wt,
    )
    #best_result = model.state_dict()
    for epoch in range(num_epochs):
        # print("training phase")
        # for idx, (data, labels) in enumerate(train_loader):
        for idx, data in enumerate(train_loader):
            x = data[0].flatten(1).to(device)
            # y = labels.to(device)
            y = data[1].to(device)
            # x = data.to(device)
            model.requires_grad_(True)
            model2.requires_grad_(True)
            model3.requires_grad_(True)
            output = model(x,y=None)
            output2 = model2(input=x,y=None)
            output3 = model3(input=x,y=None)
            loss1 = output["losses"][loss_type]
            loss2 = output2["losses"][loss_type]
            loss3 = output3["losses"][loss_type]
            q1 = output["q_y"]
            q2 = output2["q_y"]
            q3 = output3["q_y"]
            #loss_tandem = -ut.mutualInfo(q1, q2) * mi_scale
            #loss = loss1 + loss2 + loss_tandem
            loss_total_corr = -ut.totalCorrelation3(q1, q2, q3)
            loss = loss1 + loss2 + loss3 + loss_total_corr
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if loss < best_loss:
            #    best_result = model.state_dict()
            if report_interval == 0:
                continue
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("training phase")
                model.printDict(output["losses"])
                model2.printDict(output2["losses"])
                model3.printDict(output3["losses"])
                print("tandem_loss", loss_total_corr.item())
                print()
                if do_plot:
                    ut.do_plot_helper(model, device)
                if test_accuracy:
                    ut.test_accuracy_helper(model, x, y, device)
                model.train()
                model.to(device)
    model.cpu()
    model.eval()
    model2.cpu()
    model2.eval()
    model3.cpu()
    model3.eval()
    optimizer = None
    #model.load_state_dict(best_result)
    return

#__all__.append("basicTripleTrainLoop")
def basicTripleTrainLoop(
    model,
    model2,
    model3,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
    mi_scale: float = 5e0,
) -> None:
    """
    Loop for duoTrain
    """
    for lr in lrs:
        print(
            "epoch's lr = ",
            lr,
        )
        basicTripleTrain(
            model,
            model2,
            model3,
            train_loader,
            test_loader,
            num_epochs,
            lr,
            device,
            wt,
            loss_type,
            report_interval,
            do_plot=do_plot,
            test_accuracy=test_accuracy,
            mi_scale=mi_scale,
        )

    print("done training")
    return
