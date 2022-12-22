# This file contains utility functions
# For handling data, plotting, and some general purposes
# required external libraries:
# pytorch, numpy, pandas, toolz, seaborn, scanpy, anndata, networkx, scipy
import anndata as ad
import concurrent.futures
import json
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import operator
import pandas as pd
import pickle
import scanpy as sc
import seaborn as sns
import sys
import time
import toolz
import torch
import torch.utils.data
import torchvision.utils as vutils
import typing
from datetime import datetime
from math import pi, sin, cos, sqrt, log
from operator import add, mul
from scipy import stats
from toolz import groupby, count, reduce, reduceby, countby
from toolz import partial, curry
from torch import Tensor
from torch import nn, optim, distributions
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid
from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple, ClassVar
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast, NewType

__all__ = []

__all__.append("Timer")
class Timer:
    """
    Basic timer (AKA stopwatch) class for measuring runtime.
    """

    def __init__(self, state: str = "idle"):
        """
        if state != 'idle'  it will
        be counting from its initiation.
        default is idle.
        """
        self.counter = 0e0
        self._previous = -9e10
        self._state = "idle"
        if state != "idle":
            self._previous = time.perf_counter()
            self._state = "running"
        else:
            self._state = "idle"
        return

    def getState(
        self,
    ) -> str:
        return self._state

    def start(
        self,
    ) -> float:
        """
        if was idle start counting. otherwise just measure current time lapse.
        return time lapsed (counter)
        """
        if self._state == "idle":  # start counting from 0
            self._state = "running"
            self._previous = time.perf_counter()
            self.counter = 0
        else:  # had already been running
            self.counter = time.perf_counter() - self._previous
        return self.counter

    def restart(
        self,
    ) -> float:
        """
        always reset counter and initiate new counting.
        """
        self._state = "running"
        self._previous = time.perf_counter()
        self.counter = 0
        return self.counter

    def stop(
        self,
    ) -> float:
        """
        stop running if it were running.
        returns counter.
        """
        if self._state == "running":
            self.counter = time.perf_counter() - self._previous
            self._state = "idle"
        else:
            self._state = "idle"
        return self.counter

    def getCount(
        self,
    ) -> float:
        """
        returns time counter without changing _state
        """
        if self._state == "running":
            self.counter = time.perf_counter() - self._previous
        return self.counter


__all__.append("kld2normal")
def kld2normal(
    mu: Tensor,
    logvar: Tensor,
    mu2: Tensor,
    logvar2: Tensor,
):
    """
    unreduced KLD KLD(p||q) for two diagonal normal distributions.
    https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    """
    result = 0.5 * (
        -1 + (logvar.exp() + (mu - mu2).pow(2)) / logvar2.exp() + logvar2 - logvar
    )
    return result


__all__.append("soft_assign")
def soft_assign(z, mu, alpha=1):
    """
    Returns a nearly one-hot vector that indicates the nearest centroid
    to z.
    """
    q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - mu) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0
    q = q / torch.sum(q, dim=1, keepdim=True)
    return q


__all__.append("log_gaussian_prob")
def log_gaussian_prob(
    x: torch.Tensor,
    mu: torch.Tensor = torch.tensor(0),
    logvar: torch.Tensor = torch.tensor(0),
) -> torch.Tensor:
    """
    compute the log density function of a gaussian.
    user must make sure that the dimensions are aligned correctly.
    """
    return -0.5 * (log(2 * pi) + logvar + (x - mu).pow(2) / logvar.exp())


__all__.append("softclip")
def softclip(tensor, min=-6.0, max=9.0):
    """
    softly clips the tensor values at the minimum/maximimum value.
    """
    result_tensor = min + nn.functional.softplus(tensor - min)
    result_tensor = max - nn.functional.softplus(max - result_tensor)
    return result_tensor


__all__.append("SoftClip")
class SoftClip(nn.Module):
    """
    object oriented version of softclip
    """

    def __init__(self, min=-6.0, max=6.0):
        super(SoftClip, self).__init__()
        self.min = min
        self.max = max

    def forward(self, input):
        return softclip(input, self.min, self.max)


__all__.append("init_weights")
def init_weights(m: torch.nn.Module) -> None:
    """
    Initiate weights with random values, depending
    on layer type.
    In place, use the apply method.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


__all__.append("replicate")
@curry
def replicate(x: Tensor, expand=(1,)):
    """
    Replicate a tensor in a new dimension (the new 0 dimension),
    creating n=ncopies identical copies.
    """
    y = torch.ones(expand + x.shape).to(x.device)
    return y * x


__all__.append("normalize")
@curry
def normalize(
    x: Tensor,
    mu: Union[float, Tensor] = 0.5,
    sigma: Union[float, Tensor] = 0.5,
    clamp: bool = False,
) -> Tensor:
    """
    x <- (x - mu) / sigma
    """
    y = (x - mu) / sigma
    if clamp:
        y = y.clamp(0, 1)
    return y


__all__.append("denorm")
@curry
def denorm(
    x: Tensor,
    mu: Union[float, Tensor] = 0.5,
    sigma: Union[float, Tensor] = 0.5,
    clamp: bool = True,
) -> Tensor:
    """
    inverse of normalize
    x <- sigma * x + mu
    """
    y = sigma * x + mu
    if clamp:
        y = y.clamp(0, 1)
    return y


__all__.append("plot_images")
def plot_images(imgs, nrow=16, transform=nn.Identity(), out=plt):
    """
    plots input immages in a grid.
    imgs: tensor of images with dimensions (batch, channell, height, widt)
    nrow: number of rows in the grid
    out: matplotlib plot object
    outputs grid_imgs: the image grid ready to be ploted.
    also plots the result with 'out'.
    """
    imgs = transform(imgs)
    grid_imgs = make_grid(imgs, nrow=nrow).permute(1, 2, 0)
    # plt.imshow(grid_imgs)
    out.imshow(grid_imgs)
    out.grid(False)
    out.axis("off")
    plt.pause(0.05)
    return grid_imgs


__all__.append("plot_2images")
def plot_2images(
    img1,
    img2,
    nrow=16,
    transform=nn.Identity(),
):
    """
    just like 'plot_images' but takes two image sets and creates two image grids,
    which are also plotted side by side.
    """
    img1 = transform(img1)
    img2 = transform(img2)
    grid_img1 = make_grid(img1, nrow=nrow).permute(1, 2, 0)
    grid_img2 = make_grid(img2, nrow=nrow).permute(1, 2, 0)
    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(grid_img1)
    axs[1].imshow(grid_img2)
    axs[0].grid(False)
    axs[0].axis("off")
    axs[1].grid(False)
    axs[1].axis("off")
    plt.pause(0.05)
    return grid_img1, grid_img2

# used to be buildNetworkv5
__all__.append("buildNetwork")
def buildNetwork(
    layers: List[int],
    dropout: float = 0,
    activation: Optional[nn.Module] = nn.ReLU(),
    batchnorm: bool = False,
):
    """
    build a fully connected multilayer NN.
    The output layer is always linear
    """
    net = nn.Sequential()
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    for i in range(1, len(layers)-1):
        net.add_module('linear' + str(i), nn.Linear(layers[i - 1], layers[i]))
        if dropout > 0:
            net.add_module("dropout" + str(i), nn.Dropout(dropout))
        if batchnorm:
            net.add_module("batchnorm" + str(i), nn.BatchNorm1d(num_features=layers[i]))
            #net.add_module("layernotm" + str(i), nn.LayerNorm(layers[i],))
        if activation:
            net.add_module("activation" + str(i), activation)
    n = len(layers) - 1
    net.add_module("output_layer", nn.Linear(layers[n-1], layers[n]))
    return net
    #return nn.Sequential(*net)

__all__.append("mixedGaussianCircular")
@curry
def mixedGaussianCircular(k=10, sigma=0.025, rho=3.5, j=0):
    """
    Sample from a mixture of k 2d-gaussians. All have equal variance (sigma) and
    correlation coefficient (rho), with the means equally distributed on the
    unit circle.
    example:
    gauss = mixedGaussianCircular(rho=0.01, sigma=0.5, k=10, j=0)
    mix = distributions.Categorical(torch.ones(10,))
    comp = distributions.Independent(gauss, 0)
    gmm = distributions.MixtureSameFamily(mix, comp)
    """
    # cat = distributions.Categorical(torch.ones(k))
    # i = cat.sample().item()
    # theta = 2 * torch.pi * i / k
    theta = 2 * torch.pi / k
    v = torch.Tensor((1, 0))
    T = torch.Tensor([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    S = torch.stack([T.matrix_power(i) for i in range(k)])
    mu = S @ v
    # cov = sigma ** 2 * ( torch.eye(2) + rho * (torch.ones(2, 2) - torch.eye(2)))
    # cov = cov @ S
    cov = torch.eye(2) * sigma**2
    cov[1, 1] = sigma**2 * rho
    cov = torch.stack([
        T.matrix_power(i + j) @ cov @ T.matrix_power(-i - j) for i in range(k)
    ])
    gauss = distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)
    return gauss

__all__.append("randomSubset")
def randomSubset(
        s : int,
        r: float,
        ):
    """
    returns a numpy boolean 1d array of size size,
    with approximately ratio of r*s True values.
    s must be positive integer
    r must be in the range [0,1]
    """
    x = np.random.rand(s)
    x = x <= r
    return x


__all__.append("Blobs")
class Blobs:
    """
    samples gaussian blobs.
    note it is not related to the function 'blobs' ;)
    """
    def __init__(
            self,
            means: Tensor = torch.rand(5,2) * 5e0,
            scales: Tensor = torch.rand(5,2) * 2e-1,
            ) -> None:
        self.means = means
        self.scales = scales
        self.nclasses = means.shape[0]
        self.ndim = means.shape[1]
        self.comp = distributions.Normal(means, scales)
        self.mix = distributions.Categorical(torch.ones((self.nclasses,)))
        return
    def sample(self, batch_size=(100,)):
        l = self.mix.sample(batch_size)
        x = self.comp.sample(batch_size)
        s = torch.vstack(
                [x[i,l[i]] for i in range(len(x))]
                )
        return s, l, x
    def plotSample(self, batch_size=(300,)):
        s, l, x = self.sample(batch_size)
        sns.scatterplot(x=s[:,0], y=s[:,1], hue=l,)
        return

__all__.append("blobs")
def blobs(
        nx : int = 2, # dimensions
        nc : int = 2, # number of conditions
        ny : int = 5, # number of blobs per condition
        ns : int = 500, # number of samples per blop per cond
        effect : float = 15e-1
        ):
    """
    create Gaussian blobs.
    nx : dimensions of the space
    nc: number of conditions
    ny: number of components
    ns: number of samples per blob per condition
    effect: approx shift effect of treatment
    """
    mu1 = torch.rand(ny, nx)*1e1
    std1 = torch.rand(ny,nx)*5e-1
    #shift = 5e0 * torch.rand(ny,nx)
    shift = effect + torch.randn(ny,nx)*5e-1
    mu2 = mu1 + shift
    std2 = std1 + torch.randn_like(std1)*1e-2
    mu = torch.concat(
            [mu1,mu2], dim=0,).unsqueeze(0)
    std = torch.concat(
            [std1,std2], dim=0,).unsqueeze(0)
    X1 = torch.randn(ns, ny, nx)*std1 + mu1
    X2 = torch.randn(ns, ny, nx)*std2 + mu2
    X = torch.concat(
            [X1,X2], dim=0,).reshape(-1,nx).numpy()
    df = pd.DataFrame()
    adata = sc.AnnData(X=X)
    condition = ns * ny * ['ctrl'] + ns*ny*['trtmnt']
    label = [str(x) for x in toolz.concat(
        ns*nc * [np.arange(ny)]) ]
    df["label"] = label
    df["cond"] = condition
    if nx == 1:
        df[["x1",]] = X
    elif nx == 2:
        df[["x1","x2"]] = X
    else:
        df[["x1","x2", "x3"]] = X[:,:3]
    adata.obs = df
    return adata


__all__.append("SynteticSampler")
class SynteticSampler:
    """
    An object which samples from a mixture distribution.
    Should contain methods that return batched samples.
    """

    def __init__(
        self,
        means: torch.Tensor = torch.rand(5, 2),
        logscales: torch.Tensor = torch.randn(5) * 5e-1,
        noiseLevel: float = 5e-2,
    ) -> None:
        self.means = means
        self.logscales = logscales
        self.scales = logscales.exp()
        self.n_dim = means.shape[1]
        self.n_classes = means.shape[0]
        #self.m = m = distributions.Normal(
        #    loc=means,
        #    scale=logscales.exp(),
        #)
        self.noiseLevel = noiseLevel
        return

    def sample(self, batch_size=(100,)):
        m = distributions.Categorical(probs=torch.ones(self.n_classes))
        labels = m.sample(batch_size)
        locs = torch.stack(
                [self.means[labels[i]] for i in range(len(labels))], dim=0)
        scales = torch.stack(
                [self.scales[labels[i]] for i in range(len(labels))],
                dim=0).unsqueeze(1)
        noise = torch.randn_like(locs) * self.noiseLevel
        theta = torch.rand_like(labels*1e-1) * pi * 2
        data = torch.zeros_like(locs)
        data[:,0] = theta.cos()
        data[:,1] = theta.sin()
        data = data * scales + locs + noise
        return data, labels, locs, scales

    def plotData(
        self,
    ) -> None:
        # data = self.m.sample((1000,))
        data = torch.rand((300, self.n_classes)) * pi * 2
        fig = plt.figure()
        for idx in range(self.n_classes):
            color = plt.cm.Set1(idx)
            # x = data[:,idx,0]
            # y = data[:,idx,1]
            x = (
                data[:, idx].cos()
                * self.logscales[idx].exp()
                + self.means[idx, 0]
                + torch.randn(300) * self.noiseLevel
            )
            y = (
                data[:, idx].sin()
                * self.logscales[idx].exp()
                + self.means[idx, 1]
                + torch.randn(300) * self.noiseLevel
            )
            plt.scatter(
                x,
                y,
                color=color,
                s=10,
                cmap="viridis",
            )
        plt.legend([str(i) for i in range(self.n_classes)])
        plt.title("data plot")

    def plotSampleData(
        self,
    ) -> None:
        data, labels, _, _  = self.sample((1000,))
        fig = plt.figure()
        for idx in range(self.n_classes):
            color = plt.cm.Set1(idx)
            # x = data[:,idx,0]
            # y = data[:,idx,1]
            x = data[labels == idx][:, 0]
            y = data[labels == idx][:, 1]
            plt.scatter(
                x,
                y,
                color=color,
                s=10,
                cmap="viridis",
            )
        plt.legend([str(i) for i in range(self.n_classes)])
        plt.title("data plot")

# used to be SynteticDataSetV2
__all__.append("SynteticDataSet")
class SynteticDataSet(torch.utils.data.Dataset):
    """
    with arbitrary number of variables.
    initiate it with a list of tensorts representing for example
    [input, target, conditions, ,,,]
    """
    def __init__(self, dati : List[Tensor], ):
        super().__init__()
        self.dati = dati
        self.numvars = len(dati)
        return
    def __getitem__(self, idx : int):
        return [x[idx] for x in self.dati]
    def __len__(self):
        return len(self.dati[0])

__all__.append("diffMatrix")
def diffMatrix(A : np.ndarray, alpha : float = 0.25):
    """
    Returns the diffusion Kernel K for a given adjacency matrix 
    A, and restart pobability alpha.
    K = α[I - (1 - α)AD^-1]^-1
    """
    #D = np.diag(A.sum(0))
    T = A / A.sum(0)
    I = np.eye(A.shape[0])
    K = alpha * np.linalg.inv(I - (1 - alpha)*T)
    return K


__all__.append("softArgMaxOneHot")
def softArgMaxOneHot(
        x : torch.Tensor,
        factor : float = 1.2e2,
        a : float = 4,
        #one_hot : bool = True,
        ) -> torch.Tensor:
    """
    x: 1d float tensor or batch of vectors.
    factor: larger factor will make the returned result
    more similar to pure argmax (less smooth).
    returns a nearly one-hot vector indicating
    the maximal value of x. 
    Possible not currently implemented feature:
    if one_hot==False,
    returns apprixmately the argmax index itselg.
    """
    #z = 1.2e1 * x / x.norm(1)
    #z = z.exp().softmax(-1)
    #z = factor * x / x.norm(1, dim=-1).unsqueeze(-1)
    z = factor * (1 + x / x.norm(1, dim=-1).unsqueeze(-1))
    z = z.pow(a).softmax(-1)
    return z

__all__.append("SoftArgMaxOneHot")
class SoftArgMaxOneHot(nn.Module):
    """
    class version of the eponymous function.
    """
    def __init__(self, factor=1.2e2,):
        super().__init__()
        self.factor = factor
    def forward(self, input):
        return softArgMaxOneHot(input, self.factor)


__all__.append("mutualInfo")
def mutualInfo(p : torch.Tensor, q : torch.Tensor,):
    """
    p,q : n×y (batches of categorical distributions over y).
    returns their mutual information:
    I(p,q) = \int \log(P(x,y) - \logp(x) - \logq(y)dP(x,y)
    """
    batch_size, y = p.size()
    pp = p.reshape(batch_size, y, 1)
    qq = q.reshape(batch_size, 1, y)
    P = pp @ qq
    P = P.mean(0) # P(x,y)
    Pi = P.sum(1).reshape(y,1)
    Pj = P.sum(0).reshape(1,y)
    #Q = Pi @ Pj #P(x)P(y)
    #I = torch.sum(
    #        P * (P.log() - Q.log())
    #        )
    # alternatively:
    #I(X,Y) = H(X) + H(Y) - H(X,Y)
    HP = -torch.sum(
            P * P.log()
            )
    HPi = -torch.sum(
            Pi * Pi.log()
            )
    HPj = -torch.sum(
            Pj * Pj.log()
            )
    Ia = HPi + HPj - HP
    #return I, P, Q, Ia
    return Ia



__all__.append("urEntropy")
def urEntropy(p : torch.Tensor,):
    """
    returns -p * log(p) (element-wise, so unreduced).
    """
    return -p * p.log()

__all__.append("mutualInfo2")
def mutualInfo2(p : torch.Tensor, q : torch.Tensor,):
    """
    p,q : n×y (batches of categorical distributions over y).
    returns their mutual information:
    I(p,q) = \int \log(P(x,y) - \logp(x) - \logq(y)dP(x,y)
    """
    batch_size, n = p.size()
    #p(x,y):
    P = torch.einsum("...x,...y -> ...xy", p, q).mean(0)
    #p(x):
    #Px = torch.einsum("xy -> x", P)
    Px = P.sum(1,keepdim=True) #(n,1)
    #p(y):
    #Py = torch.einsum("xy -> y", P)
    Py = P.sum(0,keepdim=True) #(1,n)
    #p(x | y):
    Px_y = P / Py
    #p(y | x):
    #Py_x = P / Px
    #H(X)
    Hx = urEntropy(Px).sum()
    #H(X|Y)
    Hx_y = -(P * Px_y.log()).sum()
    return Hx - Hx_y

__all__.append("mutualInfo3")
def mutualInfo3(p : torch.Tensor, q : torch.Tensor, r : torch.Tensor):
    """
    p,q,r : n×y (batches of categorical distributions over y).
    returns their mutual information:
    I(p,q,r) = I(p,q) - I(p,q|r)
    be warned it can be negative and is hard to interpret,
    """
    Ipq = mutualInfo2(p,q)
    # p(x,y,z):
    P = torch.einsum("...x,...y,...z -> ...xyz", p, q,r).mean(0)
    # p(x,z):
    Pxz = P.sum(1, keepdim=True)
    # p(y,z)
    Pyz = P.sum(0, keepdim=True)
    # p(z):
    Pz = P.sum((0,1), keepdim=True)
    # I(x,y | z):
    #Ipq_r = (P * (P * Pz / Pxz / Pyz).log()).sum()
    temp = (P.log() + Pz.log() - Pxz.log() - Pyz.log())
    temp = temp * P
    Ipq_r = temp.sum()
    return Ipq - Ipq_r

__all__.append("totalCorrelation3")
def totalCorrelation3(p,q,r):
    """
    returns 
    KLD(p(x,y,z) || p(x)p(y)p(z))
    """
    #p(x,y,z):
    P = torch.einsum("...x,...y,...z -> ...xyz", p, q,r).mean(0)
    Px = P.sum((1,2), keepdim=True)
    Py = P.sum((0,2), keepdim=True)
    Pz = P.sum((0,1), keepdim=True)
    #p(x)p(y)p(z):
    Q = Px * Py * Pz
    tc = P * (P.log() - Q.log())
    return tc.sum()

__all__.append("checkCosineDistance")
@curry
def checkCosineDistance(
        x : torch.Tensor,
        model : nn.Module,
        ) -> torch.Tensor:
    """
    x : the input tensor
    model: the Autoencoder to feed x into.
    outputs mean cosDistance(x, y)
    where y = reconstruction of x by model.
    """
    #model.to(x.device)
    y = model(x.flatten(1),)['rec']
    cosD = torch.cosine_similarity(x.flatten(1), y, dim=-1).mean()
    return cosD



__all__.append("estimateClusterImpurity")
def estimateClusterImpurity(
        model,
        x,
        labels,
        device : str = "cpu",
        cond1 : Optional[torch.Tensor] = None,
        ):
    model.eval()
    model.to(device)
    if cond1 != None:
        output = model(x.to(device), cond1=cond1.to(device))
    else:
        output = model(x.to(device))
    model.cpu()
    y = output["q_y"].detach().to("cpu")
    del output
    n = y.shape[1] # number of clusters
    r = -np.ones(n) # homogeny index
    p = -np.ones(n) # label assignments to the clusters
    s = -np.ones(n) # label assignments to the clusters
    for i in range(n):
        c = labels[y.argmax(-1) == i]
        if c.shape[0] > 0:
            r[i] = c.sum(0).max().item() / c.shape[0] 
            p[i] = c.sum(0).argmax().item()
            s[i] = c.shape[0]
    return r, p, s


__all__.append("estimateClusterImpurityHelper")
def estimateClusterImpurityHelper(
        model,
        x,
        labels,
        device : str = "cpu",
        cond1 : Optional[Tensor] = None,
        ):
    model.eval()
    model.to(device)
    if cond1 != None:
        output = model(x.to(device), cond1=cond1.to(device))
    else:
        output = model(x.to(device))
    model.cpu()
    y = output["q_y"].detach().to("cpu")
    del output
    return y


__all__.append("estimateClusterImpurityLoop")
def estimateClusterImpurityLoop(
        model,
        xs,
        labels,
        device : str = "cpu",
        cond1 : Optional[torch.Tensor] = None, #broken for now
        ):
    y = []
    model.eval()
    model.to(device)
    if cond1 != None:
        dataset = SynteticDataSetV2(
                dati=[xs, labels, cond1,],
                )
    else:
        dataset = SynteticDataSetV2(
                dati=[xs, labels, ],
                )
    data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=128,
            shuffle=False,
            )
    for input in data_loader.__iter__():
        x = input[0]
        label = input[1]
        c = None
        if len(input) > 2:
            c = input[2]
            c.to(device)
        x.to(device)
        label.to(device)
        q_y = estimateClusterImpurityHelper(model, x, label, device, c)
        y.append(q_y.cpu())
    y = torch.concat(y, dim=0)
    n = y.shape[1] # number of clusters
    r = -np.ones(n) # homogeny index
    p = -np.ones(n) # label assignments to the clusters
    s = -np.ones(n) # label assignments to the clusters
    for i in range(n):
        c = labels[y.argmax(-1) == i]
        if c.shape[0] > 0:
            r[i] = c.sum(0).max().item() / c.shape[0] 
            p[i] = c.sum(0).argmax().item()
            s[i] = c.shape[0]
    return r, p, s

__all__.append("estimateClusterAccuracy")
def estimateClusterAccuracy(
        y : Tensor,
        labels : Tensor,
        ):
    """
    y : (relaxed) one_hot tensor (cluster indicator)
    labels: one_hot vector (ground truth class indicator)
    returns: r,p,s 
    """
    n = y.shape[1] # number of clusters
    r = -np.ones(n) # homogeny index
    p = -np.ones(n) # label assignments to the clusters
    s = -np.ones(n) # label assignments to the clusters
    for i in range(n):
        c = labels[y.argmax(-1) == i]
        if c.shape[0] > 0:
            r[i] = c.sum(0).max().item() / c.shape[0] 
            p[i] = c.sum(0).argmax().item()
            s[i] = c.shape[0]
    return r, p, s

__all__.append("do_plot_helper")
def do_plot_helper(model, device : str = "cpu",):
    """
    ploting helper function for 
    training procedures
    for gmm model
    """
    model.cpu()
    model.eval()
    w = model.w_prior.sample((16,))
    z = model.Pz(torch.cat([w, ], dim=-1))
    mu = z[:, :, : model.nz].reshape(16 * model.nclasses, model.nz)
    rec = model.Px(torch.cat([mu, ],dim=-1)).reshape(-1, 1, 28, 28)
    if model.reclosstype == "Bernoulli":
        rec = rec.sigmoid()
    plot_images(rec, model.nclasses )
    plt.pause(0.05)
    plt.savefig("tmp.png")
    #model.train()
    #model.to(device)
    return

__all__.append("do_plot_helper_cmm")
def do_plot_helper_cmm(model, device : str = "cpu",):
    """
    ploting helper function for 
    training procedures
    for cmm model
    """
    model.cpu()
    model.eval()
    w = model.w_prior.sample((16,))
    w = w.repeat_interleave(repeats=model.nclasses,dim=0)
    y = torch.eye(model.nclasses)
    y = y.repeat(16, 1)
    wy = torch.cat([w,y], dim=1)
    z = model.Pz(torch.cat([wy, ], dim=-1))
    mu = z[:, : model.nz]
    rec = model.Px(torch.cat([mu, ],dim=-1)).reshape(-1, 1, 28, 28)
    if model.reclosstype == "Bernoulli":
        rec = rec.sigmoid()
    plot_images(rec, model.nclasses )
    plt.pause(0.05)
    plt.savefig("tmp.png")
    #model.train()
    #model.to(device)
    return


__all__.append("test_accuracy_helper")
def test_accuracy_helper(model, x, y, device : str = "cpu",):
    model.cpu()
    model.eval()
    #r, p, s = estimateClusterImpurityLoop(
    r, p, s = estimateClusterImpurity(
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
    #model.train()
    #model.to(device)
    return

__all__.append("is_jsonable")
def is_jsonable(x) -> bool:
    try:
        json.dumps(x)
        return True
    except:
        return False

__all__.append("is_pickleable")
def is_pickleable(x) -> bool:
    try:
        pickle.dumps(x)
        return True
    except:
        return False

__all__.append("is_serializeable")
def is_serializeable(x, method="json",) -> bool:
    if method == "json":
        return is_jsonable(x)
    else:
        return is_pickleable(x)


__all__.append("saveModelParameters")
def saveModelParameters(
    model: nn.Module,
    fpath: str,
    method: str = "json",
) -> Dict:
    d = {}
    d['myName'] = str(model.__class__)
    for k, v in model.__dict__.copy().items():
        if is_serializeable(v, method):
            d[k] = v
    if method == "json":
        f = open(fpath, "w")
        json.dump(d, f)
        f.close()
    else:  # use pickle
        f = open(fpath, "wb")
        pickle.dump(d, f)
        f.close()
    return d

__all__.append("loadModelParameter")
def loadModelParameter(
        fpath : str,
        method : str = "json",
        ) -> Dict:
    if method == "json":
        f = open(fpath, "r")
        params = json.load(f)
        f.close()
    else: # use pickle
        f = open(fpath, "rb")
        params = pickle.load(f,)
        f.close()
    return params

__all__.append("balanceAnnData")
def balanceAnnData(
    adata: ad._core.anndata.AnnData,
    catKey: str,
    numSamples: int = 2500,
    noreps: bool = False,
    eps : float = 1e-4,
    add_noise : bool = False,
    augment_mode: bool = False,
) -> ad._core.anndata.AnnData:
    """
    creates a balanced set with numSamples objects per each
    category in the selected catKey category class
    by random selection with repetitions.
    IF noreps == True, numSamples is ignored and instead
    from each group m samples without repetitions are choses,
    where m is the size of the smallest group.
    if augment_mode is True, the original data will be included together
    with the samples, so the result dataset will not be exactly balanced.
    """
    andata_list = []
    if augment_mode:
        andata_list = [adata,]
    cats = list(np.unique(adata.obs[catKey]))
    m = 0
    if noreps:
        m = np.min(list(countby(lambda x: x, adata.obs[catKey]).values()))
    for c in cats:
        marker = adata.obs[catKey] == c
        n = np.sum(marker)
        if not noreps:
            s = np.random.randint(0, n, numSamples)  # select with repetitions
        else:
            s = np.random.permutation(n)[:m]
        andata_list.append(
            adata[marker][s].copy(),
        )
    xdata = ad.concat(
        andata_list,
        join="outer",
        label="set",
    )
    xdata.obs_names_make_unique()
    if add_noise:
        #sc.pp.scale(xdata,)
        #xdata.obs_names_make_unique()
        #noise = eps * np.random.randn(*xdata.X.shape).astype("float32")
        #xdata.X = xdata.X + noise
        xdata.X += eps * (np.random.randn(*xdata.X.shape)).astype("float32")
        #sc.pp.scale(xdata,)
        xdata.X -= (adata.X.var(0) > 0) * xdata.X.mean(0)
        xdata.X /= xdata.X.std(0)
    return xdata
    

__all__.append("randomString")
def randomString(
        n : int = 8,
        pad : str ="_",
        ) -> str:
    """
    generate a random ascii string of length n, 
    padded from both ends with pad.
    """
    ls = np.random.randint(ord("A"), ord("z")+1, n)
    ls = [chr(i) for i in ls]
    ls = reduce(add, ls)
    cs = toolz.concatv(
            np.arange(ord("A"), ord("Z")+1,1),
            np.arange(ord("a"), ord("z")+1,1),
            )
    cs = list(cs)
    ls = np.random.choice(cs, n)
    ls = reduce(
            add, toolz.map(chr, ls))
    ls = pad + ls + pad
    return ls


__all__.append("timeStamp")
def timeStamp() -> str:
    """
    generates a timestap string.
    """
    return str(datetime.timestamp(datetime.now()))

# stolen from https://github.com/theislab/scgen/blob/master/scgen/_scgen.py
__all__.append("reg_mean_plot")
def reg_mean_plot(
    adata,
    axis_keys = {"x" : "control", "y" : "stimulated"},
    labels = {"x" : "control", "y" : "stimulated"},
    condition_key : str="condition",
    path_to_save="./reg_mean.pdf",
    save=True,
    gene_list=None,
    show=False,
    top_100_genes=None,
    verbose=False,
    legend=True,
    title=None,
    x_coeff=0.30,
    y_coeff=0.8,
    fontsize=14,
    **kwargs,
):
    """
    Plots mean matching figure for a set of specific genes.
    Parameters
    ----------
    adata: `~anndata.AnnData`
        AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
        AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
        corresponding to batch and cell type metadata, respectively.
    axis_keys: dict
        Dictionary of `adata.obs` keys that are used by the axes of the plot. Has to be in the following form:
         `{"x": "Key for x-axis", "y": "Key for y-axis"}`.
    labels: dict
        Dictionary of axes labels of the form `{"x": "x-axis-name", "y": "y-axis name"}`.
    path_to_save: basestring
        path to save the plot.
    save: boolean
        Specify if the plot should be saved or not.
    gene_list: list
        list of gene names to be plotted.
    show: bool
        if `True`: will show to the plot after saving it.
    Examples
    --------
    >>> import anndata
    >>> import scgen
    >>> import scanpy as sc
    >>> train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
    >>> scgen.SCGEN.setup_anndata(train)
    >>> network = scgen.SCGEN(train)
    >>> network.train()
    >>> unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
    >>> pred, delta = network.predict(
    >>>     adata=train,
    >>>     adata_to_predict=unperturbed_data,
    >>>     ctrl_key="control",
    >>>     stim_key="stimulated"
    >>>)
    >>> pred_adata = anndata.AnnData(
    >>>     pred,
    >>>     obs={"condition": ["pred"] * len(pred)},
    >>>     var={"var_names": train.var_names},
    >>>)
    >>> CD4T = train[train.obs["cell_type"] == "CD4T"]
    >>> all_adata = CD4T.concatenate(pred_adata)
    >>> network.reg_mean_plot(
    >>>     all_adata,
    >>>     axis_keys={"x": "control", "y": "pred", "y1": "stimulated"},
    >>>     gene_list=["ISG15", "CD3D"],
    >>>     path_to_save="tests/reg_mean.pdf",
    >>>     show=False
    >>> )
    """
    plt.cla()
    plt.clf()
    plt.close()

    sns.set()
    sns.set(color_codes=True)

    diff_genes = top_100_genes
    stim = adata[adata.obs[condition_key] == axis_keys["y"]]
    ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
    if diff_genes is not None:
        if hasattr(diff_genes, "tolist"):
            diff_genes = diff_genes.tolist()
        adata_diff = adata[:, diff_genes]
        stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
        ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
        x_diff = np.asarray(np.mean(ctrl_diff.X, axis=0)).ravel()
        y_diff = np.asarray(np.mean(stim_diff.X, axis=0)).ravel()
        m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
            x_diff, y_diff
        )
        if verbose:
            print("top_100 DEGs mean: ", r_value_diff**2)
    x = np.asarray(np.mean(ctrl.X, axis=0)).ravel()
    y = np.asarray(np.mean(stim.X, axis=0)).ravel()
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    if verbose:
        print("All genes mean: ", r_value**2)
    df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
    ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
    ax.tick_params(labelsize=fontsize)
    if "range" in kwargs:
        start, stop, step = kwargs.get("range")
        ax.set_xticks(np.arange(start, stop, step))
        ax.set_yticks(np.arange(start, stop, step))
    ax.set_xlabel(labels["x"], fontsize=fontsize)
    ax.set_ylabel(labels["y"], fontsize=fontsize)
    if gene_list is not None:
        texts = []
        for i in gene_list:
            j = adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            texts.append(plt.text(x_bar, y_bar, i, fontsize=11, color="black"))
            plt.plot(x_bar, y_bar, "o", color="red", markersize=5)
            # if "y1" in axis_keys.keys():
            # y1_bar = y1[j]
            # plt.text(x_bar, y1_bar, i, fontsize=11, color="black")
    #if gene_list is not None:
    #    adjust_text(
    #        texts,
    #        x=x,
    #        y=y,
    #        arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
    #        force_points=(0.0, 0.0),
    #    )
    if legend:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if title is None:
        plt.title("", fontsize=fontsize)
    else:
        plt.title(title, fontsize=fontsize)
    ax.text(
        max(x) - max(x) * x_coeff,
        max(y) - y_coeff * max(y),
        r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2:.2f}",
        fontsize=kwargs.get("textsize", fontsize),
    )
    if diff_genes is not None:
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - (y_coeff + 0.15) * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
            + f"{r_value_diff ** 2:.2f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
    if save:
        plt.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
    if show:
        plt.show()
    plt.close()
    if diff_genes is not None:
        return r_value**2, r_value_diff**2
    else:
        return r_value**2
