import torch

import gpytorch
import numpy as np
from sklearn.preprocessing import StandardScaler
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, RQKernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.models import GP as GPytorchGP

from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)
from moonshine.gpytorch_tools import custom_index_batch, mutate_dict_to_cpu

from sklearn import cluster


def initial_values(train_dataset, feature_extractor, n_inducing_points):
    steps = 10
    check_enough_high_error = False
    if isinstance(train_dataset, dict):
        idx = torch.randperm(len(train_dataset["traj_idx"]))[:1000].chunk(steps)
        check_enough_high_error = True
    else:
        idx = torch.randperm(len(train_dataset))[:1000].chunk(steps)
    f_X_samples = []
    y_samples = []
    # a bit of a hack: ensure at least 5 "high error" samples
    min_error_samples = 7
    high_error = 0.1
    if check_enough_high_error:
        high_error_total = (train_dataset["error"][:,1] > high_error).sum()
        print("High error total", high_error_total)
        for attempt_idx in range(10):
            ys_check = []
            with torch.no_grad():
                for i in range(steps):
                    X_sample = custom_index_batch(train_dataset, idxs = idx[i])
                    ys_check.append(X_sample["error"][:,1].numpy())
                ys_check = np.vstack(ys_check)
                num_high_error =  (ys_check.flatten() > high_error).sum()
                print("num high error",num_high_error)
                if num_high_error < min_error_samples:
                    idx = torch.randperm(len(train_dataset["traj_idx"]))[:1000].chunk(steps)
                else:
                    break


    with torch.no_grad():
        for i in range(steps):
            if isinstance(train_dataset, dict):
                X_sample = custom_index_batch(train_dataset, idxs = idx[i])
                #mutate_dict_to_cpu(X_sample)
                y_samples.append(X_sample["error"][:,1].cpu())
            else:
                X_sample = torch.stack([train_dataset[j][0] for j in idx[i]])

            if torch.cuda.is_available():
                #X_sample = X_sample.cuda()
                feature_extractor = feature_extractor.cuda()

            f_X_samples.append(feature_extractor(X_sample).cpu())

    f_X_samples = torch.cat(f_X_samples)
    if len(y_samples):
        y_samples = torch.cat(y_samples)
    else:
        y_samples = torch.Tensor([])

    initial_inducing_points = _get_initial_inducing_points(
        f_X_samples.numpy(), n_inducing_points, y_samples.numpy()
    )
    initial_lengthscale = _get_initial_lengthscale(f_X_samples)

    return initial_inducing_points, initial_lengthscale


def _get_initial_inducing_points(f_X_sample, n_inducing_points, y_sample):
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_inducing_points, batch_size=n_inducing_points * 10
        )
    if len(y_sample):
        scaler = StandardScaler()
        scaler.fit(y_sample.reshape(-1,1))
        y_sample_scaled  = scaler.transform(y_sample.reshape(-1,1))
        f_X_sample = np.hstack([f_X_sample, y_sample_scaled.reshape(-1,1)])
    kmeans.fit(f_X_sample)
    ys = []
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)
    for pt in initial_inducing_points:
        index_of_point = np.abs(f_X_sample -  pt.numpy()).sum(axis=1).argmin()
        if len(y_sample):
            ys.append(y_sample[index_of_point])

    #get errors of each inducing point
    #assert (np.array(ys) > 0.1).sum() > 5
    if len(y_sample):
        return initial_inducing_points[:,:-1]
    else:
        return initial_inducing_points


def _get_initial_lengthscale(f_X_samples):
    if torch.cuda.is_available():
        f_X_samples = f_X_samples.cuda()

    initial_lengthscale = torch.pdist(f_X_samples).mean()

    return initial_lengthscale.cpu()


class GP(ApproximateGP):
    def __init__(
        self,
        num_outputs,
        initial_lengthscale,
        initial_inducing_points,
        kernel="RBF",
    ):
        n_inducing_points = initial_inducing_points.shape[0]

        if num_outputs > 1:
            batch_shape = torch.Size([num_outputs])
        else:
            batch_shape = torch.Size([])

        variational_distribution = CholeskyVariationalDistribution(
            n_inducing_points, batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self, initial_inducing_points, variational_distribution
        )

        if num_outputs > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=num_outputs
            )

        super().__init__(variational_strategy)

        kwargs = {
            "batch_shape": batch_shape,
        }

        if kernel == "RBF":
            kernel = RBFKernel(**kwargs)
        elif kernel == "Matern12":
            kernel = MaternKernel(nu=1 / 2, **kwargs)
        elif kernel == "Matern32":
            kernel = MaternKernel(nu=3 / 2, **kwargs)
        elif kernel == "Matern52":
            kernel = MaternKernel(nu=5 / 2, **kwargs)
        elif kernel == "RQ":
            kernel = RQKernel(**kwargs)
        else:
            raise ValueError("Specified kernel not known.")

        kernel.lengthscale = initial_lengthscale * torch.ones_like(kernel.lengthscale)

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(kernel, batch_shape=batch_shape)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)

        return MultivariateNormal(mean, covar)

    @property
    def inducing_points(self):
        for name, param in self.named_parameters():
            if "inducing_points" in name:
                return param

class ExactGPDKL(ExactGP):
    def __init__(
        self,
        num_outputs,
        initial_lengthscale,
        train_inputs,
        train_targets,
        likelihood,
        kernel="RBF",
    ):
        self.train_inputs = train_inputs #torch.zeros().float()
        self.train_targets = train_targets
        #super(ExactGP, self).__init__()
        super().__init__(train_inputs, train_targets, likelihood)
        if num_outputs > 1:
            batch_shape = torch.Size([num_outputs])
        else:
            batch_shape = torch.Size([])

        kwargs = {
            "batch_shape": batch_shape,
        }

        if kernel == "RBF":
            kernel = RBFKernel(**kwargs)
        elif kernel == "Matern12":
            kernel = MaternKernel(nu=1 / 2, **kwargs)
        elif kernel == "Matern32":
            kernel = MaternKernel(nu=3 / 2, **kwargs)
        elif kernel == "Matern52":
            kernel = MaternKernel(nu=5 / 2, **kwargs)
        elif kernel == "RQ":
            kernel = RQKernel(**kwargs)
        else:
            raise ValueError("Specified kernel not known.")

        kernel.lengthscale = initial_lengthscale * torch.ones_like(kernel.lengthscale)

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(kernel, batch_shape=batch_shape)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)

    def __call__(self, inputs, prior=False, **kwargs):
        return super().__call__(inputs, **kwargs)

        #return self.prediction_strategy(inputs, prior=prior, **kwargs)

    @property
    def inducing_points(self):
        for name, param in self.named_parameters():
            if "inducing_points" in name:
                return param


class DKL(gpytorch.Module):
    def __init__(self, feature_extractor, gp):
        """
        This wrapper class is necessary because ApproximateGP (above) does some magic
        on the forward method which is not compatible with a feature_extractor.
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.gp = gp

    def forward(self, x):
        #mutate_dict_to_cpu(x)
        features = self.feature_extractor(x)
        return self.gp(features)


