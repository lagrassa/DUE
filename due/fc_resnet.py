import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from due.layers import spectral_norm_fc, SpectralBatchNorm1d
from due.layers.spectral_norm_fc import compute_gt_sigma


class FCResNet(nn.Module):
    def __init__(
        self,
        preprocess,
        input_dim,
        depth,
        spectral_normalization,
        coeff=0.95,
        hidden_features = 128,
        n_power_iterations=1,
        dropout_rate=0.01,
        num_outputs=None,
        activation="relu",
    ):
        super().__init__()
        """
        ResFNN architecture

        Introduced in SNGP: https://arxiv.org/abs/2006.10108
        """
        self.preprocess = preprocess
        self.sigmas = []
        self.first =  nn.Linear(input_dim, hidden_features)
        self.residuals = nn.ModuleList(
            [nn.Linear(hidden_features, hidden_features) for i in range(depth)]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.spectral_normalization = spectral_normalization

        if False and spectral_normalization:
            self.first = spectral_norm_fc(
                self.first, coeff=coeff, n_power_iterations=n_power_iterations
            )

            for i in range(len(self.residuals)):
                self.residuals[i] = spectral_norm_fc(
                    self.residuals[i],
                    coeff=coeff,
                    n_power_iterations=n_power_iterations,
                )

        self.num_outputs = num_outputs
        self.bns = []
        for i in range(len(self.residuals)):
            self.bns.append(SpectralBatchNorm1d(hidden_features, coeff).cuda())
        if num_outputs is not None:
            self.last = nn.Linear(hidden_features, num_outputs)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError("That acivation is unknown")

    def apply(self, fn):
        for name, module in self.named_children():
            if name == "preprocess":
                continue
            module.apply(fn)
        return super().apply(fn)

    def _apply(self, fn):
        for name, module in self.named_children():
            if name == "preprocess":
                continue
            module._apply(fn)
        return self

    def _recurse_update_dict(self, submodule, destination, info):
        return 

    def _recurse_update_dict(submodule, destination, info):
        return


    def forward(self, x):
        x = self.preprocess(x)
        track_constant = False
        if track_constant:
            sigmas = []
            with torch.no_grad():
                gt_sigma = compute_gt_sigma(self.first.weight)
            if self.spectral_normalization:
                gt_sigma_orig = compute_gt_sigma(self.first.weight_orig)
            else:
                gt_sigma_orig = gt_sigma
            #print(gt_sigma, gt_sigma_orig)
            sigmas.append(gt_sigma_orig)
            if len(self.sigmas) > 100:
                plt.title("\sigma(W)")
                plt.xlabel("forward iterations")
                plt.ylabel("\sigma")
                self.sigmas = np.vstack(self.sigmas)
                ts = np.arange(len(self.sigmas))
                plt.plot(ts, self.sigmas)
                #plt.fill_between(ts, self.sigma_means + 1.96*self.sigma_stds, self.sigma_means - 1.96 * self.sigma_stds, alpha=.2)
                plt.show()
                import ipdb; ipdb.set_trace()
                assert False
        x = self.first(x.cuda())

        for i, residual in enumerate(self.residuals):
            if track_constant:
                with torch.no_grad():
                    gt_sigma = compute_gt_sigma(residual.weight)
                    sigmas.append(gt_sigma)
            x = residual(x)
            if self.spectral_normalization:
                x = self.bns[i](x)
            x = x + self.dropout(self.activation(x))
                #x = x + self.dropout(self.activation(residual(x)))
        if track_constant:
            self.sigmas.append(sigmas)

        if self.num_outputs is not None:
            x = self.last(x)

        return x
