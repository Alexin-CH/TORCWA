import torch
from ..core.torch_eig import Eig

def _eigen_decomposition_homogenous(self, eps, mu):
    # H to E transformation matirx
    self.P.append(
        torch.hstack(
            (
                torch.vstack(
                    (torch.zeros_like(self.mu_conv[-1]), -self.mu_conv[-1])
                ),
                torch.vstack(
                    (self.mu_conv[-1], torch.zeros_like(self.mu_conv[-1]))
                ),
            )
        )
        + 1
        / eps
        * torch.matmul(
            torch.vstack((self.Kx_norm, self.Ky_norm)),
            torch.hstack((self.Ky_norm, -self.Kx_norm)),
        )
    )
    # E to H transformation matrix
    self.Q.append(
        torch.hstack(
            (
                torch.vstack(
                    (torch.zeros_like(self.eps_conv[-1]), self.eps_conv[-1])
                ),
                torch.vstack(
                    (-self.eps_conv[-1], torch.zeros_like(self.eps_conv[-1]))
                ),
            )
        )
        + 1
        / mu
        * torch.matmul(
            torch.vstack((self.Kx_norm, self.Ky_norm)),
            torch.hstack((-self.Ky_norm, self.Kx_norm)),
        )
    )

    E_eigvec = torch.eye(
        self.P[-1].shape[-1], dtype=self._dtype, device=self._device
    )
    kz_norm = torch.sqrt(eps * mu - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
    kz_norm = torch.where(
        torch.imag(kz_norm) < 0, torch.conj(kz_norm), kz_norm
    )  # Normalized kz for positive mode
    kz_norm = torch.cat((kz_norm, kz_norm))

    self.kz_norm.append(kz_norm)
    self.E_eigvec.append(E_eigvec)

def _eigen_decomposition(self):
    # H to E transformation matirx
    P_tmp = torch.matmul(
        torch.vstack((self.Kx_norm, self.Ky_norm)),
        torch.linalg.inv(self.eps_conv[-1]),
    )
    self.P.append(
        torch.hstack(
            (
                torch.vstack(
                    (torch.zeros_like(self.mu_conv[-1]), -self.mu_conv[-1])
                ),
                torch.vstack(
                    (self.mu_conv[-1], torch.zeros_like(self.mu_conv[-1]))
                ),
            )
        )
        + torch.matmul(P_tmp, torch.hstack((self.Ky_norm, -self.Kx_norm)))
    )
    # E to H transformation matrix
    Q_tmp = torch.matmul(
        torch.vstack((self.Kx_norm, self.Ky_norm)),
        torch.linalg.inv(self.mu_conv[-1]),
    )
    self.Q.append(
        torch.hstack(
            (
                torch.vstack(
                    (torch.zeros_like(self.eps_conv[-1]), self.eps_conv[-1])
                ),
                torch.vstack(
                    (-self.eps_conv[-1], torch.zeros_like(self.eps_conv[-1]))
                ),
            )
        )
        + torch.matmul(Q_tmp, torch.hstack((-self.Ky_norm, self.Kx_norm)))
    )

    # Eigen-decomposition
    if self.stable_eig_grad is True:
        kz_norm, E_eigvec = Eig.apply(torch.matmul(self.P[-1], self.Q[-1]))
    else:
        kz_norm, E_eigvec = torch.linalg.eig(torch.matmul(self.P[-1], self.Q[-1]))

    kz_norm = torch.sqrt(kz_norm)
    self.kz_norm.append(
        torch.where(torch.imag(kz_norm) < 0, -kz_norm, kz_norm)
    )  # Normalized kz for positive mode
    self.E_eigvec.append(E_eigvec)