import torch

def _solve_layer_smatrix(self):
    Kz_norm = torch.diag(self.kz_norm[-1])
    phase = torch.diag(
        torch.exp(1.0j * self.omega * self.kz_norm[-1] * self.thickness[-1])
    )

    Pinv_tmp = torch.linalg.inv(self.P[-1])
    if self.avoid_Pinv_instability:

        Pinv_ins_tmp1 = torch.max(
            torch.abs(
                torch.matmul(self.P[-1].detach(), Pinv_tmp.detach())
                - torch.eye(self.P[-1].shape[-1]).to(self.P[-1])
            )
        )
        Pinv_ins_tmp2 = torch.max(
            torch.abs(
                torch.matmul(Pinv_tmp.detach(), self.P[-1].detach())
                - torch.eye(self.P[-1].shape[-1]).to(self.P[-1])
            )
        )
        Qinv_ins_tmp1 = torch.max(
            torch.abs(
                torch.matmul(
                    self.Q[-1].detach(), torch.linalg.inv(self.Q[-1]).detach()
                )
                - torch.eye(self.Q[-1].shape[-1]).to(self.Q[-1])
            )
        )
        Qinv_ins_tmp2 = torch.max(
            torch.abs(
                torch.matmul(
                    self.Q[-1].detach(), torch.linalg.inv(self.Q[-1]).detach()
                )
                - torch.eye(self.Q[-1].shape[-1]).to(self.Q[-1])
            )
        )

        self.Pinv_instability.append(torch.maximum(Pinv_ins_tmp1, Pinv_ins_tmp2))
        self.Qinv_instability.append(torch.maximum(Qinv_ins_tmp1, Qinv_ins_tmp2))

        if self.Pinv_instability[-1] < self.max_Pinv_instability:
            self.H_eigvec.append(
                torch.matmul(Pinv_tmp, torch.matmul(self.E_eigvec[-1], Kz_norm))
            )
        else:
            self.H_eigvec.append(
                torch.matmul(
                    self.Q[-1],
                    torch.matmul(self.E_eigvec[-1], torch.linalg.inv(Kz_norm)),
                )
            )
    else:
        self.H_eigvec.append(
            torch.matmul(Pinv_tmp, torch.matmul(self.E_eigvec[-1], Kz_norm))
        )

    Ctmp1 = torch.vstack(
        (
            self.E_eigvec[-1]
            + torch.matmul(torch.linalg.inv(self.Vf), self.H_eigvec[-1]),
            torch.matmul(
                self.E_eigvec[-1]
                - torch.matmul(torch.linalg.inv(self.Vf), self.H_eigvec[-1]),
                phase,
            ),
        )
    )
    Ctmp2 = torch.vstack(
        (
            torch.matmul(
                self.E_eigvec[-1]
                - torch.matmul(torch.linalg.inv(self.Vf), self.H_eigvec[-1]),
                phase,
            ),
            self.E_eigvec[-1]
            + torch.matmul(torch.linalg.inv(self.Vf), self.H_eigvec[-1]),
        )
    )
    Ctmp = torch.hstack((Ctmp1, Ctmp2))

    # Mode coupling coefficients
    self.Cf.append(
        torch.matmul(
            torch.linalg.inv(Ctmp),
            torch.vstack(
                (
                    2
                    * torch.eye(
                        2 * self.order_N, dtype=self._dtype, device=self._device
                    ),
                    torch.zeros(
                        [2 * self.order_N, 2 * self.order_N],
                        dtype=self._dtype,
                        device=self._device,
                    ),
                )
            ),
        )
    )
    self.Cb.append(
        torch.matmul(
            torch.linalg.inv(Ctmp),
            torch.vstack(
                (
                    torch.zeros(
                        [2 * self.order_N, 2 * self.order_N],
                        dtype=self._dtype,
                        device=self._device,
                    ),
                    2
                    * torch.eye(
                        2 * self.order_N, dtype=self._dtype, device=self._device
                    ),
                )
            ),
        )
    )

    self.layer_S11.append(
        torch.matmul(
            torch.matmul(self.E_eigvec[-1], phase),
            self.Cf[-1][: 2 * self.order_N, :],
        )
        + torch.matmul(self.E_eigvec[-1], self.Cf[-1][2 * self.order_N :, :])
    )
    self.layer_S21.append(
        torch.matmul(self.E_eigvec[-1], self.Cf[-1][: 2 * self.order_N, :])
        + torch.matmul(
            torch.matmul(self.E_eigvec[-1], phase),
            self.Cf[-1][2 * self.order_N :, :],
        )
        - torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
    )
    self.layer_S12.append(
        torch.matmul(
            torch.matmul(self.E_eigvec[-1], phase),
            self.Cb[-1][: 2 * self.order_N, :],
        )
        + torch.matmul(self.E_eigvec[-1], self.Cb[-1][2 * self.order_N :, :])
        - torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
    )
    self.layer_S22.append(
        torch.matmul(self.E_eigvec[-1], self.Cb[-1][: 2 * self.order_N, :])
        + torch.matmul(
            torch.matmul(self.E_eigvec[-1], phase),
            self.Cb[-1][2 * self.order_N :, :],
        )
    )

def _solve_global_smatrix(self):
    """
    Solve global S-matrix
    """

    # Initialization
    if self.layer_N > 0:
        S11 = self.layer_S11[0]
        S21 = self.layer_S21[0]
        S12 = self.layer_S12[0]
        S22 = self.layer_S22[0]
        C = [[self.Cf[0]], [self.Cb[0]]]
    else:
        S11 = torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
        S21 = torch.zeros(2 * self.order_N, dtype=self._dtype, device=self._device)
        S12 = torch.zeros(2 * self.order_N, dtype=self._dtype, device=self._device)
        S22 = torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
        C = [[], []]

    # Connection
    for i in range(self.layer_N - 1):
        [S11, S21, S12, S22], C = self._RS_prod(
            Sm=[S11, S21, S12, S22],
            Sn=[
                self.layer_S11[i + 1],
                self.layer_S21[i + 1],
                self.layer_S12[i + 1],
                self.layer_S22[i + 1],
            ],
            Cm=C,
            Cn=[[self.Cf[i + 1]], [self.Cb[i + 1]]],
        )

    if hasattr(self, "Sin"):
        # input layer coupling
        [S11, S21, S12, S22], C = self._RS_prod(
            Sm=[self.Sin[0], self.Sin[1], self.Sin[2], self.Sin[3]],
            Sn=[S11, S21, S12, S22],
            Cm=[[], []],
            Cn=C,
        )

    if hasattr(self, "Sout"):
        # output layer coupling
        [S11, S21, S12, S22], C = self._RS_prod(
            Sm=[S11, S21, S12, S22],
            Sn=[self.Sout[0], self.Sout[1], self.Sout[2], self.Sout[3]],
            Cm=C,
            Cn=[[], []],
        )

    self.S = [S11, S21, S12, S22]
    self.C = C