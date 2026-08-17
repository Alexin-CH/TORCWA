import torch

def _field_xz(self, x_axis, z_axis, y):
    """
    XZ-plane field distribution.
    Returns the field at the specific y point.

    Paramters
    - x_axis: x-direction sampling coordinates (torch.Tensor)
    - z_axis: z-direction sampling coordinates (torch.Tensor)
    - y: selected y point

    Return
    - [Ex, Ey, Ez] (list[torch.Tensor]), [Hx, Hy, Hz] (list[torch.Tensor])
    """

    if not isinstance(x_axis, torch.Tensor) or not isinstance(z_axis, torch.Tensor):
        raise TypeError("x and z axis must be torch.Tensor type.")

    x_axis = x_axis.reshape([-1, 1, 1])
    Nz = len(z_axis)

    Kx_norm, Ky_norm = self.Kx_norm, self.Ky_norm

    # Spatial phase (identical for every z point)
    xy_phase = torch.exp(
        1.0j * self.omega * (self.Kx_norm_dn * x_axis + self.Ky_norm_dn * y)
    )

    # Layer number
    zp = torch.zeros(len(self.thickness), device=self._device)
    zm = torch.zeros(len(self.thickness), device=self._device)
    layer_num = torch.zeros([Nz], dtype=torch.int64, device=self._device)
    layer_num[z_axis < 0.0] = -1

    for ti in range(len(self.thickness)):
        zp[ti:] += self.thickness[ti]
    zm[1:] = zp[0:-1]

    for bi in range(len(zp)):
        layer_num[z_axis > zp[bi]] += 1

    def synthesize(comp, idx, out):
        # comp: [order_N, Nz_region] fourier-domain field -> spatial domain
        out[:, idx] = torch.sum(
            comp.transpose(0, 1).reshape(1, len(idx), -1) * xy_phase, dim=2
        )

    Ex = torch.zeros([x_axis.shape[0], Nz], dtype=self._dtype, device=self._device)
    Ey = torch.zeros([x_axis.shape[0], Nz], dtype=self._dtype, device=self._device)
    Ez = torch.zeros([x_axis.shape[0], Nz], dtype=self._dtype, device=self._device)
    Hx = torch.zeros([x_axis.shape[0], Nz], dtype=self._dtype, device=self._device)
    Hy = torch.zeros([x_axis.shape[0], Nz], dtype=self._dtype, device=self._device)
    Hz = torch.zeros([x_axis.shape[0], Nz], dtype=self._dtype, device=self._device)

    # Input layer (z < 0)
    in_sel = layer_num == -1
    if torch.any(in_sel):
        idx = torch.nonzero(in_sel).reshape([-1])
        z_prop = torch.clamp(z_axis[idx], max=0.0)

        eps = self.eps_in if hasattr(self, "eps_in") else 1.0
        mu = self.mu_in if hasattr(self, "mu_in") else 1.0
        Vi = self.Vi if hasattr(self, "Vi") else self.Vf
        Kz_norm_dn = torch.sqrt(eps * mu - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        Kz_norm_dn = torch.where(
            torch.imag(Kz_norm_dn) > 0,
            torch.conj(Kz_norm_dn),
            Kz_norm_dn,
        ).reshape([-1, 1])
        Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))

        # Phase
        z_phase = torch.exp(1.0j * self.omega * Kz_norm_dn * z_prop.reshape([1, -1]))

        # Fourier domain fields [diffraction order]
        if self.source_direction == "forward":
            Exy_p = self.E_i * z_phase
            Hxy_p = torch.matmul(Vi, Exy_p)
            Exy_m = torch.matmul(self.S[1], self.E_i) * torch.conj(z_phase)
            Hxy_m = torch.matmul(-Vi, Exy_m)
        else:
            Exy_p = torch.zeros_like(z_phase)
            Hxy_p = torch.zeros_like(z_phase)
            Exy_m = torch.matmul(self.S[3], self.E_i) * torch.conj(z_phase)
            Hxy_m = torch.matmul(-Vi, Exy_m)

        Ex_mn = Exy_p[: self.order_N] + Exy_m[: self.order_N]
        Ey_mn = Exy_p[self.order_N :] + Exy_m[self.order_N :]
        Hz_mn = (
            torch.matmul(Kx_norm, Ey_mn) / mu
            - torch.matmul(Ky_norm, Ex_mn) / mu
        )
        Hx_mn = Hxy_p[: self.order_N] + Hxy_m[: self.order_N]
        Hy_mn = Hxy_p[self.order_N :] + Hxy_m[self.order_N :]
        Ez_mn = (
            torch.matmul(Ky_norm, Hx_mn) / eps
            - torch.matmul(Kx_norm, Hy_mn) / eps
        )

        synthesize(Ex_mn, idx, Ex)
        synthesize(Ey_mn, idx, Ey)
        synthesize(Ez_mn, idx, Ez)
        synthesize(Hx_mn, idx, Hx)
        synthesize(Hy_mn, idx, Hy)
        synthesize(Hz_mn, idx, Hz)

    # Output layer (z beyond the last internal layer)
    out_sel = layer_num == self.layer_N
    if torch.any(out_sel):
        idx = torch.nonzero(out_sel).reshape([-1])
        if len(zp) == 0:
            z_prop = z_axis[idx]
        else:
            z_prop = torch.clamp(z_axis[idx] - zp[-1], min=0.0)

        eps = self.eps_out if hasattr(self, "eps_out") else 1.0
        mu = self.mu_out if hasattr(self, "mu_out") else 1.0
        Vo = self.Vo if hasattr(self, "Vo") else self.Vf
        Kz_norm_dn = torch.sqrt(eps * mu - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        Kz_norm_dn = torch.where(
            torch.imag(Kz_norm_dn) < 0,
            torch.conj(Kz_norm_dn),
            Kz_norm_dn,
        ).reshape([-1, 1])
        Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))

        # Phase
        z_phase = torch.exp(1.0j * self.omega * Kz_norm_dn * z_prop.reshape([1, -1]))

        # Fourier domain fields [diffraction order]
        if self.source_direction == "forward":
            Exy_p = torch.matmul(self.S[0], self.E_i) * z_phase
            Hxy_p = torch.matmul(Vo, Exy_p)
            Exy_m = torch.zeros_like(z_phase)
            Hxy_m = torch.zeros_like(z_phase)
        else:
            Exy_p = torch.matmul(self.S[2], self.E_i) * z_phase
            Hxy_p = torch.matmul(Vo, Exy_p)
            Exy_m = self.E_i * torch.conj(z_phase)
            Hxy_m = torch.matmul(-Vo, Exy_m)

        Ex_mn = Exy_p[: self.order_N] + Exy_m[: self.order_N]
        Ey_mn = Exy_p[self.order_N :] + Exy_m[self.order_N :]
        Hz_mn = (
            torch.matmul(Kx_norm, Ey_mn) / mu
            - torch.matmul(Ky_norm, Ex_mn) / mu
        )
        Hx_mn = Hxy_p[: self.order_N] + Hxy_m[: self.order_N]
        Hy_mn = Hxy_p[self.order_N :] + Hxy_m[self.order_N :]
        Ez_mn = (
            torch.matmul(Ky_norm, Hx_mn) / eps
            - torch.matmul(Kx_norm, Hy_mn) / eps
        )

        synthesize(Ex_mn, idx, Ex)
        synthesize(Ey_mn, idx, Ey)
        synthesize(Ez_mn, idx, Ez)
        synthesize(Hx_mn, idx, Hx)
        synthesize(Hy_mn, idx, Hy)
        synthesize(Hz_mn, idx, Hz)

    # Internal layers
    for li in range(len(self.thickness)):
        li_sel = layer_num == li
        if not torch.any(li_sel):
            continue
        idx = torch.nonzero(li_sel).reshape([-1])
        z_prop = z_axis[idx] - zm[li]

        if self.source_direction == "forward":
            C = torch.matmul(self.C[0][li], self.E_i)
        elif self.source_direction == "backward":
            C = torch.matmul(self.C[1][li], self.E_i)

        kz_norm = self.kz_norm[li]
        E_eigvec = self.E_eigvec[li]
        H_eigvec = self.H_eigvec[li]

        Cp = C[: 2 * self.order_N, 0].reshape([-1, 1])
        Cm = C[2 * self.order_N :, 0].reshape([-1, 1])

        eps_conv_inv = torch.linalg.inv(self.eps_conv[li])
        mu_conv_inv = torch.linalg.inv(self.mu_conv[li])

        # Phase [eigenmode, z point]
        z_phase_p = torch.exp(
            1.0j * self.omega * kz_norm.reshape([-1, 1]) * z_prop.reshape([1, -1])
        )
        z_phase_m = torch.exp(
            1.0j
            * self.omega
            * kz_norm.reshape([-1, 1])
            * (self.thickness[li] - z_prop).reshape([1, -1])
        )

        # Mode coupling weights [eigenmode, z point]
        w_p = Cp * z_phase_p
        w_m = Cm * z_phase_m

        # Fourier domain fields, summed over eigenmodes [order, z point]
        Ex_mn = torch.matmul(E_eigvec[: self.order_N], w_p + w_m)
        Ey_mn = torch.matmul(E_eigvec[self.order_N :], w_p + w_m)
        Hx_mn = (
            torch.matmul(H_eigvec[: self.order_N], w_p)
            - torch.matmul(H_eigvec[: self.order_N], w_m)
        )
        Hy_mn = (
            torch.matmul(H_eigvec[self.order_N :], w_p)
            - torch.matmul(H_eigvec[self.order_N :], w_m)
        )
        Hz_mn = torch.matmul(
            mu_conv_inv, torch.matmul(Kx_norm, Ey_mn)
        ) - torch.matmul(mu_conv_inv, torch.matmul(Ky_norm, Ex_mn))
        Ez_mn = torch.matmul(
            eps_conv_inv, torch.matmul(Ky_norm, Hx_mn)
        ) - torch.matmul(eps_conv_inv, torch.matmul(Kx_norm, Hy_mn))

        synthesize(Ex_mn, idx, Ex)
        synthesize(Ey_mn, idx, Ey)
        synthesize(Ez_mn, idx, Ez)
        synthesize(Hx_mn, idx, Hx)
        synthesize(Hy_mn, idx, Hy)
        synthesize(Hz_mn, idx, Hz)

    return [Ex, Ey, Ez], [Hx, Hy, Hz]