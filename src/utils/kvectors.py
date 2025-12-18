import torch

def _kvectors(self):
    if self.angle_layer == "input":
        self.kx0_norm = (
            torch.real(torch.sqrt(self.eps_in * self.mu_in))
            * torch.sin(self.inc_ang)
            * torch.cos(self.azi_ang)
        )
        self.ky0_norm = (
            torch.real(torch.sqrt(self.eps_in * self.mu_in))
            * torch.sin(self.inc_ang)
            * torch.sin(self.azi_ang)
        )
    else:
        self.kx0_norm = (
            torch.real(torch.sqrt(self.eps_out * self.mu_out))
            * torch.sin(self.inc_ang)
            * torch.cos(self.azi_ang)
        )
        self.ky0_norm = (
            torch.real(torch.sqrt(self.eps_out * self.mu_out))
            * torch.sin(self.inc_ang)
            * torch.sin(self.azi_ang)
        )

    # Free space k-vectors and E to H transformation matrix
    self.kx_norm = self.kx0_norm + self.order_x * self.Gx_norm
    self.ky_norm = self.ky0_norm + self.order_y * self.Gy_norm

    kx_norm_grid, ky_norm_grid = torch.meshgrid(
        self.kx_norm, self.ky_norm, indexing="ij"
    )

    self.Kx_norm_dn = torch.reshape(kx_norm_grid, (-1,))
    self.Ky_norm_dn = torch.reshape(ky_norm_grid, (-1,))
    self.Kx_norm = torch.diag(self.Kx_norm_dn)
    self.Ky_norm = torch.diag(self.Ky_norm_dn)

    Kz_norm_dn = torch.sqrt(1.0 - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
    Kz_norm_dn = torch.where(
        torch.imag(Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn
    )
    tmp1 = torch.vstack(
        (
            torch.diag(-self.Ky_norm_dn * self.Kx_norm_dn / Kz_norm_dn),
            torch.diag(Kz_norm_dn + self.Kx_norm_dn**2 / Kz_norm_dn),
        )
    )
    tmp2 = torch.vstack(
        (
            torch.diag(-Kz_norm_dn - self.Ky_norm_dn**2 / Kz_norm_dn),
            torch.diag(self.Kx_norm_dn * self.Ky_norm_dn / Kz_norm_dn),
        )
    )
    self.Vf = torch.hstack((tmp1, tmp2))

    if hasattr(self, "Sin"):
        # Input layer k-vectors and E to H transformation matrix
        Kz_norm_dn = torch.sqrt(
            self.eps_in * self.mu_in - self.Kx_norm_dn**2 - self.Ky_norm_dn**2
        )
        Kz_norm_dn = torch.where(
            torch.imag(Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn
        )
        tmp1 = torch.vstack(
            (
                torch.diag(-self.Ky_norm_dn * self.Kx_norm_dn / Kz_norm_dn),
                torch.diag(Kz_norm_dn + self.Kx_norm_dn**2 / Kz_norm_dn),
            )
        )
        tmp2 = torch.vstack(
            (
                torch.diag(-Kz_norm_dn - self.Ky_norm_dn**2 / Kz_norm_dn),
                torch.diag(self.Kx_norm_dn * self.Ky_norm_dn / Kz_norm_dn),
            )
        )
        self.Vi = torch.hstack((tmp1, tmp2))

        Vtmp1 = torch.linalg.inv(self.Vf + self.Vi)
        Vtmp2 = self.Vf - self.Vi

        # Input layer S-matrix
        self.Sin.append(2 * torch.matmul(Vtmp1, self.Vi))  # Tf S11
        self.Sin.append(-torch.matmul(Vtmp1, Vtmp2))  # Rf S21
        self.Sin.append(torch.matmul(Vtmp1, Vtmp2))  # Rb S12
        self.Sin.append(2 * torch.matmul(Vtmp1, self.Vf))  # Tb S22

    if hasattr(self, "Sout"):
        # Output layer k-vectors and E to H transformation matrix
        Kz_norm_dn = torch.sqrt(
            self.eps_out * self.mu_out - self.Kx_norm_dn**2 - self.Ky_norm_dn**2
        )
        Kz_norm_dn = torch.where(
            torch.imag(Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn
        )
        tmp1 = torch.vstack(
            (
                torch.diag(-self.Ky_norm_dn * self.Kx_norm_dn / Kz_norm_dn),
                torch.diag(Kz_norm_dn + self.Kx_norm_dn**2 / Kz_norm_dn),
            )
        )
        tmp2 = torch.vstack(
            (
                torch.diag(-Kz_norm_dn - self.Ky_norm_dn**2 / Kz_norm_dn),
                torch.diag(self.Kx_norm_dn * self.Ky_norm_dn / Kz_norm_dn),
            )
        )
        self.Vo = torch.hstack((tmp1, tmp2))

        Vtmp1 = torch.linalg.inv(self.Vf + self.Vo)
        Vtmp2 = self.Vf - self.Vo

        # Output layer S-matrix
        self.Sout.append(2 * torch.matmul(Vtmp1, self.Vf))  # Tf S11
        self.Sout.append(torch.matmul(Vtmp1, Vtmp2))  # Rf S21
        self.Sout.append(-torch.matmul(Vtmp1, Vtmp2))  # Rb S12
        self.Sout.append(2 * torch.matmul(Vtmp1, self.Vo))  # Tb S22