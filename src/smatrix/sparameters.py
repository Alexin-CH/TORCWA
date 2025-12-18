import torch

def _S_parameters(
    self,
    orders,
    *,
    direction="forward",
    port="transmission",
    polarization="xx",
    ref_order=[0, 0],
    power_norm=True,
    evanscent=1e-3
):
    """
    Return S-parameters.

    Parameters
    - orders: selected orders (Recommended shape: Nx2)

    - direction: set the direction of light propagation ('f', 'forward' / 'b', 'backward')
    - port: set the direction of light propagation ('t', 'transmission' / 'r', 'reflection')
    - polarization: set the input and output polarization of light ((output,input) xy-pol: 'xx' / 'yx' / 'xy' / 'yy' , ps-pol: 'pp' / 'sp' / 'ps' / 'ss' )
    - ref_order: set the reference for calculating S-parameters (Recommended shape: Nx2)
    - power_norm: if set as True, the absolute square of S-parameters are corresponds to the ratio of power
    - evanescent: Criteria for judging the evanescent field. If power_norm=True and real(kz_norm)/imag(kz_norm) < evanscent, function returns 0 (default = 1e-3)

    Return
    - S-parameters (torch.Tensor)
    """

    orders = torch.as_tensor(
        orders, dtype=torch.int64, device=self._device
    ).reshape([-1, 2])

    if direction in ["f", "forward"]:
        direction = "forward"
    elif direction in ["b", "backward"]:
        direction = "backward"
    else:
        raise ValueError(
            "Invalid propagation direction. Set as 'forward' or 'backward'."
        )

    if port in ["t", "transmission"]:
        port = "transmission"
    elif port in ["r", "reflection"]:
        port = "reflection"
    else:
        raise ValueError("Invalid port. Set as 'transmission' or 'reflection'.")

    if polarization not in ["xx", "yx", "xy", "yy", "pp", "sp", "ps", "ss"]:
        raise ValueError(
            "Invalid polarization. Choose one of 'xx','yx','xy','yy','pp','sp','ps','ss'."
        )

    ref_order = torch.as_tensor(
        ref_order, dtype=torch.int64, device=self._device
    ).reshape([1, 2])

    # Matching order indices
    order_indices = self._matching_indices(orders)
    ref_order_index = self._matching_indices(ref_order)

    if polarization in ["xx", "yx", "xy", "yy"]:
        # Matching order indices with polarization
        if polarization == "yx" or polarization == "yy":
            order_indices = order_indices + self.order_N
        if polarization == "xy" or polarization == "yy":
            ref_order_index = ref_order_index + self.order_N

        # power normalization factor
        if power_norm:
            Kz_norm_dn_in_complex = torch.sqrt(
                self.eps_in * self.mu_in - self.Kx_norm_dn**2 - self.Ky_norm_dn**2
            )
            is_evanescent_in = (
                torch.abs(
                    torch.real(Kz_norm_dn_in_complex)
                    / torch.imag(Kz_norm_dn_in_complex)
                )
                < evanscent
            )
            Kz_norm_dn_in = torch.where(
                is_evanescent_in,
                torch.real(torch.zeros_like(Kz_norm_dn_in_complex)),
                torch.real(Kz_norm_dn_in_complex),
            )
            Kz_norm_dn_in = torch.hstack((Kz_norm_dn_in, Kz_norm_dn_in))

            Kz_norm_dn_out_complex = torch.sqrt(
                self.eps_out * self.mu_out - self.Kx_norm_dn**2 - self.Ky_norm_dn**2
            )
            is_evanescent_out = (
                torch.abs(
                    torch.real(Kz_norm_dn_out_complex)
                    / torch.imag(Kz_norm_dn_out_complex)
                )
                < evanscent
            )
            Kz_norm_dn_out = torch.where(
                is_evanescent_out,
                torch.real(torch.zeros_like(Kz_norm_dn_out_complex)),
                torch.real(Kz_norm_dn_out_complex),
            )
            Kz_norm_dn_out = torch.hstack((Kz_norm_dn_out, Kz_norm_dn_out))

            Kx_norm_dn = torch.hstack(
                (torch.real(self.Kx_norm_dn), torch.real(self.Kx_norm_dn))
            )
            Ky_norm_dn = torch.hstack(
                (torch.real(self.Ky_norm_dn), torch.real(self.Ky_norm_dn))
            )

            if polarization == "xx":
                numerator_pol, denominator_pol = Kx_norm_dn, Kx_norm_dn
            elif polarization == "xy":
                numerator_pol, denominator_pol = Kx_norm_dn, Ky_norm_dn
            elif polarization == "yx":
                numerator_pol, denominator_pol = Ky_norm_dn, Kx_norm_dn
            elif polarization == "yy":
                numerator_pol, denominator_pol = Ky_norm_dn, Ky_norm_dn

            if direction == "forward" and port == "transmission":
                numerator_kz = Kz_norm_dn_out
                denominator_kz = Kz_norm_dn_in
            elif direction == "forward" and port == "reflection":
                numerator_kz = Kz_norm_dn_in
                denominator_kz = Kz_norm_dn_in
            elif direction == "backward" and port == "reflection":
                numerator_kz = Kz_norm_dn_out
                denominator_kz = Kz_norm_dn_out
            elif direction == "backward" and port == "transmission":
                numerator_kz = Kz_norm_dn_in
                denominator_kz = Kz_norm_dn_out

            normalization = torch.sqrt(
                (
                    1
                    + (numerator_pol[order_indices] / numerator_kz[order_indices])
                    ** 2
                )
                / (
                    1
                    + (
                        denominator_pol[ref_order_index]
                        / denominator_kz[ref_order_index]
                    )
                    ** 2
                )
            )
            normalization = normalization * torch.sqrt(
                numerator_kz[order_indices] / denominator_kz[ref_order_index]
            )
        else:
            normalization = 1.0

        # Get S-parameters
        if direction == "forward" and port == "transmission":
            S = self.S[0][order_indices, ref_order_index] * normalization
        elif direction == "forward" and port == "reflection":
            S = self.S[1][order_indices, ref_order_index] * normalization
        elif direction == "backward" and port == "reflection":
            S = self.S[2][order_indices, ref_order_index] * normalization
        elif direction == "backward" and port == "transmission":
            S = self.S[3][order_indices, ref_order_index] * normalization

        S = torch.where(torch.isinf(S), torch.zeros_like(S), S)
        S = torch.where(torch.isnan(S), torch.zeros_like(S), S)

        return S

    elif polarization in ["pp", "sp", "ps", "ss"]:
        if direction == "forward" and port == "transmission":
            idx = 0
            order_sign, ref_sign = 1, 1
            order_k0_norm2 = self.eps_out * self.mu_out
            ref_k0_norm2 = self.eps_in * self.mu_in
        elif direction == "forward" and port == "reflection":
            idx = 1
            order_sign, ref_sign = -1, 1
            order_k0_norm2 = self.eps_in * self.mu_in
            ref_k0_norm2 = self.eps_in * self.mu_in
        elif direction == "backward" and port == "reflection":
            idx = 2
            order_sign, ref_sign = 1, -1
            order_k0_norm2 = self.eps_out * self.mu_out
            ref_k0_norm2 = self.eps_out * self.mu_out
        elif direction == "backward" and port == "transmission":
            idx = 3
            order_sign, ref_sign = -1, -1
            order_k0_norm2 = self.eps_in * self.mu_in
            ref_k0_norm2 = self.eps_out * self.mu_out

        order_Kx_norm_dn = self.Kx_norm_dn[order_indices]
        order_Ky_norm_dn = self.Ky_norm_dn[order_indices]
        order_Kt_norm_dn = torch.sqrt(order_Kx_norm_dn**2 + order_Ky_norm_dn**2)
        order_Kz_norm_dn = order_sign * torch.abs(
            torch.real(
                torch.sqrt(
                    order_k0_norm2 - order_Kx_norm_dn**2 - order_Ky_norm_dn**2
                )
            )
        )
        order_Kz_norm_dn_complex = torch.sqrt(
            order_k0_norm2 - order_Kx_norm_dn**2 - order_Ky_norm_dn**2
        )
        order_is_evanescent = (
            torch.abs(
                torch.real(order_Kz_norm_dn_complex)
                / torch.imag(order_Kz_norm_dn_complex)
            )
            < evanscent
        )

        order_inc_angle = torch.atan2(
            torch.real(order_Kt_norm_dn), order_Kz_norm_dn
        )
        order_azi_angle = torch.atan2(
            torch.real(order_Ky_norm_dn), torch.real(order_Kx_norm_dn)
        )

        ref_Kx_norm_dn = self.Kx_norm_dn[ref_order_index]
        ref_Ky_norm_dn = self.Ky_norm_dn[ref_order_index]
        ref_Kt_norm_dn = torch.sqrt(ref_Kx_norm_dn**2 + ref_Ky_norm_dn**2)
        ref_Kz_norm_dn = ref_sign * torch.abs(
            torch.real(
                torch.sqrt(ref_k0_norm2 - ref_Kx_norm_dn**2 - ref_Ky_norm_dn**2)
            )
        )
        ref_Kz_norm_dn_complex = torch.sqrt(
            ref_k0_norm2 - ref_Kx_norm_dn**2 - ref_Ky_norm_dn**2
        )
        ref_is_evanescent = (
            torch.abs(
                torch.real(ref_Kz_norm_dn_complex)
                / torch.imag(ref_Kz_norm_dn_complex)
            )
            < evanscent
        )

        ref_inc_angle = torch.atan2(torch.real(ref_Kt_norm_dn), ref_Kz_norm_dn)
        ref_azi_angle = torch.atan2(
            torch.real(ref_Ky_norm_dn), torch.real(ref_Kx_norm_dn)
        )

        xx = self.S[idx][order_indices, ref_order_index]
        xy = self.S[idx][order_indices, ref_order_index + self.order_N]
        yx = self.S[idx][order_indices + self.order_N, ref_order_index]
        yy = self.S[idx][
            order_indices + self.order_N, ref_order_index + self.order_N
        ]

        xx = torch.where(order_is_evanescent, torch.zeros_like(xx), xx)
        xy = torch.where(order_is_evanescent, torch.zeros_like(xy), xy)
        yx = torch.where(order_is_evanescent, torch.zeros_like(yx), yx)
        yy = torch.where(order_is_evanescent, torch.zeros_like(yy), yy)

        if ref_is_evanescent:
            S = torch.zeros_like(xx)
            return S

        if polarization == "pp":
            S = (
                torch.cos(order_azi_angle)
                / torch.cos(order_inc_angle)
                * torch.cos(ref_inc_angle)
                * torch.cos(ref_azi_angle)
                * xx
                + torch.sin(order_azi_angle)
                / torch.cos(order_inc_angle)
                * torch.cos(ref_inc_angle)
                * torch.cos(ref_azi_angle)
                * yx
                + torch.cos(order_azi_angle)
                / torch.cos(order_inc_angle)
                * torch.cos(ref_inc_angle)
                * torch.sin(ref_azi_angle)
                * xy
                + torch.sin(order_azi_angle)
                / torch.cos(order_inc_angle)
                * torch.cos(ref_inc_angle)
                * torch.sin(ref_azi_angle)
                * yy
            )
        elif polarization == "ps":
            S = (
                torch.cos(order_azi_angle)
                / torch.cos(order_inc_angle)
                * (-1)
                * torch.sin(ref_azi_angle)
                * xx
                + torch.sin(order_azi_angle)
                / torch.cos(order_inc_angle)
                * (-1)
                * torch.sin(ref_azi_angle)
                * yx
                + torch.cos(order_azi_angle)
                / torch.cos(order_inc_angle)
                * torch.cos(ref_azi_angle)
                * xy
                + torch.sin(order_azi_angle)
                / torch.cos(order_inc_angle)
                * torch.cos(ref_azi_angle)
                * yy
            )
        elif polarization == "sp":
            S = (
                -torch.sin(order_azi_angle)
                * torch.cos(ref_inc_angle)
                * torch.cos(ref_azi_angle)
                * xx
                + torch.cos(order_azi_angle)
                * torch.cos(ref_inc_angle)
                * torch.cos(ref_azi_angle)
                * yx
                + -torch.sin(order_azi_angle)
                * torch.cos(ref_inc_angle)
                * torch.sin(ref_azi_angle)
                * xy
                + torch.cos(order_azi_angle)
                * torch.cos(ref_inc_angle)
                * torch.sin(ref_azi_angle)
                * yy
            )
        elif polarization == "ss":
            S = (
                -torch.sin(order_azi_angle) * (-1) * torch.sin(ref_azi_angle) * xx
                + torch.cos(order_azi_angle) * (-1) * torch.sin(ref_azi_angle) * yx
                + -torch.sin(order_azi_angle) * torch.cos(ref_azi_angle) * xy
                + torch.cos(order_azi_angle) * torch.cos(ref_azi_angle) * yy
            )

        if power_norm:
            Kz_norm_dn_in_complex = torch.sqrt(
                self.eps_in * self.mu_in - self.Kx_norm_dn**2 - self.Ky_norm_dn**2
            )
            is_evanescent_in = (
                torch.abs(
                    torch.real(Kz_norm_dn_in_complex)
                    / torch.imag(Kz_norm_dn_in_complex)
                )
                < evanscent
            )
            Kz_norm_dn_in = torch.where(
                is_evanescent_in,
                torch.real(torch.zeros_like(Kz_norm_dn_in_complex)),
                torch.real(Kz_norm_dn_in_complex),
            )
            Kz_norm_dn_in = torch.hstack((Kz_norm_dn_in, Kz_norm_dn_in))

            Kz_norm_dn_out_complex = torch.sqrt(
                self.eps_out * self.mu_out - self.Kx_norm_dn**2 - self.Ky_norm_dn**2
            )
            is_evanescent_out = (
                torch.abs(
                    torch.real(Kz_norm_dn_out_complex)
                    / torch.imag(Kz_norm_dn_out_complex)
                )
                < evanscent
            )
            Kz_norm_dn_out = torch.where(
                is_evanescent_out,
                torch.abs(torch.real(Kz_norm_dn_out_complex)),
                torch.real(Kz_norm_dn_out_complex),
            )
            Kz_norm_dn_out = torch.hstack((Kz_norm_dn_out, Kz_norm_dn_out))

            Kx_norm_dn = torch.hstack(
                (torch.real(self.Kx_norm_dn), torch.real(self.Kx_norm_dn))
            )
            Ky_norm_dn = torch.hstack(
                (torch.real(self.Ky_norm_dn), torch.real(self.Ky_norm_dn))
            )

            if direction == "forward" and port == "transmission":
                numerator_kz = Kz_norm_dn_out
                denominator_kz = Kz_norm_dn_in
            elif direction == "forward" and port == "reflection":
                numerator_kz = Kz_norm_dn_in
                denominator_kz = Kz_norm_dn_in
            elif direction == "backward" and port == "reflection":
                numerator_kz = Kz_norm_dn_out
                denominator_kz = Kz_norm_dn_out
            elif direction == "backward" and port == "transmission":
                numerator_kz = Kz_norm_dn_in
                denominator_kz = Kz_norm_dn_out

            normalization = torch.sqrt(
                numerator_kz[order_indices] / denominator_kz[ref_order_index]
            )
        else:
            normalization = 1.0

        S = torch.where(torch.isinf(S), torch.zeros_like(S), S)
        S = torch.where(torch.isnan(S), torch.zeros_like(S), S)

        return S * normalization

    else:
        return None
