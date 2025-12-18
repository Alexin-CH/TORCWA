import torch

def _matching_indices(self, orders):
    orders[orders[:, 0] < -self.order[0], 0] = int(-self.order[0])
    orders[orders[:, 0] > self.order[0], 0] = int(self.order[0])
    orders[orders[:, 1] < -self.order[1], 1] = int(-self.order[1])
    orders[orders[:, 1] > self.order[1], 1] = int(self.order[1])
    order_indices = (
        len(self.order_y) * (orders[:, 0] + int(self.order[0]))
        + orders[:, 1]
        + int(self.order[1])
    )
    return order_indices

def _diffraction_angle(self, orders, *, layer="output", unit="radian"):
    """
    Diffraction angles for the selected orders

    Parameters
    - orders: selected diffraction orders (Recommended shape: Nx2)
    - layer: selected layer ('i', 'in', 'input' / 'o', 'out', 'output')
    - unit: unit of the output angles ('r', 'rad', 'radian' / 'd', 'deg', 'degree')

    Return
    - inclination angle (torch.Tensor), azimuthal angle (torch.Tensor)
    """

    orders = torch.as_tensor(
        orders, dtype=torch.int64, device=self._device
    ).reshape([-1, 2])

    if layer in ["i", "in", "input"]:
        layer = "input"
    elif layer in ["o", "out", "output"]:
        layer = "output"
    else:
        raise ValueError("Invalid layer selected")

    if unit in ["r", "rad", "radian"]:
        unit = "radian"
    elif unit in ["d", "deg", "degree"]:
        unit = "degree"
    else:
        raise ValueError("Invalid unit. Set as 'radian' or 'degree'.")

    # Matching indices
    order_indices = self._matching_indices(orders)

    eps = self.eps_in if layer == "input" else self.eps_out
    mu = self.mu_in if layer == "input" else self.mu_out

    kx_norm = self.Kx_norm_dn[order_indices]
    ky_norm = self.Ky_norm_dn[order_indices]
    Kt_norm_dn = torch.sqrt(kx_norm**2 + ky_norm**2)
    kz_norm = torch.sqrt(eps * mu - kx_norm**2 - ky_norm**2)
    inc_angle = torch.atan2(torch.real(Kt_norm_dn), torch.real(kz_norm))
    azi_angle = torch.atan2(torch.real(ky_norm), torch.real(kx_norm))

    if unit == "degree":
        inc_angle = (180.0 / pi) * inc_angle
        azi_angle = (180.0 / pi) * azi_angle

    return inc_angle, azi_angle

def _set_incident_angle(self, inc_ang, azi_ang, angle_layer="input"):
    """
    Set incident angle

    Parameters
    - inc_ang: incident angle (unit: radian)
    - azi_ang: azimuthal angle (unit: radian)
    - angle_layer: reference layer to calculate angle ('i', 'in', 'input' / 'o', 'out', 'output')
    """

    self.inc_ang = torch.as_tensor(inc_ang, dtype=self._dtype, device=self._device)
    self.azi_ang = torch.as_tensor(azi_ang, dtype=self._dtype, device=self._device)

    if angle_layer in ["i", "in", "input"]:
        self.angle_layer = "input"
    elif angle_layer in ["o", "out", "output"]:
        self.angle_layer = "output"
    else:
        raise ValueError("Invalid angle layer")

    self._kvectors()