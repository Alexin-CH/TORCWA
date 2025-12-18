import torch

def _RS_prod(self, Sm, Sn, Cm, Cn):
    # S11 = S[0] / S21 = S[1] / S12 = S[2] / S22 = S[3]
    # Cf = C[0] / Cb = C[1]

    tmp1 = torch.linalg.inv(
        torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
        - torch.matmul(Sm[2], Sn[1])
    )
    tmp2 = torch.linalg.inv(
        torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
        - torch.matmul(Sn[1], Sm[2])
    )

    # Layer S-matrix
    S11 = torch.matmul(Sn[0], torch.matmul(tmp1, Sm[0]))
    S21 = Sm[1] + torch.matmul(
        Sm[3], torch.matmul(tmp2, torch.matmul(Sn[1], Sm[0]))
    )
    S12 = Sn[2] + torch.matmul(
        Sn[0], torch.matmul(tmp1, torch.matmul(Sm[2], Sn[3]))
    )
    S22 = torch.matmul(Sm[3], torch.matmul(tmp2, Sn[3]))

    # Mode coupling coefficients
    C = [[], []]
    for m in range(len(Cm[0])):
        C[0].append(
            Cm[0][m]
            + torch.matmul(Cm[1][m], torch.matmul(tmp2, torch.matmul(Sn[1], Sm[0])))
        )
        C[1].append(torch.matmul(Cm[1][m], torch.matmul(tmp2, Sn[3])))

    for n in range(len(Cn[0])):
        C[0].append(torch.matmul(Cn[0][n], torch.matmul(tmp1, Sm[0])))
        C[1].append(
            Cn[1][n]
            + torch.matmul(Cn[0][n], torch.matmul(tmp1, torch.matmul(Sm[2], Sn[3])))
        )

    return [S11, S21, S12, S22], C