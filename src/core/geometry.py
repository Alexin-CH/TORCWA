import torch
import torch.fft


def _circle_level(x_grid, y_grid, R, Cx, Cy):
    return 1.0 - torch.sqrt(
        ((x_grid - Cx) / R) ** 2 + ((y_grid - Cy) / R) ** 2
    )


def _ellipse_level(x_grid, y_grid, Rx, Ry, Cx, Cy, theta):
    return 1.0 - torch.sqrt(
        (((x_grid - Cx) * torch.cos(theta) + (y_grid - Cy) * torch.sin(theta)) / Rx)
        ** 2
        + (
            (-(x_grid - Cx) * torch.sin(theta) + (y_grid - Cy) * torch.cos(theta))
            / Ry
        )
        ** 2
    )


def _square_level(x_grid, y_grid, W, Cx, Cy, theta):
    return 1.0 - torch.maximum(
        torch.abs(
            ((x_grid - Cx) * torch.cos(theta) + (y_grid - Cy) * torch.sin(theta))
            / (W / 2.0)
        ),
        torch.abs(
            (-(x_grid - Cx) * torch.sin(theta) + (y_grid - Cy) * torch.cos(theta))
            / (W / 2.0)
        ),
    )


def _rectangle_level(x_grid, y_grid, Wx, Wy, Cx, Cy, theta):
    return 1.0 - torch.maximum(
        torch.abs(
            ((x_grid - Cx) * torch.cos(theta) + (y_grid - Cy) * torch.sin(theta))
            / (Wx / 2.0)
        ),
        torch.abs(
            (-(x_grid - Cx) * torch.sin(theta) + (y_grid - Cy) * torch.cos(theta))
            / (Wy / 2.0)
        ),
    )


def _rhombus_level(x_grid, y_grid, Wx, Wy, Cx, Cy, theta):
    return 1.0 - (
        torch.abs(
            ((x_grid - Cx) * torch.cos(theta) + (y_grid - Cy) * torch.sin(theta))
            / (Wx / 2.0)
        )
        + torch.abs(
            (-(x_grid - Cx) * torch.sin(theta) + (y_grid - Cy) * torch.cos(theta))
            / (Wy / 2.0)
        )
    )


def _super_ellipse_level(x_grid, y_grid, Wx, Wy, Cx, Cy, theta, power):
    return 1.0 - (
        torch.abs(
            ((x_grid - Cx) * torch.cos(theta) + (y_grid - Cy) * torch.sin(theta))
            / (Wx / 2.0)
        )
        ** power
        + torch.abs(
            (-(x_grid - Cx) * torch.sin(theta) + (y_grid - Cy) * torch.cos(theta))
            / (Wy / 2.0)
        )
        ** power
    ) ** (1 / power)


def _level_sigmoid(edge_sharpness, level):
    return torch.sigmoid(edge_sharpness * level)


def _union(A, B):
    return torch.maximum(A, B)


def _intersection(A, B):
    return torch.minimum(A, B)


def _difference(A, B):
    return torch.minimum(A, 1.0 - B)


class geometry:
    def __init__(
        self,
        Lx: float = 1.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 100,
        edge_sharpness: float = 1000.0,
        *,
        dtype=torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Geometry

        Parameters
        - Lx: x-direction Lattice constant (float)
        - Ly: y-direction Lattice constant (float)
        - x: x-axis sampling number (int)
        - y: y-axis sampling number (int)
        - edge_sharpness: sharpness of edge (float)

        Keyword Parameters
        - dtype: geometry data type (only torch.complex64 and torch.complex128 are allowed.)
        - device: geometry device (only torch.device('cpu') and torch.device('cuda') are allowed.)
        """
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.edge_sharpness = edge_sharpness

        self.dtype = dtype
        self.device = device

    def grid(self):
        """
        Update grid
        """

        self.x = (self.Lx / self.nx) * (
            torch.arange(self.nx, dtype=self.dtype, device=self.device) + 0.5
        )
        self.y = (self.Ly / self.ny) * (
            torch.arange(self.ny, dtype=self.dtype, device=self.device) + 0.5
        )
        self.x_grid, self.y_grid = torch.meshgrid(
            self.x, self.y, indexing="ij"
        )

    def circle(self, R, Cx, Cy):
        """
        R: radius
        Cx: x center
        Cy: y center
        """

        self.grid()
        return _level_sigmoid(
            self.edge_sharpness,
            _circle_level(self.x_grid, self.y_grid, R, Cx, Cy),
        )

    def ellipse(self, Rx, Ry, Cx, Cy, theta=0.0):
        """
        Rx: x direction radius
        Ry: y direction radius
        Cx: x center
        Cy: y center
        """

        theta = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        self.grid()
        return _level_sigmoid(
            self.edge_sharpness,
            _ellipse_level(self.x_grid, self.y_grid, Rx, Ry, Cx, Cy, theta),
        )

    def square(self, W, Cx, Cy, theta=0.0):
        """
        W: width
        Cx: x center
        Cy: y center
        theta: rotation angle / center: [Cx, Cy] / axis: z-axis
        """

        theta = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        self.grid()
        return _level_sigmoid(
            self.edge_sharpness,
            _square_level(self.x_grid, self.y_grid, W, Cx, Cy, theta),
        )

    def rectangle(self, Wx, Wy, Cx, Cy, theta=0.0):
        """
        Wx: x width
        Wy: y width
        Cx: x center
        Cy: y center
        theta: rotation angle / center: [Cx, Cy] / axis: z-axis
        """

        theta = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        self.grid()
        return _level_sigmoid(
            self.edge_sharpness,
            _rectangle_level(self.x_grid, self.y_grid, Wx, Wy, Cx, Cy, theta),
        )

    def rhombus(self, Wx, Wy, Cx, Cy, theta=0.0):
        """
        Wx: x diagonal
        Wy: y diagonal
        Cx: x center
        Cy: y center
        theta: rotation angle / center: [Cx, Cy] / axis: z-axis
        """

        theta = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        self.grid()
        return _level_sigmoid(
            self.edge_sharpness,
            _rhombus_level(self.x_grid, self.y_grid, Wx, Wy, Cx, Cy, theta),
        )

    def super_ellipse(self, Wx, Wy, Cx, Cy, theta=0.0, power=2.0):
        """
        Wx: x width
        Wy: y width
        Cx: x center
        Cy: y center
        theta: rotation angle / center: [Cx, Cy] / axis: z-axis
        power: elliptic power
        """

        theta = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        self.grid()
        return _level_sigmoid(
            self.edge_sharpness,
            _super_ellipse_level(
                self.x_grid, self.y_grid, Wx, Wy, Cx, Cy, theta, power
            ),
        )

    def union(self, A, B):
        """
        A U B
        """

        return _union(A, B)

    def intersection(self, A, B):
        """
        A n B
        """

        return _intersection(A, B)

    def difference(self, A, B):
        """
        A - B = A n Bc
        """

        return _difference(A, B)


class rcwa_geo:
    edge_sharpness = 100.0  # sharpness of edge
    Lx = 1.0  # x-direction Lattice constant
    Ly = 1.0  # y-direction Lattice constant
    nx = 100  # x-axis sampling number
    ny = 100  # y-axis sampling number
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        pass

    @classmethod
    def grid(cls):
        """
        Update grid
        """

        cls.x = (cls.Lx / cls.nx) * (
            torch.arange(cls.nx, dtype=cls.dtype, device=cls.device) + 0.5
        )
        cls.y = (cls.Ly / cls.ny) * (
            torch.arange(cls.ny, dtype=cls.dtype, device=cls.device) + 0.5
        )
        cls.x_grid, cls.y_grid = torch.meshgrid(cls.x, cls.y, indexing="ij")

    @classmethod
    def circle(cls, R, Cx, Cy):
        """
        R: radius
        Cx: x center
        Cy: y center
        """

        cls.grid()
        return _level_sigmoid(
            cls.edge_sharpness,
            _circle_level(cls.x_grid, cls.y_grid, R, Cx, Cy),
        )

    @classmethod
    def ellipse(cls, Rx, Ry, Cx, Cy, theta=0.0):
        """
        Rx: x direction radius
        Ry: y direction radius
        Cx: x center
        Cy: y center
        """

        theta = torch.as_tensor(theta, dtype=cls.dtype, device=cls.device)
        cls.grid()
        return _level_sigmoid(
            cls.edge_sharpness,
            _ellipse_level(cls.x_grid, cls.y_grid, Rx, Ry, Cx, Cy, theta),
        )

    @classmethod
    def square(cls, W, Cx, Cy, theta=0.0):
        """
        W: width
        Cx: x center
        Cy: y center
        theta: rotation angle / center: [Cx, Cy] / axis: z-axis
        """

        theta = torch.as_tensor(theta, dtype=cls.dtype, device=cls.device)
        cls.grid()
        return _level_sigmoid(
            cls.edge_sharpness,
            _square_level(cls.x_grid, cls.y_grid, W, Cx, Cy, theta),
        )

    @classmethod
    def rectangle(cls, Wx, Wy, Cx, Cy, theta=0.0):
        """
        Wx: x width
        Wy: y width
        Cx: x center
        Cy: y center
        theta: rotation angle / center: [Cx, Cy] / axis: z-axis
        """

        theta = torch.as_tensor(theta, dtype=cls.dtype, device=cls.device)
        cls.grid()
        return _level_sigmoid(
            cls.edge_sharpness,
            _rectangle_level(cls.x_grid, cls.y_grid, Wx, Wy, Cx, Cy, theta),
        )

    @classmethod
    def rhombus(cls, Wx, Wy, Cx, Cy, theta=0.0):
        """
        Wx: x diagonal
        Wy: y diagonal
        Cx: x center
        Cy: y center
        theta: rotation angle / center: [Cx, Cy] / axis: z-axis
        """

        theta = torch.as_tensor(theta, dtype=cls.dtype, device=cls.device)
        cls.grid()
        return _level_sigmoid(
            cls.edge_sharpness,
            _rhombus_level(cls.x_grid, cls.y_grid, Wx, Wy, Cx, Cy, theta),
        )

    @classmethod
    def super_ellipse(cls, Wx, Wy, Cx, Cy, theta=0.0, power=2.0):
        """
        Wx: x width
        Wy: y width
        Cx: x center
        Cy: y center
        theta: rotation angle / center: [Cx, Cy] / axis: z-axis
        power: elliptic power
        """

        theta = torch.as_tensor(theta, dtype=cls.dtype, device=cls.device)
        cls.grid()
        return _level_sigmoid(
            cls.edge_sharpness,
            _super_ellipse_level(
                cls.x_grid, cls.y_grid, Wx, Wy, Cx, Cy, theta, power
            ),
        )

    @classmethod
    def union(cls, A, B):
        """
        A U B
        """

        return _union(A, B)

    @classmethod
    def intersection(cls, A, B):
        """
        A n B
        """

        return _intersection(A, B)

    @classmethod
    def difference(cls, A, B):
        """
        A - B = A n Bc
        """

        return _difference(A, B)
