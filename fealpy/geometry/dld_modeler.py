
import matplotlib.pyplot as plt
from typing import Any

from fealpy.backend import bm
from fealpy.typing import TensorLike
from fealpy.decorator import variantmethod


from typing import Any, Dict, List, Tuple, Optional
import math
import numpy as np


class DLDModeler:
    """
    A parametric geometry builder for Deterministic Lateral Displacement (DLD) microfluidic chips.

    This class generates a 2D/3D layout of a DLD device, including inlet/outlet straight
    sections and a central pillar lattice assembled from multiple *stages* (tiles).
    Each stage is a rectangular pillar array that may be rotated by a per-stage angle
    and placed with a configurable gap along the main flow direction. The lattice
    inside a stage supports the canonical DLD row-shift scheme.

    Parameters:
        options (Dict[str, Any]):
            Configuration dictionary controlling chip scale sizes, stage assembly,
            and the lattice pattern. See :meth:`get_options` for the full list of
            keys and default values. Missing keys are filled with defaults.

    Attributes:
        options (Dict[str, Any]):
            A validated copy of the input options after applying defaults and
            normalizations (e.g., expanding scalar angle into a per-stage list).

        circles_ (TensorLike | None):
            Cached pillar array of shape ``(NC, 3)`` where each row is ``[x, y, r]``.
            Populated after calling :meth:`build`.

        boundary_ (np.ndarray | None):
            Outer boundary polygon of shape ``(NB, 2)`` in counter-clockwise order.
            Populated after calling :meth:`build`.

    Methods:
        get_options() -> Dict[str, Any]:
            Return a dictionary of default options for the model.

        build(gmsh: Any | None = None) -> Tuple[np.ndarray, np.ndarray]:
            Construct the geometry from :attr:`options` and return ``(circles, boundary)``.
    """

    @staticmethod
    def get_options() -> Dict[str, Any]:
        """
        Return the default option dictionary for the DLD geometry.

        Geometry (chip scale)
        ---------------------
        chip_height (float):
            Total channel height used to size/validate the lattice (unit-consistent, e.g., micrometers).

        inlet_length (float):
            Straight duct length before the first stage.

        outlet_length (float):
            Straight duct length after the last stage.

        wall_margin_x (float):
            Extra horizontal margin added on the outlet side when building the boundary.

        wall_margin_y (float):
            Extra vertical margin added above the top envelope and below the bottom envelope.

        Stage assembly
        --------------
        n_stages (int):
            Number of pillar stages (tiles).

        stage_length (float):
            Stage length in local x **before rotation**.

        stage_height (float):
            Stage height in local y **before rotation**.

        stage_gap (float):
            Gap in x between successive stages (center-to-center offset is ``stage_length + stage_gap``).

        angles (float | List[float]):
            Rotation angle(s) in degrees for each stage. If a scalar is provided,
            it will be expanded to all stages; see ``alternate_sign``.

        alternate_sign (bool):
            When ``angles`` is scalar and this flag is ``True``, the sign is
            alternated as ``+a, -a, +a, -a, ...`` across stages.

        Pillar lattice inside each stage (local frame, before rotation)
        ----------------------------------------------------------------
        n_rows (int):
            Number of lattice rows in local y within a stage.

        n_cols (int):
            Number of lattice columns in local x within a stage.

        pitch_x (float):
            Lattice spacing in local x between adjacent columns.

        pitch_y (float):
            Lattice spacing in local y between adjacent rows.

        pillar_diam (float):
            Pillar diameter (radius is ``pillar_diam / 2``).

        row_shift_period (int):
            DLD row-shift period ``P`` (typically ``P >= 2``).

        row_shift_dx (float):
            Per-row x-shift applied as ``(row_index % P) * row_shift_dx``.
            For classical DLD, ``row_shift_dx = pitch_x / P``.

        Lattice origin and centering
        ----------------------------
        stage_origin (Tuple[float, float]):
            Local origin offset ``(ox, oy)`` for the lattice inside a stage.

        center_stage (bool):
            If ``True``, center the lattice within the stage rectangle prior to rotation.

        Returns:
            Dict[str, Any]: A dictionary containing all default options.
        """
        return dict(
            # chip-scale
            chip_height=800.0,
            inlet_length=1200.0,
            outlet_length=1200.0,
            wall_margin_x=100.0,
            wall_margin_y=80.0,

            # stage assembly
            n_stages=7,
            stage_length=900.0,
            stage_height=600.0,
            stage_gap=40.0,
            angles=12.0,          # deg; can be a list like [10,10,10,...]
            alternate_sign=True,

            # lattice inside a stage (local before rotation)
            n_rows=9,
            n_cols=18,
            pitch_x=45.0,
            pitch_y=45.0,
            pillar_diam=24.0,
            row_shift_period=4,
            row_shift_dx=45.0/4.0,

            # centering
            stage_origin=(0.0, 0.0),
            center_stage=True,
        )

    def __init__(self, options: Dict[str, Any]):
        """
        Initialize the modeler and normalize options.

        Parameters:
            options (Dict[str, Any]):
                User-specified options. Any missing keys will be populated from
                :meth:`get_options`. If ``angles`` is a scalar and ``alternate_sign``
                is ``True``, it is expanded to ``[+a, -a, +a, ...]`` with length ``n_stages``.
        """
        defaults = self.get_options()
        cfg = {**defaults, **(options or {})}

        angles = cfg["angles"]
        if isinstance(angles, (int, float)):
            a = float(angles)
            if cfg["alternate_sign"]:
                seq = [a if (k % 2 == 0) else -a for k in range(cfg["n_stages"])]
            else:
                seq = [a] * cfg["n_stages"]
            cfg["angles"] = seq
        else:
            if len(angles) != cfg["n_stages"]:
                raise ValueError("`angles` list length must equal `n_stages`.")

        self.options: Dict[str, Any] = cfg
        self.circles_: Optional[np.ndarray] = None
        self.boundary_: Optional[np.ndarray] = None

    def __str__(self) -> str:
        """
        Return a concise string summary of the current configuration.

        Returns:
            str: One-line summary including stage count, stage size, lattice pitch,
                 lattice size, the first stage angle, and pillar diameter.
        """
        o = self.options
        return (
            "DLDModeler("
            f"stages={o['n_stages']}, stage(LxH)={o['stage_length']}x{o['stage_height']}, "
            f"pitch=({o['pitch_x']},{o['pitch_y']}), rows/cols={o['n_rows']}/{o['n_cols']}, "
            f"angle0={o['angles'][0]}deg, pillar_d={o['pillar_diam']})"
        )

    def build(self, gmsh: Any | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the DLD geometry and return pillar circles and the outer boundary.

        The construction proceeds in two steps. First, for each stage, a local lattice
        of size ``(n_rows, n_cols)`` is created in the stage frame, a per-row DLD
        shift is applied, and the lattice is optionally centered within the stage
        rectangle. Second, the lattice is rotated by the per-stage angle and translated
        to its global position along the main flow direction. A simplified outer
        boundary polygon is then assembled by padding the top/bottom envelopes with
        the specified wall margins and extending inlet/outlet sections.

        Parameters:
            gmsh (Any | None, optional):
                A Gmsh modeling engine instance (e.g., ``gmsh.model.occ`` wrapper).
                In this initial version the parameter is unused; geometry is returned
                as arrays. It is reserved for future extensions to emit Gmsh entities.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - ``circles``: ``(NC, 3)`` array where each row is ``[x, y, r]``.
                - ``boundary``: ``(NB, 2)`` polygon vertices in counter-clockwise order.

        Notes:
            - All coordinates use consistent length units (e.g., micrometers).
            - The boundary is a simplified “belt” around the rotated stages; if a
              sawtooth envelope is required, connect the rotated rectangle edges
              stage-by-stage to form upper and lower polylines before padding.
        """
        o = self.options

        # --- precompute stage centers and angles (in radians)
        n_stages = o["n_stages"]
        Ls, Hs = o["stage_length"], o["stage_height"]
        G = o["stage_gap"]
        angle_list = [math.radians(a) for a in o["angles"]]

        x0 = o["inlet_length"] + Ls / 2.0
        stage_centers = [(x0 + i * (Ls + G), 0.0) for i in range(n_stages)]

        # --- local lattice (before rotation)
        nr, nc = o["n_rows"], o["n_cols"]
        px, py = o["pitch_x"], o["pitch_y"]
        P = max(1, int(o["row_shift_period"]))
        dx = float(o["row_shift_dx"])
        r = 0.5 * o["pillar_diam"]

        yy = np.arange(nr) * py
        xx = np.arange(nc) * px
        X0, Y0 = np.meshgrid(xx, yy)  # (nr, nc)

        # per-row DLD shift
        shifts = (np.arange(nr) % P) * dx
        X0 = X0 + shifts[:, None]

        # centering/origin in stage box
        if o["center_stage"]:
            bbox_w = (nc - 1) * px + (P - 1) * dx
            bbox_h = (nr - 1) * py
            ox = (Ls - bbox_w) / 2.0 + o["stage_origin"][0]
            oy = (Hs - bbox_h) / 2.0 + o["stage_origin"][1]
        else:
            ox, oy = o["stage_origin"]

        X0 = X0 + ox
        Y0 = Y0 + oy
        local_pts = np.c_[X0.ravel(), Y0.ravel()]  # (nr*nc, 2)

        # --- accumulate pillars across stages
        circle_list: List[np.ndarray] = []

        for k in range(n_stages):
            cx, cy = stage_centers[k]
            ang = angle_list[k]
            ca, sa = math.cos(ang), math.sin(ang)

            # rotate around (Ls/2, Hs/2) then translate to stage center along x
            lc = local_pts - np.array([Ls / 2.0, Hs / 2.0])
            R = np.array([[ca, -sa], [sa, ca]])
            rot = lc @ R.T
            glob = rot + np.array([cx, cy])

            circles_k = np.c_[glob, np.full((glob.shape[0], 1), r)]
            circle_list.append(circles_k)

        circles = np.vstack(circle_list)

        # --- simplified outer boundary using top/bottom envelopes + margins
        def rect_corners(center: Tuple[float, float], L: float, H: float, ang_rad: float) -> np.ndarray:
            """
            Compute the four rotated corner points of a rectangle.

            Parameters:
                center (Tuple[float, float]): Rectangle center ``(cx, cy)``.
                L (float): Rectangle length along local x.
                H (float): Rectangle height along local y.
                ang_rad (float): Rotation angle in radians.

            Returns:
                np.ndarray: Array of shape ``(4, 2)`` listing the corner coordinates.
            """
            cx, cy = center
            ca, sa = math.cos(ang_rad), math.sin(ang_rad)
            dx = L / 2.0
            dy = H / 2.0
            rel = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
            R = np.array([[ca, -sa], [sa, ca]])
            return rel @ R.T + np.array([cx, cy])

        top_pts, bot_pts = [], []
        for k in range(n_stages):
            pts = rect_corners(stage_centers[k], Ls, Hs, angle_list[k])
            top_pts.append(pts[np.argmax(pts[:, 1])])
            bot_pts.append(pts[np.argmin(pts[:, 1])])

        top_poly = np.array(top_pts)
        bot_poly = np.array(bot_pts)

        x_min = o["inlet_length"]
        x_max = stage_centers[-1][0] + Ls / 2.0 + o["outlet_length"]

        y_top = float(np.max(top_poly[:, 1]) + o["wall_margin_y"])
        y_bot = float(np.min(bot_poly[:, 1]) - o["wall_margin_y"])

        boundary = np.array([
            [x_min - o["inlet_length"], y_bot],
            [x_min, y_bot],
            [x_max, y_bot],
            [x_max + o["wall_margin_x"], y_bot],
            [x_max + o["wall_margin_x"], y_top],
            [x_min - o["inlet_length"], y_top],
        ], dtype=float)

        self.circles_ = circles
        self.boundary_ = boundary
        return circles, boundary

    def add_plot(self, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """
        Plot the DLD geometry (boundary polygon and pillar circles) using Matplotlib.

        Parameters:
            ax (matplotlib.axes.Axes, optional):
                An existing axes object to plot on. If None, a new figure and
                axes are created.
            
            **kwargs:
                Additional keyword arguments passed to the circle patch
                (e.g., ``facecolor``, ``edgecolor``, ``alpha``).

        Returns:
            matplotlib.axes.Axes:
                The axes containing the plotted geometry.

        Notes:
            - Requires that :meth:`build` has already been called so that
              :attr:`circles_` and :attr:`boundary_` are populated.
            - The boundary is drawn as a closed polygon with a blue edge,
              and circles are drawn with white fill and black edge by default.
        """
        if self.circles_ is None or self.boundary_ is None:
            raise RuntimeError("Geometry not built yet. Call `build()` first.")

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # plot boundary
        bnd = self.boundary_
        ax.plot(
            np.append(bnd[:, 0], bnd[0, 0]),
            np.append(bnd[:, 1], bnd[0, 1]),
            color="blue",
            linewidth=1.5,
        )

        # plot circles
        from matplotlib.patches import Circle
        for (x, y, r) in self.circles_:
            circ = Circle(
                (x, y),
                r,
                facecolor=kwargs.get("facecolor", "white"),
                edgecolor=kwargs.get("edgecolor", "black"),
                linewidth=0.5,
                alpha=kwargs.get("alpha", 1.0),
            )
            ax.add_patch(circ)

        ax.set_aspect("equal", "box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("DLD Geometry")
        return ax


if __name__ == "__main__":
    opts = DLDModeler.get_options()
    opts.update(dict(n_cols=22, angles=10.0, alternate_sign=True))
    modeler = DLDModeler(opts)
    circles, boundary = modeler.build()
    ax = modeler.add_plot()
    plt.show()

    print(modeler)
    print("circles:", circles.shape, " boundary:", boundary.shape)

