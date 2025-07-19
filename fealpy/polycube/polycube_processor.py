from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, TetrahedronMesh

from .mesh_deformation import MeshNormalSmoothDeformation, MeshNormalAlignmentDeformation
from .polycube_segmentator import PolyCubeSegmentator


class PolyCubeProcessor:
    """

    """
    def __init__(self, origin_mesh):
        """
        Initialize the PolyCubeProcessor with the original mesh.

        Parameters
        ----------
        origin_mesh : TriangleMesh or TetrahedronMesh
            The original mesh to be processed.
        """

        self.mesh = origin_mesh
        self.segmentator:PolyCubeSegmentator = None

    def mesh_normal_smooth_deformation(self, sigma=0.1, s=7, alpha=0.5, max_epochs=100000, error_threshold=1e-3, weights=None):
        """
        Apply normal smooth deformation to the original mesh.

        Parameters
        ----------
        sigma : float
            The standard deviation for the Gaussian kernel.
        s : int
            The number of nearest neighbors to consider.
        alpha : float
            The weight for the smoothness term.
        max_epochs : int
            The maximum number of epochs for optimization.
        error_threshold : float
            The threshold for convergence.
        weights : dict
            Weights for different components in the deformation.

        Returns
        -------
        TriangleMesh or TetrahedronMesh
            The deformed mesh after normal smoothing.
        """

        normal_smooth_deformer = MeshNormalSmoothDeformation(self.mesh,
                                                              sigma=sigma, s=s, alpha=alpha,
                                                              max_epochs=max_epochs,
                                                              error_threshold=error_threshold,
                                                              weights=weights)
        self.mesh = normal_smooth_deformer.optimize()
        return self.mesh

    def polycube_segmentator(self,
                        straighten_max_iter=5,
                        merge_min_size=5,
                        laplacian_smooth_alpha=0.3,
                        laplacian_smooth_max_iter=5):
        """
        Segmentation of the polycube mesh.

        Parameters
            straighten_max_iter : int
                Maximum iterations for edge straightening.
            merge_min_size : int
                Minimum size for merging small charts.
            laplacian_smooth_alpha : float
                Alpha value for Laplacian smoothing.
            laplacian_smooth_max_iter : int
                Maximum iterations for Laplacian smoothing.

        Returns
            is_valid : bool
                Whether the resulting topology is valid.
        """
        polycube_segmentator = PolyCubeSegmentator(self.mesh)
        polycube_segmentator.build_candidate_charts()
        polycube_segmentator.extract_candidate_edges_vertices()
        polycube_segmentator.straighten_edges(max_iter=straighten_max_iter)
        polycube_segmentator.merge_small_charts(min_size=merge_min_size)
        is_valid = polycube_segmentator.validate_topology()
        polycube_segmentator.laplacian_smooth(alpha=laplacian_smooth_alpha, max_iter=laplacian_smooth_max_iter)
        polycube_segmentator.edge_projection()
        polycube_segmentator.detect_turning_points()
        self.segmentator = polycube_segmentator

        return is_valid

    def mesh_normal_alignment_deformation(self, gamma=1e3, s=6, alpha=0.5, max_epochs=100000, error_threshold=1e-3, weights=None):
        """
        Apply normal alignment deformation to the mesh.

        Parameters
            gamma : float
                The weight for the alignment term.
            s : int
                The number of nearest neighbors to consider.
            alpha : float
                The weight for the smoothness term.
            max_epochs : int
                The maximum number of epochs for optimization.
            error_threshold : float
                The threshold for convergence.
            weights : dict
                Weights for different components in the deformation.

        Returns
            TetrahedronMesh
                The deformed mesh after normal alignment.
        """

        normal_alignment_deformer = MeshNormalAlignmentDeformation(self.mesh,
                                                                   gamma=gamma, s=s, alpha=alpha,
                                                                   max_epochs=max_epochs,
                                                                   error_threshold=error_threshold,
                                                                   weights=weights)
        self.mesh = normal_alignment_deformer.optimize()
        return self.mesh
