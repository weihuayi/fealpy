from ..backend import backend_manager as bm
from ..mesh import (TriangleMesh, QuadrangleMesh, TetrahedronMesh, HexahedronMesh, 
                        LagrangeTriangleMesh, LagrangeQuadrangleMesh)


class ViZiRPlotter:
    def __init__(self,mesh, 
                 solution = None,
                 sol_p = None,
                 sol_num = 1,
                 is_scalar = True,
                 filename='vizirploter', 
                 version=3,
                 colormap='jet',
                 linewidth=0.2,
                 output_dir=None):
        """
        Initializes the ViZiRPloter class for visualizing meshes and solutions.
        Parameters:
            mesh : Mesh object or list of Mesh objects
                The mesh or list of meshes to visualize.
                spotify the mesh type, e.g., 
                TriangleMesh, QuadrangleMesh, 
                TetrahedronMesh, HexahedronMesh,
                LagrangeTriangleMesh, LagrangeQuadrangleMesh

            solution : array-like or list of array-like, optional
                The solution or list of solutions to visualize.

            sol_p : int or list of int, optional
                The polynomial order of the solution. 
            sol_num : int, optional
                The number of components in the solution.
            is_scalar : bool, optional
                Whether the solution is scalar or vector.
            filename : str, optional
                The base filename for the output files.
            version : int, optional
                The version number.
            colormap : str, optional
                The colormap to use for visualization.
            linewidth : float, optional
            output_dir : str, optional
                The directory to save the output files.
        """
        self.mesh = mesh if isinstance(mesh, list) else [mesh]
        self.solution = solution if isinstance(solution, list) else [solution]
        self.sol_p = sol_p if isinstance(sol_p, list) else [sol_p]
        self.filename = filename
        self.version = version
        self.movie_list = []
        self.output_dir = output_dir

        # make sure the output directory exists
        self._ensure_output_dir()

        self.colormap = colormap
        self.linewidth = linewidth

        self.sol_num = sol_num
        self.is_scalar = is_scalar
        
        self.mesh_properties = [self._judge_mesh_type(m) for m in self.mesh]

        # initialize mesh and solution hash values
        self._mesh_hash = self._compute_hash(self.mesh)
        self._solution_hash = self._compute_hash(self.solution)

        # initialize version counter
        self._version_counter = 0
        # installation check
        self.is_install , t = self._check_vizir4_installation()
        print(t)
        if not self.is_install:
            self._tips()
        self._state_setup()

    def _ensure_output_dir(self):
        """
        Ensure the output directory exists.
        """
        import os
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def _get_file_path(self, filename, ext=None):
        """
        Get the full file path for the given filename and extension.

        Parameters:
            filename : str
            ext : str, optional
                file extension
            
            Returns:
                str : the full file path
        """
        if self.output_dir is None:
            return f"{filename}{ext if ext else ''}"
        else:
            import os
            return os.path.join(self.output_dir, f"{filename}{ext if ext else ''}")
    
    def _compute_hash(self, obj):
        """
        Compute a hash value for the given object.
        """
        import hashlib
        obj_str = str(obj).encode('utf-8')
        return hashlib.md5(obj_str).hexdigest()

    def _has_changed(self):
        """
        Check if the mesh or solution has changed.
        """
        current_mesh_hash = self._compute_hash(self.mesh)
        current_solution_hash = self._compute_hash(self.solution)
        if current_mesh_hash != self._mesh_hash or current_solution_hash != self._solution_hash:
            # update the internal state
            self._mesh_hash = current_mesh_hash
            self._solution_hash = current_solution_hash
            return True
        return False
    
    def _tips(self):
        """
        Prompt the user to install Vizir4 software.
        """
        print("="*80)
        print("Warning: Vizir4 installation not detected.")
        print("Vizir4 is a powerful tool for visualizing meshes and solutions, essential for analyzing computational results.")
        print("\nInstallation Instructions:")
        print("   - Visit https://pyamg.saclay.inria.fr/vizir4.html#download to download the Vizir4 package.")
        print("\nAfter installation, ensure that Vizir4 is added to the system PATH environment variable.")
        print("If you encounter any issues, please visit the official website: https://pyamg.saclay.inria.fr/vizir4.html")
        print("\nNote: Without installing Vizir4, visualization features will not be available, but other computational functionalities will remain unaffected.")
        print("="*80)

    def _check_vizir4_installation(self):
        """
        检测系统是否安装了vizir4软件
        
        returns:
            bool: 是否安装了vizir4
            str: 安装路径或错误信息
        """
        import os
        import subprocess
        import platform
        import shutil
        
        system = platform.system()
        
        try:
            # use shutil.which to find the executable path (works for Windows and Linux)
            vizir_path = shutil.which('vizir4')

            if vizir_path is not None:
                return True, f"Vizir4已安装于: {vizir_path}"
            
            # if shutil.which didn't find it, try system-specific methods
            if system == 'Windows':
                # try to find the executable in common installation directories
                try:
                    result = subprocess.run(['vizir4', '-version'], 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE, 
                                        text=True,
                                        timeout=2)
                    if result.returncode == 0:
                        return True, "Vizir4已安装，但无法确定路径"
                except Exception:
                    common_paths = [
                        r'C:\\Program Files\\vizir4',
                        r'C:\\Program Files (x86)\\vizir4',
                        os.path.expanduser(r'~\\AppData\\Local\\vizir4')
                    ]
                    for path in common_paths:
                        if os.path.exists(os.path.join(path, 'vizir4.exe')):
                            return True, f"Vizir4可能已安装于: {path}"
                    
            elif system == 'Linux' or system == 'Darwin':
                # check if the executable is in the PATH
                try:
                    result = subprocess.run(['which', 'vizir4'], 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE, 
                                        text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        return True, f"Vizir4已安装于: {result.stdout.strip()}"
                except Exception:
                    pass
                
                common_paths = [
                    '/usr/bin/vizir4',
                    '/usr/local/bin/vizir4',
                    '/opt/vizir4/bin/vizir4'
                ]
                for path in common_paths:
                    if os.path.exists(path):
                        return True, f"Vizir4已安装于: {path}"
                        
            return False, "未找到vizir4安装,请确保它已安装并添加到系统PATH中"
        
        except Exception as e:
            return False, f"检查vizir4安装时出错: {str(e)}"
        
    def update(self, mesh=None, solution=None,sol_p=None,filename=None):
        """
        Update the mesh and/or solution and refresh internal states.

        Parameters:
            mesh : list of Mesh objects, optional
                The new mesh to update.
            solution : list of array-like, optional
                The new solution to update.
        """
        if filename is not None:
            self.filename = filename
        if mesh is not None:
            self.mesh = mesh if isinstance(mesh, list) else [mesh]
            self.mesh_properties = [self._judge_mesh_type(m) for m in self.mesh]
            self._mesh_hash = self._compute_hash(self.mesh)  

        if solution is not None:
            self.solution = solution if isinstance(solution, list) else [solution]
            self._solution_hash = self._compute_hash(self.solution)
            self.sol_p = sol_p if isinstance(sol_p, list) else [sol_p]  

    def _judge_mesh_type(self,mesh):
        """
        Generate properties for a single mesh.

        Parameters:
            mesh : Mesh object
                The mesh to process.

            Returns:
            dict : A dictionary containing the mesh properties.
        """
        if isinstance(mesh, (TriangleMesh, LagrangeTriangleMesh)):
            cell_type = 'Triangles'
            face_type = None
            g_type = 'Simplex'
            pro_od = 'P'
        elif isinstance(mesh, (QuadrangleMesh, LagrangeQuadrangleMesh)):
            cell_type = 'Quadrilaterals'
            face_type = None
            g_type = 'Tensor'
            pro_od = 'Q'
        elif isinstance(mesh, TetrahedronMesh):
            cell_type = 'Tetrahedra'
            face_type = 'Triangles'
            g_type = 'Simplex'
            pro_od = 'P'
        elif isinstance(mesh, HexahedronMesh):
            cell_type = 'Hexahedra'
            face_type = 'Quadrilaterals'
            g_type = 'Tensor'
            pro_od = 'Q'
        else:
            raise ValueError("Unsupported mesh type.")

        return {
            'cell_type': cell_type,
            'face_type': face_type,
            'g_type': g_type,
            'pro_od': pro_od
        }
    
    def _judge_solution_type(self,sol_p,cell_type, pro_od):
        """
        Judge the type of solution and determine its storage format.
        """
        if sol_p == 0:
            sol_type = 'SolAt' + cell_type
            N = self.NC  # 分片常数
        elif sol_p == 1:
            sol_type = 'SolAtVertices'
            N = self.NN  # 分片线性
        elif sol_p >= 2:
            sol_type = 'HOSolAt' + cell_type + pro_od + str(sol_p)
            N = self.NC  # 高阶解
        else:
            raise ValueError("Unsupported solution type: sol_p must be 0, 1, or >= 2.")
        case = 1 if self.is_scalar else 2
        return {
            'pro_sol_p': sol_p,
            'sol_type': sol_type,
            'N': N,
            'case': case
        }

    def _mesh_order_control(self,pro_od):
        """
        Control the local order of the mesh.
        """
        if pro_od == 'P':
            multi_index = bm.multi_index_matrix(self.p , self.TD)
        elif pro_od == 'Q':
            ranges = [bm.arange(self.p + 1) for _ in range(self.TD)]
            G = bm.meshgrid(*ranges, indexing='ij')
            multi_index = bm.stack([g.flatten() for g in G], axis=-1)
        return multi_index
    
    def _solution_order_control(self,sol_p,pro_od):
        """
        Control the local order of the solution.
        """
        if pro_od == 'P':
            bc = bm.multi_index_matrix(sol_p , self.TD)/ sol_p
        elif pro_od == 'Q':
            ranges = [bm.arange(sol_p + 1) for _ in range(self.TD)]
            G = bm.meshgrid(*ranges, indexing='ij')
            bc = bm.stack([g.flatten() for g in G], axis=-1)/ (sol_p+1)
        return bc
            
    def _fetch_mesh_data(self,mesh):
        """
        Fetch the mesh data such as node, cell, and face data.
        """
        self.node = mesh.node
        self.cell = mesh.cell
        self.NN = mesh.number_of_nodes()
        self.NC = mesh.number_of_cells()
        self.GD = mesh.GD
        self.TD = mesh.TD
        self.p = getattr(mesh, 'p', 1)
        if self.TD == 3:
            self.NF = mesh.number_of_faces()
            self.face = mesh.face

    def _mesh_writer(self,mesh,filename,properties,multi_index):
        """
        Write the mesh data to a file.
        """
        import numpy as np
        cell_type = properties['cell_type']
        pro_od = properties['pro_od']
        self._fetch_mesh_data(mesh)
        cell_type = cell_type + pro_od + str(self.p) if self.p > 1 else cell_type
        indices_n = np.arange(self.NN).reshape(-1, 1)+1
        node_data = np.hstack((self.node, indices_n))
        indices_c = np.arange(self.NC).reshape(-1, 1)
        cell_data = np.hstack((self.cell, indices_c))+1
        if self.TD == 3:
            face_type = properties['face_type']
            indices_f = np.arange(self.NF).reshape(-1, 1)
            face_data = np.hstack((self.face, indices_f))+1
            face_type = face_type + pro_od + str(self.p) if self.p > 1 else face_type
        
        with open(self._get_file_path(filename, '.mesh'), 'w') as f:
            f.write(f"MeshVersionFormatted {self.version}\n")
            f.write(f"Dimension {self.GD}\n")
            f.write("Vertices\n")
            f.write(f"{self.NN}\n")
            np.savetxt(f ,node_data , fmt="%.8f " * self.GD + "%d")
            f.write(f"{cell_type}Ordering\n")
            f.write(f"{len(multi_index)}\n")
            np.savetxt(f ,multi_index , fmt="%d " * multi_index.shape[1])
            if self.TD == 3:
                f.write(f"{face_type}\n")
                f.write(f"{self.NF}\n")
                np.savetxt(f ,face_data , fmt="%d " * self.face.shape[1] + "%d")
            f.write(f"{cell_type}\n")
            f.write(f"{self.NC}\n")
            np.savetxt(f ,cell_data , fmt="%d " * self.cell.shape[1] + "%d")
            f.write("scrolling\n")
            f.write("1\n")
            f.write("End\n")
     
    def _solution_writer(self,mesh,sol,filename,sol_properties,bc = None):
        """
        Write the solution data to a file.
        """
        import numpy as np
        sol_type = sol_properties['sol_type']
        N = sol_properties['N']
        case = sol_properties['case']
        sol_p = sol_properties['pro_sol_p']
        sol_ndarray = np.array(sol,dtype=np.float64)
        NLI = mesh.number_of_local_ipoints(self.p)

        with open(self._get_file_path(filename, '.sol'), 'w') as f:
            f.write(f"MeshVersionFormatted {self.version}\n")
            f.write(f"Dimension {self.GD}\n")
            if sol_p >=2:
                f.write(f"{sol_type}NodesPositions\n")
                f.write(f"{len(bc)}\n")
                np.savetxt(f ,bc , fmt="%.8f " * bc.shape[1])
            f.write(f"{sol_type}\n")
            f.write(f"{N}\n")
            f.write(f"{self.sol_num} {case}\n")
            if sol_p < 2:
                np.savetxt(f ,sol_ndarray.reshape(-1, self.sol_num) , fmt="%.8f " * self.sol_num)
            else:
                f.write(f"{sol_p} {NLI}\n")
                np.savetxt(f ,sol_ndarray[self.cell] , fmt="%.8f " * self.sol_num)
            f.write("End\n")

    def plot(self, plot_mesh=True, plot_solution=True, is_visual=True,multi_index_list=None):
        """
        Plot the mesh and/or solution based on user preferences.

        Parameters:
            plot_mesh : bool, optional
                Whether to plot the mesh.
            plot_solution : bool, optional
                Whether to plot the solution.
            is_visual : bool, optional
                Whether to visualize the output using an external tool.
            multi_index_list : list of arrays, optional
                List of multi-index arrays for each mesh.
        """
        if self._has_changed():
            self._version_counter += 1

        filename_list = []
        for i, (mesh,solution, m_properties) in enumerate(zip(self.mesh, self.solution, 
                                                              self.mesh_properties)):
            mesh_filename = f"{self.filename}_{i}_v{self._version_counter}"
            self._fetch_mesh_data(mesh)
            if multi_index_list is not None:
                multi_index = multi_index_list[i]
            else:
                multi_index = self._mesh_order_control(m_properties['pro_od'])
            if plot_mesh:
                self._mesh_writer(mesh, mesh_filename, m_properties, multi_index)
                filename_list.append(mesh_filename)

            if plot_solution:
                s_properties = self._judge_solution_type(self.sol_p[i], 
                                                         m_properties['cell_type'], 
                                                         m_properties['pro_od'])
                bc = self._solution_order_control(self.sol_p[i], m_properties['pro_od'])
                solution_filename = f"{self.filename}_{i}_v{self._version_counter}"
                self._solution_writer(mesh, solution, solution_filename,s_properties,bc)

        self.movie_list.append(mesh_filename)
        if is_visual:
            self._visualize(filename_list)

    def _visualize(self,mesh_filename_list):
        """Visualize the output using an external tool."""
        import subprocess
        filenames = " ".join([self._get_file_path(filename) for filename in mesh_filename_list])
        state_path = self._get_file_path('vizir', '.state')
        process = f'vizir4 -in {filenames} -state {state_path}'
        subprocess.run(process, shell=True)
        print(f"Visualization completed using {process}")

    def _state_setup(self):
        """
        Set up the internal state of the class.
        """
        p = getattr(self.mesh[0], 'p', 1)
        with open(self._get_file_path('vizir', '.state'), 'w') as f:
            if p>1:
                f.write('TessOn 0\n')
                f.write(f'TessLevel {p +4}\n')
            f.write('SolOn 1\n')
            f.write(f'Colormap {self.colormap}\n')
            f.write('Scrolling 1\n')
            f.write(f'WireSiz {self.linewidth}\n')

    def movie_generator(self,is_visual=False, fps=15, movie_name='output'):
        """
        Generate a movie file for the mesh and solution.

        Parameters:
            is_visual : bool, optional
                Whether to visualize the movie using an external tool.
        """
        import numpy as np
        mesh_files = [f"{self._get_file_path(filename,'.mesh')}" for filename in self.movie_list]
        sol_files = [f"{self._get_file_path(filename, '.sol')}" for filename in self.movie_list]
        combined_array = np.column_stack((mesh_files, sol_files))
        combined_array = np.column_stack((mesh_files, sol_files))
        with open(self._get_file_path('vizir', '.movie'), 'w') as f:
            np.savetxt(f, combined_array, fmt="%s %s")
        if is_visual:
            import subprocess
            import os,re

            process = f'vizir4 -movie {self._get_file_path("vizir", ".movie")}'
            subprocess.run(process)
            def extract_number(filename):
                match = re.search(r'_v(\d+)\.sol\.jpg$', filename)
                return int(match.group(1)) if match else -1

            files = sorted(
                [f for f in os.listdir(self.output_dir) 
                 if f.startswith(self.filename) and f.endswith('.sol.jpg')],
                key=extract_number
            )
            for i, file in enumerate(files):
                new_name = f"frame_{i:04d}.jpg"
                old_path = os.path.join(self.output_dir, file)
                new_path = os.path.join(self.output_dir, new_name)        
                # 检查目标文件是否存在
                if os.path.exists(new_path):
                    os.remove(new_path)  # 删除旧文件
                
                os.rename(old_path, new_path)
            print(f"Renamed {len(files)} files for ffmpeg.")
            command = [
                "ffmpeg",
                "-y",  
                "-framerate", str(fps), 
                "-i", f"{self.output_dir}/frame_%04d.jpg", 
                "-vf", "scale=800:-1:flags=lanczos", 
                "-loop", "0",  
                self._get_file_path(movie_name, '.gif')
            ]
            subprocess.run(command, check=True)
            print(f"GIF saved to {movie_name}")