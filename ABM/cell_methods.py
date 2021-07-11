import random as r

from cell_backend import *
from pythonabm import record_time, normal_vector


class CellMethods:
    """ The methods in this class are meant to be inherited by the CellSimulation
        class so that CellSimulation objects can call these methods.
    """
    @record_time
    def cell_motility(self):
        """ Gives the cells a motive force depending on set rules
            for the cell types.
        """
        # motility force for each cell
        motility_force = 0.000000002

        # add random vector for motility
        for index in range(self.number_agents):
            self.motility_forces[index] += self.random_vector() * motility_force

    def calculate_jkr(self):
        """ Goes through all contacting cells and quantifies any resulting
            adhesive or repulsion forces.
        """
        # contact mechanics parameter
        adhesion_const = 0.000107    # the adhesion constant in kg/s from P Pathmanathan et al.
        poisson = 0.5    # Poisson's ratio for the cells, 0.5 means incompressible
        youngs = 1000    # Young's modulus for the cells in Pa

        # get the edges as an array, count them, and create holder used to delete edges
        jkr_edges = np.array(self.jkr_graph.get_edgelist())
        number_edges = len(jkr_edges)
        delete_edges = np.zeros(number_edges, dtype=bool)

        # only continue if edges exist
        if number_edges > 0:
            # if using CUDA GPU
            if self.cuda:
                # allow the following arrays to be passed to the GPU
                delete_edges = cuda.to_device(delete_edges)
                forces = cuda.to_device(self.jkr_forces)

                # specify threads-per-block and blocks-per-grid values
                tpb = 72
                bpg = math.ceil(number_edges / tpb)

                # call the CUDA kernel, sending arrays to GPU
                jkr_forces_gpu[bpg, tpb](cuda.to_device(jkr_edges), delete_edges, cuda.to_device(self.locations),
                                         cuda.to_device(self.radii), forces, cuda.to_device(poisson),
                                         cuda.to_device(youngs), cuda.to_device(adhesion_const))

                # return the following arrays back from the GPU
                forces = forces.copy_to_host()
                delete_edges = delete_edges.copy_to_host()

            # otherwise use parallelized JIT function
            else:
                forces, delete_edges = jkr_forces_cpu(number_edges, jkr_edges, delete_edges, self.locations, self.radii,
                                                      self.jkr_forces, poisson, youngs, adhesion_const)

            # update the graph to remove any edges that have be broken and update the JKR forces array
            self.jkr_graph.delete_edges(np.arange(number_edges)[delete_edges])
            self.jkr_forces = forces

    @record_time
    def apply_forces(self):
        """ Calls multiple methods in an attempt to move the cells to an
            equilibrium between repulsive, adhesive, and motility forces.
        """
        # constant for calculating stokes friction
        stokes = 10000

        # calculate the number of steps and the last step time if it doesn't divide nicely
        steps, last_dt = divmod(self.step_dt, self.move_dt)
        total_steps = int(steps) + 1  # add extra step for the last dt, if divides nicely last_dt will equal zero

        # go through all move steps, calculating the physical interactions and applying the forces
        for step in range(total_steps):
            # update graph for pairs of contacting cells
            self.get_neighbors("jkr_graph", 2 * np.amax(self.radii), clear=False)

            # calculate the JKR forces based on the JKR graph edges
            self.calculate_jkr()

            # if on the last step use, that dt
            if step == total_steps - 1:
                move_dt = last_dt
            else:
                move_dt = self.move_dt

            # turn size into numpy array
            size = np.asarray(self.size)

            # if using CUDA GPU
            if self.cuda:
                # allow the following arrays to be passed to the GPU
                locations = cuda.to_device(self.locations)

                # specify threads-per-block and blocks-per-grid values
                tpb = 72
                bpg = math.ceil(self.number_agents / tpb)

                # call the CUDA kernel, sending arrays to GPU
                apply_forces_gpu[bpg, tpb](cuda.to_device(self.jkr_forces), cuda.to_device(self.motility_forces),
                                           locations, cuda.to_device(self.radii), cuda.to_device(stokes),
                                           cuda.to_device(size), cuda.to_device(move_dt))

                # return the following arrays back from the GPU
                new_locations = locations.copy_to_host()

            # otherwise use parallelized JIT function
            else:
                new_locations = apply_forces_cpu(self.number_agents, self.jkr_forces, self.motility_forces,
                                                 self.locations, self.radii, stokes, size, move_dt)

            # update the locations and reset the JKR forces back to zero
            self.locations = new_locations
            self.jkr_forces[:, :] = 0

        # reset motility forces back to zero
        self.motility_forces[:, :] = 0

    @record_time
    def update_diffusion(self):
        """ Solves the PDE for BMP4 and NOGGIN concentrations.
        """
        # pad the sides of the array with zeros for holding ghost points
        BMP_base = np.zeros((self.size[0] + 2, self.size[1] + 2))
        NOG_base = np.zeros((self.size[0] + 2, self.size[1] + 2))
        BMP_base[1:-1, 1:-1] = self.BMP
        NOG_base[1:-1, 1:-1] = self.NOG

        # call the JIT diffusion function, remove ghost points
        BMP_base, NOG_base = update_diffusion_jit(BMP_base, NOG_base)
        self.BMP = BMP_base[1:-1, 1:-1]
        self.NOG = NOG_base[1:-1, 1:-1]

    def get_concentration(self, gradient_name, index):
        """ Get the concentration of a gradient for a cell's
            location from the nearest diffusion point.
        """
        # get the gradient array
        gradient = self.__dict__[gradient_name]

        # find the nearest diffusion point
        half_indices = np.floor(2 * self.locations[index] / self.spat_res)
        indices = np.ceil(half_indices / 2).astype(int)
        x, y, z = indices[0], indices[1], indices[2]

        # return the value of the gradient at the diffusion point
        return gradient[x][y][z]

    def adjust_morphogens(self, gradient_name, index, amount):
        """ Adjust the concentration of the gradient based on
            the amount and the location of the cell.
        """
        # get the gradient array
        gradient = self.__dict__[gradient_name]

        # divide the location for a cell by the spatial resolution then take the floor function of it
        indices = np.floor(self.locations[index] / self.spat_res).astype(int)
        x, y, z = indices[0], indices[1], indices[2]

        # adjust the gradient
        gradient[x][y][z] += amount
