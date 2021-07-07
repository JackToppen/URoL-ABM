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
    def update_diffusion(self, gradient_name):
        """ Approximates the diffusion of the morphogen for the
            extracellular gradient specified.
        """
        # calculate the number of steps and the last step time if it doesn't divide nicely
        steps, last_dt = divmod(self.step_dt, self.diffuse_dt)
        steps = int(steps) + 1  # make sure steps is an int, add extra step for the last dt if it's less

        # all gradients are held as 3D arrays for simplicity, get the gradient as a 2D array
        gradient = self.__dict__[gradient_name][:, :, 0]

        # set max and min concentration values
        gradient[gradient > self.max_concentration] = self.max_concentration
        gradient[gradient < 0] = 0

        # pad the sides of the array with zeros for holding ghost points
        base = np.pad(gradient, 1)

        # call the JIT diffusion function, remove ghost points
        base = update_diffusion_jit(base, steps, self.diffuse_dt, last_dt, self.diffuse_const, self.spat_res2)
        gradient = base[1:-1, 1:-1]

        # degrade the morphogen concentrations
        gradient *= 1 - self.degradation

        # update the simulation with the updated gradient
        self.__dict__[gradient_name][:, :, 0] = gradient

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

        # get the four nearest points to the cell in 2D and make array for holding distances
        points = np.array([[x, y, 0], [x + 1, y, 0], [x, y + 1, 0], [x + 1, y + 1, 0]], dtype=int)
        if_nearby = np.zeros(4, dtype=bool)

        # go through potential nearby diffusion points
        for i in range(4):
            # get point and make sure it's in bounds
            point = points[i]
            if point[0] < self.gradient_size[0] and point[1] < self.gradient_size[1]:
                # get location of point
                point_location = point * self.spat_res

                # see if point is in diffuse radius, if so update if_nearby index to True
                if np.linalg.norm(self.locations[index] - point_location) < self.spat_res:
                    if_nearby[i] = 1

        # get the number of points within diffuse radius
        total_nearby = np.sum(if_nearby)

        # if at least one diffusion point nearby, go back through points adding morphogen
        if total_nearby > 0:
            point_amount = amount / total_nearby
            for i in range(4):
                if if_nearby[i]:
                    x, y, z = points[i][0], points[i][1], 0
                    gradient[x][y][z] += point_amount
