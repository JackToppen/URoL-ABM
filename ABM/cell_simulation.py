import numpy as np
import random as r
import math

from cell_methods import CellMethods

from pythonabm import Simulation
from pythonabm import *


class CellSimulation(CellMethods, Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """
    def __init__(self):
        # initialize the Simulation object
        Simulation.__init__(self)

        # read parameters from YAML file and add them to instance variables
        self.yaml_parameters("templates\\general.yaml")

        # the temporal resolution for the simulation
        self.step_dt = 1800  # dt of each simulation step (1800 sec)
        self.move_dt = 180  # dt for incremental movement (180 sec)

        # make gradient arrays for BMP4 and NOGGIN
        self.BMP = np.random.rand(self.size[0], self.size[1])
        self.NOG = np.random.rand(self.size[0], self.size[1])

    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """
        # add agents to the simulation
        self.add_agents(self.num_to_start)

        # create function for giving random location on circle with uniform distribution
        def uniform_on_circle():
            center_x, center_y = self.size[0] / 2, self.size[1] / 2
            radius = 100 * math.sqrt(r.random())
            angle = math.tau * r.random()
            return np.array([center_x + radius * math.cos(angle), center_y + radius * math.sin(angle), 0])

        # create the following agent arrays with initial conditions
        self.agent_array("locations", func=uniform_on_circle)
        self.agent_array("radii", func=lambda: 5)
        self.agent_array("motility_forces", vector=3)
        self.agent_array("jkr_forces", vector=3)

        # create graph for holding agent neighbors
        self.agent_graph("neighbor_graph")
        self.agent_graph("jkr_graph")

        # record initial values
        self.step_values()
        self.step_image()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # prints info about the current step and records the starting time of the step
        self.info()

        # get all neighbors within radius of 2
        self.get_neighbors("neighbor_graph", 5)

        # call the following methods that update agent values
        # self.die()
        # self.reproduce()
        # self.cell_motility()

        self.update_diffusion()

        # add/remove agents from the simulation
        self.update_populations()

        # apply any adhesive or repulsive forces to the cells
        self.apply_forces()

        # save multiple forms of information about the simulation at the current step
        self.step_values()
        self.step_image()
        self.temp()
        self.data()

        import cv2
        grad_image = 255 * self.BMP
        grad_image = grad_image.astype(np.uint8)  # use unsigned int8
        grad_image = cv2.applyColorMap(grad_image, cv2.COLORMAP_OCEAN)
        cv2.imwrite("test_bmp.png", grad_image)

    @record_time
    def die(self):
        """ Updates an agent based on the presence of neighbors.
        """
        # determine which agents are being removed
        for index in range(self.number_agents):
            if r.random() < 0.1:
                self.mark_to_remove(index)

    @record_time
    def move(self):
        """ Assigns new location to agent.
        """
        for index in range(self.number_agents):
            # get new location position
            new_location = self.locations[index] + 5 * self.random_vector()

            # check that the new location is within the space, otherwise use boundary values
            for i in range(3):
                if new_location[i] > self.size[i]:
                    self.locations[index][i] = self.size[i]
                elif new_location[i] < 0:
                    self.locations[index][i] = 0
                else:
                    self.locations[index][i] = new_location[i]

    @record_time
    def reproduce(self):
        """ If the agent meets criteria, hatch a new agent.
        """
        # determine which agents are hatching
        for index in range(self.number_agents):
            if r.random() < 0.1:
                self.mark_to_hatch(index)
