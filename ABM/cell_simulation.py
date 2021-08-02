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
        x, y = int(self.size[0] / 10), int(self.size[1] / 10)
        self.BMP = np.zeros((x, y))
        self.NOG = np.zeros((x, y))

    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """
        # add agents to the simulation
        self.add_agents(self.num_to_start)

        # create function for giving random location on circle with uniform distribution
        def uniform_on_circle():
            center_x, center_y = self.size[0] / 2, self.size[1] / 2
            radius = 500 * math.sqrt(r.random())
            angle = math.tau * r.random()
            return np.array([center_x + radius * math.cos(angle), center_y + radius * math.sin(angle), 0])

        # create the following agent arrays with initial conditions
        self.agent_array("locations", func=uniform_on_circle)
        self.agent_array("radii", func=lambda: 5)
        self.agent_array("motility_forces", vector=3)
        self.agent_array("jkr_forces", vector=3)
        self.agent_array("states", dtype=int)
        self.agent_array("BMP_counter")

        # create graph for holding agent neighbors
        self.agent_graph("neighbor_graph")
        self.agent_graph("jkr_graph")

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # get all neighbors within radius of 2
        self.get_neighbors("neighbor_graph", 5)

        # calculate diffusion of BMP4 and NOGGIN
        self.update_diffusion()

        # call the following methods that update agent values
        self.update()

        # add/remove agents from the simulation
        self.update_populations()

        # apply any adhesive or repulsive forces to the cells
        self.apply_forces()

        # save multiple forms of information about the simulation at the current step
        self.step_values()
        self.step_image()
        self.temp()
        self.data()

    @record_time
    def update(self):
        """ Updates cells based on their state.
        """
        for index in range(self.number_agents):
            # find the nearest diffusion point
            half_indices = np.floor(2 * self.locations[index] / 10)
            indices = np.ceil(half_indices / 2).astype(int)
            x, y = indices[0], indices[1]

            # find the nearest diffusion point
            if self.BMP[x][y] > 0.02:
                self.BMP_counter[index] += 1

                if self.BMP_counter[index] >= 5:
                    self.states[index] = 1
                if self.BMP_counter[index] >= 20:
                    self.states[index] = 2
            #
            # if self.BMP_counter[index] >= 30:
            #     self.states[index] = 2    # CDX2
            # elif 30 > self.BMP_counter[index] >= 15:
            #     self.states[index] = 1    # BRA
            # else:
            #     self.states[index] = 0    # SOX2

    @record_time
    def step_image(self, background=(0, 0, 0), origin_bottom=True):
        """ Overrides default step_image() method.
        """
        # only continue if outputting images
        if self.output_images:
            # get path and make sure directory exists
            check_direct(self.images_path)

            # get the size of the array used for imaging in addition to the scaling factor
            x_size = self.image_quality
            scale = x_size / self.size[0]
            y_size = math.ceil(scale * self.size[1])

            # create the agent space background image and apply background color
            image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
            image[:, :] = background

            # go through all of the agents
            for index in range(self.number_agents):
                # get xy coordinates, the axis lengths, and color of agent
                x, y = int(scale * self.locations[index][0]), int(scale * self.locations[index][1])
                major = int(scale * self.radii[index])
                minor = int(scale * self.radii[index])
                if self.states[index] == 0:
                    color = (255, 50, 50)
                elif self.states[index] == 1:
                    color = (50, 255, 50)
                else:
                    color = (50, 50, 255)

                # draw the agent and a black outline to distinguish overlapping agents
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, color, -1)
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, (0, 0, 0), 1)

            # if the origin should be bottom-left flip it, otherwise it will be top-left
            if origin_bottom:
                image = cv2.flip(image, 0)

            # save the image as a PNG
            image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)
            file_name = f"{self.name}_image_{self.current_step}.png"
            cv2.imwrite(self.images_path + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression])
