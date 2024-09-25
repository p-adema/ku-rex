"""
some simple first-order robot dynamics models
"""
import numpy as np

class RobotModel:

    def __init__(self, ctrl_range) -> None:
        #record the range of action that can be used to integrate the state
        self.ctrl_range = ctrl_range
        return
    
    def forward_dyn(self, x, u, T):
        #need to return intergated path of X with u along the horizon of T
        return NotImplementedError

    def inverse_dyn(self, x, x_goal, T):
        #return dynamically feasible path to move to the x_goal as close as possible
        return NotImplementedError

class PointMassModel(RobotModel):
    #Note Arlo is differential driven and may be simpler to avoid Dubins car model by rotating in-place to direct and executing piecewise straight path  
    def forward_dyn(self, x, u, T):
        path = [x]
        #note u must have T ctrl to apply
        for i in range(T):
            x_new = path[-1] + u[i] #u is velocity command here
            path.append(x_new)    
        
        return path[1:]

    def inverse_dyn(self, x, x_goal, T):
        #for point mass, the path is just a straight line by taking full ctrl_range at each step
        dir = (x_goal-x)/np.linalg.norm(x_goal-x)

        u = np.array([dir*self.ctrl_range[1] for _ in range(T)])

        return self.forward_dyn(x, u, T)