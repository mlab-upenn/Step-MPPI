import numpy as np
import jax.numpy as jnp

from f1tenth_gym.envs.track import Track, Boundary
from f1tenth_planning.control.controllers.mpc.mpc import MPCController
from f1tenth_planning.control.config.dynamics_config import (
    DynamicsConfig,
    f1tenth_params,
)
from f1tenth_planning.control.controllers.mpc.mpc import MPCController
from f1tenth_planning.control.config.controller_config import (
    MPCConfig,
    dynamic_mpc_config,
)
from f1tenth_planning.control.dynamics_models.dynamic_model import DynamicBicycleModel
from f1tenth_planning.control.solvers.nonlinear_mpc_solver import NonlinearMPCSolver
from f1tenth_planning.control.dynamics_model import DynamicsModel
from f1tenth_planning.utils.utils import calc_interpolated_reference_trajectory
from .DPC_solver import DPC_Solver
from .constraints import *

class DPC_Planner(MPCController):
    def __init__(
        self,
        track: Track,
        boundary: Boundary,
        params: DynamicsConfig = None,
        model: DynamicBicycleModel = None,
        solver: DPC_Solver = None,
    ):
        """
        Convenience class that uses Differentiable MPC solver with dynamic bicycle model.

        Args:
            track (f1tenth_gym_ros:Track): track object, contains the reference raceline
            config (MPCConfig, optional): MPC configuration object, contains MPC costs and constraints
            params (DynamicsConfig, optional): Vehicle parameters for the dynamic model. If none,
            default f1tenth_params() will be used.
        """
        if params is None:
            params = f1tenth_params()
        if model is None:
            model = DynamicBicycleModel(params)

        super(DPC_Planner, self).__init__(track, solver, model, params)
        self.track_boundary_coeffs = boundary.get_coefficients()

    def update_info(self, state: dict):
        """
        Updates on current state, references, and constraint coefficients
        """
        x = state["pose_x"]
        y = state["pose_y"]
        v = state["linear_vel_x"]
        yaw = state["pose_theta"]
        # x0 of shape (nx,)
        x0 = np.array([x, y, state["delta"], v, yaw, state["ang_vel_z"], state["beta"]])

        cx = self.waypoints[:, 0]
        cy = self.waypoints[:, 1]
        cv = self.waypoints[:, 3]

        # Clip the reference velocity to the operational speed limits
        clipped_velocity = np.clip(cv, a_min=self.ref_v_min, a_max=self.ref_v_max)

        ref_traj, idx_list = calc_interpolated_reference_trajectory(
            x,
            y,
            yaw,
            cx,
            cy,
            clipped_velocity,
            self.solver.config.dt,
            self.solver.config.N,
            self.waypoints,
        )
        
        ref_traj = ref_traj.T.copy()
        idx_list = idx_list
        a = self.track_boundary_coeffs["a"][idx_list]
        b = self.track_boundary_coeffs["b"][idx_list]
        c_left = self.track_boundary_coeffs["c_left"][idx_list]
        c_right = self.track_boundary_coeffs["c_right"][idx_list]
        
        const_coeffs = np.vstack((a, b, c_left, c_right))

        return x0, ref_traj, const_coeffs

    def plan(
        self,
        state: dict,
    ):
        # Get the updates on current state, references, and constraint coefficients
        x0, ref_traj, bdr_const_coeffs = self.update_info(state)
        self.ref_traj = ref_traj
        self.const_coeffs = bdr_const_coeffs
        # Call the solver
        self.x_pred, self.u_pred = self.solver.solve(x0, self.ref_traj, self.const_coeffs)
        self.local_plan = self.ref_traj[:2].T
        self.control_solution = np.array(self.x_pred[:2, :])
        return np.array(self.u_pred[:, 0]).flatten()
