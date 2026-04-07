import numpy as np
import jax.numpy as jnp

from f1tenth_gym.envs.track import Track, Boundary
from f1tenth_planning.control.controllers.mpc.mpc import MPCController
from f1tenth_planning.control.config.dynamics_config import (
    DynamicsConfig,
    f1tenth_params,
)
from f1tenth_planning.control.dynamics_models.dynamic_model import DynamicBicycleModel
from f1tenth_planning.control.config.controller_config import (
    APMPPIConfig,
    dynamic_ap_mppi_config,
)
from f1tenth_planning.control.dynamics_model import DynamicsModel
from f1tenth_planning.utils.utils import jnp_to_np, calc_interpolated_reference_trajectory

# from f1tenth_planning.control.solvers import Const_MPPI_Solver
from .MPPI_solver import Const_MPPI_Solver
from .constraints import *

class Const_MPPI_Planner(MPCController):
    """
    Convenience class that uses Constrained-MPPI solver with dynamic bicycle model.

    This planner automatically sets up state limit constraints based on vehicle parameters
    when no config is provided. To customize bounds, pass your own APMPPIConfig with
    x_min, x_max, u_min, u_max set to your desired values.

    State: x = [x, y, delta, v, yaw, yaw_rate, beta]
    Control: u = [delta_v, a]

    Args:
        track: Track object containing the reference raceline.
        params: Vehicle parameters for the dynamic model. Defaults to f1tenth_params().
        model: Dynamics model object. Defaults to DynamicBicycleModel.
        config: AP-MPPI configuration. If None, creates default with state limit constraints
            and bounds from vehicle parameters. Pass your own config to customize.
        solver: AP-MPPI solver. If None, creates from config and model.
        pre_processing_fn: Optional preprocessing function for observations.
        use_state_limits: Whether to automatically add state limit constraints (only when config is None).
        ref_velocity_bounds: (v_min, v_max) for reference trajectory clipping. If None, uses config.x_min/x_max[3].
    """

    def __init__(
        self,
        track: Track,
        boundary: Boundary,
        params: DynamicsConfig = None,
        model: DynamicsModel = None,
        config: APMPPIConfig = None,
        solver: Const_MPPI_Solver = None,
    ):
        '''
        Initialize Dynamic AP-MPPI Planner.

        Args:
            track (Track): Track object containing the reference raceline.
            params (DynamicsConfig, optional): Vehicle parameters for the dynamic model. Defaults to f1tenth_params().
            model (DynamicsModel, optional): Dynamics model object. Defaults to DynamicBicycleModel.
            config (APMPPIConfig, optional): AP-MPPI configuration. If None, creates default with state limit
                constraints and bounds from vehicle parameters. Pass your own config to customize bounds.
            solver (Const_MPPI_Solver, optional): AP-MPPI solver. If None, creates from config and model.
            pre_processing_fn (callable, optional): Optional preprocessing function for observations.
            use_state_limits (bool, optional): Whether to automatically add state limit constraints
                from vehicle parameters. Only used when config is None. Defaults to True.
            ref_velocity_bounds (tuple[float, float], optional): (v_min, v_max) bounds for clipping
                reference trajectory velocities. Use this to set operational speed limits that differ
                from the physical limits in config.x_min/x_max. If None, uses config bounds.
        '''
        if not isinstance(solver, Const_MPPI_Solver) and solver is not None:
            raise ValueError("Solver must be an instance of Const_MPPI_Solver")
        if not isinstance(model, DynamicsModel) and model is not None:
            raise ValueError("Model must be an instance of DynamicsModel")

        if params is None:
            params = f1tenth_params()
        if model is None:
            model = DynamicBicycleModel(params)

        if config is None:
            raise NotImplementedError("Const_MPPI_Planner requires an explicit config")
        
        if solver is None:
            solver = Const_MPPI_Solver(config, model)
                        
        super(Const_MPPI_Planner, self).__init__(
            track,
            solver,
            model,
            params,
        )
        # Build state constraints from vehicle parameters
        # x = [x, y, delta, v, yaw, yaw_rate, beta]
        x_min_constrained = np.array([
            -np.inf,           # x position: unconstrained
            -np.inf,           # y position: unconstrained  
            params.MIN_STEER,  # steering angle
            params.MIN_SPEED,  # velocity
            -np.inf,           # yaw: unconstrained
            -np.inf,           # yaw_rate: unconstrained
            -np.inf,           # beta: unconstrained
        ])
        x_max_constrained = np.array([
            np.inf,            # x position: unconstrained
            np.inf,            # y position: unconstrained
            params.MAX_STEER,  # steering angle
            params.MAX_SPEED,  # for testing, set max speed to 5 m/s
            np.inf,            # yaw: unconstrained
            np.inf,            # yaw_rate: unconstrained
            np.inf,            # beta: unconstrained
        ])

        # Construct the constraint coefficients
        N = self.solver.config.N
        A_state, b_state = st_limit_constraint_coeffs(
            x_min_constrained,
            x_max_constrained,
            horizon=N,
        )
        self.const_coeffs = [[A_state, b_state], [None, None]]
        # self.const_coeffs = [[A_state, b_state]]

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
        idx_list = idx_list[1:]
        a = self.track_boundary_coeffs["a"][idx_list]
        b = self.track_boundary_coeffs["b"][idx_list]
        c_left = self.track_boundary_coeffs["c_left"][idx_list]
        c_right = self.track_boundary_coeffs["c_right"][idx_list]
        
        const_coeffs = boundary_constraint_coeffs(a, b, c_left, c_right)

        return x0, ref_traj, const_coeffs

    def plan(
        self,
        state: dict,
    ):
        # Get the updates on current state, references, and constraint coefficients
        x0, ref_traj, bdr_const_coeffs = self.update_info(state)
        self.ref_traj = ref_traj
        self.const_coeffs[1] = [bdr_const_coeffs[0], bdr_const_coeffs[1]]
        # Call the solver
        self.x_pred, self.u_pred = self.solver.solve(x0, self.ref_traj, self.const_coeffs)
        self.x_pred = jnp_to_np(self.x_pred)
        self.u_pred = jnp_to_np(self.u_pred)
        self.local_plan = self.ref_traj[:2].T
        self.control_solution = np.array(self.x_pred[:2, :])
        return np.array(self.u_pred[:, 0]).flatten()
