import argparse
import copy
import json
import os
import pathlib
import sys
import time


def _requested_device_from_argv(argv):
    for idx, arg in enumerate(argv):
        if arg.startswith("--device="):
            return arg.split("=", 1)[1].lower()
        if arg == "--device" and idx + 1 < len(argv):
            return argv[idx + 1].lower()
    return "gpu"

if _requested_device_from_argv(sys.argv[1:]) == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco
import jax
import numpy as np
import yaml
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
from gym_quadruped.utils.quadruped_utils import LegsAttr
from tqdm import tqdm

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
QUADRUPED_PYMPC_PATH = REPO_ROOT / "Quadruped-PyMPC"
if str(QUADRUPED_PYMPC_PATH) not in sys.path:
    sys.path.insert(0, str(QUADRUPED_PYMPC_PATH))

from quadruped_pympc import config as cfg
from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco
from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper


def block_tree_until_ready(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def load_config(config_path: pathlib.Path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_repo_path(path: pathlib.Path):
    if path.is_absolute():
        return path
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return CURRENT_DIR / path


def normalize_policy_config(policy_config: dict):
    normalized = dict(policy_config)
    if "num_layers" in normalized and normalized["num_layers"] is not None:
        normalized["num_layers"] = int(normalized["num_layers"])
    if "hidden_dim" in normalized and normalized["hidden_dim"] is not None:
        normalized["hidden_dim"] = int(normalized["hidden_dim"])
    if "activation" in normalized and normalized["activation"] is not None:
        normalized["activation"] = str(normalized["activation"])
    return normalized


def normalize_dmppi_config(dmppi_config: dict):
    normalized = dict(dmppi_config)
    if "dmppi_samples" in normalized and normalized["dmppi_samples"] is not None:
        normalized["dmppi_samples"] = int(normalized["dmppi_samples"])
    if "dmppi_temperature" in normalized and normalized["dmppi_temperature"] is not None:
        normalized["dmppi_temperature"] = float(normalized["dmppi_temperature"])
    return normalized


def sample_goal(qpympc_cfg, rng: np.random.Generator, initial_base_pos: np.ndarray):
    ref_z = qpympc_cfg.simulation_params["ref_z"]
    return np.array(
        [
            initial_base_pos[0] + rng.uniform(-2.0, 2.0),
            initial_base_pos[1] + rng.uniform(-2.0, 2.0),
            ref_z,
        ],
        dtype=float,
    )


def load_goal_set(goal_file: pathlib.Path | None):
    if goal_file is None:
        return None
    with np.load(goal_file) as data:
        return {
            "goals": np.asarray(data["goal_base_positions"], dtype=float),
            "initial_base_positions": np.asarray(data["initial_base_positions"], dtype=float)
            if "initial_base_positions" in data
            else None,
        }


def reset_episode_base_position(env, goal_set, episode_num: int):
    if goal_set is None or goal_set.get("initial_base_positions") is None:
        return
    initial_base_pos = np.asarray(
        goal_set["initial_base_positions"][min(episode_num, len(goal_set["initial_base_positions"]) - 1)],
        dtype=float,
    )
    env.mjData.qpos[0:3] = initial_base_pos
    env.mjData.qvel[0:6] = 0.0
    mujoco.mj_forward(env.mjModel, env.mjData)


def save_summary_json(script_name: str, summary: dict, seed: int, goal_file: pathlib.Path | None):
    statistics_dir = REPO_ROOT / "statistics"
    statistics_dir.mkdir(parents=True, exist_ok=True)
    output_path = statistics_dir / f"{pathlib.Path(script_name).stem}_summary_seed_{seed}.json"
    payload = {
        "script": script_name,
        "seed": int(seed),
        "goal_file": None if goal_file is None else str(goal_file),
        "goals_reached": int(summary["goals_reached"]),
        "timeout": int(summary["timeout"]),
        "truncated": int(summary["truncated"]),
        "terminated": int(summary["terminated"]),
        "success_rate": float(summary["success_rate"]),
        "lap_times": np.asarray(summary["lap_times"], dtype=float).tolist(),
        "episode_avg_compute_times_ms": np.asarray(summary["episode_avg_compute_times_ms"], dtype=float).tolist(),
        "mean_compute_time_ms": float(summary["mean_compute_time_ms"]),
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Saved summary to {output_path}")
    return output_path


def configure_dmppi_controller(
    policy_file_path: pathlib.Path,
    policy_config: dict,
    device: str,
    dmppi_num_samples: int,
    dmppi_temperature: float,
    use_neural_updater: bool,
):
    cfg.mpc_params["type"] = "dmppi"
    cfg.mpc_params["use_neural_updater"] = bool(use_neural_updater)
    cfg.mpc_params["device"] = device
    cfg.mpc_params["dpc_policy_path"] = str(policy_file_path)
    cfg.mpc_params["dpc_num_layers"] = policy_config.get("num_layers", 5)
    cfg.mpc_params["dpc_hidden_dim"] = policy_config.get("hidden_dim", 256)
    cfg.mpc_params["dpc_activation"] = policy_config.get("activation", "gelu")
    cfg.mpc_params["dmppi_samples"] = int(dmppi_num_samples)
    cfg.mpc_params["dmppi_temperature"] = float(dmppi_temperature)


def run_dmppi_test(
    policy_file_path,
    policy_config,
    qpympc_cfg,
    process=0,
    num_episodes=None,
    num_seconds_per_episode=30,
    ref_base_lin_vel=(0.0, 2.0),
    ref_base_ang_vel=(-0.4, 0.4),
    friction_coeff=(0.5, 1.0),
    base_vel_command_type="forward",
    goal_base_pos=None,
    goal_set=None,
    random_goals=False,
    goal_kp=1.0,
    goal_max_lin_vel=0.2,
    goal_position_tolerance=0.1,
    seed=0,
    render=True,
    device="gpu",
    dmppi_num_samples=64,
    dmppi_temperature=0.2,
    use_neural_updater=True,
):
    del process
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    configure_dmppi_controller(
        policy_file_path,
        policy_config,
        device,
        dmppi_num_samples,
        dmppi_temperature,
        use_neural_updater,
    )

    print(f"Loaded policy checkpoint: {policy_file_path}")
    print(
        "Policy architecture: "
        f"NeuralGRFDistributionPolicy(num_layers={policy_config.get('num_layers', 5)}, "
        f"hidden_dim={policy_config.get('hidden_dim', 256)}, "
        f"activation='{policy_config.get('activation', 'gelu')}')"
    )
    print(f"Controller mode: {cfg.mpc_params['type']}")
    print(
        "DMPPI settings: "
        f"dmppi_samples={cfg.mpc_params['dmppi_samples']}, "
        f"dmppi_temperature={cfg.mpc_params['dmppi_temperature']}"
    )
    print(f"Neural updater: {'enabled' if use_neural_updater else 'disabled (plain DMPPI correction)'}")

    robot_name = qpympc_cfg.robot
    hip_height = qpympc_cfg.hip_height
    scene_name = qpympc_cfg.simulation_params["scene"]
    simulation_dt = qpympc_cfg.simulation_params["dt"]

    env = QuadrupedEnv(
        robot=robot_name,
        scene=scene_name,
        sim_dt=simulation_dt,
        ref_base_lin_vel=np.asarray(ref_base_lin_vel) * hip_height,
        ref_base_ang_vel=ref_base_ang_vel,
        ground_friction_coeff=friction_coeff,
        base_vel_command_type=base_vel_command_type,
        state_obs_names=tuple(),
    )
    env.mjModel.opt.gravity[2] = -qpympc_cfg.gravity_constant

    if qpympc_cfg.qpos0_js is not None:
        env.mjModel.qpos0 = np.concatenate((env.mjModel.qpos0[:7], qpympc_cfg.qpos0_js))

    env.reset(random=False)
    if render:
        env.render()
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    tau_soft_limits_scalar = 0.9
    tau_limits = LegsAttr(
        FL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FL] * tau_soft_limits_scalar,
        FR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FR] * tau_soft_limits_scalar,
        RL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RL] * tau_soft_limits_scalar,
        RR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RR] * tau_soft_limits_scalar,
    )

    feet_traj_geom_ids, feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
    legs_order = ("FL", "FR", "RL", "RR")
    goal_geom_id = -1

    wrapper = QuadrupedPyMPC_Wrapper(
        initial_feet_pos=env.feet_pos,
        legs_order=legs_order,
        feet_geom_id=env._feet_geom_id,
    )
    print(f"Controller device: {wrapper.srbd_controller_interface.controller.device}")

    goal_base_pos = None if goal_base_pos is None else np.asarray(goal_base_pos, dtype=float)
    if goal_base_pos is not None and goal_base_pos.shape not in {(2,), (3,)}:
        raise ValueError("goal_base_pos must be a 2D or 3D position.")

    if qpympc_cfg.simulation_params["visual_foothold_adaptation"] != "blind":
        from gym_quadruped.sensors.heightmap import HeightMap

        resolution_heightmap = 0.04
        num_rows_heightmap = 7
        num_cols_heightmap = 7
        heightmaps = LegsAttr(
            FL=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=env.mjModel, mj_data=env.mjData),
            FR=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=env.mjModel, mj_data=env.mjData),
            RL=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=env.mjModel, mj_data=env.mjData),
            RR=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=env.mjModel, mj_data=env.mjData),
        )
    else:
        heightmaps = None

    render_freq = 30
    steps_per_episode = int(num_seconds_per_episode // simulation_dt)
    control_update_interval = max(1, round(1 / (wrapper.mpc_frequency * simulation_dt)))
    last_render_time = time.time()
    stats = {
        "goal": 0,
        "terminated": 0,
        "truncated": 0,
        "timeout": 0,
    }
    lap_times = []
    episode_avg_compute_times = []

    if goal_set is not None:
        num_episodes = len(goal_set["goals"])
    elif num_episodes is None:
        num_episodes = 1

    for episode_num in range(num_episodes):
        env.reset(random=False)
        reset_episode_base_position(env, goal_set, episode_num)
        wrapper.reset(initial_feet_pos=env.feet_pos(frame="world"))
        tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
        if render and goal_geom_id != -1:
            goal_geom_id = render_sphere(
                viewer=env.viewer,
                position=np.array([0.0, 0.0, -10.0], dtype=float),
                diameter=0.001,
                color=[0, 0, 0, 0.0],
                geom_id=goal_geom_id,
            )

        if goal_set is not None:
            sampled_goal = np.asarray(goal_set["goals"][min(episode_num, len(goal_set["goals"]) - 1)], dtype=float)
        elif random_goals:
            sampled_goal = sample_goal(qpympc_cfg, rng, np.asarray(env.base_pos, dtype=float))
        else:
            sampled_goal = goal_base_pos

        episode_outcome = "timeout"
        episode_compute_time_sum = 0.0
        episode_compute_steps = 0
        for _ in tqdm(range(steps_per_episode), desc=f"Ep:{episode_num:d}-steps:", total=steps_per_episode):
            feet_pos = env.feet_pos(frame="world")
            feet_vel = env.feet_vel(frame="world")
            hip_pos = env.hip_positions(frame="world")
            base_lin_vel = env.base_lin_vel(frame="world")
            base_ang_vel = env.base_ang_vel(frame="base")
            base_ori_euler_xyz = env.base_ori_euler_xyz
            base_pos = copy.deepcopy(env.base_pos)
            com_pos = copy.deepcopy(env.com)

            if sampled_goal is not None:
                goal_position_world = sampled_goal if sampled_goal.shape == (3,) else np.array(
                    [sampled_goal[0], sampled_goal[1], base_pos[2]],
                    dtype=float,
                )
                position_error = goal_position_world - base_pos
                position_error[2] = 0.0
                distance_to_goal = np.linalg.norm(position_error[:2])

                if distance_to_goal <= goal_position_tolerance:
                    ref_base_lin_vel_cmd = np.zeros(3)
                    ref_base_ang_vel_cmd = np.zeros(3)
                else:
                    ref_base_lin_vel_cmd = goal_kp * position_error
                    ref_base_lin_vel_cmd[2] = 0.0
                    planar_speed = np.linalg.norm(ref_base_lin_vel_cmd[:2])
                    if planar_speed > goal_max_lin_vel:
                        ref_base_lin_vel_cmd[:2] *= goal_max_lin_vel / planar_speed
                    ref_base_ang_vel_cmd = np.zeros(3)
            else:
                ref_base_lin_vel_cmd, ref_base_ang_vel_cmd = env.target_base_vel()
                distance_to_goal = None
                goal_position_world = None

            if qpympc_cfg.simulation_params["use_inertia_recomputation"]:
                inertia = env.get_base_inertia().flatten()
            else:
                inertia = qpympc_cfg.inertia.flatten()

            qpos, qvel = env.mjData.qpos, env.mjData.qvel
            legs_qvel_idx = env.legs_qvel_idx
            legs_qpos_idx = env.legs_qpos_idx
            joints_pos = LegsAttr(
                FL=qpos[legs_qpos_idx.FL].copy(),
                FR=qpos[legs_qpos_idx.FR].copy(),
                RL=qpos[legs_qpos_idx.RL].copy(),
                RR=qpos[legs_qpos_idx.RR].copy(),
            )
            legs_mass_matrix = env.legs_mass_matrix
            legs_qfrc_bias = env.legs_qfrc_bias
            legs_qfrc_passive = env.legs_qfrc_passive
            feet_jac = env.feet_jacobians(frame="world", return_rot_jac=False)
            feet_jac_dot = env.feet_jacobians_dot(frame="world", return_rot_jac=False)

            is_control_update_step = (env.step_num % control_update_interval) == 0
            if is_control_update_step:
                compute_start_time = time.perf_counter()
                tau = wrapper.compute_actions(
                    com_pos,
                    base_pos,
                    base_lin_vel,
                    base_ori_euler_xyz,
                    base_ang_vel,
                    feet_pos,
                    hip_pos,
                    joints_pos,
                    heightmaps,
                    legs_order,
                    simulation_dt,
                    ref_base_lin_vel_cmd,
                    ref_base_ang_vel_cmd,
                    env.step_num,
                    qpos,
                    qvel,
                    feet_jac,
                    feet_jac_dot,
                    feet_vel,
                    legs_qfrc_passive,
                    legs_qfrc_bias,
                    legs_mass_matrix,
                    legs_qpos_idx,
                    legs_qvel_idx,
                    tau,
                    inertia,
                    env.mjData.contact,
                )
                block_tree_until_ready(tau)
                episode_compute_time_sum += time.perf_counter() - compute_start_time
                episode_compute_steps += 1
            else:
                tau = wrapper.compute_actions(
                    com_pos,
                    base_pos,
                    base_lin_vel,
                    base_ori_euler_xyz,
                    base_ang_vel,
                    feet_pos,
                    hip_pos,
                    joints_pos,
                    heightmaps,
                    legs_order,
                    simulation_dt,
                    ref_base_lin_vel_cmd,
                    ref_base_ang_vel_cmd,
                    env.step_num,
                    qpos,
                    qvel,
                    feet_jac,
                    feet_jac_dot,
                    feet_vel,
                    legs_qfrc_passive,
                    legs_qfrc_bias,
                    legs_mass_matrix,
                    legs_qpos_idx,
                    legs_qvel_idx,
                    tau,
                    inertia,
                    env.mjData.contact,
                )

            for leg in legs_order:
                tau_min, tau_max = tau_limits[leg][:, 0], tau_limits[leg][:, 1]
                tau[leg] = np.clip(tau[leg], tau_min, tau_max)

            action = np.zeros(env.mjModel.nu)
            action[env.legs_tau_idx.FL] = tau.FL
            action[env.legs_tau_idx.FR] = tau.FR
            action[env.legs_tau_idx.RL] = tau.RL
            action[env.legs_tau_idx.RR] = tau.RR

            _, _, is_terminated, is_truncated, _ = env.step(action=action)

            if render and (time.time() - last_render_time > 1.0 / render_freq or env.step_num == 1):
                _, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)

                feet_traj_geom_ids = plot_swing_mujoco(
                    viewer=env.viewer,
                    swing_traj_controller=wrapper.wb_interface.stc,
                    swing_period=wrapper.wb_interface.stc.swing_period,
                    swing_time=LegsAttr(
                        FL=wrapper.wb_interface.stc.swing_time[0],
                        FR=wrapper.wb_interface.stc.swing_time[1],
                        RL=wrapper.wb_interface.stc.swing_time[2],
                        RR=wrapper.wb_interface.stc.swing_time[3],
                    ),
                    lift_off_positions=wrapper.wb_interface.frg.lift_off_positions,
                    nmpc_footholds=wrapper.nmpc_footholds,
                    ref_feet_pos=wrapper.nmpc_footholds,
                    early_stance_detector=wrapper.wb_interface.esd,
                    geom_ids=feet_traj_geom_ids,
                )

                for leg_name in legs_order:
                    feet_GRF_geom_ids[leg_name] = render_vector(
                        env.viewer,
                        vector=feet_GRF[leg_name],
                        pos=feet_pos[leg_name],
                        scale=np.linalg.norm(feet_GRF[leg_name]) * 0.005,
                        color=np.array([0, 1, 0, 0.5]),
                        geom_id=feet_GRF_geom_ids[leg_name],
                    )

                if goal_position_world is not None:
                    goal_geom_id = render_sphere(
                        viewer=env.viewer,
                        position=goal_position_world,
                        diameter=0.18,
                        color=[1, 0, 0, 0.75],
                        geom_id=goal_geom_id,
                    )

                env.render()
                last_render_time = time.time()

            reached_goal = sampled_goal is not None and distance_to_goal is not None and distance_to_goal <= goal_position_tolerance
            if reached_goal:
                episode_outcome = "goal"
                lap_times.append(float(env.simulation_time))
                break

            if is_terminated or is_truncated or env.step_num >= steps_per_episode:
                if is_terminated:
                    episode_outcome = "terminated"
                elif is_truncated:
                    episode_outcome = "truncated"
                else:
                    episode_outcome = "timeout"
                break

        stats[episode_outcome] += 1
        avg_compute_time_ms = 1e3 * (
            episode_compute_time_sum / episode_compute_steps
            if episode_compute_steps > 0
            else 0.0
        )
        episode_avg_compute_times.append(avg_compute_time_ms)

    env.close()
    total = sum(stats.values())
    success_rate = stats["goal"] / total if total > 0 else 0.0
    summary = {
        "goals_reached": stats["goal"],
        "timeout": stats["timeout"],
        "truncated": stats["truncated"],
        "terminated": stats["terminated"],
        "success_rate": success_rate,
        "lap_times": np.asarray(lap_times, dtype=float),
        "episode_avg_compute_times_ms": np.asarray(episode_avg_compute_times, dtype=float),
        "mean_compute_time_ms": float(np.mean(episode_avg_compute_times)) if episode_avg_compute_times else 0.0,
    }
    print("Evaluation summary:")
    print(f"  total_episodes: {total}")
    print(f"  goals_reached: {summary['goals_reached']}")
    print(f"  terminated: {summary['terminated']}")
    print(f"  truncated: {summary['truncated']}")
    print(f"  timeout: {summary['timeout']}")
    print(f"  success_rate: {summary['success_rate']:.3f}")
    print(f"  lap_times: {summary['lap_times']}")
    print(f"  episode_avg_compute_times_ms: {summary['episode_avg_compute_times_ms']}")
    print(f"  mean_compute_time_ms: {summary['mean_compute_time_ms']:.3f} ms")
    return summary


def main():
    default_policy_file = REPO_ROOT / "training" / "neural_dmppi_constrained_policy.pkl"
    default_config_path = REPO_ROOT / "training" / "dmppi_neural_config.yaml"
    parser = argparse.ArgumentParser(description="Test a trained neural-updater DMPPI policy through QuadrupedPyMPC_Wrapper.")
    parser.add_argument("--policy_file", type=pathlib.Path, default=default_policy_file, help="Path to the trained neural DMPPI checkpoint.")
    parser.add_argument("--config", type=pathlib.Path, default=default_config_path, help="Path to the neural DMPPI YAML config.")
    parser.add_argument("--device", type=str, default="gpu", choices=("cpu", "gpu"), help="JAX device preference for the DMPPI policy.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--goal-x", type=float, default=1.0, help="Goal x position in world frame.")
    parser.add_argument("--goal-y", type=float, default=0.0, help="Goal y position in world frame.")
    parser.add_argument("--random-goals", action="store_true", help="Sample a fresh random goal per episode, using the same seeded sampler as dataset generation.")
    parser.add_argument("--goal-file", type=pathlib.Path, default=REPO_ROOT / "training" / "test_goals.npz", help="Optional .npz file with pre-generated goal_base_positions.")
    parser.add_argument(
        "--goal-max-lin-vel",
        type=float,
        default=0.2,
        help="Maximum planar linear velocity command used when driving toward the goal.",
    )
    parser.add_argument("--dmppi-num-samples", type=int, default=None, help="Override the number of one-step DMPPI force samples.")
    parser.add_argument("--dmppi-temperature", type=float, default=None, help="Override the softmax temperature for DMPPI weighting.")
    parser.add_argument(
        "--plain-dmppi-correction",
        action="store_true",
        help="Disable the neural updater and use plain analytic DMPPI correction with the learned proposal.",
    )
    parser.add_argument("--no-render", action="store_true", help="Disable Mujoco rendering.")
    args = parser.parse_args()

    policy_file_path = resolve_repo_path(args.policy_file)
    config_path = resolve_repo_path(args.config)
    goal_file_path = resolve_repo_path(args.goal_file) if args.goal_file is not None else None

    config = load_config(config_path)
    policy_config = normalize_policy_config(dict(config.get("policy", {})))
    dmppi_config = normalize_dmppi_config(dict(config.get("dmppi", {})))
    goal_base_pos = np.array([args.goal_x, args.goal_y, cfg.simulation_params["ref_z"]], dtype=float)
    goal_set = load_goal_set(goal_file_path) if goal_file_path is not None and goal_file_path.exists() else None

    summary = run_dmppi_test(
        policy_file_path=policy_file_path,
        policy_config=policy_config,
        qpympc_cfg=cfg,
        goal_base_pos=goal_base_pos,
        goal_set=goal_set,
        random_goals=args.random_goals,
        goal_max_lin_vel=args.goal_max_lin_vel,
        seed=args.seed,
        render=not args.no_render,
        device=args.device,
        dmppi_num_samples=(
            args.dmppi_num_samples
            if args.dmppi_num_samples is not None
            else dmppi_config.get("dmppi_samples", 64)
        ),
        dmppi_temperature=(
            args.dmppi_temperature
            if args.dmppi_temperature is not None
            else dmppi_config.get("dmppi_temperature", 0.2)
        ),
        use_neural_updater=not args.plain_dmppi_correction,
    )
    save_summary_json(pathlib.Path(__file__).name, summary, args.seed, goal_file_path if goal_set is not None else None)


if __name__ == "__main__":
    main()
