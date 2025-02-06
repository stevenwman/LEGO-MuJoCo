from collections import deque
import numpy as np
import mujoco
import os
import pickle as pkl
from src.sim import MjcSim, ProgressCallback
from utils.sim_args import arg_parser
from utils.recorder import Recorder
from typing import Callable, Any
import yaml

class Duplo(MjcSim):
    def __init__(self, config: dict) -> None:
        """Initialize the Duplo simulation environment."""
        # scene_path = f"{config['robot_dir']}/duplo_ballfeet_mjcf/scene_motor.xml"
        scene_path = f"{config['robot_dir']}/duplo_hip_offset_mjcf/scene_hip_offset.xml"
        self.camera_params = {
            # 'tracking': "leg_1",
            'tracking': "leg_h",
            'distance': 5,
            'xyaxis': [1, 0, 0, 0, 0, 1],
        }

        new_scene_path = self.update_xml(scene_path, config['design_params'])
        super().__init__(new_scene_path, config)
        self.get_hip_idx()
        self.init_ctrl_params(config["ctrl_dict"])
        self.step_sim() # Take the first sim step to initialize the data

    def get_hip_idx(self) -> None:
        """Get the joint ID and qpos index for the hip joint."""
        self.ctrl_joint_names = ['hip']
        n_ctrl_joints = self.setup_ctrl_joints()
        print(f"Number of control joints: {n_ctrl_joints}")
        self.hip_qpos_idx = self.model.jnt_qposadr[self.ctrl_joint_ids[0]] # qpos index for the hip joint
        self.hip_dof_idx = self.ctrl_dof_addrs[0] # dof index for the hip joint (for qfrc)
        self.hip_qvel_idx = self.hip_dof_idx
        self.action = None

    def init_ctrl_params(self, ctrl_dict: dict[str, Any]={}) -> None: 
        # Default values
        self.Kp = 0
        self.Kd = 0
        self.leg_amp_deg = 0
        self.hip_omega = None
        # self.pend_len = 0.63 # default from a while back
        for k,v in ctrl_dict.items(): 
            setattr(self, k, v)
            print(f"{k} set to {getattr(self, k)}.")

    @property
    def leg_amp_rad(self) -> float:
        """Calculate the angular frequency of the hip joint."""
        return np.deg2rad(self.leg_amp_deg) 

    def pendulum_length(self) -> tuple[float, float]:
        """Calculate the length and z offset of the pendulum."""
        hip_pos = self.data.joint(self.ctrl_joint_names[0]).xanchor
        com_pos = self.mass_center()
        pedulum_length = np.linalg.norm(hip_pos - com_pos)
        pendulum_z = hip_pos[2] - com_pos[2]
        return pedulum_length, pendulum_z

    def calculate_sine_reference(self, waittime: float=1.0, b:float=1.0) -> None:
        """Calculate the sine wave control signal for the hip joint."""
        # b defines how sharp a sine wave is, higher the sharper
        wave = np.sin(self.hip_omega * (self.data.time-waittime))
        wave_val = np.sqrt((1 + b**2) / (1 + (b**2) * wave**2))*wave
        self.reference = self.leg_amp_rad * wave_val
        if self.data.time < waittime: self.reference = 0

    def calculate_cosine_reference(self, 
                                   wait_time: float=1.0, 
                                   init_time: float=1.0,
                                   b:float=1.0) -> None:
        """Calculate the cosine wave control signal for the hip joint."""
        # b defines how sharp a cosine wave is, higher the sharper
        wave = np.cos(self.hip_omega * (self.data.time-wait_time))
        wave_val = np.sqrt((1 + b**2) / (1 + (b**2) * wave**2))*wave
        self.reference = self.leg_amp_rad * wave_val
        if self.data.time < init_time + wait_time: self.reference = self.leg_amp_rad
        if self.data.time < wait_time: self.reference = 0
        
    def apply_ctrl(self) -> None:
        """Apply the calculated control signal to the hip joint."""
        self.data.actuator("hip_joint_act").ctrl = self.action

    def calculate_mujoco_position_ctrl(self) -> None:
        """Calculate the position control signal for the hip joint."""
        self.action = self.reference

    def calculate_pd_ctrl(self, hist_window: int=10) -> None:
        """Calculate the PID control signal for the hip joint."""
        self.calculate_sine_reference()

        if self.action is None: # Initialize the control signal
            # start a queue 
            self.p_hist = deque([self.data.qpos[self.hip_qpos_idx]], maxlen=hist_window)  # Fixed-size queue
            self.p_ref_hist = deque([self.reference], maxlen=hist_window)  # Fixed-size queue
            self.action = 0
            return
            
        # update the queue
        self.p_hist.append(self.data.qpos[self.hip_qpos_idx])
        self.p_ref_hist.append(self.reference)
        p_err_hist = np.array(self.p_ref_hist) - np.array(self.p_hist)

        # average the derivative over the entire queue
        p_err = p_err_hist[-1] # most recent entry is at the end
        p_err_d = np.mean(np.diff(p_err_hist) / self.model.opt.timestep)

        # calculate the control signal
        self.action = self.Kp * p_err + self.Kd * p_err_d

    def data_log(self) -> None:
        """Log the data from the simulation."""
        self.actuator_setpoints = self.reference
        self.actuator_actual_pos = self.data.qpos[self.hip_qpos_idx]
        self.actuator_torque = self.data.qfrc_actuator[self.hip_dof_idx]
        self.applied_torque = self.data.qfrc_applied[self.hip_dof_idx]
        self.actuator_speed = self.data.qvel[self.hip_dof_idx]

    def record_contact_points(self) -> None:
        for k,v in self.contact_bodies.items():
            body_id = v['body_id']
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                geom1_body = self.model.geom_bodyid[contact.geom1]
                geom2_body = self.model.geom_bodyid[contact.geom2]
                if geom1_body == body_id or geom2_body == body_id:
                    # Get world-frame contact position
                    pos_world = contact.pos
                    
                    # Body's current pose
                    body_pos = self.data.body(body_id).xpos
                    body_quat = self.data.body(body_id).xquat

                    mesh_pos = v['pos']
                    mesh_quat = v['quat']
                    mesh_offset = v['mesh_offset']
                    
                    # Convert quaternion to rotation matrix
                    R_body = np.zeros(9)
                    mujoco.mju_quat2Mat(R_body, body_quat)
                    R_body = R_body.reshape(3, 3)

                    R_mesh = np.zeros(9)
                    mujoco.mju_quat2Mat(R_mesh, mesh_quat)
                    R_mesh = R_mesh.reshape(3, 3)
                    
                    # Transform to body frame: p_body = R^T (p_world - body_pos)
                    p_body = R_body.T @ (pos_world - body_pos)
                    p_mesh = R_mesh.T @ (p_body - mesh_pos)

                    timed_p_mesh = np.hstack([self.data.time, p_mesh])

                    if k in self.con_dict.keys():
                        # vstack
                        self.con_dict[k]['t_coords'] = np.vstack([self.con_dict[k]['t_coords'], 
                                                                      timed_p_mesh.copy()])
                    else:
                        self.con_dict[k] = {}
                        self.con_dict[k]['t_coords'] = [timed_p_mesh.copy()]
                        self.con_dict[k]['pos'] = mesh_pos
                        self.con_dict[k]['quat'] = mesh_quat
                        self.con_dict[k]['mesh'] = v['mesh']
                        self.con_dict[k]['mesh_offset'] = mesh_offset


    def run_sim(self, callbacks: dict[str, Callable]=None) -> None:
        """Run the simulation for the specified time."""
        if self.hip_omega is None:
            self.pend_len = self.pendulum_length()[0]
            self.hip_omega = np.sqrt(9.81 / self.pend_len)

        print(f"hip freq: {self.hip_omega/(2*np.pi)}")
        loop = range(int(self.simtime // self.model.opt.timestep))

        quats = []

        self.contact_bodies = {
            'leg_v': {
                'pos': np.array([0.14908, -0.3625, -0.0124026]),
                'mesh_offset' : np.array([-0.265, 0, 0]),
                'quat': np.array([0, 0, -0.707107, 0.707107]),
                'mesh': 'part_1'
                },
            'leg_v_2': {
                'pos': np.array([-0.14908, -0.3625, -0.0124026]),
                'mesh_offset' : np.array([0.265, 0, 0]),
                'quat': np.array([0.707107, 0.707107, 0, 0]),
                'mesh': 'part_1'
                }
            }

        for k in self.contact_bodies.keys():
            self.contact_bodies[k]['body_id'] = mujoco.mj_name2id(self.model,
                                                                  mujoco.mjtObj.mjOBJ_BODY, 
                                                                  k)
        self.con_dict: dict[str,dict[str,list|np.ndarray|str]] = {}

        for _ in loop:
            self.calculate_sine_reference()
            self.calculate_pd_ctrl()    
            self.apply_ctrl()
            self.step_sim()
            self.data_log()

            quats.append(self.data.qpos[3:7].copy())
            # print(f"quats: {quats[-1]}")

            self.record_contact_points()

            if callbacks:
                for name, func in callbacks.items():
                    func(self)  # Call function dynamically

        mean_quat = np.mean(quats, axis=0)
        # print(quats)
        print(f"mean quat: {mean_quat}")
        print(f"last quat: {quats[-1]}")
        
def main():
    args = arg_parser("Duplo Sim Args")

    # Define the variables and their properties
    plot_attributes = {
        "actuator_actual_pos"   : {"title": "Joint Angle", "unit": "Rad"},
        "actuator_torque"       : {"title": "Joint Torque", "unit": "Nm"},
        "actuator_setpoints"    : {"title": "Joint Setpoint", "unit": "Rad"},
        "actuator_speed"        : {"title": "Joint Speed", "unit": "Rad/s"},
        "time"                  : {"title": "Time", "unit": "s"},  
    }

    # Define the structure of the plots
    plot_structure = [
        ["time", "actuator_actual_pos", "actuator_setpoints"],  # Subplot 1: X = time, Y = angle & setpoint
        ["time", "actuator_torque"],  # Subplot 2: X = time, Y = torque
        ["actuator_actual_pos", "actuator_torque"],  # Subplot 3: X = angle, Y = torque
        # ["actuator_speed", "actuator_torque"],
    ]

    # dictionary of control parameters
    args['ctrl_dict'] = {
        'Kp': 15,
        'Kd': 12,
        'leg_amp_deg': 35,
        # 'leg_amp_deg': 0,
        # 'hip_omega': 0.6 * 2 * np.pi,
    }

    args['design_params'] = {
        'body_pos_offset': {'leg_v' : [-0.0, 0, 0], 
                            'leg_v_2' : [-0.0, 0, 0]},
        # 'body_quat' : {'motor' : [1, 0, 0, 0]},
        'body_quat' : {'motor' : [9.91243386e-01, 1.22932829e-01, -3.19655647e-05, 2.30992995e-03]},
        # 'body_quat' : {'motor' : [-8.60502024e-04, -6.59361173e-07, 4.95928642e-02, 9.98769146e-01]},
        'mesh_scale' : {'part_1' : [1.3, 1, 1]}
    }

    robot = Duplo(args)
    progress_cb = ProgressCallback(args['sim_time'])  # Initialize progress tracker
    callbacks_dict = {
        "progress_bar" : progress_cb.update
        }

    if args["record"]:
        recorder = Recorder(args['video_fps'], plot_attributes, plot_structure)
        callbacks_dict["record_frame"] = recorder.record_frame
        callbacks_dict["record_plot_data"] = recorder.record_plot_data

    robot.run_sim(callbacks=callbacks_dict)

    if args["record"]:
        v_dir = f"{args['video_dir']}/{robot.__class__.__name__}/{args['name']}"
        os.makedirs(v_dir, exist_ok=True)
        recorder.generate_plot_video(output_path=f"{v_dir}/live_plot.mp4")
        recorder.generate_robot_video(output_path=f"{v_dir}/robot_walking.mp4")
        recorder.stack_video_frames(recorder.plot_frames, 
                                    recorder.robot_frames,
                                    output_path=f"{v_dir}/combined.mp4")
    # Save to file

    with open("contact_dict.pkl", "wb") as f:
        pkl.dump(robot.con_dict, f)

    robot.close()
        
if __name__ == "__main__":
    # from pyinstrument import Profiler

    # profiler = Profiler()
    # profiler.start()

    main()

    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))