from collections import deque
import numpy as np
import os
from src.sim import MjcSim, ProgressCallback
import time
from typing import Callable, Any
from utils.recorder import Recorder
from utils.sim_args import arg_parser

class Zippy(MjcSim):
    def __init__(self, config: dict) -> None:
        """Initialize the Zippy simulation environment."""
        scene_path = f"{config['robot_dir']}/zippy_mjcf/scene_motor.xml"
        self.camera_params = {
            'tracking': "r_leg",
            'distance': 0.15,
            'xyaxis': [1, 0, 0, 0, 0, 1],
        }

        new_scene_path = self.update_xml(scene_path)
        super().__init__(new_scene_path, config)
        
        self.model.opt.enableflags |= 1 << 0  # enable override
        self.model.opt.timestep = 0.001

        # solreflimit="4e-3 1" solimplimit=".95 .99 1e-3"
        # self.model.opt.iterations = 200
        self.model.opt.o_solref[0] = 4e-3
        self.model.opt.o_solref[1] = 1
        self.model.opt.o_solimp[0] = 0.95 
        self.model.opt.o_solimp[1] = 0.99 
        self.model.opt.o_solimp[2] = 1e-3

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
        self.j_damping = 0
        self.v_mean = 0
        # self.pend_len = 0.63 # default from a while back
        for k,v in ctrl_dict.items(): 
            setattr(self, k, v)
            print(f"{k} set to {getattr(self, k)}.")

    def set_j_damping(self, damping: float) -> None:
        """Set the damping of the joint."""
        self.model.dof_damping[self.hip_dof_idx] = damping

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
        # self.reference = self.leg_amp_rad * wave_val
        self.reference = self.leg_amp_rad * (wave_val + self.v_mean)
        if self.data.time < waittime: self.reference = 0

    def direct_ctrl(self) -> None:
        """Calculate the position control signal for the hip joint."""
        # self.action = self.reference
        trq_lim = np.rad2deg(0.01059232775)
        self.action = np.clip(self.reference, a_min=-trq_lim, a_max=trq_lim)

    def apply_ctrl(self) -> None:
        """Apply the calculated control signal to the hip joint."""
        self.data.actuator("hip_joint_act").ctrl = self.action

    def data_log(self) -> None:
        """Log the data from the simulation."""
        self.actuator_setpoints = self.reference
        self.actuator_actual_pos = self.data.qpos[self.hip_qpos_idx]
        self.actuator_torque = self.data.qfrc_actuator[self.hip_dof_idx]
        self.applied_torque = self.data.qfrc_applied[self.hip_dof_idx]
        self.actuator_speed = self.data.qvel[self.hip_dof_idx]

    def run_sim(self, callbacks: dict[str, Callable]=None) -> None:
        """Run the simulation for the specified time."""
        if self.hip_omega is None:
            self.pend_len = self.pendulum_length()[0]
            self.hip_omega = np.sqrt(9.81 / self.pend_len)

        self.set_j_damping(self.j_damping)

        print(f"hip freq: {self.hip_omega/(2*np.pi)}")
        loop = range(int(self.simtime // self.model.opt.timestep))

        for i in loop:
            self.calculate_sine_reference(b=5)
            self.direct_ctrl()
            self.apply_ctrl()
            self.step_sim()
            self.data_log()

            # time.sleep(self.model.opt.timestep * 4)

            if i % 2 == 0:  # Update the progress bar every 2 steps
                print(f"Time: {self.data.time:.5f} / {self.simtime} s", end="\r")
            else:
                print(f"Time: {self.data.time:.5f} s", end=" -> ")

            if callbacks:
                for name, func in callbacks.items():
                    func(self)  # Call function dynamically

            # if not self.viewer.is_running():
            #     return

            
        
def main():
    args = arg_parser("Zippy Sim Args")

    # Define the variables and their properties
    plot_attributes = {
        "actuator_actual_pos"   : {"title": "Joint Angle", "unit": "Rad"},
        "actuator_torque"       : {"title": "Joint Torque", "unit": "Nm"},
        "reference"             : {"title": "Joint Setpoint", "unit": "Rad"},
        "actuator_speed"        : {"title": "Joint Speed", "unit": "Rad/s"},
        "time"                  : {"title": "Time", "unit": "s"},  
    }

    # Define the structure of the plots
    plot_structure = [
        ["time", "actuator_actual_pos"],  # Subplot 1: X = time, Y = angle & setpoint
        ["time", "reference"],  # Subplot 1: X = time, Y = angle & setpoint
        ["time", "actuator_torque"],  # Subplot 2: X = time, Y = torque
        # ["actuator_actual_pos", "actuator_torque"],  # Subplot 3: X = angle, Y = torque
        # ["actuator_speed", "actuator_torque"],
    ]

    for i in np.arange(1,15,0.25):
        print(f"Running simulation for hip freq: {i} Hz")

        # dictionary of control parameters
        args['ctrl_dict'] = {
            # 'Kp': 0.1,
            # 'Kd': 0.00007,
            'leg_amp_deg': - np.rad2deg(0.01059232775) / 2,
            # 'leg_amp_deg': 0,
            # 'hip_omega': 3 * 2 * np.pi,
            'hip_omega': i * 2 * np.pi,
            'j_damping': 0.1,
            'v_mean' : -0.1,
        }

        robot = Zippy(args)
        progress_cb = ProgressCallback(args['sim_time'])  # Initialize progress tracker
        callbacks_dict = {
            # "progress_bar" : progress_cb.update
            }

        if args["record"]:
            recorder = Recorder(args['video_fps'], plot_attributes, plot_structure)
            callbacks_dict["record_frame"] = recorder.record_frame
            # callbacks_dict["record_plot_data"] = recorder.record_plot_data

        robot.run_sim(callbacks=callbacks_dict)

        if args["record"]:
            v_dir = f"{args['video_dir']}/{robot.__class__.__name__}/{args['name']}"
            os.makedirs(v_dir, exist_ok=True)
            # recorder.generate_plot_video(output_path=f"{v_dir}/live_plot.mp4")
            recorder.generate_robot_video(output_path=f"{v_dir}/robot_walking_{i}hz.mp4")
            # recorder.stack_video_frames(recorder.plot_frames, 
            #                             recorder.robot_frames,
            #                             output_path=f"{v_dir}/combined.mp4")
            
        robot.close()
        
def main2():
    args = arg_parser("Zippy Sim Args")

    # Define the variables and their properties
    plot_attributes = {
        "actuator_actual_pos"   : {"title": "Joint Angle", "unit": "Rad"},
        "actuator_torque"       : {"title": "Joint Torque", "unit": "Nm"},
        "reference"             : {"title": "Joint Setpoint", "unit": "Rad"},
        "actuator_speed"        : {"title": "Joint Speed", "unit": "Rad/s"},
        "time"                  : {"title": "Time", "unit": "s"},  
    }

    # Define the structure of the plots
    plot_structure = [
        ["time", "actuator_actual_pos"],  # Subplot 1: X = time, Y = angle & setpoint
        ["time", "reference"],  # Subplot 1: X = time, Y = angle & setpoint
        ["time", "actuator_torque"],  # Subplot 2: X = time, Y = torque
        # ["actuator_actual_pos", "actuator_torque"],  # Subplot 3: X = angle, Y = torque
        # ["actuator_speed", "actuator_torque"],
    ]

    plot_time_range = [0.75,7]

    # dictionary of control parameters
    args['ctrl_dict'] = {
        # 'Kp': 0.1,
        # 'Kd': 0.00007,
        'leg_amp_deg': np.rad2deg(0.01059232775)/2,
        # 'leg_amp_deg': 0,
        # 'hip_omega': 0.1 * 2 * np.pi,
        'hip_omega': 4.6 * 2 * np.pi,
        'j_damping': 0.001,
        # 'v_mean' : -0.01,
    }

    robot = Zippy(args)
    progress_cb = ProgressCallback(args['sim_time'])  # Initialize progress tracker
    callbacks_dict = {
        # "progress_bar" : progress_cb.update
        }

    if args["record"]:
        recorder = Recorder(args['video_fps'], 
                            plot_attributes, 
                            plot_structure, 
                            plot_time_range)
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
        
    robot.close()

if __name__ == "__main__":
    # from pyinstrument import Profiler

    # profiler = Profiler()
    # profiler.start()

    main2()

    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))