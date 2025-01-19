import numpy as np
import os
from src.sim import MjcSim, ProgressCallback
from utils.sim_args import arg_parser
from utils.recorder import Recorder
from typing import Callable
from utils.xml_handler import MJCFHandler

class Duplo(MjcSim):
    def __init__(self, config: dict) -> None:
        """Initialize the Duplo simulation environment."""
        scene_path = f"{config['robot_dir']}/duplo_ballfeet_mjcf/scene.xml"
        self.camera_params = {
            'tracking': "leg_1",
            'distance': 5,
            'xyaxis': [-1, 0, 0, 0, 0, 1],
        }

        self.mjcf_handler = MJCFHandler(scene_path)
        self.mjcf_handler.update_mass()
        self.mjcf_handler.export_xml_scene()
        scene_path = self.mjcf_handler.new_scene_path

        super().__init__(scene_path, config)
        self.ctrl_joint_names = ['hip']
        n_ctrl_joints = self.setup_ctrl_joints()
        print(f"Number of control joints: {n_ctrl_joints}")

        self.dof_addr = self.ctrl_dof_addrs[0]
        self.action = None

        self.leg_amp_deg = 35
        self.leg_amp_rad = np.deg2rad(self.leg_amp_deg)
        self.pend_len = 0.63 # default from a while backs

        self.step_sim() # Take the first sim step to initialize the data

    def pendulum_length(self) -> tuple[float, float]:
        """Calculate the length and z offset of the pendulum."""
        hip_pos = self.data.joint(self.ctrl_joint_names[0]).xanchor
        com_pos = self.mass_center()
        pedulum_length = np.linalg.norm(hip_pos - com_pos)
        pendulum_z = hip_pos[2] - com_pos[2]
        return pedulum_length, pendulum_z

    def calculate_sine_ctrl(self, waittime: float=1.0, b:float =1.0) -> None:
        """Calculate the sine wave control signal for the hip joint."""
        if self.data.time < waittime: return
        # b defines how sharp a sine wave is, higher the sharper
        wave = np.sin(self.hip_omega * (self.data.time-waittime))
        wave_val = np.sqrt((1 + b**2) / (1 + (b**2) * wave**2))*wave
        self.action = self.leg_amp_rad * wave_val

    def apply_ctrl(self) -> None:
        """Apply the calculated control signal to the hip joint."""
        if self.action is not None:
            self.data.actuator("hip_joint_act").ctrl = self.action

    def data_log(self) -> None:
        """Log the data from the simulation."""
        self.actuator_setpoints = self.data.actuator("hip_joint_act").ctrl[0]
        self.actuator_actual_pos = self.data.qpos[7]
        self.actuator_torque = self.data.qfrc_actuator[self.dof_addr]

    def run_sim(self, callbacks: dict[str, Callable]=None) -> None:
        """Run the simulation for the specified time."""
        self.pend_len = self.pendulum_length()[0]
        self.hip_omega = np.sqrt(9.81 / self.pend_len)

        print(f"hip freq: {self.hip_omega/(2*np.pi)}")

        loop = range(int(self.simtime // self.model.opt.timestep))
        for _ in loop:
            self.calculate_sine_ctrl()
            self.apply_ctrl()
            self.step_sim()
            self.data_log()

            if callbacks:
                for name, func in callbacks.items():
                    func(self)  # Call function dynamically

if __name__ == "__main__":
    args = arg_parser("duplo")

    # Define the variables and their properties
    plot_attributes = {
        "actuator_actual_pos"   : {"title": "Joint Angle", "unit": "Rad"},
        "actuator_torque"       : {"title": "Joint Torque", "unit": "Nm"},
        "actuator_setpoints"    : {"title": "Joint Setpoint", "unit": "Rad"},
        "time"                  : {"title": "Time", "unit": "s"},  
    }

    # Define the structure of the plots
    plot_structure = [
        ["time", "actuator_actual_pos", "actuator_setpoints"],  # Subplot 1: X = time, Y = angle & setpoint
        ["time", "actuator_torque"],  # Subplot 2: X = time, Y = torque
        ["actuator_actual_pos", "actuator_torque"],  # Subplot 3: X = angle, Y = torque
    ]

    duplo = Duplo(args)
    progress_cb = ProgressCallback(args["sim_time"])  # Initialize progress tracker
    callbacks_dict = {"progress_bar" : progress_cb.update}

    if args["record"]:
        recorder = Recorder(args['video_fps'], plot_attributes, plot_structure)
        callbacks_dict["record_frame"] = recorder.record_frame
        callbacks_dict["record_plot_data"] = recorder.record_plot_data

    duplo.run_sim(callbacks=callbacks_dict)
    duplo.close()

    if args["record"]:
        v_dir = f"{args['video_dir']}/{args['name']}"
        os.makedirs(v_dir, exist_ok=True)
        recorder.generate_plot_video(output_path=f"{v_dir}/live_plot.mp4")
        recorder.generate_robot_video(output_path=f"{v_dir}/robot_walking.mp4")
        recorder.stack_video_frames(recorder.plot_frames, 
                                    recorder.robot_frames,  
                                    output_path=f"{v_dir}/padded.mp4",
                                    pad=True)