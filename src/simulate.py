import argparse
import mujoco
import mujoco.viewer
import numpy as np
import sys
import time
from typing import Callable, Any
import os

os.environ['QT_QPA_PLATFORM'] = "offscreen"
print(f"Using Qt platform: {os.environ.get('QT_QPA_PLATFORM')}")


class MjcSim:
    def __init__(self, model_path: str, config: dict) -> None:
        """ Initialize the Mujoco simulation environment."""
        self.simtime = config['sim_time']
        self.gui = config['gui']
        self.v_fps = config['video_fps']

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.setup_gui()
        self.ctrl_joint_names = [] # names of the joints to control

    def setup_ctrl_joints(self) -> int:
        """Convert actuator joint names to joint ids and dof addresses."""
        self.ctrl_joint_ids = [mujoco.mj_name2id(self.model, 
                                                 mujoco.mjtObj.mjOBJ_JOINT, 
                                                 j_name) for j_name in self.ctrl_joint_names]
        self.ctrl_dof_addrs = [self.model.jnt_dofadr[j_id] for j_id in self.ctrl_joint_ids]
        return len(self.ctrl_joint_ids)

    def setup_gui(self):
        """Setup the GUI for the simulation."""
        self.width, self.height = 1920, 1080
        self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.camera = mujoco.MjvCamera()

        tracking_obj = self.camera_params['tracking']
        # copy over non-tracking camera parameters
        normal_cam_attrs = {"distance", "lookat", "elevation", "azimuth"}
        for attr in normal_cam_attrs & set(self.camera_params.keys()): 
            setattr(self.camera, attr, self.camera_params[attr])

        if tracking_obj is not None:
            # calculate tracking camera parameters
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, tracking_obj)
            xyaxis = self.camera_params['xyaxis']
            z_axis = np.cross(xyaxis[:3], xyaxis[3:])
            azimuth = np.degrees(np.arctan2(z_axis[1], z_axis[0]))
            elevation = np.degrees(np.arctan2(z_axis[2], np.linalg.norm(z_axis[0:2])))

            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = body_id
            self.camera.azimuth = azimuth
            self.camera.elevation = elevation            

        if self.gui:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            for attr in dir(self.camera):
                if not attr.startswith("__"): # copy camera public attributes
                    setattr(self.viewer.cam, attr, getattr(self.camera, attr))

    def get_image(self):
        """Get the current camera image from the simulation."""
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()
    
    def mass_center(self):
        """Calculate the center of mass of the robot."""
        mass = np.expand_dims(self.model.body_mass, axis=1)
        xpos = self.data.xipos
        return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:3].copy()

    def step_sim(self):
        """Step the simulation forward by one time step."""
        mujoco.mj_step(self.model, self.data)
        if self.gui: 
            self.viewer.sync()
            time.sleep(self.model.opt.timestep)
            if not self.viewer.is_running(): sys.exit()

    def close(self):
        """Close the simulation environment."""
        self.renderer.close()
        if self.gui: self.viewer.close()

class Duplo(MjcSim):
    def __init__(self, config: dict) -> None:
        """Initialize the Duplo simulation environment."""
        scene_path = 'duplo_ballfeet_mjcf/scene.xml'
        self.camera_params = {
            'tracking': "leg_1",
            'distance': 5,
            'xyaxis': [-1, 0, 0, 0, 0, 1],
        }
        super().__init__(scene_path, config)
        self.ctrl_joint_names = ['hip']
        n_ctrl_joints = self.setup_ctrl_joints()
        print(f"Number of control joints: {n_ctrl_joints}")

        self.dof_addr = self.ctrl_dof_addrs[0]
        self.action = None

        self.leg_amp_deg = 35
        self.leg_amp_rad = np.deg2rad(self.leg_amp_deg)
        self.pend_len = 0.63

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

        loop = range(int(self.simtime // self.model.opt.timestep))
        for _ in loop:
            self.calculate_sine_ctrl()
            self.apply_ctrl()
            self.step_sim()
            self.data_log()

            if callbacks:
                for name, func in callbacks.items():
                    func(self)  # Call function dynamically


import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from moviepy import ImageSequenceClip, clips_array
import cv2

class Recorder:
    """Class to record and generate videos from simulation data."""
    def __init__(self, 
                 video_fps: float, 
                 plot_attributes: dict[str, dict[str, str]], 
                 plot_structure: list[list[str]]) -> None:
        """
        - plot_attributes: Dict mapping variable names to {title}.
        - plot_structure: List of subplots, each containing [x, y1, y2, ...].
        """
        # Determine the output interval based on the video fps
        self.video_fps = video_fps
        self.output_interval = 1 / video_fps
        self.next_vid_output_time = 0
        self.plt_data = {}
        self.robot_frames = []
        self.plt_attr = plot_attributes  # { "varName": {"title": "plotTitle", "color": (G,B,R), "line_style": "--"} }
        self.plt_struc = plot_structure  # [[xvar, yvar1, yvar2], [xvar2, yvar3], ...]
        # Initialize empty lists for each variable in plot_attributes
        for var in self.plt_attr:
            self.plt_data[var] = []

        # assert time is always a key in the plot_attributes
        assert "time" in self.plt_attr, "Time variable not in plot_attributes!"

        self.color_dict = {
            "white"       : (255, 255, 255), 
            "green"       : (  0, 255,   0), 
            "red"         : (  0,   0, 255), 
            "light_blue"  : (255, 165,   0)
            }
        self.colors = [color for color, gbr in self.color_dict.items()]

    def record_plot_data(self, sim: MjcSim) -> None:
        """Log specified variables."""
        # self.plt_data['time'].append(sim.data.time) # Always log time

        for var in self.plt_attr:
            value = sim.data.time if var == "time" else getattr(sim, var, None)
            assert value is not None, f"Variable {var} not found."
            self.plt_data[var].append(value)

    def record_frame(self, sim: MjcSim) -> None:
        """Record robot frames at specified intervals."""
        if sim.data.time >= self.next_vid_output_time:
            self.robot_frames.append(sim.get_image())
            self.next_vid_output_time += self.output_interval

    def export_clip(self, 
                    frames: list, 
                    vid_type: str, 
                    output_path: str) -> None:
        """Export video from frames."""
        clip = ImageSequenceClip(frames, fps=self.video_fps)
        from proglog import TqdmProgressBarLogger
        custom_logger = TqdmProgressBarLogger(logged_bars='all', print_messages=False)
        clip.write_videofile(output_path, 
                             codec="h264_nvenc",  # NVIDIA GPU encoding (for NVENC-compatible GPUs)
                             preset="fast",       # Adjust for speed vs. quality
                             logger=custom_logger)

        print(f"{vid_type} Video saved to {output_path}")

    def generate_robot_video(self, output_path: str) -> str:
        """Create a video from stored robot frames."""
        assert self.robot_frames is not [], "No frames recorded!"
        self.export_clip(self.robot_frames, "Scene", output_path)
        return output_path

    def setup_plot(self) -> None:
        """Setup the plot format for the recorded data."""
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="Live Plot Video")

        plot_w = 1080 # Width of the plot
        plot_hs = [] # Heights of the plots

        self.plots_and_curves = []
        # Iterate over plot structure to create subplots
        for row, subplot_vars in enumerate(self.plt_struc):
            x = subplot_vars[0]  # First variable is the x-axis
            ys = subplot_vars[1:]  # Remaining variables are y-axes
            plot_title_list = [self.plt_attr[y]['title'] for y in ys]
            plot = self.win.addPlot(row=row, col=0, title=", ".join(plot_title_list)) # Plots stacked vertically
        
            # Create appropriate x/y labels for the plot
            first_y_unit = self.plt_attr[ys[0]]['unit']
            if len(ys) == 1:
                ylabel = (f"{self.plt_attr[ys[0]]['title']}"
                          f"({self.plt_attr[ys[0]]['unit']})")
            elif all(self.plt_attr[y]['unit'] == first_y_unit for y in ys):
                ylabel = (f"Values ({self.plt_attr[ys[0]]['unit']})")

            xlabel = (f"{self.plt_attr[x]['title']}"
                      f"({self.plt_attr[x]['unit']})")
            plot.setLabels(left=ylabel, bottom=xlabel)
            plot.showGrid(x=True, y=True, alpha=0.5)
            plot.addLegend(offset=(10, 10))

            # Square aspect ratio for non-time plots, else 2:1
            plot_h = plot_w // 2 if self.plt_attr[x]['unit'] == 's' else plot_w
            plot.setPreferredSize(plot_w, plot_h)
            plot_hs.append(plot_h)

            curves = []
            for idx, y in enumerate(ys):
                attrs = self.plt_attr[y]      
                curve_color = self.color_dict[self.colors[idx % len(self.colors)]] # Cycle through colors          
                curve = plot.plot(pen=pg.mkPen(color=curve_color, width=2), name=attrs['title'])
                curves.append((curve, y))
            self.plots_and_curves.append((plot, curves, x))

        for (plot, curves, x) in self.plots_and_curves:
            xmin, xmax = min(self.plt_data[x]), max(self.plt_data[x])
            ymin = min(min(self.plt_data[y]) for _, y in curves)
            ymax = max(max(self.plt_data[y]) for _, y in curves)
            plot.setRange(xRange=(xmin, xmax), yRange=(ymin, ymax))

        self.win.resize(plot_w, sum(plot_hs))
        self.win.show()
        self.app.processEvents()

    def generate_plot_video(self, output_path: str) -> str:
        """Create a video from stored plot data."""
        self.setup_plot()
        self.plot_frames = []
        next_plot_output_time = 0

        for i, t in enumerate(self.plt_data['time']):
            if t < next_plot_output_time: continue
            
            for (_, curves, x) in self.plots_and_curves:
                x_data = self.plt_data[x][:i + 1]
                for curve, y in curves:
                    curve.setData(x_data, self.plt_data[y][:i + 1])

            img = self.win.grab().toImage()
            buffer = np.array(img.bits().asarray(img.byteCount()))
            buffer.shape = (img.height(), img.width(), 4)
            self.plot_frames.append(buffer[:, :, :3])
            next_plot_output_time += self.output_interval

        self.win.close()
        self.export_clip(self.plot_frames, "Plot", output_path)
        return output_path

    def stack_video_frames(self, 
                           left_frames: list, 
                           right_frames: list, 
                           output_path: str, 
                           pad: bool=True):
        """Stack two lists of frames side by side, using either padding (default) or resizing."""
        def adjust_height(frame, target_h, pad_mode):
            """Helper to adjust frame height using either padding or resizing."""
            h, w, _ = frame.shape
            if h == target_h:
                return frame
            if pad_mode:
                pad_width = (target_h - h) // 2
                return np.pad(frame, ((pad_width, pad_width), (0, 0), (0, 0)))
            return cv2.resize(frame, (w * target_h // h, target_h))  # Preserve aspect ratio

        target_h = max(left_frames[0].shape[0], right_frames[0].shape[0])  # Determine max height
        combined_frames = [cv2.hconcat([adjust_height(left, target_h, pad),
                                        adjust_height(right, target_h, pad)])
                                        for left, right in zip(left_frames, right_frames)]
        self.export_clip(combined_frames, "Combined", output_path)


from tqdm import tqdm

class ProgressCallback:
    """Tracks simulation progress using tqdm."""
    def __init__(self, total_time: float) -> None:
        self.pbar = tqdm(total=total_time, desc="Simulation Progress", unit="s")

    def update(self, sim: MjcSim) -> None:
        """Update the progress bar using the simulation time step."""
        curr_time = round(sim.data.time, 5) 
        self.pbar.n = min(curr_time, self.pbar.total)  # Prevents overshooting
        self.pbar.refresh()
        if curr_time >= self.pbar.total:
            self.pbar.close()
            print("Simulation finished.")
            
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="duplo sim")
    parser.add_argument("-n", "--name", type=str, default="duplo_sim_default", help="name of the video")
    parser.add_argument("-t", "--sim_time", type=float, default=5, help="total simulation time")
    parser.add_argument("-gui", "--gui", action="store_true", help="enable GUI")
    parser.add_argument("-vfps", "--video_fps", type=int, default=30, help="FPS for video recording")
    args = vars(parser.parse_args())

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
    recorder = Recorder(args['video_fps'], plot_attributes, plot_structure)
    progress_cb = ProgressCallback(args["sim_time"])  # Initialize progress tracker

    callbacks_dict = {
        "record_frame"      : recorder.record_frame,
        "record_plot_data"  : recorder.record_plot_data,
        "progress_bar"      : progress_cb.update
        }

    duplo.run_sim(callbacks=callbacks_dict)
    duplo.close()

    v_dir = f"data/videos/{args['name']}"
    os.makedirs(v_dir, exist_ok=True)

    recorder.generate_plot_video(output_path=f"{v_dir}/live_plot.mp4")
    recorder.generate_robot_video(output_path=f"{v_dir}/robot_walking.mp4")
    recorder.stack_video_frames(recorder.plot_frames, 
                                 recorder.robot_frames,  
                                 output_path=f"{v_dir}/padded.mp4",
                                 pad=True)
