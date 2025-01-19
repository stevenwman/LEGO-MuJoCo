import argparse
import mujoco
import mujoco.viewer
import numpy as np
import sys
import os 
os.environ["QT_QPA_PLATFORM"] = "offscreen"
print(f"Using Qt platform: {os.environ.get('QT_QPA_PLATFORM')}")
import time

class Duplo:
    def __init__(self, args):
        self.simtime = args['time']
        self.gui = args['gui']
        self.v_fps = args['video_fps']
        scene_path = 'duplo_ballfeet_mjcf/scene.xml'
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        leg_amp_deg = 35
        self.leg_amp_rad = np.deg2rad(leg_amp_deg)
        self.hip_omega = np.sqrt(9.81/0.63)

        self.joint_name = "hip"
        self.joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_name)
        self.dof_addr = self.model.jnt_dofadr[self.joint_id]

        self.camera_params = {
            'tracking': "leg_1",
            'distance': 5,
            'xyaxis': [-1, 0, 0, 0, 0, 1],
        }

        self.setup_gui()

        # Your existing initialization code
        self.plot_data = {"time": [], "setpoint": [], "actual_pos": [], "torque": []}
        self.plot_recorder = None  # For recording plot updates

    def apply_sine_ctrl(self, waittime=1, b=1):
        if self.data.time < waittime: return
        # b defines how sharp a sine wave is, higher the sharper
        wave = np.sin(self.hip_omega*(self.data.time-waittime))
        wave_val = np.sqrt((1+b**2)/(1+(b**2)*wave**2))*wave
        self.acutator_control = self.leg_amp_rad * wave_val
        self.data.actuator("hip_joint_act").ctrl = self.acutator_control

    def setup_gui(self):
        self.width, self.height = 640, 480
        self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.camera = mujoco.MjvCamera()

        tracking = self.camera_params["tracking"]
        self.camera.distance = self.camera_params["distance"]

        if tracking is None:
            self.camera.lookat = self.camera_params["lookat"]
            self.camera.elevation = self.camera_params["elevation"]
            self.camera.azimuth = self.camera_params["azimuth"]
        else:
            body_id = mujoco.mj_name2id(self.model,
                                        mujoco.mjtObj.mjOBJ_BODY,
                                        tracking)
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = body_id
            xyaxis = self.camera_params["xyaxis"]
            z_axis = np.cross(xyaxis[:3], xyaxis[3:])
            azimuth = np.degrees(np.arctan2(z_axis[1], z_axis[0]))
            elevation = np.degrees(np.arctan2(z_axis[2], np.linalg.norm(z_axis[0:2])))
            self.camera.azimuth = azimuth
            self.camera.elevation = elevation

        if self.gui:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if tracking is not None:
                self.viewer.cam.type = self.camera.type
                self.viewer.cam.trackbodyid = self.camera.trackbodyid
            self.viewer.cam.distance = self.camera.distance
            self.viewer.cam.lookat = self.camera.lookat
            self.viewer.cam.elevation = self.camera.elevation
            self.viewer.cam.azimuth = self.camera.azimuth

    def get_image(self):
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def step_sim(self):
        mujoco.mj_step(self.model, self.data)
        self.actuator_setpoints = self.data.actuator("hip_joint_act").ctrl[0]
        self.actuator_actual_pos = self.data.qpos[7]
        # adding 6 because body wrenches are also included in qfrc_actuator
        self.actuator_torque = self.data.qfrc_actuator[self.dof_addr]
        print(f"t = {self.data.time:.3f}s")

        if self.gui: 
            self.viewer.sync()
            time.sleep(self.model.opt.timestep)
            if not self.viewer.is_running(): sys.exit()

    def run_sim(self):
        output_interval = 1 / self.v_fps
        next_output_time = 0
        # Lists to store data
        self.robot_frames = []  # Store rendered frames for robot video
        self.plot_data = {"time"       : [], 
                          "setpoint"   : [], 
                          "actual_pos" : [], 
                          "torque"     : []}

        loop = range(int(self.simtime // self.model.opt.timestep))
        for _ in loop:
            self.apply_sine_ctrl()
            self.step_sim()  # Step simulation at full frequency

            if self.data.time >= next_output_time:
                self.robot_frames.append(self.get_image())  # Store rendered frame   
                next_output_time += output_interval

            self.plot_data["time"].append(self.data.time)
            self.plot_data["setpoint"].append(self.actuator_setpoints)
            self.plot_data["actual_pos"].append(self.actuator_actual_pos)
            self.plot_data["torque"].append(self.actuator_torque)

    def generate_robot_video(self, output_path="robot_walking.mp4"):
        from moviepy import ImageSequenceClip
        clip = ImageSequenceClip(self.robot_frames, fps=self.v_fps)
        clip.write_videofile(output_path, codec="libx264")
        print(f"Robot video saved to {output_path}")
        return output_path

    def generate_plot_video(self, output_path, fps, stack_keys):
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets
        from moviepy import ImageSequenceClip
        import numpy as np

        # Initialize PyQtGraph app
        app = QtWidgets.QApplication([])

        # Create the layout window
        win = pg.GraphicsLayoutWidget(title="Live Plot Video")

        # Assign colors to the curves
        colors = [
            (255, 255, 255),  # White
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 165, 0),    # Orange
        ]

        # Create plots for each group and store their curve objects
        plots_and_curves = []  # List of (plot, [curves]) tuples
        for row, group in enumerate(stack_keys):  # Use rows for vertical alignment
            plot = win.addPlot(row=row, col=0, title=", ".join(group))
            plot.setLabels(left="Value", bottom="Time (s)")
            plot.showGrid(x=True, y=True, alpha=0.5)
            plot.addLegend(offset=(10, 10))  # Add legend with a slight offset

            curves = []
            for idx, key in enumerate(group):
                color = colors[idx % len(colors)]  # Cycle through colors
                curve = plot.plot(
                    pen=pg.mkPen(color=color, width=2),
                    name=key  # Add the name to show in the legend
                )
                curves.append(curve)

            plots_and_curves.append((plot, curves))

        app.processEvents()  # Ensure all pending events are processed

        # Make plotting range static using max and min data values
        time_min, time_max = min(self.plot_data["time"]), max(self.plot_data["time"])
        group_ranges = []
        for group in stack_keys:
            group_min = min(min(self.plot_data[key]) for key in group)
            group_max = max(max(self.plot_data[key]) for key in group)
            group_ranges.append((group_min, group_max))

        # Set the range for each plot once
        for (plot, _), (data_min, data_max) in zip(plots_and_curves, group_ranges):
            plot.setRange(xRange=(time_min, time_max), yRange=(data_min, data_max))

        # Resize window and fit layout
        win.resize(800, 300 * len(stack_keys))  # Adjust height based on number of plots
        win.show()
        app.processEvents()

        # Generate video frames
        plot_frames = []
        output_interval = 1 / fps
        next_output_time = 0

        for i, t in enumerate(self.plot_data["time"]):
            if t >= next_output_time:
                # Update curves within each group
                for (plot, curves), group in zip(plots_and_curves, stack_keys):
                    for curve, key in zip(curves, group):
                        curve.setData(self.plot_data["time"][:i + 1], self.plot_data[key][:i + 1])

                # Render the widget to an image
                img = win.grab().toImage()
                buffer = img.bits().asarray(img.byteCount())
                plot_frames.append(np.array(buffer).reshape((img.height(), img.width(), 4))[:, :, :3])

                next_output_time += output_interval

        # Close the window
        win.close()

        # Create video from frames
        clip = ImageSequenceClip(plot_frames, fps=fps)
        clip.write_videofile(output_path, codec="libx264")
        print(f"Plot video saved to {output_path}")
        return output_path



    def close(self):
        self.renderer.close()
        if self.gui: self.viewer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="duplo sim")
    parser.add_argument("-n", "--name", type=str, default="duplo_sim_default", help="name of the video")
    parser.add_argument("-t", "--time", type=float, default=5, help="time to run the simulation")
    parser.add_argument("-g", "--gui", action="store_true", help="enable GUI")
    parser.add_argument("-vfps", "--video_fps", type=int, default=30, help="fps of the video")
    parser.add_argument("-stk", "--stack_videos", action="store_true", help="stack videos")
    args = vars(parser.parse_args())

    duplo = Duplo(args)

    # Run simulation and store data
    duplo.run_sim()

    # Generate videos
    v_dir = f"data/videos/{args['name']}"
    os.makedirs(v_dir, exist_ok=True)

    stack_keys = [["setpoint", "actual_pos"], ["torque"]]
    plot_video_path = duplo.generate_plot_video(output_path=f"{v_dir}/live_plot.mp4", fps=30, stack_keys=stack_keys)
    robot_video_path = duplo.generate_robot_video(output_path=f"{v_dir}/robot_walking.mp4")

    if args["stack_videos"]:
        from moviepy import VideoFileClip, clips_array

        # Load your video files
        videoL = VideoFileClip(plot_video_path)
        videoR = VideoFileClip(robot_video_path)

        # Ensure both videos have the same height
        if videoL.h != videoR.h:
            max_height = max(videoL.h, videoR.h)
            videoL = videoL.resized(height=max_height)
            videoR = videoR.resized(height=max_height)

        # Combine videos side by side
        final_video = clips_array([[videoL, videoR]])
        # Write the output to a file
        final_video.write_videofile(f"{v_dir}/stacked.mp4", codec="libx264")

    print("All videos generated!")

    duplo.close()
