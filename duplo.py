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
        self.plot_data = {"time": [], "setpoint": [], "actual_pos": [], "torque": []}

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
        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(self.robot_frames, fps=self.v_fps)
        clip.write_videofile(output_path, codec="libx264")
        print(f"Robot video saved to {output_path}")

    def generate_plot_video(self, output_path="live_plot.mp4", fps=30):
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets, QtGui
        from moviepy.editor import ImageSequenceClip
        import numpy as np

        # Initialize PyQtGraph app
        app = QtWidgets.QApplication([])

        # Create a PyQtGraph layout
        titles = ["Actuator Setpoint", "Actuator Actual Position", "Actuator Torque"]
        data_keys = ["setpoint", "actual_pos", "torque"]

        win = pg.GraphicsLayoutWidget()
        win.setWindowTitle("Live Plot Video")

        # Automatically adjust spacing and scaling
        win.ci.setSpacing(10)  # Minimal manual spacing
        plots = []  # To store plot objects for later reference
        curves = []  # To store curve objects for later updates

        # Dynamically create plots
        for i, title in enumerate(titles):
            plot = win.addPlot(row=i, col=0, title=title)  # Specify row=i for vertical stacking
            plot.setLabel("bottom", "Time (s)", color="white", size="10pt")
            plot.setLabel("left", title, color="white", size="10pt")
            plot.showGrid(x=True, y=True, alpha=0.5)
            plot.setTitle(title, color="white", size="12pt")

            # Adjust the left axis label offset
            plot.getAxis('left').setWidth(75)

            curve = plot.plot(pen=pg.mkPen(color=(0, 255, 0), width=2))
            plots.append(plot)
            curves.append(curve)

        # Compute the global min and max values for time and each data key
        time_min, time_max = min(self.plot_data["time"]), max(self.plot_data["time"])
        data_ranges = {
            key: (min(self.plot_data[key]), max(self.plot_data[key])) for key in data_keys
        }

        # Set the range for each plot once
        for j, plot in enumerate(plots):
            data_min, data_max = data_ranges[data_keys[j]]
            plot.setRange(xRange=(time_min, time_max), yRange=(data_min, data_max))

        # Force auto-adjustment to fit all subplots
        win.ci.layout.setContentsMargins(10, 10, 10, 10)  # Remove extra margins
        win.ci.layout.activate()  # Update layout to reflect changes
        win.updateGeometry()  # Trigger automatic size adjustment
        win.show()  # Ensure everything is displayed correctly
        app.processEvents()  # Allow layout adjustments

        # Dynamically determine the optimal window size
        width = 800
        # height = win.sizeHint().height()
        # win.resize(width, height)

        total_height = 300 * len(titles)
        win.resize(width, total_height)


        # Generate frames for the video
        output_interval = 1 / fps
        next_output_time = 0
        plot_frames = []

        for i in range(len(self.plot_data["time"]) - 1):
            if next_output_time <= self.plot_data["time"][i + 1]:
                # Update curves with new data
                for j, key in enumerate(data_keys):
                    curves[j].setData(self.plot_data["time"][:i + 1], self.plot_data[key][:i + 1])

                # Render the widget directly to a QImage
                image = QtGui.QImage(win.size().width(), win.size().height(), QtGui.QImage.Format_RGB888)
                painter = QtGui.QPainter(image)
                win.render(painter)
                painter.end()

                # Convert the QImage to a numpy array
                width = image.width()
                height = image.height()
                ptr = image.bits()
                ptr.setsize(width * height * 3)
                img_array = np.array(ptr).reshape((height, width, 3))
                plot_frames.append(img_array)

                next_output_time += output_interval

        # Close the PyQtGraph window
        win.close()

        # Save the plot video
        clip = ImageSequenceClip(plot_frames, fps=fps)
        clip.write_videofile(output_path, codec="libx264")
        print(f"Plot video saved to {output_path}")


    def close(self):
        self.renderer.close()
        if self.gui: self.viewer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="duplo sim")
    parser.add_argument("-n", "--name", type=str, default="duplo_sim_default", help="name of the video")
    parser.add_argument("-t", "--time", type=float, default=5, help="time to run the simulation")
    parser.add_argument("-g", "--gui", action="store_true", help="enable GUI")
    parser.add_argument("-vfps", "--video_fps", type=int, default=30, help="fps of the video")
    args = vars(parser.parse_args())

    duplo = Duplo(args)

    # Run simulation and store data
    duplo.run_sim()

    # Generate videos
    v_dir = f"data/videos/{args['name']}"
    os.makedirs(v_dir, exist_ok=True)

    duplo.generate_plot_video(output_path=f"{v_dir}/live_plot.mp4", fps=30)
    duplo.generate_robot_video(output_path=f"{v_dir}/robot_walking.mp4")
    
    print("All videos generated!")

    duplo.close()
