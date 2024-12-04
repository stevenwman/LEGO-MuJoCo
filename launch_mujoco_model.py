import os
import mujoco
import mujoco.viewer

curr_dir = os.path.dirname(os.path.abspath(__file__))
file_rel_path = "Mugatu/scene.xml"
file_rel_path = "my-robot-fixed/robot.urdf"
file_rel_path = "my-robot-fixed/scene3.xml"
# file_rel_path = "robot/fullasm.mjcf"
# file_rel_path = "robot.xml"
file_path = os.path.join(curr_dir, file_rel_path)

# Load your model
model = mujoco.MjModel.from_xml_path(file_path)

# Create a simulation data structure
data = mujoco.MjData(model)

# mujoco.mj_saveLastXML("output.xml", model)

# Launch the viewer (GUI)
mujoco.viewer.launch(model, data)
