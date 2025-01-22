import os
import mujoco
import mujoco.viewer

# curr_dir = os.path.dirname(os.path.abspath(__file__))
file_path = "robots/zippy_mjcf/scene_motor.xml"
# file_path = "robots/duplo_ballfeet_mjcf/scene_motor_temp.xml"
# file_path = os.path.join(curr_dir, file_rel_path)

# Load your model
model = mujoco.MjModel.from_xml_path(file_path)
# Create a simulation data structure
data = mujoco.MjData(model)
# mujoco.mj_saveLastXML("robots/zippy_newmass/zippy.xml", model)

# Launch the viewer (GUI)
mujoco.viewer.launch(model, data)
