import os
import mujoco
import mujoco.viewer

# curr_dir = os.path.dirname(os.path.abspath(__file__))
file_path = "old/old_robot_files/duplo_hip_offset2/robot.urdf"
# file_path = "robots/duplo_ballfeet_mjcf/scene_motor_temp.xml"
# file_path = os.path.join(curr_dir, file_rel_path)

# Load your model
model = mujoco.MjModel.from_xml_path(file_path)
# Create a simulation data structure
data = mujoco.MjData(model)
mujoco.mj_saveLastXML("robots/duplo_hip_offset_mjcf/duplo_hip_offset2.xml", model)

# Launch the viewer (GUI)
mujoco.viewer.launch(model, data)
