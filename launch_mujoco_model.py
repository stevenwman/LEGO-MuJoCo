import mujoco
import mujoco.viewer

# Load your model
model = mujoco.MjModel.from_xml_path('Mugatu/scene.xml')
# model = mujoco.MjModel.from_xml_path('robotis_op3/scene.xml')

# Create a simulation data structure
data = mujoco.MjData(model)

# Launch the viewer (GUI)
mujoco.viewer.launch(model, data)
