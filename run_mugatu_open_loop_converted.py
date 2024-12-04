import mujoco as mjc
import mujoco_viewer as mjcv
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

old_robot_path = 'Mugatu/mugatu.xml'
new_robot_path = 'Mugatu/mugatu2.xml'
new_scene_path = 'Mugatu/scene2.xml'
new_scene_path = 'my-robot-fixed/robot.xml'
new_scene_path = 'my-robot-fixed/scene3.xml'

new_foot_mass = '0.13'
com_height = 0.066
joint_height = 0.15

leg_amp_deg, hip_freq = 42, 1.7


round_to = 4
leg_amp_rad = np.round(np.deg2rad(leg_amp_deg), round_to)
hip_omega = np.round(2*np.pi*hip_freq, round_to)

model = mjc.MjModel.from_xml_path(new_scene_path)
data = mjc.MjData(model)

# model.opt.timestep = 0.001  # Set a custom timestep
# model.opt.enableflags |= 1 << 0  # enable override
# model.opt.cone = 1

viewer = mjcv.MujocoViewer(model, data, width=int(1920/2), height=int(1080/2))

mjc.mj_step(model, data)
max_time_range = 50
wait_time = 3

# store actuator setpoints empty array and append later
actuator_setpoints = np.zeros((0, 1))
# store actuator actual pos empty array and append later
actuator_actual_pos = np.zeros((0, 1))

while data.time < max_time_range:
    if viewer.is_alive:
        mjc.mj_step(model, data)
        if data.time > wait_time:
            # acutator_control = leg_amp_rad * \
            #     np.sin(np.pi/2*np.cos(hip_omega*(data.time-wait_time)))
            b = 1
            wave = np.cos(hip_omega*(data.time-wait_time))
            wave_val = np.sqrt((1+b**2)/(1+(b**2)*wave**2))*wave

            wave_val = -np.cos(hip_omega*(data.time-wait_time))
            acutator_control = leg_amp_rad * wave_val

            data.actuator("hip_joint_act").ctrl = acutator_control
            # data.actuator("hip").ctrl = acutator_control
            actuator_setpoints = np.append(
                actuator_setpoints, acutator_control)
            actuator_actual_pos = np.append(actuator_actual_pos, data.qpos[7])

        # data.joint("").xanchor to find joint location
        # data.subtree_com() to find com location
        viewer.render()
    else:
        break
print("done!")

viewer.close()

# plot actuator setpoints and actual pos
plt.plot(actuator_setpoints, label='setpoints')
plt.plot(actuator_actual_pos, label='actual pos')
plt.legend()
plt.show()
