import mujoco as mjc
import mujoco_viewer as mjcv
import numpy as np
import matplotlib.pyplot as plt

new_scene_path = 'duplo_mjcf/scene.xml'
# for mjcf, scale = "1.1 1 1" to scale = "1.3 1 1" worked
new_scene_path = 'duplo_ballfeet_mjcf/scene.xml'

leg_amp_deg = 35
hip_freq = np.sqrt(9.81/0.63)

leg_amp_rad = np.deg2rad(leg_amp_deg)
hip_omega = np.sqrt(9.81/0.63)

model = mjc.MjModel.from_xml_path(new_scene_path)
data = mjc.MjData(model)
viewer = mjcv.MujocoViewer(model,
                           data,
                           width=int(1920/2),
                           height=int(1080/2))

mjc.mj_step(model, data)
max_time_range = 500000
wait_time = 1

# store actuator setpoints empty array and append later
actuator_setpoints = np.zeros((0, 1))
actuator_actual_pos = np.zeros((0, 1))
actuator_torque = np.zeros((0, 1))
joint_vel = np.zeros((0, 1))
# j_pos = np.zeros((1, 4))

joint_name = "hip"
joint_id = mjc.mj_name2id(model, mjc.mjtObj.mjOBJ_JOINT, joint_name)
dof_addr = model.jnt_dofadr[joint_id]

while data.time < max_time_range:
    if viewer.is_alive:
        mjc.mj_step(model, data)
        if data.time > wait_time:
            # acutator_control = leg_amp_rad * \
            #     np.sin(np.pi/2*np.cos(hip_omega*(data.time-wait_time)))
            b = 1
            wave = np.sin(hip_omega*(data.time-wait_time))
            wave_val = np.sqrt((1+b**2)/(1+(b**2)*wave**2))*wave
            acutator_control = leg_amp_rad * wave_val

            data.actuator("hip_joint_act").ctrl = acutator_control
            actuator_setpoints = np.append(actuator_setpoints,
                                           acutator_control)
            actuator_actual_pos = np.append(actuator_actual_pos,
                                            data.qpos[7])
            # adding 6 because body wrenches are also included in qfrc_actuator
            actuator_torque = np.append(actuator_torque,
                                        data.qfrc_actuator[dof_addr])
            joint_vel = np.append(joint_vel, data.qvel[dof_addr])

        # data.joint("").xanchor to find joint location
        # j_pos = np.vstack([j_pos, data.qpos[3:7]])

        # # take mean along each column
        # print(np.mean(j_pos, axis=0))

        print(f"{data.time:.3f}")
        viewer.render()
    else:
        break
print("done!")

viewer.close()

# plot actuator setpoints and actual pos
plt.figure()
plt.plot(actuator_setpoints, label='setpoints')
plt.plot(actuator_actual_pos, label='actual pos')
plt.legend()

# plot joint torque
plt.figure()
plt.plot(actuator_torque, label='joint torque')
plt.legend()

# plot joint velocity
plt.figure()
plt.plot(joint_vel, label='joint velocity')
plt.legend()
plt.show()

# plt.plot(j_pos[:, 0], label='joint 1')
# plt.plot(j_pos[:, 1], label='joint 2')
# plt.plot(j_pos[:, 2], label='joint 3')
# plt.plot(j_pos[:, 3], label='joint 4')
# plt.legend()
# plt.show()
