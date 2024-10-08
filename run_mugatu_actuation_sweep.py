import mujoco as mjc
import mujoco_viewer as mjcv
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

old_robot_path = 'Mugatu/mugatu.xml'
new_robot_path = 'Mugatu/mugatu2.xml'
new_scene_path = 'Mugatu/scene2.xml'

new_foot_mass = '0.13'
# com_height = 0.066
joint_height = 0.15
# leg_amp_deg = 42.2
# leg_amp_rad = np.deg2rad(leg_amp_deg)
# hip_omega = np.sqrt(9.81/(joint_height - com_height))

robot_tree = ET.parse(old_robot_path)
robot_root = robot_tree.getroot()
right_foot_body = robot_root.find(".//body[@name='right_foot']")
left_foot_body = robot_root.find(".//body[@name='left_foot']")
right_foot_body.find('inertial').set('mass', new_foot_mass)
left_foot_body.find('inertial').set('mass', new_foot_mass)

robot_tree.write(new_robot_path)
max_time_range = 25

round_to = 4

amp_params, freq_params = (50, 100)
amp_range = np.round(np.linspace(
    24.2 * 0.75, 42.2 * 1.25, amp_params), round_to)
amp_range_rad =  np.round(np.deg2rad(amp_range), round_to)
freq_range = np.round(np.linspace(1, 2, freq_params), round_to)
freq_range_rad = np.round(2*np.pi*freq_range, round_to)

tot_params = amp_params*freq_params

param_data = np.zeros((tot_params, 3))
count = 0

for cnt_amp, amp in enumerate(amp_range_rad):
    for cnt_freq, freq in enumerate(freq_range_rad):
        count += 1
        failed = False
        model = mjc.MjModel.from_xml_path(new_scene_path)
        data = mjc.MjData(model)
        model.opt.timestep = 0.001

        mjc.mj_step(model, data)
        trial_init_pos = data.qpos.copy()

        while data.time < max_time_range:
            mjc.mj_step(model, data)
            if data.time > 3:
                data.actuator("hip_joint_act").ctrl = amp * \
                    np.sin(freq*data.time)
            if data.qpos[2] < joint_height / 2:
                print(f"fell! ({count}/{tot_params})")
                failed = True
                param_data[count-1, :] = [0,
                                          amp_range[cnt_amp], freq_range[cnt_freq]]
                break

        if not failed:
            print(f"done! ({count}/{tot_params})")
            dist_traveled = np.linalg.norm(
                data.qpos[0:2] - trial_init_pos[0:2])
            param_data[count-1, :] = [dist_traveled,
                                      amp_range[cnt_amp], freq_range[cnt_freq]]

np.savetxt('act_sweep.csv', param_data, delimiter=',')
