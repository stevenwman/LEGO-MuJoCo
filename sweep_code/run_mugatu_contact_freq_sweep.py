import mujoco as mjc
import mujoco_viewer as mjcv
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

old_robot_path = 'Mugatu/mugatu.xml'
new_robot_path = 'Mugatu/mugatu2.xml'
new_scene_path = 'Mugatu/scene2.xml'

round_to = 4
new_foot_mass = '0.13'
leg_amp_deg = 42.2
leg_amp_rad = np.round(np.deg2rad(leg_amp_deg), round_to)
f_slide = 1
joint_height = 0.15

robot_tree = ET.parse(old_robot_path)
robot_root = robot_tree.getroot()
right_foot_body = robot_root.find(".//body[@name='right_foot']")
left_foot_body = robot_root.find(".//body[@name='left_foot']")
right_foot_body.find('inertial').set('mass', new_foot_mass)
left_foot_body.find('inertial').set('mass', new_foot_mass)
robot_tree.write(new_robot_path)

max_time_range = 25
sol_stiff_params, sol_damp_params, freq_params = (15, 15, 15)
sol_stiff_range = np.round(np.linspace(0.005, 0.2, sol_stiff_params), round_to)
sol_damp_range = np.round(np.linspace(0.1, 2.0, sol_damp_params), round_to)
freq_range = np.round(np.linspace(1.3, 2.2, freq_params), round_to)
freq_range_rad = np.round(2*np.pi*freq_range, round_to)
tot_params = sol_stiff_params*sol_damp_params*freq_params

param_data = np.zeros((tot_params, 4))
count = 0

for cnt_stiff, stiffs in enumerate(sol_stiff_range):
    for cnt_damp, damps in enumerate(sol_damp_range):
        for cnt_freq, freq in enumerate(freq_range_rad):
            count += 1
            failed = False
            model = mjc.MjModel.from_xml_path(new_scene_path)
            data = mjc.MjData(model)
            model.opt.enableflags |= 1 << 0  # enable override
            model.opt.timestep = 0.0001
            model.opt.o_solref[0] = stiffs
            model.opt.o_solref[1] = damps

            for item in model.geom_friction:
                item[0] = f_slide

            mjc.mj_step(model, data)
            trial_init_pos = data.qpos.copy()

            while data.time < max_time_range:
                mjc.mj_step(model, data)
                if data.time > 3:
                    data.actuator("hip_joint_act").ctrl = leg_amp_rad * \
                        np.sin(freq*data.time)
                if data.qpos[2] < joint_height / 2:
                    print(f"fell! ({count}/{tot_params})")
                    failed = True
                    param_data[count-1, :] = [0, stiffs,
                                              damps, freq_range[cnt_freq]]
                    break

            if not failed:
                print(f"done! ({count}/{tot_params})")
                dist_traveled = np.linalg.norm(
                    data.qpos[0:2] - trial_init_pos[0:2])
                param_data[count-1, :] = [dist_traveled,
                                          stiffs, damps, freq_range[cnt_freq]]

np.savetxt('contact_freq_sweep.csv', param_data, delimiter=',')
