import mujoco as mjc
import mujoco_viewer as mjcv
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

old_robot_path = 'Mugatu/mugatu.xml'
new_robot_path = 'Mugatu/mugatu2.xml'
new_scene_path = 'Mugatu/scene2.xml'

new_foot_mass = '0.13'
com_height = 0.066
joint_height = 0.15
leg_amp_deg = 42.2
leg_amp_rad = np.deg2rad(leg_amp_deg)
hip_omega = np.sqrt(9.81/(joint_height - com_height))

robot_tree = ET.parse(old_robot_path)
robot_root = robot_tree.getroot()
right_foot_body = robot_root.find(".//body[@name='right_foot']")
left_foot_body = robot_root.find(".//body[@name='left_foot']")
right_foot_body.find('inertial').set('mass', new_foot_mass)
left_foot_body.find('inertial').set('mass', new_foot_mass)

robot_tree.write(new_robot_path)
max_time_range = 25

f_slide_params, f_spin_params, f_roll_params = (25,10,50)
# f_slide_range = np.geomspace(0.1, 2, f_slide_params)
f_spin_range = np.geomspace(5.e-03, 1, f_spin_params)
f_roll_range = np.geomspace(1.e-04, 1, f_roll_params)

f_slide_range = np.linspace(0.1, 2, f_slide_params)
# f_spin_range = np.linspace(5.e-04, 1, f_spin_params)
# f_roll_range = np.linspace(1.e-05, 1, f_roll_params)

tot_params = f_slide_params*f_spin_params*f_roll_params

param_data = np.zeros((tot_params,4))
count = 0

for cnt_slide, f_slide in enumerate(f_slide_range):
    for cnt_spin, f_spin in enumerate(f_spin_range):
        for cnt_roll, f_roll in enumerate(f_roll_range):
            count += 1
            failed = False
            model = mjc.MjModel.from_xml_path(new_scene_path)
            data = mjc.MjData(model)
            model.opt.timestep = 0.001

            for item in model.geom_friction:
                item[:] = [f_slide,f_spin,f_roll]
            mjc.mj_step(model, data)
            trial_init_pos = data.qpos.copy()

            while data.time < max_time_range:
                mjc.mj_step(model, data)
                if data.time > 3:
                    data.actuator("hip_joint_act").ctrl = leg_amp_rad * \
                        np.sin(hip_omega*data.time)
                if data.qpos[2] < joint_height / 3:
                    print(f"fell! ({count}/{tot_params})")
                    failed = True
                    param_data[count-1,:] = [0,f_slide, f_spin, f_roll]
                    break
                
            if not failed:
                print(f"done! ({count}/{tot_params})")
                dist_traveled = np.linalg.norm(data.qpos[0:2] - trial_init_pos[0:2])
                param_data[count-1,:] = [dist_traveled,f_slide, f_spin, f_roll]

np.savetxt('friction_sweep.csv', param_data, delimiter=',')



            