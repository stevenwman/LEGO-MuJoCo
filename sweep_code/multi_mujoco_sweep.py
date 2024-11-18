import numpy as np
import concurrent.futures
import csv
import mujoco as mjc
import xml.etree.ElementTree as ET

new_robot_path = 'Mugatu/mugatu2.xml'
new_scene_path = 'Mugatu/scene2.xml'

round_to = 4
new_foot_mass = '0.13'
leg_amp_deg = 42.2
leg_amp_rad = np.round(np.deg2rad(leg_amp_deg), round_to)
f_slide = 1
joint_height = 0.15

max_time_range = 25
wait_time = 3
run_time = max_time_range - wait_time

n_freqs, n_amps = (5, 5)
freq_range = np.round(np.linspace(1.3, 2.2, n_freqs), round_to)
freq_range_rad = np.round(2*np.pi*freq_range, round_to)
amp_range = np.round(np.linspace(24.2, 42.2, n_amps), round_to)
amp_range_rad = np.round(np.deg2rad(amp_range), round_to)

tot_params = n_freqs*n_amps
params = [(freq, amp) for freq in freq_range_rad for amp in amp_range_rad]
results = np.zeros((tot_params, 3))

count = 0


def run_mjc_instance(params):
    freq, amp = params
    freq_label, amp_label = (
        np.round(freq/2/np.pi, round_to), np.round(np.rad2deg(amp), round_to))
    model = mjc.MjModel.from_xml_path(new_scene_path)
    data = mjc.MjData(model)
    model.opt.timestep = 0.0001

    mjc.mj_step(model, data)
    trial_init_pos = data.qpos.copy()

    while data.time < max_time_range:
        mjc.mj_step(model, data)
        if data.time > wait_time:
            data.actuator("hip_joint_act").ctrl = amp * np.sin(freq*data.time)
        if data.qpos[2] < joint_height / 2:
            print("fell!")
            return [0, freq_label, amp_label]

    print("done!")
    dist_traveled = np.linalg.norm(data.qpos[0:2] - trial_init_pos[0:2])
    return [dist_traveled / run_time, freq_label, amp_label]


# Number of threads to run in parallel (e.g., 5 cores)
num_workers = 14  # You can set this to the number of CPU cores you have

# Run the additions in parallel using threads
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(run_mjc_instance, param) for param in params]
    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        print(i)
        results[i, :] = future.result()

# Optionally, save results to CSV
with open('multi_mjc_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['speed', 'freq', 'amp'])  # Header
    writer.writerows(results)

print(results)
