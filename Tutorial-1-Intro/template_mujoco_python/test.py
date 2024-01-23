import os

urdf_file_path = '/home/sman/Work/CMU/Research/LEGO-project/LEGO-MuJoCo/Tutorial-1-Intro/template_mujoco_python/mugatu_urdf/mugatu_urdf/urdf/mugatu_urdf.urdf'
absolute_path = os.path.abspath(urdf_file_path)
print("Absolute Path:", absolute_path)

# Check if the file exists
if not os.path.exists(absolute_path):
    print("File does not exist at this path")
else:
    print("File exists")
