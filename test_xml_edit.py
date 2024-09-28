import xml.etree.ElementTree as ET

old_robot_path = 'Mugatu/mugatu.xml'
new_robot_path = 'Mugatu/mugatu2.xml'
new_scene_path = 'Mugatu/scene2.xml'

robot_tree = ET.parse(old_robot_path)
robot_root = robot_tree.getroot()
right_foot_body = robot_root.find(".//body[@name='right_foot']")
left_foot_body = robot_root.find(".//body[@name='left_foot']")
new_foot_mass = 0.1
right_foot_body.find('inertial').set('mass', str(new_foot_mass))
left_foot_body.find('inertial').set('mass', str(new_foot_mass))
robot_tree.write(new_robot_path)
