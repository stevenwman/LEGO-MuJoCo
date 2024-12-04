import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from xml.etree.ElementTree import Element, SubElement, ElementTree
from onshape_client.client import Client

# Replace these with your Onshape API keys
ONSHAPE_ACCESS_KEY = 'Q9CQgVoSEChaMUoTsD0a24KS'
ONSHAPE_SECRET_KEY = '17jQsMCWyYGM3wOPYz9Ozrje3p0oKR47knLlQWLm4dgG4YpX'
DOCUMENT_ID = 'bf016333eb87030ff171a82f'
WORKSPACE_ID = '54e50e04943a1e9f2924bf70'
ELEMENT_ID = '2f23535faa32137a2d7728a2'

# API base URL
BASE_URL = 'https://cad-usw2.onshape.com:443'

# Initialize the client
client = Client(configuration={
    "base_url": BASE_URL,
    "access_key": ONSHAPE_ACCESS_KEY,
    "secret_key": ONSHAPE_SECRET_KEY
})

# Import the necessary API modules
assemblies_api = client.assemblies_api
part_studios_api = client.part_studios_api  # Use PartStudiosApi


def transform_to_pos_quat(transform):
    # Convert the 4x4 transform matrix to position and quaternion
    # Transform is in row-major order
    # Convert to column-major for consistency
    transform = np.array(transform).reshape(4, 4).T

    # Extract rotation matrix (upper-left 3x3)
    rot_matrix = transform[:3, :3]

    # Extract position (last column of the first three rows)
    pos = transform[:3, 3]

    # Convert rotation matrix to quaternion
    rot = R.from_matrix(rot_matrix)
    quat = rot.as_quat()  # Returns (x, y, z, w)

    # Reorder quaternion to (w, x, y, z) for MuJoCo
    quat = [quat[3], quat[0], quat[1], quat[2]]

    return pos, quat


def get_assembly_definition(did, wid, eid):
    """Retrieves the assembly definition from Onshape."""
    response = assemblies_api.get_assembly_definition(
        did=did,
        wvm='w',
        wvmid=wid,
        eid=eid,
        include_mate_features=True,  # Include mates in the assembly definition
        _preload_content=False
    )
    return json.loads(response.data)


def get_part_mesh(did, wid, eid, partid, filename):
    """Retrieves the mesh for a part and saves it as an STL file."""
    try:
        response = part_studios_api.export_stl1(
            did=did,
            wvm='w',
            wvmid=wid,
            eid=eid,
            part_ids=partid,
            units='meter',
            mode='binary',
            _preload_content=False
        )
        # Save the mesh to a file
        with open(filename, 'wb') as f:
            f.write(response.data)
    except Exception as e:
        print(f"An error occurred while exporting STL for part {partid}: {e}")


def get_mass_properties(did, wid, eid, partid):
    """Retrieves the mass properties for a part."""
    response = client.parts_api.get_mass_properties(
        did=did,
        wvm='w',
        wvmid=wid,
        eid=eid,
        partid=partid,
        _preload_content=False
    )
    return json.loads(response.data)


def extract_mates_from_features(features):
    mates = []
    for feature in features:
        if feature.get('featureType') == 'mate':
            mates.append(feature)
    return mates


def main():
    # Document, workspace, and element IDs
    did = DOCUMENT_ID
    wid = WORKSPACE_ID
    eid = ELEMENT_ID

    # Output directory for meshes
    mesh_dir = 'meshes'
    os.makedirs(mesh_dir, exist_ok=True)

    # Get assembly definition
    assembly = get_assembly_definition(did, wid, eid)

    # Start building the MJCF XML
    mjcf = Element('mujoco', model='OnshapeRobot')
    compiler = SubElement(mjcf, 'compiler', angle='degree')
    option = SubElement(mjcf, 'option', timestep='0.005')
    worldbody = SubElement(mjcf, 'worldbody')
    asset = SubElement(mjcf, 'asset')

    # Map of instance IDs to body elements and their transforms
    part_bodies = {}
    part_transforms = {}

    # Process each instance in the assembly
    for instance in assembly['rootAssembly']['instances']:
        if instance['type'] != 'Part':
            continue  # Skip non-part instances

        partid = instance['partId']
        instance_id = instance['id']
        name = instance['name'].replace(' ', '_')
        mesh_file = f"{mesh_dir}/{name}_{partid}.stl"

        # Get and save the mesh
        get_part_mesh(did, wid, instance['elementId'], partid, mesh_file)

        # Get mass properties
        mass_properties = get_mass_properties(
            did, wid, instance['elementId'], partid)

        # Check if the part ID exists in the mass properties
        if partid in mass_properties.get('bodies', {}):
            body_properties = mass_properties['bodies'][partid]

            # Extract mean mass (second value in the 'mass' array)
            mass = body_properties['mass'][1]

            # Extract principal inertia
            inertia = body_properties['principalInertia']

            # Add mesh to assets
            mesh_asset = SubElement(asset, 'mesh', name=name, file=mesh_file)

            # Extract instance transform
            transform = instance.get('transform', None)
            if transform is None:
                # If no transform is provided, default to identity
                transform = [1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0,
                             0, 0, 0, 1]

            # Convert the transform to position and quaternion
            pos, quat = transform_to_pos_quat(transform)

            # If units are in millimeters, convert to meters
            pos = pos * 0.001  # Adjust based on the units in Onshape

            # Create body element with position and orientation
            body = SubElement(worldbody, 'body', name=name,
                              pos=f"{pos[0]} {pos[1]} {pos[2]}",
                              quat=f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")  # w x y z

            # Add inertial properties with pos attribute
            SubElement(body, 'inertial', mass=str(mass),
                       diaginertia=f"{inertia[0]} {inertia[1]} {inertia[2]}",
                       pos="0 0 0")

            # Add visual geometry
            SubElement(body, 'geom', type='mesh',
                       mesh=name, rgba='0.8 0.6 0.4 1')

            # Store the body element and its transform for later reference
            part_bodies[instance_id] = body
            part_transforms[instance_id] = {
                'pos': pos,
                'quat': quat
            }
        else:
            print(f"Mass properties for part ID {partid} not found.")
            continue  # Skip this part if mass properties are missing

    # Process each mate (joint)
    root_assembly = assembly['rootAssembly']

    # Extract mates from features
    features = root_assembly.get('features', [])
    mates = extract_mates_from_features(features)

    if not mates:
        print("No mates found in the assembly features.")
    else:
        for mate in mates:
            # Extract mate information from 'featureData'
            mate_data = mate['featureData']
            mate_type = mate_data['mateType']
            if mate_type not in ['REVOLUTE', 'FASTENED']:
                continue

            # Get the instances connected by this mate
            mated_entities = mate_data['matedEntities']
            if len(mated_entities) != 2:
                continue  # Skip if not connecting exactly two parts

            # Extract instance IDs from matedOccurrences
            instance_ids = []
            for entity in mated_entities:
                occurrence_path = entity['matedOccurrence']
                if not occurrence_path:
                    continue
                instance_id = occurrence_path[-1].split('/')[-1]
                instance_ids.append(instance_id)

            if len(instance_ids) != 2:
                continue  # Unable to get both instance IDs

            instance_id1, instance_id2 = instance_ids
            body1 = part_bodies.get(instance_id1)
            body2 = part_bodies.get(instance_id2)

            if not body1 or not body2:
                continue  # Skip if bodies are not found

            # Get transforms of bodies
            transform1 = part_transforms[instance_id1]
            transform2 = part_transforms[instance_id2]

            # Compute relative position and orientation of body2 with respect to body1
            pos_body1 = transform1['pos']
            quat_body1 = transform1['quat']
            pos_body2 = transform2['pos']
            quat_body2 = transform2['quat']

            # Relative position
            pos_relative = pos_body2 - pos_body1

            # Rotate the relative position into body1's frame
            rot_body1_inv = R.from_quat(
                [quat_body1[1], quat_body1[2], quat_body1[3], quat_body1[0]]).inv()
            pos_relative_in_body1 = rot_body1_inv.apply(pos_relative)

            # Relative orientation
            rot_body1 = R.from_quat(
                [quat_body1[1], quat_body1[2], quat_body1[3], quat_body1[0]])
            rot_body2 = R.from_quat(
                [quat_body2[1], quat_body2[2], quat_body2[3], quat_body2[0]])
            rot_relative = rot_body1.inv() * rot_body2
            quat_relative = rot_relative.as_quat()  # x, y, z, w
            quat_relative = [quat_relative[3], quat_relative[0],
                             quat_relative[1], quat_relative[2]]  # w x y z

            # Update body2's position and orientation
            body2.attrib['pos'] = f"{pos_relative_in_body1[0]} {pos_relative_in_body1[1]} {pos_relative_in_body1[2]}"
            body2.attrib['quat'] = f"{quat_relative[0]} {quat_relative[1]} {quat_relative[2]} {quat_relative[3]}"

            # Remove body2 from worldbody if it's there, and reassign it under body1
            if body2 in worldbody:
                worldbody.remove(body2)
                body1.append(body2)

            if mate_type == 'REVOLUTE':
                # Use the mate origin as joint position
                mated_cs1 = mated_entities[0]['matedCS']
                mated_cs2 = mated_entities[1]['matedCS']

                # Use the matedCS of body2
                # Convert units if necessary
                joint_origin = np.array(mated_cs2['origin']) * 0.001

                # Compute joint position relative to body2
                pos_body2_world = pos_body2
                rot_body2_world = R.from_quat(
                    [quat_body2[1], quat_body2[2], quat_body2[3], quat_body2[0]])

                joint_pos_in_body2 = rot_body2_world.inv().apply(joint_origin - pos_body2_world)

                # Joint axis in body2's frame
                axis_vector_world = np.array(mated_cs2['zAxis'])
                joint_axis_in_body2 = rot_body2_world.inv().apply(axis_vector_world)

                SubElement(body2, 'joint', name=f"joint_{body2.attrib['name']}",
                           type='hinge',
                           pos=f"{joint_pos_in_body2[0]} {joint_pos_in_body2[1]} {joint_pos_in_body2[2]}",
                           axis=f"{joint_axis_in_body2[0]} {joint_axis_in_body2[1]} {joint_axis_in_body2[2]}")
            elif mate_type == 'FASTENED':
                # Fixed joint; no joint element needed in MJCF
                pass

    # Write the MJCF to a file
    tree = ElementTree(mjcf)
    with open('robot.xml', 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    main()
