import xml.etree.ElementTree as ET
import yaml
import os

class MJCFHandler:
    def __init__(self, path: str, mass_config_path: str=None) -> None:
        """Initialize the XMLHandler class."""
        self.scene_path = path
        self.mass_config_path = mass_config_path
        self.scene_path_to_robot_path(self.scene_path)
        self.model_tree = ET.parse(self.model_path)
        self.model_root = self.model_tree.getroot()

        self.worldbody = self.model_root.find('worldbody')
        assert self.worldbody is not None, "worldbody tag not found"

        self.geoms = self.worldbody.findall('.//geom')
        self.mass_dict: dict[str,dict[str,float]] = {}
        self.mass_config_export_path = self.scene_dir + "/mass_config.yaml"
        # if mass_config doesn't exist, save new yaml file
        if self.mass_config_path is not None:
            self.mass_dict = self.load_mass_config(self.mass_config_path)
        elif os.path.exists(self.mass_config_export_path):
            self.mass_dict = self.load_mass_config(self.mass_config_export_path)
        else:
            self.get_mass_of_geoms()
            self.save_mass_config()

    def scene_path_to_robot_path(self, scene_path: str) -> str:
        """Convert the scene path to robot path."""
        self.scene_tree = ET.parse(scene_path)
        self.scene_root = self.scene_tree.getroot()
        # extract the file value of include tag
        model_subpath = self.scene_root.find("include").get("file")
        # get directory from scene path
        self.scene_dir = "/".join(scene_path.split("/")[:-1])
        self.model_path = f"{self.scene_dir}/{model_subpath}"

    def get_mass_of_geoms(self) -> dict[str, float]:
        """Get the mass of the geoms in the model."""
        for geom in self.geoms:
            mass = geom.get('mass')
            assert mass is not None, "mass attribute not found"
            name = geom.get('mesh')
            self.mass_dict[name] = {"mass": float(mass), "scale": 1}

    def load_mass_config(self, path):
        """Load the mass config from path"""
        assert path is not None, "mass_config_path not set"
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        return data
    
    def save_mass_config(self) -> None:
        """Save the mass config to a yaml file"""
        with open(self.mass_config_export_path, "w") as file:
            yaml.dump(self.mass_dict, file)

    def update_mass(self) -> None:
        """Apply the mass scale to the geoms in the xml."""
        for geom in self.geoms:
            name = geom.get('mesh')
            new_mass = self.mass_dict[name]["mass"] * self.mass_dict[name]["scale"]
            geom.set('mass', str(new_mass))

    def export_xml_scene(self) -> None:
        """Export the modified xml to a file."""
        new_model_path = self.model_path.replace(".xml", "_temp.xml")
        self.model_tree.write(new_model_path)
        new_model_name = new_model_path.split("/")[-1]
        self.scene_root.find("include").set("file", new_model_name)
        self.new_scene_path = self.scene_path.replace(".xml", "_temp.xml")
        self.scene_tree.write(self.new_scene_path)


if __name__ == "__main__":
    pass