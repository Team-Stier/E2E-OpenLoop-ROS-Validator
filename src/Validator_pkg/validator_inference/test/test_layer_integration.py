"""Static regression tests for the preprocessing/inference/visualization contract.

These tests deliberately avoid importing ROS nodes.  They are intended to catch
cross-package wiring regressions on developer hosts that do not have ROS sourced.
"""

import ast
import json
from pathlib import Path
from xml.etree import ElementTree


def _repository_root() -> Path:
    return next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "README.md").is_file() and (parent / "src").is_dir()
    )


REPOSITORY_ROOT = _repository_root()
VALIDATOR_GROUP = REPOSITORY_ROOT / "src" / "Validator_pkg"
INFERENCE_PACKAGE = VALIDATOR_GROUP / "validator_inference"
PIPELINE_PACKAGE = VALIDATOR_GROUP / "validator_pipeline"
PROFILE_PATH = VALIDATOR_GROUP / "config" / "profile.json"


def _profiles_by_id():
    document = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    return {profile["id"]: profile for profile in document["profiles"]}


def _required_inputs_by_key(profile):
    return {item["key"]: item for item in profile["required_inputs"]}


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a literal assignment in {path}")


def _image_publisher_topics():
    """Evaluate literal/image format publisher arguments without importing ROS."""
    path = PIPELINE_PACKAGE / "scripts" / "image_adaptor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    topics = set()

    class PublisherVisitor(ast.NodeVisitor):
        def __init__(self):
            self.environment = {}

        def visit_DictComp(self, node):
            if len(node.generators) != 1:
                return self.generic_visit(node)
            generator = node.generators[0]
            if not isinstance(generator.target, ast.Name):
                return self.generic_visit(node)
            try:
                values = ast.literal_eval(generator.iter)
            except (ValueError, TypeError):
                return self.generic_visit(node)

            variable = generator.target.id
            previous = self.environment.get(variable)
            had_previous = variable in self.environment
            for value in values:
                self.environment[variable] = value
                self.visit(node.key)
                self.visit(node.value)
            if had_previous:
                self.environment[variable] = previous
            else:
                self.environment.pop(variable, None)

        def visit_Call(self, node):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "Publisher"
                and node.args
            ):
                topic = self._evaluate_string(node.args[0])
                if topic is not None:
                    topics.add(topic)
            self.generic_visit(node)

        def _evaluate_string(self, node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "format"
                and isinstance(node.func.value, ast.Constant)
                and isinstance(node.func.value.value, str)
            ):
                arguments = []
                for argument in node.args:
                    if not isinstance(argument, ast.Name):
                        return None
                    if argument.id not in self.environment:
                        return None
                    arguments.append(self.environment[argument.id])
                return node.func.value.value.format(*arguments)
            return None

    PublisherVisitor().visit(tree)
    return topics


def _simple_yaml_mapping(path: Path):
    """Parse the scalar/list subset used by the checked geometry config."""
    root = {}
    stack = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indentation = len(content) - len(content.lstrip())
        key, separator, raw_value = content.strip().partition(":")
        assert separator, f"unsupported YAML line: {raw_line!r}"

        while indentation <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        raw_value = raw_value.strip()
        if not raw_value:
            value = {}
            parent[key] = value
            stack.append((indentation, value))
            continue

        lowered = raw_value.lower()
        if lowered in {"true", "false"}:
            value = lowered == "true"
        else:
            value = ast.literal_eval(raw_value)
        parent[key] = value
    return root


def _function_linspace_arguments(path: Path, function_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    arguments = []
    for node in ast.walk(function):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "linspace"
        ):
            continue
        arguments.append(tuple(ast.literal_eval(arg) for arg in node.args[:3]))
    return arguments


def test_preprocessing_topics_match_every_model_profile():
    profiles = _profiles_by_id()
    image_topics = _image_publisher_topics()
    lidar_topics = set(
        _literal_assignment(
            PIPELINE_PACKAGE / "scripts" / "lidar_bev_processor.py",
            "MODEL_OUTPUT_TOPICS",
        ).values()
    )

    expected_topics = {
        "tfv6": {"camera_main": "/image/tfv6/processed"},
        "interfuser": {
            "camera_front_224": "/image/interfuser/front",
            "camera_left_128": "/image/interfuser/left",
            "camera_right_128": "/image/interfuser/right",
            "camera_front_center_128": "/image/interfuser/center",
            "lidar_bev": "/lidar/interfuser/processed",
        },
        "diffusiondrive_v2": {
            "camera_main": "/image/diffusiondrive_v2/processed",
            "lidar_bev": "/lidar/diffusiondrive_v2/processed",
        },
        "sparsedrive_v2": {
            "camera_left": "/image/sparsedrive_v2/left",
            "camera_main": "/image/sparsedrive_v2/front",
            "camera_right": "/image/sparsedrive_v2/right",
        },
    }

    for model_id, expected in expected_topics.items():
        actual = _required_inputs_by_key(profiles[model_id])
        assert {key: actual[key]["topic"] for key in expected} == expected
        for key, topic in expected.items():
            producer_topics = lidar_topics if key == "lidar_bev" else image_topics
            assert topic in producer_topics, f"no preprocessing producer for {model_id}:{key}"


def test_tfv6_keeps_raw_ten_sweep_lidar_contract_in_inference():
    profile = _profiles_by_id()["tfv6"]
    required = _required_inputs_by_key(profile)
    histories = {
        item["key"]: item for item in profile["buffer_requirements"]["history_inputs"]
    }

    assert required["lidar_raw"]["topic"] == "/lidar3D"
    assert required["lidar_raw"]["ros_type"] == "sensor_msgs/PointCloud2"
    assert "lidar_bev" not in required
    assert histories["lidar_raw"]["topic"] == "/lidar3D"
    assert histories["lidar_raw"]["fifo_depth"] == 10
    assert histories["lidar_raw"]["paired_pose"] == "map_to_base_link at each sweep stamp"
    assert histories["lidar_raw"]["pose_interpolation"].startswith(
        "linear translation plus quaternion slerp"
    )

    # The preprocessing package may expose a diagnostic TFv6 raster, but the
    # inference profile must never consume it in place of raw sweep history.
    diagnostic_topic = _literal_assignment(
        PIPELINE_PACKAGE / "scripts" / "lidar_bev_processor.py",
        "MODEL_OUTPUT_TOPICS",
    )["tfv6"]
    assert diagnostic_topic == "/lidar/tfv6/processed"
    assert all(item["topic"] != diagnostic_topic for item in profile["required_inputs"])


def test_interfuser_records_rig_compatibility_and_exact_histogram_bounds():
    profile = _profiles_by_id()["interfuser"]
    assert "camera_rig_differs_from_carla_training" in profile["compatibility_flags"]

    lidar_contract = _required_inputs_by_key(profile)["lidar_bev"][
        "preprocess_contract"
    ].replace(" ", "")
    assert "xbins=np.linspace(-28,29,225)" in lidar_contract
    assert "ybins=np.linspace(-28,0,225)" in lidar_contract
    assert "3x224x224" in lidar_contract

    linspace_arguments = _function_linspace_arguments(
        PIPELINE_PACKAGE / "scripts" / "lidar_bev_processor.py",
        "prepare_interfuser",
    )
    assert (-28.0, 29.0, 225) in linspace_arguments
    assert (-28.0, 0.0, 225) in linspace_arguments


def test_integrated_launch_wires_one_model_id_through_all_layers():
    launch_path = PIPELINE_PACKAGE / "launch" / "openloop_validator.launch"
    root = ElementTree.parse(launch_path).getroot()
    top_level_args = {element.attrib["name"]: element for element in root.findall("arg")}
    assert "model_id" in top_level_args
    assert "project_root" in top_level_args
    assert top_level_args["project_root"].attrib["default"] == (
        "$(eval '/'.join(find('Validator_pkg').split('/')[:-3]))"
    )
    assert top_level_args["model_env_root"].attrib["default"] == (
        "$(arg project_root)/.venvs"
    )
    assert top_level_args["record_dir"].attrib["default"] == (
        "$(arg project_root)/records/openloop"
    )
    assert top_level_args["sensor_config"].attrib["default"] == (
        "$(arg project_root)/src/Validator_pkg/validator_pipeline/config/0716demo.json"
    )

    includes = {
        Path(element.attrib["file"]).name: element for element in root.findall("include")
    }
    assert set(includes) == {
        "camera_bev_pipeline.launch",
        "openloop_inference.launch",
        "bev_visualization.launch",
    }

    preprocessing_args = {
        arg.attrib["name"]: arg.attrib.get("value")
        for arg in includes["camera_bev_pipeline.launch"].findall("arg")
    }
    inference_args = {
        arg.attrib["name"]: arg.attrib.get("value")
        for arg in includes["openloop_inference.launch"].findall("arg")
    }
    visualization_args = {
        arg.attrib["name"]: arg.attrib.get("value")
        for arg in includes["bev_visualization.launch"].findall("arg")
    }
    assert preprocessing_args["model_profiles"] == "$(arg model_id)"
    assert preprocessing_args["config_path"] == "$(arg sensor_config)"
    assert inference_args["model_id"] == "$(arg model_id)"
    assert inference_args["profile_file"] == "$(arg profile_file)"
    assert inference_args["project_root"] == "$(arg project_root)"
    assert inference_args["model_env_root"] == "$(arg model_env_root)"
    assert inference_args["record_dir"] == "$(arg record_dir)"
    assert visualization_args["profile_file"] == "$(arg profile_file)"
    for geometry_argument in ("forward_range", "lateral_range", "resolution"):
        expected = f"$(arg {geometry_argument})"
        assert preprocessing_args[geometry_argument] == expected
        assert visualization_args[geometry_argument] == expected

    preprocessing = ElementTree.parse(
        PIPELINE_PACKAGE / "launch" / "camera_bev_pipeline.launch"
    ).getroot()
    image_adaptor = next(
        node
        for node in preprocessing.findall("node")
        if node.attrib["name"] == "image_adaptor"
    )
    image_parameters = {
        parameter.attrib["name"]: parameter.attrib["value"]
        for parameter in image_adaptor.findall("param")
    }
    assert image_parameters["config_path"] == "$(arg config_path)"


def test_visualization_geometry_matches_preprocessing_and_path_frame_transform():
    geometry = _simple_yaml_mapping(
        PIPELINE_PACKAGE / "config" / "bev_geometry.yaml"
    )
    assert geometry["frame_id"] == "base_link"
    assert (geometry["width"], geometry["height"]) == (600, 500)
    assert (geometry["x_min"], geometry["x_max"]) == (0.0, 25.0)
    assert (geometry["y_min"], geometry["y_max"]) == (-15.0, 15.0)
    assert geometry["resolution"] == 0.05

    profile_document = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    transform = profile_document["common_inference_policy"][
        "static_transforms"
    ]["T_base_from_lidar"]
    assert transform == {
        "parent_frame_id": "base_link",
        "child_frame_id": "Lidar3D-4",
        "translation_m": [2.0, 0.0, 1.5],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }

    visualizer_source = (
        PIPELINE_PACKAGE / "scripts" / "bev_visualizer.py"
    ).read_text(encoding="utf-8")
    assert 'payload["common_inference_policy"]["static_transforms"]' in visualizer_source
    assert '"/bev_geometry/waypoint_transform"' not in visualizer_source


def test_validator_group_contains_three_discoverable_sibling_packages():
    assert not (VALIDATOR_GROUP / "package.xml").exists()
    assert not (VALIDATOR_GROUP / "CMakeLists.txt").exists()

    expected_directories = {
        "validator_inference": "validator_inference",
        "validator_msgs": "validator_msgs",
        "validator_pipeline": "Validator_pkg",
    }
    actual = {}
    for directory_name, expected_package_name in expected_directories.items():
        package_directory = VALIDATOR_GROUP / directory_name
        package_xml = package_directory / "package.xml"
        cmake_lists = package_directory / "CMakeLists.txt"
        assert package_xml.is_file()
        assert cmake_lists.is_file()
        actual[directory_name] = ElementTree.parse(package_xml).findtext("name")
        assert actual[directory_name] == expected_package_name

    assert set(actual.values()) == {
        "Validator_pkg",
        "validator_inference",
        "validator_msgs",
    }
