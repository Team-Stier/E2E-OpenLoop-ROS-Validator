from pathlib import Path
from xml.etree import ElementTree


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_cmake_installs_opt_feeder_and_local_profile():
    package_root = _package_root()
    cmake = (package_root / "CMakeLists.txt").read_text()
    local_profile = package_root.parent / "config" / "profile.json"

    assert "catkin_install_python(PROGRAMS scripts/opt_feeder_node.py" in cmake
    assert "DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}" in cmake
    assert "${CMAKE_CURRENT_SOURCE_DIR}/../config/profile.json" in cmake
    assert "DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config" in cmake
    assert local_profile.is_file()


def test_cmake_installs_environment_cli_wrapper_and_reproducibility_inputs():
    """환경 CLI와 wrapper 및 lock 자료가 install space에 보존되는지 검사한다."""
    package_root = _package_root()
    cmake = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "scripts/openloop_cli.py" in cmake
    assert "catkin_install_python(" in cmake
    assert "scripts/model_env_exec" in cmake
    assert "DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts" in cmake
    assert "install(DIRECTORY requirements/" in cmake
    assert "DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/requirements" in cmake
    assert "config/model_environments.json" in cmake
    assert "DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config" in cmake


def test_runtime_package_declares_ros_smoke_dependencies():
    """설치된 호환성 smoke가 사용하는 ROS runtime 의존성을 선언하는지 검사한다."""
    package_xml = ElementTree.parse(_package_root() / "package.xml").getroot()
    runtime_dependencies = {
        element.text for element in package_xml.findall("exec_depend")
    }

    assert {"rosbag", "rosgraph", "rosgraph_msgs", "roslaunch"} <= runtime_dependencies
