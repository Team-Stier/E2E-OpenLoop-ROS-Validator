from pathlib import Path


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
