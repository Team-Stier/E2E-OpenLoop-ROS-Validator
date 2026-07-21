from pathlib import Path
from xml.etree import ElementTree


def _repository_root() -> Path:
    return next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "README.md").is_file() and (parent / "src").is_dir()
    )


def test_validator_group_exposes_three_sibling_catkin_packages():
    validator_group = _repository_root() / "src" / "Validator_pkg"

    assert not (validator_group / "package.xml").exists()

    package_names = {
        ElementTree.parse(package_xml).findtext("name")
        for package_xml in validator_group.glob("*/package.xml")
    }
    assert package_names == {"Validator_pkg", "validator_inference", "validator_msgs"}


def test_environment_install_inputs_exist_in_source_package():
    """환경 manifest와 네 모델의 입력·hash lock이 source package에 존재하는지 검사한다."""
    package_root = _repository_root() / "src" / "Validator_pkg" / "validator_inference"

    assert (package_root / "config" / "model_environments.json").is_file()
    for model_id in (
        "interfuser",
        "tfv6",
        "diffusiondrive_v2",
        "sparsedrive_v2",
    ):
        assert (package_root / "requirements" / f"{model_id}.in").is_file()
        assert (package_root / "requirements" / f"{model_id}.lock").is_file()


def test_operator_documentation_uses_canonical_three_command_workflow():
    """운영 문서가 setup, run, bag의 단순 명령을 기본 절차로 제시하는지 검사한다."""
    root = _repository_root()
    canonical = (
        "./openloop setup interfuser",
        "./openloop interfuser --rviz",
        "./openloop bag",
    )
    forbidden = (
        "rosrun validator_inference fetch_model_assets.py --model-id interfuser",
        "roslaunch Validator_pkg openloop_validator.launch model_id:=interfuser",
        "rosbag play --clock rosbag/2026-07-16-17-13-14.bag",
    )

    for document in (root / "README.md", root / "docs" / "openloop_validator.md"):
        text = document.read_text(encoding="utf-8")
        for command in canonical:
            assert command in text, document
        for command in forbidden:
            assert command not in text, document

        assert "./openloop verify interfuser" in text
        assert "./openloop prune interfuser" in text
        assert "project_root:=/" in text
        assert "model_env_root:=/" in text
        assert "third-party compatibility artifact" in text
        assert "uv 0.11.14" in text
        assert "Ubuntu 20.04" in text
        assert "ROS Noetic" in text
        assert "profile_file" in text
        assert "record_dir" in text


def test_pipeline_documentation_separates_normal_cli_from_direct_launch():
    """BEV 문서도 단순 CLI를 기본으로 두고 raw launch는 고급 절차로 격리한다."""
    root = _repository_root()
    document = (
        root
        / "src/Validator_pkg/validator_pipeline/README_BEV_VISUALIZATION_SETUP.md"
    )
    text = document.read_text(encoding="utf-8")

    assert "### 통합 실행(권장)" in text
    assert "### 고급 직접 launch" in text
    recommended = text.split("### 통합 실행(권장)", 1)[1].split(
        "### 고급 직접 launch", 1
    )[0]
    assert "./openloop setup interfuser" in recommended
    assert "./openloop interfuser --rviz" in recommended
    assert "./openloop bag" in recommended
    assert "roslaunch Validator_pkg openloop_validator.launch" not in recommended

    advanced = text.split("### 고급 직접 launch", 1)[1]
    assert "roslaunch Validator_pkg openloop_validator.launch" in advanced
    assert "project_root:=/" in advanced
    assert "model_env_root:=/" in advanced
    assert "profile_file:=/" in advanced
    assert "sensor_config:=/" in advanced
