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
