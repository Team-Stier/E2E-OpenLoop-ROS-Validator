"""README 코드 헤더와 운영 코드 문서화 컨벤션을 회귀 검사한다."""

import ast
from pathlib import Path


def test_production_python_headers_describe_role_and_ros_io():
    """모든 운영 Python module header가 README의 ROS 입출력 설명을 갖는지 검사한다."""
    package_root = Path(__file__).parents[1]
    production_files = sorted(
        list((package_root / "src" / "validator_inference").rglob("*.py"))
        + list((package_root / "scripts").glob("*.py"))
    )

    assert production_files
    for path in production_files:
        text = path.read_text(encoding="utf-8")
        header = text[:800]
        assert "입력:" in header, path
        assert "출력(pub):" in header, path
        assert "구독(sub):" in header, path


def test_every_production_class_and_function_has_a_korean_docstring():
    """모든 운영 class·function이 역할과 동작을 설명하는 한국어 docstring을 갖는지 검사한다."""
    package_root = Path(__file__).parents[1]
    production_files = sorted(
        list((package_root / "src" / "validator_inference").rglob("*.py"))
        + list((package_root / "scripts").glob("*.py"))
    )

    for path in production_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            documentation = ast.get_docstring(node)
            assert documentation, f"{path}:{node.lineno}:{node.name}"
            assert any("가" <= character <= "힣" for character in documentation), (
                f"{path}:{node.lineno}:{node.name}"
            )
