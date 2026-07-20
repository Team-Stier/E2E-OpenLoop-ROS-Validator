"""README 코드 헤더 컨벤션을 회귀 검사한다."""

from pathlib import Path


def test_production_python_headers_describe_role_and_ros_io():
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
