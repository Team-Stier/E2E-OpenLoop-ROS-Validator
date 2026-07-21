"""
모델 환경 candidate와 순수 Python ROS overlay의 원자적 lifecycle을 관리한다.
입력: 검증된 environment paths, build ID, ROS Python source tree와 overlay 경로.
출력(pub): 없음. candidate directory, overlay manifest와 활성화·정리 결과를 반환한다.
구독(sub): 없음.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import tempfile
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, Sequence

from validator_inference.model_environment import (
    ROS_OVERLAY_MANIFEST_FILENAME,
    SUPPORTED_MODEL_IDS,
    EnvironmentPaths,
    sha256_file,
)


PURE_ROS_PACKAGES = (
    "rospy",
    "rosgraph",
    "roslib",
    "genpy",
    "genmsg",
    "catkin",
    "roscpp",
    "rosgraph_msgs",
    "std_msgs",
    "sensor_msgs",
    "geometry_msgs",
    "nav_msgs",
    "tf2_msgs",
    "actionlib_msgs",
)
ROS_OVERLAY_MANIFEST_SCHEMA = 1
_BUILD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PACKAGE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_KEYS = {"schema", "packages", "files"}
_FILE_KEYS = {"source", "destination", "sha256"}
_NATIVE_SUFFIXES = (".so", ".pyd", ".dylib", ".dll")
_LIFECYCLE_LOCK_FILENAME = ".lifecycle.lock"


class EnvironmentSetupError(RuntimeError):
    """candidate 또는 overlay action 실패 원인과 단계를 보관한다."""

    def __init__(self, action: str, message: str) -> None:
        """실패 action과 사람이 조치할 수 있는 원인을 초기화한다."""
        self.action = action
        super().__init__(f"{action}: {message}")


@dataclass(frozen=True)
class RosOverlayFile:
    """복사한 ROS Python 파일의 source, 상대 destination과 hash를 보관한다."""

    source_path: Path
    destination_path: Path
    sha256: str


@dataclass(frozen=True)
class RosOverlayManifest:
    """한 candidate의 허용 ROS package와 파일 provenance를 보관한다."""

    schema: int
    packages: tuple[str, ...]
    files: tuple[RosOverlayFile, ...]
    manifest_path: Path


def _raise(action: str, message: str) -> None:
    """일관된 action prefix를 가진 setup domain 오류를 발생시킨다."""
    raise EnvironmentSetupError(action, message)


def _validate_environment_paths(paths: EnvironmentPaths, action: str) -> None:
    """EnvironmentPaths가 project/.venvs/MODEL에서 안전하게 유도됐는지 검사한다."""
    model_root = Path(paths.model_root)
    if (
        not model_root.is_absolute()
        or model_root.name not in SUPPORTED_MODEL_IDS
        or model_root.parent.name != ".venvs"
        or Path(paths.builds) != model_root / "builds"
        or Path(paths.current) != model_root / "current"
    ):
        _raise(action, "environment path layout is invalid")


def _validate_component(value: str, context: str, action: str) -> None:
    """build 또는 package 이름이 안전한 단일 filesystem component인지 검사한다."""
    pattern = _BUILD_ID if context == "build_id" else _PACKAGE_NAME
    if not isinstance(value, str) or not pattern.fullmatch(value) or value in {".", ".."}:
        _raise(action, f"{context} must be a safe single component")


def _canonical_absolute_posix_path(value: Any, context: str, action: str) -> Path:
    """입력 문자열을 정규화하지 않고 canonical absolute POSIX path인지 검사한다."""
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise EnvironmentSetupError(
            action, f"{context} must be a canonical absolute POSIX path"
        ) from exc
    if (
        not isinstance(raw, str)
        or not raw
        or "\\" in raw
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        _raise(action, f"{context} must be a canonical absolute POSIX path")
    pure = PurePosixPath(raw)
    if (
        not pure.is_absolute()
        or raw.startswith("//")
        or raw != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _raise(action, f"{context} must be a canonical absolute POSIX path")
    return Path(*pure.parts)


def _reject_existing_symlink_chain(path: Path, action: str, context: str) -> None:
    """존재하는 path와 상위 component 중 symlink가 있으면 따라가기 전에 거부한다."""
    value = Path(path)
    candidates = [value, *value.parents]
    for candidate in reversed(candidates):
        if candidate == Path(candidate.anchor):
            continue
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise EnvironmentSetupError(
                action, f"cannot inspect {context} path: {candidate}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            _raise(action, f"{context} path contains a symlink: {candidate}")


def _require_real_directory(path: Path, action: str, context: str) -> None:
    """path가 symlink가 아닌 기존 directory인지 lstat 기준으로 검사한다."""
    _reject_existing_symlink_chain(path, action, context)
    try:
        metadata = Path(path).lstat()
    except FileNotFoundError as exc:
        raise EnvironmentSetupError(action, f"{context} directory is missing: {path}") from exc
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot inspect {context}: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        _raise(action, f"{context} must be a directory: {path}")


def _ensure_real_directory(path: Path, action: str, context: str) -> None:
    """새 directory를 만들거나 기존 실제 directory만 재사용한다."""
    _reject_existing_symlink_chain(path, action, context)
    try:
        Path(path).mkdir()
    except FileExistsError:
        pass
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot create {context}: {path}") from exc
    _require_real_directory(path, action, context)


@contextmanager
def _lifecycle_lock(paths: EnvironmentPaths, action: str) -> Iterator[None]:
    """model root의 안전한 advisory lock으로 lifecycle 변경을 직렬화한다."""
    _validate_environment_paths(paths, action)
    model_root = Path(paths.model_root)
    _require_real_directory(model_root, action, "model root")
    lock_path = model_root / _LIFECYCLE_LOCK_FILENAME
    _reject_existing_symlink_chain(lock_path, action, "lifecycle lock")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if not no_follow:
        _raise(action, "lifecycle lock requires O_NOFOLLOW support")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | no_follow
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(str(lock_path), flags, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            _raise(action, "lifecycle lock must be a regular unlinked file")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        path_metadata = lock_path.lstat()
        descriptor_metadata = os.fstat(descriptor)
        if (
            stat.S_ISLNK(path_metadata.st_mode)
            or path_metadata.st_dev != descriptor_metadata.st_dev
            or path_metadata.st_ino != descriptor_metadata.st_ino
        ):
            _raise(action, "lifecycle lock path changed during acquisition")
    except EnvironmentSetupError:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise EnvironmentSetupError(
            action, f"cannot acquire lifecycle lock: {lock_path}"
        ) from exc

    body_failed = False
    try:
        yield
    except BaseException:
        body_failed = True
        raise
    finally:
        release_error: Optional[OSError] = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError as exc:
            release_error = exc
        try:
            os.close(descriptor)
        except OSError as exc:
            if release_error is None:
                release_error = exc
        if release_error is not None and not body_failed:
            raise EnvironmentSetupError(
                action, f"cannot release lifecycle lock: {lock_path}"
            ) from release_error


def create_candidate(paths: EnvironmentPaths, build_id: str) -> Path:
    """고유 build ID의 candidate를 움직이지 않을 최종 builds 경로에 만든다."""
    action = "create_candidate"
    _validate_environment_paths(paths, action)
    _validate_component(build_id, "build_id", action)
    model_root = Path(paths.model_root)
    venvs_root = model_root.parent
    _ensure_real_directory(venvs_root, action, ".venvs")
    _ensure_real_directory(model_root, action, "model root")
    _ensure_real_directory(Path(paths.builds), action, "builds")
    candidate = Path(paths.builds) / build_id
    try:
        candidate.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise EnvironmentSetupError(action, f"candidate already exists: {candidate}") from exc
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot create candidate: {candidate}") from exc
    _require_real_directory(candidate, action, "candidate")
    return candidate


def _validate_packages(packages: Sequence[str], action: str) -> tuple[str, ...]:
    """overlay package 목록이 exact 고정 allowlist인지 검사한다."""
    if isinstance(packages, (str, bytes)):
        _raise(action, "packages must be a sequence of package names")
    selected = tuple(packages)
    if not selected:
        _raise(action, "packages must not be empty")
    for package in selected:
        _validate_component(package, "package", action)
    if len(set(selected)) != len(selected):
        _raise(action, "packages must not contain duplicate names")
    if selected != PURE_ROS_PACKAGES:
        _raise(action, "packages must match the exact PURE_ROS_PACKAGES allowlist")
    return selected


def _is_native_filename(name: str) -> bool:
    """파일 이름이 Python extension 또는 native shared library인지 판별한다."""
    lower = name.lower()
    return lower.endswith(_NATIVE_SUFFIXES) or ".so." in lower


def _scan_python_tree(
    directory: Path,
    relative_directory: Path,
    action: str,
) -> list[tuple[Path, Path]]:
    """한 실제 package tree를 symlink 없이 순회해 Python source 목록을 만든다."""
    discovered: list[tuple[Path, Path]] = []
    try:
        with os.scandir(str(directory)) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot scan ROS package: {directory}") from exc
    for entry in ordered:
        source = directory / entry.name
        relative = relative_directory / entry.name
        _canonical_absolute_posix_path(source, "ROS source entry", action)
        if entry.is_symlink():
            _raise(action, f"ROS package tree contains a symlink: {source}")
        try:
            is_directory = entry.is_dir(follow_symlinks=False)
            is_file = entry.is_file(follow_symlinks=False)
        except OSError as exc:
            raise EnvironmentSetupError(action, f"cannot inspect ROS entry: {source}") from exc
        if is_directory:
            if entry.name == "__pycache__":
                continue
            discovered.extend(_scan_python_tree(source, relative, action))
            continue
        if not is_file:
            _raise(action, f"ROS package tree contains a special file: {source}")
        if _is_native_filename(entry.name):
            _raise(action, f"ROS package tree contains a native extension: {source}")
        if entry.name.endswith(".pyc"):
            continue
        if entry.name.endswith(".py"):
            discovered.append((source, relative))
    return discovered


def _source_package_entry(source_root: Path, package: str, action: str) -> Path:
    """source root에서 package entry를 follow 없이 찾아 실제 directory인지 확인한다."""
    try:
        with os.scandir(str(source_root)) as entries:
            entry = next((item for item in entries if item.name == package), None)
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot scan ROS source root: {source_root}") from exc
    if entry is None:
        _raise(action, f"declared ROS package is missing: {package}")
    if entry.is_symlink():
        _raise(action, f"declared ROS package is a symlink: {package}")
    try:
        is_directory = entry.is_dir(follow_symlinks=False)
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot inspect ROS package: {package}") from exc
    if not is_directory:
        _raise(action, f"declared ROS package must be a directory: {package}")
    return source_root / package


def _require_package_init(package_root: Path, package: str, action: str) -> None:
    """선언 ROS package가 symlink가 아닌 일반 __init__.py를 갖는지 검사한다."""
    initializer = package_root / "__init__.py"
    try:
        metadata = initializer.lstat()
    except FileNotFoundError as exc:
        raise EnvironmentSetupError(
            action, f"declared ROS package requires __init__.py: {package}"
        ) from exc
    except OSError as exc:
        raise EnvironmentSetupError(
            action, f"cannot inspect ROS package __init__.py: {package}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode):
        _raise(action, f"ROS package __init__.py is a symlink: {package}")
    if not stat.S_ISREG(metadata.st_mode):
        _raise(action, f"ROS package __init__.py must be a regular file: {package}")


def _module_is_copied(module: Sequence[str], copied: set[Path]) -> bool:
    """dotted module이 복사 예정 .py 또는 package/__init__.py로 존재하는지 판별한다."""
    if not module:
        return False
    module_path = Path(*module)
    return module_path.with_suffix(".py") in copied or (
        module_path / "__init__.py"
    ) in copied


@dataclass(frozen=True)
class _ParsedModule:
    """한 copied module의 AST와 static export 판정 자료를 보관한다."""

    relative_path: Path
    tree: ast.Module
    direct_bindings: frozenset[str]
    conditional_bindings: frozenset[str]
    wildcard_imports: tuple[ast.ImportFrom, ...]
    has_conditional_wildcard: bool
    has_dynamic_getattr: bool
    static_all_known: bool
    static_all: Optional[frozenset[str]]


def _target_names(target: ast.AST) -> set[str]:
    """assignment target가 현재 scope에 묶는 단순 name set을 반환한다."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, (ast.List, ast.Tuple)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_target_names(element))
        return names
    return set()


def _statement_binding_names(statement: ast.stmt) -> set[str]:
    """한 statement가 실행될 때 현재 scope에 묶을 수 있는 name을 찾는다."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {statement.name}
    if isinstance(statement, ast.Assign):
        names: set[str] = set()
        for target in statement.targets:
            names.update(_target_names(target))
        return names
    if isinstance(statement, ast.AnnAssign):
        return _target_names(statement.target) if statement.value is not None else set()
    if isinstance(statement, ast.AugAssign):
        return _target_names(statement.target)
    if isinstance(statement, ast.Import):
        return {
            alias.asname if alias.asname is not None else alias.name.split(".")[0]
            for alias in statement.names
        }
    if isinstance(statement, ast.ImportFrom):
        return {
            alias.asname if alias.asname is not None else alias.name
            for alias in statement.names
            if alias.name != "*"
        }
    return set()


def _nested_statement_bindings(statements: Sequence[ast.stmt]) -> tuple[set[str], bool]:
    """conditional module statement 아래의 잠재 binding과 wildcard 여부를 찾는다."""
    names: set[str] = set()
    has_wildcard = False
    for statement in statements:
        names.update(_statement_binding_names(statement))
        if isinstance(statement, ast.ImportFrom) and any(
            alias.name == "*" for alias in statement.names
        ):
            has_wildcard = True
        nested_groups: list[Sequence[ast.stmt]] = []
        if isinstance(statement, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            nested_groups.extend((statement.body, statement.orelse))
            if isinstance(statement, (ast.For, ast.AsyncFor)):
                names.update(_target_names(statement.target))
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            nested_groups.append(statement.body)
            for item in statement.items:
                if item.optional_vars is not None:
                    names.update(_target_names(item.optional_vars))
        elif isinstance(statement, ast.Try):
            nested_groups.extend((statement.body, statement.orelse, statement.finalbody))
            nested_groups.extend(handler.body for handler in statement.handlers)
        for group in nested_groups:
            nested_names, nested_wildcard = _nested_statement_bindings(group)
            names.update(nested_names)
            has_wildcard = has_wildcard or nested_wildcard
    return names, has_wildcard


def _literal_string_collection(value: ast.AST) -> Optional[frozenset[str]]:
    """literal string list/tuple/set이면 순서와 중복을 제거한 값을 반환한다."""
    try:
        evaluated = ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    if not isinstance(evaluated, (list, tuple, set, frozenset)) or not all(
        isinstance(item, str) for item in evaluated
    ):
        return None
    return frozenset(evaluated)


def _static_all_policy(
    tree: ast.Module, conditional_bindings: set[str]
) -> tuple[bool, Optional[frozenset[str]]]:
    """module의 static __all__ set, 기본 정책 또는 동적 여부를 판정한다."""
    known = True
    explicit: Optional[frozenset[str]] = None
    for statement in tree.body:
        assigned_value: Optional[ast.AST] = None
        assigns_all = False
        if isinstance(statement, ast.Assign):
            assigns_all = any("__all__" in _target_names(target) for target in statement.targets)
            if assigns_all:
                assigned_value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            assigns_all = "__all__" in _target_names(statement.target)
            assigned_value = statement.value
        if assigns_all:
            if assigned_value is None:
                known = False
                explicit = None
                continue
            literal = _literal_string_collection(assigned_value)
            known = literal is not None
            explicit = literal
            continue
        if isinstance(statement, ast.AugAssign) and "__all__" in _target_names(
            statement.target
        ):
            addition = _literal_string_collection(statement.value)
            if (
                known
                and explicit is not None
                and isinstance(statement.op, ast.Add)
                and addition is not None
            ):
                explicit = explicit | addition
            else:
                known = False
                explicit = None
            continue
        if isinstance(statement, ast.Delete) and any(
            "__all__" in _target_names(target) for target in statement.targets
        ):
            known = True
            explicit = None
            continue
        if "__all__" in _statement_binding_names(statement):
            known = False
            explicit = None
    if "__all__" in conditional_bindings:
        return False, None
    return known, explicit


def _parsed_module(relative: Path, tree: ast.Module) -> _ParsedModule:
    """module body에서 static/conditional export 판정 자료를 구성한다."""
    direct: set[str] = set()
    conditional: set[str] = set()
    wildcard_imports: list[ast.ImportFrom] = []
    has_conditional_wildcard = False
    for statement in tree.body:
        if isinstance(statement, ast.Delete):
            for target in statement.targets:
                direct.difference_update(_target_names(target))
            continue
        direct.update(_statement_binding_names(statement))
        if isinstance(statement, ast.ImportFrom) and any(
            alias.name == "*" for alias in statement.names
        ):
            wildcard_imports.append(statement)
        if isinstance(
            statement,
            (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try),
        ):
            nested, nested_wildcard = _nested_statement_bindings([statement])
            conditional.update(nested)
            has_conditional_wildcard = (
                has_conditional_wildcard or nested_wildcard
            )
    static_all_known, static_all = _static_all_policy(tree, conditional)
    return _ParsedModule(
        relative_path=relative,
        tree=tree,
        direct_bindings=frozenset(direct),
        conditional_bindings=frozenset(conditional),
        wildcard_imports=tuple(wildcard_imports),
        has_conditional_wildcard=has_conditional_wildcard,
        has_dynamic_getattr=(
            "__getattr__" in direct or "__getattr__" in conditional
        ),
        static_all_known=static_all_known,
        static_all=static_all,
    )


def _module_name(relative: Path) -> tuple[str, ...]:
    """overlay-relative Python path를 import module parts로 변환한다."""
    if relative.name == "__init__.py":
        return tuple(relative.parent.parts)
    return tuple(relative.with_suffix("").parts)


def _relative_import_module(
    source: Path, node: ast.ImportFrom, action: str
) -> tuple[str, ...]:
    """한 relative ImportFrom의 기준 package와 명시 module을 절대 parts로 바꾼다."""
    package = list(source.parent.parts)
    if node.level > len(package):
        _raise(
            action,
            f"import closure escapes top-level package in {source.as_posix()}",
        )
    parent_count = node.level - 1
    if parent_count:
        package = package[:-parent_count]
    if node.module:
        package.extend(node.module.split("."))
    return tuple(package)


def _require_imported_module(
    module: Sequence[str], copied: set[Path], source: Path, action: str
) -> None:
    """참조한 선택 package module이 copied set에 없으면 closure 오류를 낸다."""
    if not _module_is_copied(module, copied):
        _raise(
            action,
            "import closure is missing module "
            + ".".join(module)
            + f" referenced by {source.as_posix()}",
        )


class _StaticExportResolver:
    """copied module의 static export를 cache와 cycle guard로 판정한다."""

    def __init__(
        self,
        modules: Mapping[tuple[str, ...], _ParsedModule],
        packages: tuple[str, ...],
    ) -> None:
        """parsed module map과 selected package set을 보관한다."""
        self._modules = modules
        self._selected = set(packages)
        self._cache: dict[tuple[tuple[str, ...], str], Optional[bool]] = {}
        self._visiting: set[tuple[tuple[str, ...], str]] = set()

    def _wildcard_target(
        self, source: Path, node: ast.ImportFrom, action: str
    ) -> Optional[tuple[str, ...]]:
        """local/selected wildcard target은 parts, 외부 target은 None으로 반환한다."""
        if node.level:
            return _relative_import_module(source, node, action)
        if node.module:
            module = tuple(node.module.split("."))
            if module[0] in self._selected:
                return module
        return None

    def _star_export_status(
        self, module: tuple[str, ...], symbol: str, action: str
    ) -> Optional[bool]:
        """provider module의 wildcard import가 symbol을 묶는지 판정한다."""
        parsed = self._modules.get(module)
        if parsed is None:
            return False
        if parsed.static_all_known:
            if parsed.static_all is None:
                allowed = not symbol.startswith("_")
            else:
                allowed = symbol in parsed.static_all
            if not allowed:
                return False
            return self.export_status(module, symbol, action)
        status = self.export_status(module, symbol, action)
        return False if status is False else None

    def export_status(
        self, module: tuple[str, ...], symbol: str, action: str
    ) -> Optional[bool]:
        """module symbol을 known present/absent 또는 runtime unknown으로 판정한다."""
        key = (module, symbol)
        if key in self._cache:
            return self._cache[key]
        if key in self._visiting:
            return False
        parsed = self._modules.get(module)
        if parsed is None:
            self._cache[key] = False
            return False
        self._visiting.add(key)
        try:
            if symbol in parsed.direct_bindings:
                status: Optional[bool] = True
            else:
                unknown = (
                    symbol in parsed.conditional_bindings
                    or parsed.has_conditional_wildcard
                    or parsed.has_dynamic_getattr
                )
                status = None if unknown else False
                for wildcard in parsed.wildcard_imports:
                    target = self._wildcard_target(
                        parsed.relative_path, wildcard, action
                    )
                    if target is None:
                        status = None
                        continue
                    inherited = self._star_export_status(target, symbol, action)
                    if inherited is True:
                        status = True
                        break
                    if inherited is None:
                        status = None
        finally:
            self._visiting.remove(key)
        self._cache[key] = status
        return status


def _validate_imported_aliases(
    node: ast.ImportFrom,
    module: tuple[str, ...],
    source: Path,
    copied: set[Path],
    resolver: _StaticExportResolver,
    action: str,
) -> None:
    """from-import alias가 copied submodule 또는 base static export인지 검사한다."""
    for alias in node.names:
        if alias.name == "*":
            continue
        candidate = (*module, *alias.name.split("."))
        if _module_is_copied(candidate, copied):
            continue
        if resolver.export_status(module, alias.name, action) is False:
            _raise(
                action,
                "import closure is missing export "
                + alias.name
                + " from "
                + ".".join(module)
                + f" referenced by {source.as_posix()}",
            )


def _validate_import_node(
    node: ast.AST,
    source: Path,
    packages: tuple[str, ...],
    copied: set[Path],
    resolver: _StaticExportResolver,
    action: str,
) -> None:
    """한 AST import node가 선택 package copied closure 안에서 해석되는지 검사한다."""
    selected = set(packages)
    if isinstance(node, ast.Import):
        for alias in node.names:
            module = tuple(alias.name.split("."))
            if module[0] in selected:
                _require_imported_module(module, copied, source, action)
        return
    if not isinstance(node, ast.ImportFrom):
        return
    if node.level:
        module = _relative_import_module(source, node, action)
        _require_imported_module(module, copied, source, action)
        _validate_imported_aliases(
            node, module, source, copied, resolver, action
        )
        return
    if not node.module:
        return
    module = tuple(node.module.split("."))
    if module[0] in selected:
        _require_imported_module(module, copied, source, action)
        _validate_imported_aliases(
            node, module, source, copied, resolver, action
        )


def _validate_import_closure(
    discovered: Sequence[tuple[Path, Path]],
    packages: tuple[str, ...],
    action: str,
) -> None:
    """모든 복사 예정 Python source의 선택 package와 relative import closure를 검사한다."""
    copied = {relative for _source, relative in discovered}
    parsed_modules: dict[tuple[str, ...], _ParsedModule] = {}
    for source_path, relative in discovered:
        try:
            tree = ast.parse(source_path.read_bytes(), filename=str(source_path))
        except SyntaxError as exc:
            raise EnvironmentSetupError(
                action, f"Python syntax error in ROS source: {relative.as_posix()}"
            ) from exc
        except OSError as exc:
            raise EnvironmentSetupError(
                action, f"cannot read ROS Python source: {source_path}"
            ) from exc
        module = _module_name(relative)
        if module in parsed_modules:
            _raise(action, f"import closure has duplicate module: {'.'.join(module)}")
        parsed_modules[module] = _parsed_module(relative, tree)
    resolver = _StaticExportResolver(parsed_modules, packages)
    for parsed in parsed_modules.values():
        for node in ast.walk(parsed.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                _validate_import_node(
                    node,
                    parsed.relative_path,
                    packages,
                    copied,
                    resolver,
                    action,
                )


def _overlay_payload(
    packages: tuple[str, ...], files: tuple[RosOverlayFile, ...]
) -> dict[str, Any]:
    """overlay manifest dataclass를 deterministic JSON 기본 자료형으로 변환한다."""
    return {
        "schema": ROS_OVERLAY_MANIFEST_SCHEMA,
        "packages": list(packages),
        "files": [
            {
                "source": str(record.source_path),
                "destination": record.destination_path.as_posix(),
                "sha256": record.sha256,
            }
            for record in files
        ],
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any], action: str) -> None:
    """canonical JSON을 sibling 임시 파일에 fsync한 뒤 원자적으로 교체한다."""
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        + b"\n"
    )
    temporary: Optional[Path] = None
    try:
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), delete=False
        )
        temporary = Path(handle.name)
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        temporary = None
        directory_fd = os.open(str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot write ROS overlay manifest: {path}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def build_ros_overlay(
    source_root: Path,
    destination: Path,
    packages: Sequence[str] = PURE_ROS_PACKAGES,
) -> RosOverlayManifest:
    """허용 ROS package의 Python source만 새 candidate overlay로 복사한다."""
    action = "build_ros_overlay"
    source = _canonical_absolute_posix_path(source_root, "source_root", action)
    target = _canonical_absolute_posix_path(destination, "destination", action)
    _require_real_directory(source, action, "ROS source root")
    _reject_existing_symlink_chain(target.parent, action, "overlay parent")
    if target.is_symlink() or target.exists():
        _raise(action, f"destination already exists: {target}")
    selected_packages = _validate_packages(packages, action)
    manifest_path = target.parent / ROS_OVERLAY_MANIFEST_FILENAME
    if manifest_path.is_symlink() or manifest_path.exists():
        _raise(action, f"overlay manifest already exists: {manifest_path}")

    discovered: list[tuple[Path, Path]] = []
    for package in selected_packages:
        package_root = _source_package_entry(source, package, action)
        _require_package_init(package_root, package, action)
        package_files = _scan_python_tree(package_root, Path(package), action)
        if not package_files:
            _raise(action, f"declared ROS package has no Python source: {package}")
        discovered.extend(package_files)
    discovered.sort(key=lambda item: item[1].as_posix())
    _validate_import_closure(discovered, selected_packages, action)

    try:
        target.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot create ROS overlay: {target}") from exc
    records: list[RosOverlayFile] = []
    for source_file, relative in discovered:
        destination_file = target / relative
        try:
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                str(source_file), str(destination_file), follow_symlinks=False
            )
            metadata = destination_file.lstat()
        except OSError as exc:
            raise EnvironmentSetupError(
                action, f"cannot copy ROS Python source: {source_file}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            _raise(action, f"copied ROS entry is not a regular file: {destination_file}")
        records.append(
            RosOverlayFile(
                source_path=source_file.absolute(),
                destination_path=relative,
                sha256=sha256_file(destination_file),
            )
        )

    # 전체-file AST closure는 Noetic의 optional rosnode/rosservice/em import 때문에
    # 거짓 실패한다. 실제 도달 가능한 import closure는 후속 target runtime smoke가 검증한다.
    files = tuple(records)
    _atomic_write_json(
        manifest_path, _overlay_payload(selected_packages, files), action
    )
    return RosOverlayManifest(
        schema=ROS_OVERLAY_MANIFEST_SCHEMA,
        packages=selected_packages,
        files=files,
        manifest_path=manifest_path.resolve(strict=True),
    )


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """overlay manifest JSON object의 duplicate key를 손상으로 거부한다."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _raise("verify_ros_overlay", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str, action: str
) -> None:
    """manifest object의 누락 key와 알 수 없는 key를 거부한다."""
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        _raise(action, f"{context} missing keys: {', '.join(sorted(missing))}")
    if unknown:
        _raise(action, f"{context} unknown keys: {', '.join(sorted(unknown))}")


def _safe_destination(value: Any, packages: tuple[str, ...], action: str) -> Path:
    """manifest destination을 package 아래 canonical 상대 POSIX Python path로 검증한다."""
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        _raise(action, "file destination must be a safe relative POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or value != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or not pure.parts
        or pure.parts[0] not in packages
        or not value.endswith(".py")
    ):
        _raise(action, "file destination must stay inside a declared package")
    return Path(*pure.parts)


def _safe_source(value: Any, destination: Path, action: str) -> Path:
    """manifest source를 destination suffix와 일치하는 절대 경로로 검증한다."""
    source = _canonical_absolute_posix_path(value, "file source", action)
    if len(source.parts) < len(destination.parts) or tuple(
        source.parts[-len(destination.parts) :]
    ) != destination.parts:
        _raise(action, "file source must match its destination suffix")
    return source


def _read_overlay_manifest(path: Path, action: str) -> RosOverlayManifest:
    """overlay manifest bytes를 strict immutable record로 읽고 검증한다."""
    if not path.is_absolute():
        _raise(action, "manifest path must be absolute")
    _reject_existing_symlink_chain(path, action, "manifest")
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise EnvironmentSetupError(action, f"manifest is missing: {path}") from exc
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot inspect manifest: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        _raise(action, f"manifest is not a regular file: {path}")
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot read manifest: {path}") from exc
    try:
        payload = json.loads(
            encoded.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except EnvironmentSetupError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise EnvironmentSetupError(action, "manifest is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        _raise(action, "manifest must be a JSON object")
    _require_exact_keys(payload, _MANIFEST_KEYS, "manifest", action)
    if (
        not isinstance(payload["schema"], int)
        or isinstance(payload["schema"], bool)
        or payload["schema"] != ROS_OVERLAY_MANIFEST_SCHEMA
    ):
        _raise(action, f"manifest schema must be {ROS_OVERLAY_MANIFEST_SCHEMA}")
    raw_packages = payload["packages"]
    if not isinstance(raw_packages, list):
        _raise(action, "manifest packages must be an array")
    packages = _validate_packages(raw_packages, action)
    raw_files = payload["files"]
    if not isinstance(raw_files, list) or not raw_files:
        _raise(action, "manifest files must be a nonempty array")
    records: list[RosOverlayFile] = []
    destinations: set[Path] = set()
    sources: set[Path] = set()
    package_counts = {package: 0 for package in packages}
    for index, raw_file in enumerate(raw_files):
        context = f"manifest.files[{index}]"
        if not isinstance(raw_file, dict):
            _raise(action, f"{context} must be an object")
        _require_exact_keys(raw_file, _FILE_KEYS, context, action)
        destination = _safe_destination(raw_file["destination"], packages, action)
        source = _safe_source(raw_file["source"], destination, action)
        digest = raw_file["sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            _raise(action, f"{context} SHA-256 is invalid")
        if destination in destinations:
            _raise(action, f"duplicate file destination: {destination.as_posix()}")
        if source in sources:
            _raise(action, f"duplicate file source: {source}")
        destinations.add(destination)
        sources.add(source)
        package_counts[destination.parts[0]] += 1
        records.append(RosOverlayFile(source, destination, digest))
    missing_packages = sorted(
        package for package, count in package_counts.items() if count == 0
    )
    if missing_packages:
        _raise(action, f"manifest package has no files: {', '.join(missing_packages)}")
    if tuple(record.destination_path.as_posix() for record in records) != tuple(
        sorted(record.destination_path.as_posix() for record in records)
    ):
        _raise(action, "manifest file records must be sorted")
    return RosOverlayManifest(
        schema=ROS_OVERLAY_MANIFEST_SCHEMA,
        packages=packages,
        files=tuple(records),
        manifest_path=path,
    )


def _scan_overlay_tree(destination: Path, action: str) -> tuple[set[Path], set[Path]]:
    """overlay 실제 directory와 file set을 symlink를 따라가지 않고 수집한다."""
    actual_files: set[Path] = set()
    actual_directories: set[Path] = set()

    def scan(directory: Path, relative_directory: Path) -> None:
        """현재 실제 directory의 자식 entry를 재귀적으로 검사한다."""
        try:
            with os.scandir(str(directory)) as entries:
                ordered = sorted(entries, key=lambda entry: entry.name)
        except OSError as exc:
            raise EnvironmentSetupError(action, f"cannot scan overlay: {directory}") from exc
        for entry in ordered:
            path = directory / entry.name
            relative = relative_directory / entry.name
            if entry.is_symlink():
                _raise(action, f"overlay contains a symlink: {relative.as_posix()}")
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError as exc:
                raise EnvironmentSetupError(action, f"cannot inspect overlay: {path}") from exc
            if is_directory:
                actual_directories.add(relative)
                scan(path, relative)
            elif is_file:
                if _is_native_filename(entry.name):
                    _raise(action, f"overlay contains a native extension: {relative.as_posix()}")
                actual_files.add(relative)
            else:
                _raise(action, f"overlay contains a special file: {relative.as_posix()}")

    scan(destination, Path())
    return actual_files, actual_directories


def _expected_overlay_directories(files: Sequence[RosOverlayFile]) -> set[Path]:
    """manifest file의 모든 overlay-relative parent directory set을 만든다."""
    expected: set[Path] = set()
    for record in files:
        parent = record.destination_path.parent
        while parent != Path("."):
            expected.add(parent)
            parent = parent.parent
    return expected


def _verify_overlay_file(record: RosOverlayFile, destination: Path, action: str) -> None:
    """한 provenance source와 복사 destination이 일반 파일이고 hash가 같은지 검사한다."""
    source = record.source_path
    target = destination / record.destination_path
    for path, context in ((source, "source"), (target, "destination")):
        _reject_existing_symlink_chain(path, action, context)
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise EnvironmentSetupError(
                action, f"recorded {context} file is missing: {path}"
            ) from exc
        except OSError as exc:
            raise EnvironmentSetupError(
                action, f"cannot inspect recorded {context} file: {path}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            _raise(action, f"recorded {context} is not a regular file: {path}")
        try:
            digest = sha256_file(path)
        except OSError as exc:
            raise EnvironmentSetupError(action, f"cannot hash {context} file: {path}") from exc
        if digest != record.sha256:
            _raise(action, f"recorded {context} file hash changed: {path}")


def verify_ros_overlay(destination: Path, manifest_path: Path) -> str:
    """overlay manifest와 exact tree를 검증하고 manifest 파일 SHA256을 반환한다."""
    action = "verify_ros_overlay"
    target = Path(destination)
    manifest_file = Path(manifest_path)
    if not target.is_absolute():
        _raise(action, "destination must be an absolute path")
    if manifest_file != target.parent / ROS_OVERLAY_MANIFEST_FILENAME:
        _raise(action, "manifest path must be the fixed overlay sibling")
    _require_real_directory(target, action, "overlay destination")
    manifest = _read_overlay_manifest(manifest_file, action)
    actual_files, actual_directories = _scan_overlay_tree(target, action)
    expected_files = {record.destination_path for record in manifest.files}
    expected_directories = _expected_overlay_directories(manifest.files)
    missing_files = expected_files - actual_files
    extra_files = actual_files - expected_files
    missing_directories = expected_directories - actual_directories
    extra_directories = actual_directories - expected_directories
    if missing_files:
        _raise(
            action,
            f"overlay file is missing: {sorted(path.as_posix() for path in missing_files)[0]}",
        )
    if extra_files:
        _raise(
            action,
            f"overlay contains an extra file: {sorted(path.as_posix() for path in extra_files)[0]}",
        )
    if missing_directories:
        _raise(
            action,
            "overlay directory is missing: "
            + sorted(path.as_posix() for path in missing_directories)[0],
        )
    if extra_directories:
        _raise(
            action,
            "overlay contains an extra directory: "
            + sorted(path.as_posix() for path in extra_directories)[0],
        )
    for record in manifest.files:
        _verify_overlay_file(record, target, action)
    try:
        return sha256_file(manifest_file)
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot hash manifest: {manifest_file}") from exc


def _validate_candidate(
    paths: EnvironmentPaths, candidate: Path, action: str
) -> tuple[Path, Path]:
    """candidate가 builds immediate 실제 directory인지 검증해 상대 target을 반환한다."""
    _validate_environment_paths(paths, action)
    model_root = Path(paths.model_root)
    builds = Path(paths.builds)
    selected = Path(candidate)
    if not selected.is_absolute() or selected.parent != builds:
        _raise(action, "candidate must be an immediate child of builds")
    _validate_component(selected.name, "build_id", action)
    try:
        metadata = selected.lstat()
    except FileNotFoundError as exc:
        raise EnvironmentSetupError(action, f"candidate is missing: {selected}") from exc
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot inspect candidate: {selected}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        _raise(action, f"candidate must not be a symlink: {selected}")
    if not stat.S_ISDIR(metadata.st_mode):
        _raise(action, f"candidate must be a real directory: {selected}")
    _require_real_directory(model_root, action, "model root")
    _require_real_directory(builds, action, "builds")
    relative = Path("builds") / selected.name
    return selected, relative


def activate_candidate(paths: EnvironmentPaths, candidate: Path) -> None:
    """candidate를 임시 상대 symlink와 os.replace로 원자적으로 활성화한다."""
    action = "activate_candidate"
    with _lifecycle_lock(paths, action):
        _, relative = _validate_candidate(paths, Path(candidate), action)
        current = Path(paths.current)
        model_root = Path(paths.model_root)
        if not current.is_symlink() and current.exists():
            _raise(action, f"current exists and is not a symlink: {current}")
        temporary: Optional[Path] = None
        try:
            for _attempt in range(32):
                candidate_temp = (
                    model_root / f".current.{os.getpid()}.{secrets.token_hex(8)}"
                )
                try:
                    os.symlink(relative.as_posix(), str(candidate_temp))
                    temporary = candidate_temp
                    break
                except FileExistsError:
                    continue
            if temporary is None:
                _raise(action, "cannot allocate a temporary current symlink")
            os.replace(str(temporary), str(current))
            temporary = None
        except EnvironmentSetupError:
            raise
        except OSError as exc:
            raise EnvironmentSetupError(
                action, f"activate replace failed for {current}"
            ) from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


def _active_build(paths: EnvironmentPaths, action: str) -> Optional[Path]:
    """current가 없으면 None, 있으면 안전한 active immediate build를 반환한다."""
    current = Path(paths.current)
    if current.is_symlink():
        try:
            relative = Path(os.readlink(str(current)))
        except OSError as exc:
            raise EnvironmentSetupError(action, f"cannot read current symlink: {current}") from exc
        if (
            relative.is_absolute()
            or len(relative.parts) != 2
            or relative.parts[0] != "builds"
        ):
            _raise(action, "current symlink target is malformed")
        _validate_component(relative.parts[1], "build_id", action)
        active = Path(paths.model_root) / relative
        try:
            metadata = active.lstat()
        except FileNotFoundError as exc:
            raise EnvironmentSetupError(action, "current symlink is dangling") from exc
        except OSError as exc:
            raise EnvironmentSetupError(action, f"cannot inspect active build: {active}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            _raise(action, "current must select a real immediate build directory")
        return active
    if current.exists():
        _raise(action, "current exists and is not a symlink")
    return None


def prune_inactive_builds(paths: EnvironmentPaths) -> tuple[Path, ...]:
    """active build를 보존하고 검증된 inactive immediate directory만 정리한다."""
    action = "prune_inactive_builds"
    _validate_environment_paths(paths, action)
    model_root = Path(paths.model_root)
    builds = Path(paths.builds)
    if not model_root.exists() and not model_root.is_symlink():
        return ()
    with _lifecycle_lock(paths, action):
        active = _active_build(paths, action)
        if not builds.exists() and not builds.is_symlink():
            if active is not None:
                _raise(action, "active build exists while builds directory is missing")
            return ()
        _require_real_directory(builds, action, "builds")
        candidates: list[Path] = []
        try:
            with os.scandir(str(builds)) as entries:
                ordered = sorted(entries, key=lambda entry: entry.name)
        except OSError as exc:
            raise EnvironmentSetupError(action, f"cannot scan builds: {builds}") from exc
        for entry in ordered:
            candidate = builds / entry.name
            _validate_component(entry.name, "build_id", action)
            if entry.is_symlink():
                _raise(action, f"suspicious symlink in builds: {candidate}")
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError as exc:
                raise EnvironmentSetupError(
                    action, f"cannot inspect build: {candidate}"
                ) from exc
            if not is_directory:
                _raise(action, f"suspicious non-directory entry in builds: {candidate}")
            candidates.append(candidate)
        if active is not None and active not in candidates:
            _raise(action, "active build is not an immediate builds entry")
        inactive = tuple(candidate for candidate in candidates if candidate != active)
        for candidate in inactive:
            try:
                shutil.rmtree(str(candidate))
            except OSError as exc:
                raise EnvironmentSetupError(
                    action, f"cannot prune build: {candidate}"
                ) from exc
        return inactive
