"""ROS TF에 의존하지 않는 quaternion 보간 수학을 검사한다."""

import numpy as np
import pytest

from validator_inference.quaternion_math import quaternion_slerp


def test_slerp_normalizes_endpoints():
    assert quaternion_slerp([0, 0, 0, 2], [0, 0, 2, 0], 0.0) == pytest.approx(
        [0, 0, 0, 1]
    )
    assert quaternion_slerp([0, 0, 0, 2], [0, 0, 2, 0], 1.0) == pytest.approx(
        [0, 0, 1, 0]
    )


def test_slerp_uses_shortest_path_when_dot_is_negative():
    result = quaternion_slerp([0, 0, 0, 1], [0, 0, -1, -1], 0.5)

    expected_rotation = [0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)]
    assert abs(float(np.dot(result, expected_rotation))) == pytest.approx(1.0)


def test_slerp_returns_known_half_turn_midpoint():
    result = quaternion_slerp([0, 0, 0, 1], [0, 0, 1, 0], 0.5)

    assert result == pytest.approx([0, 0, 2**-0.5, 2**-0.5])


def test_slerp_nearly_identical_quaternions_return_finite_unit_midpoint():
    angle = 0.01
    result = quaternion_slerp(
        [0, 0, 0, 1],
        [0, 0, np.sin(angle), np.cos(angle)],
        0.5,
    )

    assert np.all(np.isfinite(result))
    assert np.linalg.norm(result) == pytest.approx(1.0, rel=0, abs=1e-12)
    assert result == pytest.approx(
        [0, 0, np.sin(angle / 2), np.cos(angle / 2)],
        rel=0,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    "q0,q1,fraction,expected",
    [
        (
            [0.2, -0.3, 0.4, 0.5],
            [-0.6, 0.1, 0.2, 0.7],
            0.37,
            [
                -0.08879411841550619,
                -0.24921883182949775,
                0.49162927971131254,
                0.82964222995631642,
            ],
        ),
        (
            [1.0, 2.0, 3.0, 4.0],
            [-4.0, 3.0, -2.0, 1.0],
            0.25,
            [
                -0.1107959109285632,
                0.54695745520570171,
                0.36629342818051769,
                0.74457432992829964,
            ],
        ),
    ],
)
def test_slerp_matches_tf_transformations_golden_values(
    q0, q1, fraction, expected
):
    assert quaternion_slerp(q0, q1, fraction) == pytest.approx(
        expected, rel=0, abs=1e-12
    )


@pytest.mark.parametrize(
    "invalid_quaternion",
    [
        [0, 0, 1],
        [0, 0, 0, 1, 2],
        np.zeros((2, 2)),
        1.0,
    ],
)
@pytest.mark.parametrize("invalid_index", [0, 1])
def test_slerp_rejects_inputs_without_exact_quaternion_shape(
    invalid_quaternion, invalid_index
):
    quaternions = [[0, 0, 0, 1], [0, 0, 1, 0]]
    quaternions[invalid_index] = invalid_quaternion

    with pytest.raises(ValueError, match="shape"):
        quaternion_slerp(quaternions[0], quaternions[1], 0.5)


@pytest.mark.parametrize(
    "invalid_quaternion",
    [
        [np.nan, 0, 0, 1],
        [0, np.inf, 0, 1],
        [0, 0, -np.inf, 1],
    ],
)
@pytest.mark.parametrize("invalid_index", [0, 1])
def test_slerp_rejects_nonfinite_quaternion_components(
    invalid_quaternion, invalid_index
):
    quaternions = [[0, 0, 0, 1], [0, 0, 1, 0]]
    quaternions[invalid_index] = invalid_quaternion

    with pytest.raises(ValueError, match="finite components"):
        quaternion_slerp(quaternions[0], quaternions[1], 0.5)


@pytest.mark.parametrize("invalid_index", [0, 1])
def test_slerp_rejects_overflowed_quaternion_norm(invalid_index):
    maximum = np.finfo(np.float64).max
    quaternions = [[0, 0, 0, 1], [0, 0, 1, 0]]
    quaternions[invalid_index] = [maximum, maximum, maximum, maximum]

    with pytest.raises(ValueError, match="finite nonzero norm"):
        quaternion_slerp(quaternions[0], quaternions[1], 0.5)


@pytest.mark.parametrize("zero_index", [0, 1])
def test_slerp_rejects_zero_norm_quaternion(zero_index):
    quaternions = [[0, 0, 0, 1], [0, 0, 1, 0]]
    quaternions[zero_index] = [0, 0, 0, 0]

    with pytest.raises(ValueError, match="nonzero"):
        quaternion_slerp(quaternions[0], quaternions[1], 0.5)


@pytest.mark.parametrize(
    "fraction",
    [np.nan, np.inf, -np.inf, [0.5], np.asarray([0.5])],
)
def test_slerp_rejects_nonfinite_or_nonscalar_fraction(fraction):
    with pytest.raises(ValueError, match="finite scalar"):
        quaternion_slerp([0, 0, 0, 1], [0, 0, 1, 0], fraction)
