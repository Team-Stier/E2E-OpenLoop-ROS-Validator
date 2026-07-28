#!/usr/bin/env python3
"""Contract tests for the repository-local ``bev`` launcher."""

from pathlib import Path
import os
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "run_lidar_bev_debug.sh"


class BevShellCommandContractTest(unittest.TestCase):
    def test_launcher_exists_is_executable_and_has_valid_bash_syntax(self):
        self.assertTrue(SCRIPT_PATH.is_file(), str(SCRIPT_PATH))
        self.assertTrue(os.access(str(SCRIPT_PATH), os.X_OK))

        result = subprocess.run(
            ["bash", "-n", str(SCRIPT_PATH)],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_launcher_contains_the_required_runtime_contract(self):
        self.assertTrue(SCRIPT_PATH.is_file(), str(SCRIPT_PATH))
        script = SCRIPT_PATH.read_text(encoding="utf-8")

        required_fragments = (
            '${BASH_SOURCE[0]}',
            "rosbag/2026-07-16-17-13-14.bag",
            "source /opt/ros/noetic/setup.bash",
            'source "$REPOSITORY_ROOT/devel/setup.bash"',
            "trap cleanup EXIT INT TERM HUP",
            "rosnode list",
            "/lidar_attitude_debugger",
            "rosparam set /use_sim_time true",
            "roslaunch Validator_pkg lidar_attitude_debug.launch show_rviz:=true",
            "rosbag play --clock --loop",
            'kill -INT "$ROSLAUNCH_PID"',
            'kill -INT "$ROSCORE_PID"',
        )
        for fragment in required_fragments:
            self.assertIn(fragment, script)

        self.assertNotIn("pkill", script)
        self.assertNotIn("killall", script)


if __name__ == "__main__":
    unittest.main()
