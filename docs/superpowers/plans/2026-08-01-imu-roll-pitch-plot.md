# IMU Roll/Pitch Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible CSV-to-PNG command and check in a readable roll/pitch result under `horizontal_pkg/docs`.

**Architecture:** A focused offline Python module validates the analyzer CSV into an immutable time-series value and renders two headless Matplotlib axes. Package documentation records the exact regeneration command and interpretation boundary; no ROS node or sensor-processing path changes.

**Tech Stack:** Ubuntu, ROS1 Noetic, Catkin, Python 3, standard-library `csv`/`unittest`, Matplotlib with the `Agg` backend.

## Global Constraints

- Input is the CSV output of `horizontal_pkg.imu_orientation_analyzer`.
- Required columns are exactly `elapsed_s`, `roll_deg`, and `pitch_deg`; unrelated analyzer columns remain accepted.
- Required values must be numeric and finite, and elapsed timestamps must be nondecreasing.
- The documentation image path is `src/Validator_pkg/horizontal_pkg/docs/imu_roll_pitch.png`.
- The documentation notes path is `src/Validator_pkg/horizontal_pkg/docs/README.md`.
- The checked-in image uses `records/horizontal/imu_roll_pitch.csv`, currently 10,882 samples over 307.153 seconds.
- The graph describes raw IMU orientation-derived roll/pitch, not LiDAR horizontalization or validated compensation signs.
- No ROS node, topic, PointCloud, BEV output, model input, launch file, or analyzer CSV-schema changes.
- Work stays local on `feature/imu-orientation-analyzer`; do not push, merge, or open a PR.

---

## File map

| File | Responsibility |
| --- | --- |
| `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/imu_orientation_plotter.py` | Validate analyzer CSV, summarize series, render PNG, and expose the CLI. |
| `src/Validator_pkg/horizontal_pkg/test/test_imu_orientation_plotter.py` | Unit tests for CSV rejection, rendering, and CLI behavior. |
| `src/Validator_pkg/horizontal_pkg/package.xml` | Declare the `python3-matplotlib` runtime dependency. |
| `src/Validator_pkg/horizontal_pkg/CMakeLists.txt` | Register the plotter test with Catkin. |
| `src/Validator_pkg/horizontal_pkg/docs/README.md` | Explain the image, observed ranges, and regeneration command. |
| `src/Validator_pkg/horizontal_pkg/docs/imu_roll_pitch.png` | Actual two-axis plot generated from the current CSV. |

---

### Task 1: Validated CSV Loader

**Files:**
- Create: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/imu_orientation_plotter.py`
- Create: `src/Validator_pkg/horizontal_pkg/test/test_imu_orientation_plotter.py`

**Interfaces:**
- Produces: `PlotError(RuntimeError)` for rejected input.
- Produces: frozen `OrientationSeries(elapsed_s: tuple, roll_deg: tuple, pitch_deg: tuple)`.
- Produces: `load_orientation_csv(input_path) -> OrientationSeries`.

- [ ] **Step 1: Write failing loader tests**

Create a temporary CSV helper and tests equivalent to:

```python
def write_csv(path, rows, fields=("elapsed_s", "roll_deg", "pitch_deg")):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def test_loads_valid_rows_in_order(self):
    write_csv(self.path, [
        {"elapsed_s": "0.0", "roll_deg": "1.0", "pitch_deg": "-0.5"},
        {"elapsed_s": "0.1", "roll_deg": "2.0", "pitch_deg": "0.5"},
    ])
    series = plotter.load_orientation_csv(self.path)
    self.assertEqual(series.elapsed_s, (0.0, 0.1))
    self.assertEqual(series.roll_deg, (1.0, 2.0))
    self.assertEqual(series.pitch_deg, (-0.5, 0.5))
```

Add separate assertions that a nonexistent file, header-only CSV, missing
`pitch_deg`, `nan`, and elapsed values `[0.1, 0.0]` each raise `PlotError`
with a message that names the rejected condition.

- [ ] **Step 2: Run the test to verify it fails**

```bash
source /opt/ros/noetic/setup.bash
PYTHONPATH=src/Validator_pkg/horizontal_pkg/src \
  /usr/bin/python3 src/Validator_pkg/horizontal_pkg/test/test_imu_orientation_plotter.py
```

Expected: FAIL because `horizontal_pkg.imu_orientation_plotter` does not exist.

- [ ] **Step 3: Implement the minimal loader**

Use `csv.DictReader`, `math.isfinite`, and `Path`. Require all three column
names, convert values using `float`, reject an empty row set, and reject a
sample when `elapsed_s < previous_elapsed_s`. Preserve equal timestamps and
return tuples in their input order.

Use stable messages beginning with `CSV does not exist`, `CSV is empty`,
`CSV is missing required columns`, `CSV row N contains an invalid value`, and
`elapsed_s decreases at CSV row N` so CLI failures remain understandable.

- [ ] **Step 4: Run the loader tests and verify they pass**

Run the Step 2 command. Expected: every loader test passes.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/imu_orientation_plotter.py \
  src/Validator_pkg/horizontal_pkg/test/test_imu_orientation_plotter.py
git commit -m "feat: validate IMU attitude plot input"
```

---

### Task 2: Headless PNG Renderer and CLI

**Files:**
- Modify: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/imu_orientation_plotter.py`
- Modify: `src/Validator_pkg/horizontal_pkg/test/test_imu_orientation_plotter.py`
- Modify: `src/Validator_pkg/horizontal_pkg/package.xml`
- Modify: `src/Validator_pkg/horizontal_pkg/CMakeLists.txt`

**Interfaces:**
- Consumes: `OrientationSeries` from Task 1.
- Produces: frozen `AngleStats(minimum: float, median: float, maximum: float)`.
- Produces: `summarize(values) -> AngleStats`.
- Produces: `render_orientation_plot(series, output_path) -> None`.
- Produces: `main(argv=None) -> int` with required `--input-csv` and `--output-png`.

- [ ] **Step 1: Write failing summary, renderer, and CLI tests**

Add tests that assert `summarize((2.0, 1.0, 3.0))` returns min/median/max
`1.0/2.0/3.0`. Render a two-sample `OrientationSeries` into a nested temporary
directory and assert the file starts with the eight-byte PNG signature
`b"\x89PNG\r\n\x1a\n"` and has nonzero length. Assert a valid CLI call returns
0 and creates a PNG, while a missing CSV returns 2 and writes an `error:` line
to stderr.

- [ ] **Step 2: Run the tests and verify the new cases fail**

Run the Task 1 test command. Expected: the loader cases pass and the new cases
fail because `summarize`, `render_orientation_plot`, and `main` are missing.

- [ ] **Step 3: Implement headless rendering**

Select `matplotlib.use("Agg")` before importing `matplotlib.pyplot`. Create a
`12 x 7` inch figure at 150 DPI with two vertically stacked axes sharing X.
Render roll in blue and pitch in orange, add dashed zero-degree lines, grids,
legends, `Angle [deg]` Y labels, and `Elapsed time [s]` on the lower axis.
Set the figure title to `Raw IMU Orientation: Roll and Pitch`.

Add a compact monospace figure annotation containing sample count, duration,
and min/median/max to three decimals for both signals. Use `tight_layout` with
space for the annotation. Create the output parent directory and save explicitly
as PNG before closing the figure in a `finally` block.

- [ ] **Step 4: Implement the CLI and package declarations**

Parse required `--input-csv` and `--output-png`. On success print input path,
output path, sample count, duration, and both angle summaries. Catch
`PlotError`, `OSError`, and `ValueError`, print `error: ...` to stderr, and
return 2; finish module execution with `raise SystemExit(main())`.

Add `<exec_depend>python3-matplotlib</exec_depend>` to `package.xml`. Register
`test/test_imu_orientation_plotter.py` with a second `catkin_add_nosetests`
entry in the existing `CATKIN_ENABLE_TESTING` block.

- [ ] **Step 5: Run unit and Catkin-native tests**

```bash
source /opt/ros/noetic/setup.bash
PYTHONPATH=src/Validator_pkg/horizontal_pkg/src \
  /usr/bin/python3 src/Validator_pkg/horizontal_pkg/test/test_imu_orientation_plotter.py
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
catkin_make run_tests_horizontal_pkg -DPYTHON_EXECUTABLE=/usr/bin/python3
catkin_test_results --verbose
```

Expected: plotter unit tests, existing analyzer tests, Catkin build, and the
`horizontal_pkg` Catkin test target all pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/Validator_pkg/horizontal_pkg/package.xml \
  src/Validator_pkg/horizontal_pkg/CMakeLists.txt \
  src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/imu_orientation_plotter.py \
  src/Validator_pkg/horizontal_pkg/test/test_imu_orientation_plotter.py
git commit -m "feat: render IMU roll pitch plot"
```

---

### Task 3: Generate and Document the Actual Result

**Files:**
- Create: `src/Validator_pkg/horizontal_pkg/docs/README.md`
- Create: `src/Validator_pkg/horizontal_pkg/docs/imu_roll_pitch.png`

**Interfaces:**
- Consumes: `records/horizontal/imu_roll_pitch.csv` and the Task 2 CLI.
- Produces: the checked-in documentation image and exact regeneration command.

- [ ] **Step 1: Generate the documentation PNG**

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
/usr/bin/python3 -m horizontal_pkg.imu_orientation_plotter \
  --input-csv records/horizontal/imu_roll_pitch.csv \
  --output-png src/Validator_pkg/horizontal_pkg/docs/imu_roll_pitch.png
```

Expected: one 1800 x 1050 PNG generated from 10,882 samples over 307.153
seconds, and terminal min/median/max values matching the CSV.

- [ ] **Step 2: Write the package documentation**

In `horizontal_pkg/docs/README.md`, embed `![IMU roll and pitch](imu_roll_pitch.png)`
and include the Step 1 command. Record these CSV-derived ranges:

```text
roll  min=-2.911 deg, median=0.310 deg, max=3.665 deg
pitch min=-0.834 deg, median=0.088 deg, max=2.133 deg
```

State that positive/negative signs follow the IMU frame and ROS quaternion
conversion, and that the image alone does not validate LiDAR compensation.

- [ ] **Step 3: Inspect the image and verify its file contract**

Open the PNG and confirm both traces, zero lines, labels, legends, title, and
statistics are readable. Then run:

```bash
file src/Validator_pkg/horizontal_pkg/docs/imu_roll_pitch.png
/usr/bin/python3 - <<'PY'
from pathlib import Path
p = Path("src/Validator_pkg/horizontal_pkg/docs/imu_roll_pitch.png")
data = p.read_bytes()
assert data.startswith(b"\x89PNG\r\n\x1a\n")
assert len(data) > 10_000
print(p, len(data))
PY
git diff --check
```

Expected: `file` reports a 1800 x 1050 PNG, the signature/size assertion
passes, and `git diff --check` reports no errors.

- [ ] **Step 4: Run final regression verification**

Repeat Task 2 Step 5 from a freshly sourced Noetic environment. Expected: all
analyzer and plotter package tests pass without changing any ROS topic or model
contract.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/Validator_pkg/horizontal_pkg/docs
git commit -m "docs: add IMU roll pitch result"
```
