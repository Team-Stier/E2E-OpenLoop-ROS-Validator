# Offline IMU Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested offline 4 Hz zero-phase roll/pitch filter and generate a raw-versus-filtered documentation plot from the existing analyzer CSV.

**Architecture:** One Python module owns CSV validation, irregular-time resampling, Butterworth filtering, atomic CSV output, and atomic headless plot output. It preserves every source row and column, appending only filtered roll/pitch columns; it does not alter any ROS runtime interface.

**Tech Stack:** ROS Noetic system Python 3, standard-library `csv` and `argparse`, NumPy, SciPy, Matplotlib, Catkin nosetests.

## Global Constraints

- Runtime source: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/imu_filtering.py`.
- Default filter: second-order Butterworth, cutoff `4.0 Hz`, applied with `sosfiltfilt`.
- Input timestamps and values must be finite and strictly increasing.
- Source CSV is never overwritten; output files are atomically replaced.
- Preserve every source row and column; append only `roll_deg_filtered` and `pitch_deg_filtered`.
- The PNG overlays raw and filtered roll/pitch and is documentation-only evidence.
- Do not change ROS topics, PointCloud, TF, BEV, model inputs, launch files, inference, or control.

---

### Task 1: Filtering Core

**Files:**
- Create: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/imu_filtering.py`
- Create: `src/Validator_pkg/horizontal_pkg/test/test_imu_filtering.py`

**Interfaces:**
- Produces: `FilteringError`, `FilterResult`, and `filter_orientation(header_stamp_ns, roll_deg, pitch_deg, cutoff_hz=4.0)`.
- `FilterResult` provides original-time filtered arrays, `sample_rate_hz`, and `cutoff_hz`.

- [ ] **Step 1: Write failing core tests**

Add tests that call `filter_orientation` with irregular timestamps and assert low-frequency preservation, high-frequency attenuation, and no measurable peak shift:

```python
result = filtering.filter_orientation(stamps, roll, pitch, cutoff_hz=4.0)
self.assertLess(rms(result.roll_deg_filtered - low), 0.05)
self.assertLess(rms(result.roll_deg_filtered - low), rms(roll - low) * 0.25)
self.assertLess(abs(peak_time(result.roll_deg_filtered) - peak_time(low)), 0.03)
```

Add separate tests for non-finite values, duplicate timestamps, reversed timestamps, non-positive cutoff, cutoff at Nyquist, and too-short input. Each raises `FilteringError` with a specific diagnostic fragment.

- [ ] **Step 2: Run core tests and verify RED**

```bash
source /opt/ros/noetic/setup.bash
PYTHONPATH=src/Validator_pkg/horizontal_pkg/src \
  /usr/bin/python3 -m unittest -v \
  src/Validator_pkg/horizontal_pkg/test/test_imu_filtering.py
```

Expected: import failure because `horizontal_pkg.imu_filtering` does not exist.

- [ ] **Step 3: Implement the minimal filter core**

Implement validation, median timestamp interval derivation, uniform-grid `numpy.interp`, `scipy.signal.butter(2, ..., output="sos")`, `scipy.signal.sosfiltfilt`, and interpolation back to original timestamps. Convert nanosecond timestamps to relative seconds before floating-point math.

- [ ] **Step 4: Run core tests and verify GREEN**

Run the Step 2 command. Expected: all core tests pass.

- [ ] **Step 5: Commit the filtering core**

```bash
git add src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/imu_filtering.py \
  src/Validator_pkg/horizontal_pkg/test/test_imu_filtering.py
git commit -m "feat: filter offline IMU orientation"
```

---

### Task 2: CSV, Plot, and CLI

**Files:**
- Modify: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/imu_filtering.py`
- Modify: `src/Validator_pkg/horizontal_pkg/test/test_imu_filtering.py`
- Modify: `src/Validator_pkg/horizontal_pkg/CMakeLists.txt`
- Modify: `src/Validator_pkg/horizontal_pkg/package.xml`
- Modify: `README.md`

**Interfaces:**
- Consumes: `filter_orientation(...) -> FilterResult`.
- Produces: `filter_csv(input_csv, output_csv, output_png, cutoff_hz=4.0)`, `main(argv=None) -> int`, and CLI arguments `--input-csv`, `--output-csv`, `--output-png`, `--cutoff-hz`.

- [ ] **Step 1: Write failing CSV, plot, and CLI tests**

Create a temporary analyzer-format CSV with extra columns. Assert that `filter_csv` preserves row count, order, values, and field order while appending the two filtered fields. Verify the PNG signature and non-zero size. Add tests that reject missing columns and identical input/output paths. Patch `os.replace` to fail and assert the old destination remains unchanged and the temporary file is removed. Add CLI success and expected-error exit tests.

- [ ] **Step 2: Run focused tests and verify RED**

Run the Task 1 test command. Expected: failures because CSV, plot, and CLI interfaces do not exist.

- [ ] **Step 3: Implement CSV, plot, and CLI behavior**

Use `csv.DictReader` and `csv.DictWriter`, temporary files in each destination directory, `os.replace`, and Matplotlib's `Agg` backend. Render two shared-x panels with gray raw traces and blue/orange filtered traces. Put `2nd-order Butterworth, cutoff 4.0 Hz, zero-phase` in the title. Print sample count, sample rate, cutoff, and raw/filtered standard deviations on success. Print `error: ...` to stderr and return `2` for expected failures.

- [ ] **Step 4: Declare dependencies and Catkin test discovery**

Add `test/test_imu_filtering.py` to `catkin_add_nosetests`. Declare `python3-numpy`, `python3-scipy`, and `python3-matplotlib` as runtime and test dependencies in `package.xml`. Document the exact CLI and output paths in the offline IMU section of `README.md`.

- [ ] **Step 5: Run focused and package tests and verify GREEN**

```bash
source /opt/ros/noetic/setup.bash
PYTHONPATH=src/Validator_pkg/horizontal_pkg/src \
  /usr/bin/python3 -m unittest -v \
  src/Validator_pkg/horizontal_pkg/test/test_imu_filtering.py \
  src/Validator_pkg/horizontal_pkg/test/test_imu_orientation_analyzer.py
```

Expected: all tests pass without warnings or errors.

- [ ] **Step 6: Commit CSV, plotting, CLI, and package wiring**

```bash
git add README.md src/Validator_pkg/horizontal_pkg
git commit -m "feat: export filtered IMU comparison"
```

---

### Task 3: Actual CSV Artifact and Full Verification

**Files:**
- Create: `src/Validator_pkg/horizontal_pkg/docs/imu_roll_pitch_filtered.png`
- Runtime output only, ignored: `records/horizontal/imu_roll_pitch_filtered.csv`

**Interfaces:**
- Consumes: Task 2 CLI and `records/horizontal/imu_roll_pitch.csv`.
- Produces: checked-in comparison PNG and console validation metrics.

- [ ] **Step 1: Run the tool on the current CSV**

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
/usr/bin/python3 -m horizontal_pkg.imu_filtering \
  --input-csv records/horizontal/imu_roll_pitch.csv \
  --output-csv records/horizontal/imu_roll_pitch_filtered.csv \
  --output-png src/Validator_pkg/horizontal_pkg/docs/imu_roll_pitch_filtered.png \
  --cutoff-hz 4.0
```

Expected: 10,882 rows, cutoff 4.0 Hz, and both output files created.

- [ ] **Step 2: Validate numerical attenuation and artifacts**

Read both angle series with system Python. Resample by the same median timestamp interval and use Welch PSD to calculate RMS above 4 Hz. Assert each filtered high-frequency RMS is below its raw counterpart. Use `file` and Pillow to verify that the PNG decodes and has non-zero dimensions.

- [ ] **Step 3: Run full workspace verification**

```bash
source /opt/ros/noetic/setup.bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
catkin_make run_tests
catkin_test_results --all build/test_results
/usr/bin/python3 -m py_compile \
  src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/imu_filtering.py
git diff --check
```

Expected: build and all tests pass, `catkin_test_results` reports zero failures, Python compilation succeeds, and `git diff --check` is silent.

- [ ] **Step 4: Visually inspect the generated image**

Open `src/Validator_pkg/horizontal_pkg/docs/imu_roll_pitch_filtered.png` and confirm two readable panels, correct raw/filtered legends, units, time axis, and filter title without clipped text.

- [ ] **Step 5: Commit the generated documentation artifact**

```bash
git add src/Validator_pkg/horizontal_pkg/docs/imu_roll_pitch_filtered.png
git commit -m "docs: add filtered IMU comparison plot"
```
