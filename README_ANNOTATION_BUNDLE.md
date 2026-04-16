# Annotation Bundle

This bundle contains the code and current annotation state needed for group members to continue the SUNRGBD labeling work.

It does **not** include the `SUNRGBD` dataset itself. Each teammate should download and extract SUNRGBD on their own machine.

## What Is Included

- `smart_home_rgbd/`
  - the annotation UI
  - data loading and manifest logic
- `artifacts/manual_state_labels_seed.csv`
  - the selected subset of scenes for the project
- `artifacts/manual_state_instance_manifest.json`
  - the per-object annotation manifest used by the UI
- `artifacts/manual_object_state_labels.json`
  - the current saved labels, including partial work already completed
- `artifacts/sunrgbd_scene_stats.json`
  - a small summary of the selected project data

## What Teammates Need To Do

1. Put this bundle in a working folder.
2. Download and extract the SUNRGBD dataset.
3. Make sure the extracted dataset folder is named `SUNRGBD` and sits next to this README.

Expected layout:

```text
some_folder/
  README_ANNOTATION_BUNDLE.md
  requirements.txt
  launch_annotation_ui.sh
  SUNRGBD/
  artifacts/
  smart_home_rgbd/
```

## Setup

Create a Python environment and install the minimal dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Launch

Option 1:

```bash
./launch_annotation_ui.sh
```

Option 2:

```bash
.venv/bin/python -m smart_home_rgbd.cli label-ui --port 8765
```

Then open:

```text
http://127.0.0.1:8765/
```

## Recommended Workflow

- Select an object instance on the right.
- If the polygon is slightly misplaced, use `move` mode and drag it.
- If the polygon is fundamentally wrong, check `Bad Polygon / Unreliable Shape`.
- Label the object state:
  - `lamp`: `on` / `off` / `unclear`
  - `monitor_or_tv`: `on` / `off` / `unclear`
  - `door`: `open` / `closed` / `unclear`
- Add optional scene-level energy label and notes.
- Click `Save`.

## Important Note About Collaboration

The file `artifacts/manual_object_state_labels.json` stores the current shared annotation progress.

If multiple people are labeling in parallel, do **not** each overwrite the same file independently forever.
Best practice:

- each person works on their own copy
- commit/push regularly if using GitHub
- merge JSON carefully, or coordinate by split / assigned scene ranges

## Current Saved Work

This bundle already includes the partial annotation work saved so far, so teammates should start from the existing JSON rather than from scratch.
