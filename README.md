# 3D Spatial Intelligence for Smart-Home RGB-D Scenes

This repo contains the shared annotation pipeline and current label state for the project.

It does **not** include the `SUNRGBD` dataset itself. Each teammate should download the dataset separately and place it in a local folder named `SUNRGBD/` next to this README.

## Included

- `smart_home_rgbd/`
  - browser-based annotation UI
  - scene/instance manifest logic
- `artifacts/manual_state_labels_seed.csv`
  - selected subset of scenes for the project
- `artifacts/manual_state_instance_manifest.json`
  - per-object target manifest used by the UI
- `artifacts/manual_object_state_labels.json`
  - current saved labels and in-progress annotation work
- `requirements.txt`
- `launch_annotation_ui.sh`

## Expected Layout

```text
repo_root/
  README.md
  requirements.txt
  launch_annotation_ui.sh
  SUNRGBD/
  artifacts/
  smart_home_rgbd/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
./launch_annotation_ui.sh
```

Then open:

```text
http://127.0.0.1:8765/
```

## Annotation Notes

- Use `move` mode if the SUNRGBD overlay is slightly misaligned.
- Check `Bad Polygon / Unreliable Shape` when the original polygon is clearly poor.
- Click `Save` often.
- The shared progress lives in `artifacts/manual_object_state_labels.json`.

## Collaboration

- Do not commit the full SUNRGBD dataset.
- Commit only code and the shared annotation files.
- If multiple people annotate in parallel, merge `artifacts/manual_object_state_labels.json` carefully.
