from contextlib import ExitStack

import mujoco

from mujoco_video_recorder import MujocoVideoRecorder


ENABLE_VISUALIZATION = True
ENABLE_VIDEO_RECORDING = True
SIMULATION_DURATION = 10.0


def main() -> None:
    model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)

    with ExitStack() as stack:
        recorder = stack.enter_context(
            MujocoVideoRecorder(model, data, enabled=ENABLE_VIDEO_RECORDING)
        )
        viewer = None
        if ENABLE_VISUALIZATION:
            viewer = stack.enter_context(mujoco.viewer.launch_passive(model, data))

        if recorder.enabled:
            recorder.capture_frame(force=True)

        while data.time < SIMULATION_DURATION and (viewer is None or viewer.is_running()):
            mujoco.mj_step(model, data)
            if recorder.enabled:
                recorder.capture_frame()
            if viewer is not None:
                viewer.sync()

    if recorder.enabled:
        print(f"Saved MuJoCo video to {recorder.output_path}")
    else:
        print("Video recording disabled; no video file was created.")


if __name__ == "__main__":
    main()
