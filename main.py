import mujoco

from mujoco_video_recorder import MujocoVideoRecorder


def main() -> None:
    model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)

    with MujocoVideoRecorder(model, data) as recorder:
        recorder.capture_frame(force=True)
        while data.time < 10.0:
            mujoco.mj_step(model, data)
            recorder.capture_frame()

    print(f"Saved MuJoCo video to {recorder.output_path}")


if __name__ == "__main__":
    main()
