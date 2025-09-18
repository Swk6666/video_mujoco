import mujoco

from mujoco_video_recorder import MujocoSimulationRunner


ENABLE_VISUALIZATION = True
ENABLE_VIDEO_RECORDING = True
SIMULATION_DURATION = 10.0


def main() -> None:
    runner = MujocoSimulationRunner.from_xml(
        "franka_emika_panda/scene.xml",
        duration=SIMULATION_DURATION,
        enable_visualization=ENABLE_VISUALIZATION,
        enable_video_recording=ENABLE_VIDEO_RECORDING,
    )

    with runner as sim:
        while sim.should_continue():
            # Add any additional per-step logic here
            mujoco.mj_step(sim.model, sim.data)
            sim.post_step()

    if runner.recorder_enabled:
        print(f"Saved MuJoCo video to {runner.output_path}")
    else:
        print("Video recording disabled; no video file was created.")


if __name__ == "__main__":
    main()
