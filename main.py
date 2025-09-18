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
            #指定控制：sim.data.ctrl=np.array([0, 0, 0, 0, 0, 0, 0, 0])
            sim.step() #mujoco.mj_step(sim.model, sim.data)和保存帧

    if runner.recorder_enabled:
        print(f"Saved MuJoCo video to {runner.output_path}")
    else:
        print("Video recording disabled; no video file was created.")


if __name__ == "__main__":
    main()
