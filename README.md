# video_mujoco

一个使用 MuJoCo 渲染 Franka 机械臂场景并通过 FFmpeg 录制 MP4 视频的最小示例。

## 外部依赖
- FFmpeg：用于从渲染图像流生成视频。确保命令行执行 `ffmpeg -version` 可以看到版本信息。
  - macOS：`brew install ffmpeg`
  - Ubuntu/Debian：`sudo apt-get install ffmpeg`
  - Windows：下载官方发行版或通过 `choco install ffmpeg`

## 运行示例
1. 确保已安装依赖并可以访问 `franka_emika_panda/scene.xml`。
2. 在仓库根目录执行：

   python main.py

3. 程序会模拟 10 秒钟并录制视频，默认输出文件为 `simulation.mp4`。

## Tips：
1.最好预先在xml中定义相机配置
2.最好预先在xml中定义分辨率

## 目录结构
```
.
├── main.py                      # 入口脚本
├── mujoco_video_recorder.py     # 视频录制工具类
├── franka_emika_panda/scene.xml # Franka 机械臂场景描述
└── simulation.mp4               # 示例运行后的输出视频（可选）
```
