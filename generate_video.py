from moviepy.editor import ColorClip


def generate_video(output_path: str):
    """Placeholder video generator — creates a short blank clip."""
    clip = ColorClip(size=(640, 360), color=(
        50, 50, 50), duration=3)  # 3s gray clip
    clip.write_videofile(output_path, fps=24)
    clip.close()
