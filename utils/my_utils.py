import numpy as np
from PIL import Image , ImageDraw , ImageFont
import supervision as sv

MAX_VIDEO_LENGTH_SEC = 1

def draw_text_on_image(
        frame: np.ndarray,
        text: str = "",
) -> Image:
    image = Image.fromarray(frame)

    draw = ImageDraw.Draw(image)

    image_width, image_height = image.size
    em_value = image_height / 50

    text_box_width, text_box_height = 10 * em_value, 4 * em_value
    margin = em_value
    font_size = em_value

    font = ImageFont.truetype("arial.ttf", font_size)
    
    x = image_width - text_box_width - margin
    y = image_height - text_box_height - margin
    draw.text((x, y), text, fill="white", font=font)

    return image



def calculate_end_frame_index(source_video_path: str) -> int:
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    return min(
        video_info.total_frames,
        video_info.fps * MAX_VIDEO_LENGTH_SEC
    )

