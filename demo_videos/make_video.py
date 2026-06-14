import os
import cv2
import imageio
from glob import glob

# ==========================
# 설정
# ==========================
episode_dir = "/home/lsk/physicalai/batch_outputs/episode_000037"  # 이미지가 저장된 디렉토리 경로
fps = 20

output_mp4 = os.path.join(episode_dir, "episode_video.mp4")
output_gif = os.path.join(episode_dir, "episode_video.gif")

# ==========================
# 이미지 목록 읽기
# ==========================
image_extensions = ["*.png", "*.jpg", "*.jpeg"]

image_files = []
for ext in image_extensions:
    image_files.extend(glob(os.path.join(episode_dir, ext)))

image_files = sorted(image_files)

if len(image_files) == 0:
    raise ValueError(f"이미지를 찾을 수 없습니다: {episode_dir}")

print(f"총 {len(image_files)}장의 이미지 발견")

# ==========================
# MP4 생성
# ==========================
first_img = cv2.imread(image_files[0])
height, width = first_img.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(
    output_mp4,
    fourcc,
    fps,
    (width, height)
)

for img_path in image_files:
    frame = cv2.imread(img_path)

    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height))

    video_writer.write(frame)

video_writer.release()

print(f"MP4 저장 완료: {output_mp4}")

# ==========================
# GIF 생성
# ==========================
gif_frames = []

for img_path in image_files:
    img = imageio.imread(img_path)
    gif_frames.append(img)

imageio.mimsave(
    output_gif,
    gif_frames,
    fps=fps
)

print(f"GIF 저장 완료: {output_gif}")