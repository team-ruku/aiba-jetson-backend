import ffmpeg
import cv2
import subprocess
import shlex
import time

cap = cv2.VideoCapture(0)
rtsp_url = "rtsp://localhost:8554/stream"


def spawn_ffmpeg_proc():
    proc = (
        ffmpeg.input("FaceTime", format="avfoundation", pix_fmt="uyvy422", framerate=30)
        .output(
            rtsp_url,
            codec="libx264",
            listen=1,
            pix_fmt="yuv420p",
            preset="ultrafast",
            format="rtsp",
        )
        .global_args("-re")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    return proc


def spawn_ffmpeg_proc_alternative():  # using subprocess
    proc_command = f"ffmpeg -report -re -y -f rawvideo -vcodec rawvideo -s 420x360 -r 30 -pix_fmt 24 -i - -an -vcodec libx264 -f rtsp {rtsp_url}"

    proc = subprocess.Popen(shlex.split(proc_command), stdin=subprocess.PIPE)

    return proc


if __name__ == "__main__":
    # proc = spawn_ffmpeg_proc()
    proc_alternative = spawn_ffmpeg_proc_alternative()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        proc_alternative.stdin.write(frame.tobytes())
        time.sleep(0.03)

    cap.release()
    # proc.stdin.close()
    # proc.wait()

    proc_alternative.stdin.close()
    proc_alternative.wait()
