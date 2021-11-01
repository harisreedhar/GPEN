import os
import cv2
import uuid
import argparse
from pathlib import Path
import subprocess, platform
from face_enhancement import FaceEnhancement

def getTempDirectory():
    from pathlib import Path
    import platform
    import tempfile
    tempdir = Path("/tmp" if platform.system() == "Darwin" else tempfile.gettempdir())
    return tempdir

def inferVideo(args):
    os.makedirs(args.output_dir, exist_ok=True)

    faceEnhancer = FaceEnhancement(
        size=args.size,
        model=args.model,
        use_sr=args.use_sr,
        sr_model=args.sr_model,
        channel_multiplier=args.channel_multiplier,
        narrow=args.narrow
    )

    inputVideo = args.input_video
    baseName = Path(inputVideo).stem
    outNoAudio = os.path.join(args.output_dir, baseName + "_restored_noaudio.mp4")
    outWithAudio = os.path.join(args.output_dir, baseName + "_restored.mp4")

    # write video
    videoWriter = None
    frameCount = 0
    cap = cv2.VideoCapture(inputVideo)
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    while True:
        _, frame = cap.read(cv2.IMREAD_COLOR)
        if frame is None:
            break
        else:
            frameCount += 1
            print(f'Processing frame {frameCount} ...')
            restored, orig_faces, enhanced_faces = faceEnhancer.process(frame)
            if videoWriter == None:
                frameSize = (restored.shape[1], restored.shape[0])
                videoWriter = cv2.VideoWriter(outNoAudio, codec, fps, frameSize, True)
            videoWriter.write(restored)
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

    # transfer audio
    print("Transfer audio")
    tempAudio = os.path.join(args.output_dir, f"temp_{uuid.uuid4().hex}.mp3" )
    command1 = f"ffmpeg -i {inputVideo} -map 0:a {tempAudio}"
    command2 = f'ffmpeg -y -i {tempAudio} -i {outNoAudio} -strict -2 -q:v 1 {outWithAudio}'
    subprocess.call(" && ".join([command1, command2]), shell=platform.system() != 'Windows')
    os.remove(tempAudio)
    os.remove(outNoAudio)
    print(f'Results are in the [{args.output_dir}] folder.')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GPEN-BFR-512', help='GPEN model')
    parser.add_argument('--size', type=int, default=512, help='resolution of GPEN')
    parser.add_argument('--channel_multiplier', type=int, default=2, help='channel multiplier of GPEN')
    parser.add_argument('--narrow', type=float, default=1, help='channel narrow scale')
    parser.add_argument('--use_sr', action='store_true', help='use sr or not')
    parser.add_argument('--sr_model', type=str, default='rrdb_realesrnet_psnr', help='SR model')
    parser.add_argument('--sr_scale', type=int, default=2, help='SR scale')
    parser.add_argument('--input_video', type=str, default='inputs/test.mp4')
    parser.add_argument('--output_dir', type=str, default='results/outs-BFR', help='output folder')
    parser.add_argument('--save_frames', action='store_true')
    args = parser.parse_args()

    inferVideo(args)
