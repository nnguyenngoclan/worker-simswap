# from flask import Flask, request, jsonify
import subprocess
import uuid
import os

# app = Flask(__name__)

def swap():
    # video_file = request.files['video']
    # target_image = request.files['target']

    session_dir = "./demo_file"    

    input_video_path = os.path.join(session_dir, "trimmed_output.mp4")
    source_image_path = os.path.join(session_dir, "specific1.png")
    target_image_path = os.path.join(session_dir, "Iron_man.jpg")
    output_video_path = os.path.join("output", "output_test.mp4")
    temp_result_path = "./temp_results"

    # video_file.save(input_video_path)
    # target_image.save(target_image_path)

    try:
        # Build the command for SimSwap
        command = [
            "python", "test_video_swapspecific.py",
            "--crop_size", "224",
            "--use_mask",
            "--pic_specific_path", source_image_path,
            "--name", "people",
            "--Arc_path", "arcface_model/arcface_checkpoint.tar",
            "--pic_a_path", target_image_path,
            "--video_path", input_video_path,
            "--output_path", output_video_path,
            "--temp_path", temp_result_path
        ]

        subprocess.run(command, check=True, cwd="/Users/lannn/Desktop/AI/SimSwap")
        
    except subprocess.CalledProcessError as e:
        print(e)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    swap()