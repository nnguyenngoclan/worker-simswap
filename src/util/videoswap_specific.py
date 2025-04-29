import os 
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
# import moviepy.editor as mp
from moviepy.audio.io.AudioFileClip import AudioFileClip #, VideoFileClip 
from moviepy.video.io.VideoFileClip import VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import  time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
import torch.nn.functional as F
from parsing_model.model import BiSeNet

import av
import time
import concurrent.futures

def extract_frames_pyav(video_path, threads = 4):
    container = av.open(video_path)
    
    # Set multi-threaded decoding
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    
    total_frames = stream.frames  # get total frame count
    if total_frames == 0:  # fallback if not available
        total_frames = int(container.streams.video[0].duration * container.streams.video[0].average_rate)

    frames = []
    
    for idx, frame in enumerate(container.decode(video=0)):
        img = frame.to_ndarray(format='bgr24')
        frames.append(img)        
         # Progress calculation
        percent = (idx + 1) / total_frames * 100
        print(f"Extract frame progress: {percent:.2f}% ({idx + 1}/{total_frames})", end='\r')

    container.close()
    print()
    return frames

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def process_one_frame(frame_index, frame, frame_align_crop_list, frame_mat_list, specific_person_id_nonorm, id_thres, swap_model, spNorm, mse, temp_results_dir, crop_size, logoclass, no_simswaplogo, id_vetor, net, use_mask, total_frames):

    try:
        if frame_align_crop_list is not None and frame_mat_list is not None and len(frame_align_crop_list) > 0:
            if not os.path.exists(temp_results_dir):
                os.makedirs(temp_results_dir, exist_ok=True)

            frame_align_crop_tensors = []

            for frame_align_crop in frame_align_crop_list:
                frame_tensor = _totensor(cv2.cvtColor(frame_align_crop, cv2.COLOR_BGR2RGB))
                frame_align_crop_tensors.append(frame_tensor)

            frame_align_crop_batch = torch.stack(frame_align_crop_tensors).cuda()
            frame_align_crop_batch_arcnorm = spNorm(frame_align_crop_batch)
            frame_align_crop_batch_downsampled = F.interpolate(frame_align_crop_batch_arcnorm, size=(112, 112))
            frame_align_crop_batch_id_nonorm = swap_model.netArc(frame_align_crop_batch_downsampled)

            specific_person_id_nonorm = specific_person_id_nonorm.squeeze().to(frame_align_crop_batch_id_nonorm.device)
            specific_person_id_nonorm_batch = specific_person_id_nonorm.unsqueeze(0).expand(frame_align_crop_batch_id_nonorm.size(0), -1)

            mse_losses = F.mse_loss(frame_align_crop_batch_id_nonorm, specific_person_id_nonorm_batch, reduction='none')
            mse_per_sample = mse_losses.mean(dim=1)
            id_compare_values = mse_per_sample.detach().cpu().numpy()

            frame_align_crop_tenor_list = [frame_align_crop_batch[i:i+1] for i in range(frame_align_crop_batch.size(0))]

            min_index = np.argmin(id_compare_values)
            min_value = id_compare_values[min_index]

            if min_value < id_thres:
                swap_result = swap_model(None, frame_align_crop_tenor_list[min_index], id_vetor, None, True)[0]

                reverse2wholeimage([frame_align_crop_tenor_list[min_index]], [swap_result], [frame_mat_list[min_index]], crop_size, frame, logoclass,
                    os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), no_simswaplogo, pasring_model=net, use_mask=use_mask, norm=spNorm)
            else:
                frame = frame.astype(np.uint8)
                if not no_simswaplogo:
                    frame = logoclass.apply_frames(frame)
                cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)

        else:
            if not os.path.exists(temp_results_dir):
                os.makedirs(temp_results_dir, exist_ok=True)
            frame = frame.astype(np.uint8)
            if not no_simswaplogo:
                frame = logoclass.apply_frames(frame)
            cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)

    except Exception as e:
        print(f"\nError processing frame {frame_index}: {e}")

    # Progress report
    percent = (frame_index + 1) * 100 / total_frames
    print(f"Swap Face Progress: {percent:.2f}% ({frame_index + 1}/{total_frames})", end='\r')

def video_swap(video_path, id_vetor,specific_person_id_nonorm,id_thres, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False,use_mask =False):
    video_forcheck = VideoFileClip(video_path)
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)
    
    print(f"Got detect model type: {type(detect_model)}")

    video = cv2.VideoCapture(video_path)
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    ret = True
    frame_index = 0

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = video.get(cv2.CAP_PROP_FPS)
    if  os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)

    spNorm =SpecificNorm()
    mse = torch.nn.MSELoss().cuda()

    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net = None
    
    start_time = time.time()

    frames = extract_frames_pyav(video_path)
    bboxes_batch, kpss_batch = detect_model.batch_detect(frames, crop_size)
    total_frames = len(frames)
    
    print("Do swapping faces\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
    
        for frame_index, (frame_align_crop_list, frame_mat_list) in enumerate(zip(bboxes_batch, kpss_batch)):
            frame = frames[frame_index]
    
            future = executor.submit(
                process_one_frame,
                frame_index, frame,
                frame_align_crop_list, frame_mat_list,
                specific_person_id_nonorm, id_thres, swap_model,
                spNorm, mse,
                temp_results_dir, crop_size, logoclass,
                no_simswaplogo, id_vetor, net, use_mask,
                total_frames
            )
            futures.append(future)
    
        # (optional) Wait for all to complete
        concurrent.futures.wait(futures)
    
    print()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Method took {elapsed_time:.4f} seconds to run.")

    video.release()

    # image_filename_list = []
    path = os.path.join(temp_results_dir,'*.jpg')
    image_filenames = sorted(glob.glob(path))

    clips = ImageSequenceClip(image_filenames,fps = fps)

    if not no_audio:
        clips = clips.with_audio(video_audio_clip)


    clips.write_videofile(save_path,audio_codec='aac')

