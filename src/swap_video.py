import test_video_swapspecific
from options.test_options import TestOptions

def swap_video(source_image, target_image, video_path, output_path):
    opt = TestOptions().parse()
    opt.crop_size = 224
    opt.use_mask = True
    opt.name = "people"
    # opt.gpu_ids = []
    opt.pic_specific_path = source_image
    opt.pic_a_path = target_image
    opt.video_path = video_path
    opt.output_path = output_path
    opt.temp_path = "./temp_results"

    test_video_swapspecific.__execute(opt)
