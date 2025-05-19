import os.path
import argparse
import torch

from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
from datetime import datetime


# Script that performs inference on captured images using trained model.

# Define the path to test.json file and adjust the code (see comments) depending on which network you're using.
def main(json_path='options/test_dncnn_tempest.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    opt = option.dict_to_nonedict(opt)

    model_path = opt['path']['pretrained_netG']
    device = torch.device('mps')

    opt_netG = opt['netG']

    in_nc = opt_netG['in_nc']
    out_nc = opt_netG['out_nc']
    nc = opt_netG['nc']
    nb = opt_netG['nb']
    act_mode = opt_netG['act_mode']
    # bias = opt_netG['bias']    # drunet

    # from models.network_unet import UNetRes as net # drunet
    from models.network_dncnn import DnCNN as net  # dncnn
    # model = net(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, bias=bias)    # drunet
    model = net(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode)  # dncnn
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    L_paths = util.get_image_paths(opt['datasets']['test']['dataroot_L'])

    for L_path in L_paths:
        image_name_ext = os.path.basename(L_path)
        img_name, ext = os.path.splitext(image_name_ext)

        capture_dir = os.path.join(opt['datasets']['test']['dataroot_L'], img_name)
        if not os.path.isdir(capture_dir):
            continue

        img_dir = os.path.join(opt['path']['images'])
        util.mkdir(img_dir)

        estimate_dir = os.path.join(img_dir, f"{img_name}")
        util.mkdir(estimate_dir)

        for capture_file in os.listdir(capture_dir):
            capture_path = os.path.join(capture_dir, capture_file)
            if not capture_file.endswith('.png'):
                continue

            img_L_original = util.imread_uint(capture_path, n_channels=1)
            # img_L_original = util.imread_uint(capture_path, n_channels=3)  # drunet
            # img_L = img_L_original[:, :, :2]   # drunet
            img_L = util.uint2single(img_L_original)
            img_L = util.single2tensor4(img_L).to(device)

            with torch.no_grad():
                img_E = model(img_L)
            img_E = util.tensor2uint(img_E)

            save_img_path = os.path.join(estimate_dir, capture_file)
            util.imsave(img_E, save_img_path)

            print(
                f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")} - Inference of {capture_file} completed. Saved at {estimate_dir}.')


if __name__ == '__main__':
    main()
