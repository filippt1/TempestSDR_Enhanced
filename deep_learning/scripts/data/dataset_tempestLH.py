import os
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import itertools

# Dataloader for desired directory structure. Adjusted from Deep-Tempest for DnCNN.

class DatasetTempestLH(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denoising without sigma (TEMPEST image is already noisy)
    # Only dataroot_H is needed.
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetTempestLH, self).__init__()
        print("Loading TEMPEST dataloader.")
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        # self.n_channels_datasetload = opt['n_channels_datasetload'] if opt['n_channels_datasetload'] else 3
        self.patch_size = self.opt['H_size'] if opt['H_size'] else 64
        # self.sigma = opt['sigma'] if opt['sigma'] else [0, 75]
        # self.sigma_min, self.sigma_max = self.sigma[0], self.sigma[1]
        # self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else 0
        self.use_all_patches = opt['use_all_patches'] if opt['use_all_patches'] else False
        self.num_patches_per_image = opt['num_patches_per_image'] if opt['num_patches_per_image'] else 100
        # self.skip_natural_patches = opt['skip_natural_patches'] if opt['skip_natural_patches'] else False
        # self.use_abs_value = opt['use_abs_value'] if opt['use_abs_value'] else False

        # -------------------------------------
        # Dataset path contains all H images and subfolders for every single one with one or more captures
        # -------------------------------------

        """  
        Dataset path includes all H images and L-subfolders.
        Every H image has one L-subfolder assosiated, which
        contains one or more L representations of the H image.
        """

        assert os.path.isdir(opt['dataroot_H']), f"{opt['dataroot_H']} is not a directory"
        self.paths_H = [f for f in os.listdir(opt['dataroot_H']) if os.path.isfile(os.path.join(opt['dataroot_H'], f))]
        # ------------------------------------------------------------------------------------------------------
        # For the above step you can use util.get_image_paths(), but it goes recursevely throught the tree dirs
        # ------------------------------------------------------------------------------------------------------
        paths_H_aux = []
        self.paths_L = []

        # Iterate over all image paths
        for H_file in self.paths_H:
            # filename = os.path.basename(H_file)
            filename = H_file.split(".")[0]  # TODO: the correct way to do it is with os.path.basename()
            L_folder = os.path.join(opt['dataroot_H'], filename)
            # For image at subfolder, append to L paths and repeat current H path
            for L_file in os.listdir(L_folder):
                L_filepath = os.path.join(L_folder, L_file)
                paths_H_aux.append(os.path.join(opt['dataroot_H'], H_file))
                self.paths_L.append(L_filepath)

        # Update H paths
        self.paths_H = paths_H_aux


        # Repeat every image in path list to get more than one patch per image
        if self.opt['phase'] == 'train':
            listOfLists = [list(itertools.repeat(path, self.num_patches_per_image)) for path in self.paths_H]
            self.paths_H = list(itertools.chain.from_iterable(listOfLists))

            listOfLists = [list(itertools.repeat(path, self.num_patches_per_image)) for path in self.paths_L]
            self.paths_L = list(itertools.chain.from_iterable(listOfLists))


    def __getitem__(self, index):

        # -------------------------------------
        # get H and L image
        # -------------------------------------
        H_path = self.paths_H[index]
        L_path = self.paths_L[index]

        img_H = util.imread_uint(H_path, self.n_channels)
        img_L = util.imread_uint(L_path, self.n_channels)

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H
            # --------------------------------
            """
            H, W = img_H.shape[:2]

            if self.use_all_patches or (img_H.shape[0] <= self.patch_size) or (img_H.shape[1] <= self.patch_size):

                # ---------------------------------
                # Start or continue image patching
                # ---------------------------------
                img_patch_index = index % self.num_patches_per_image  # Resets to 0 every time index overflows num_patches

                # Upper-left corner of patch
                h_index = self.patch_size * ((img_patch_index * self.patch_size) // W)
                w_index = self.patch_size * (((img_patch_index * self.patch_size) % W) // self.patch_size)

                # Don't exceed the image limit
                h_index = min(h_index, H - self.patch_size)
                w_index = min(w_index, W - self.patch_size)

            else:
                # ---------------------------------
                # randomly crop the patch
                # ---------------------------------
                h_index = random.randint(0, max(0, H - self.patch_size))
                w_index = random.randint(0, max(0, W - self.patch_size))

            # Ground-truth patch
            patch_H = img_H[h_index:h_index + self.patch_size, w_index:w_index + self.patch_size, :]
            # Captured, TEMPEST patch
            patch_L = img_L[h_index:h_index + self.patch_size, w_index:w_index + self.patch_size]

            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            img_H = util.uint2tensor3(img_H)
            img_L = util.uint2tensor3(img_L)

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
