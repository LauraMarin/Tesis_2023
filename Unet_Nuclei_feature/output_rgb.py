#v2
#7/11/2018

import argparse
import os
import glob
import numpy as np
import cv2
import torch
import sklearn.feature_extraction.image
from colortransfert import color_transfer

from unet import UNet
from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.preprocessing.color_conversion import lab_mean_std

def zoom(img, zoom_factor=2):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)



#-----helper function to split data into batches
def divide_batch(l, n): 
    for i in range(0, l.shape[0], n):  
        yield l[i:i + n,::]

def unet_nuclei(indice_slide):
    suffix = str(indice_slide).zfill(3)
    print(suffix)
    basepath ='C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/Unet_nuclei/wsi/tiles_png'

#basepath ='E:/These/NucleiMaskCNN/NucleiMaskCNN/WholeSLide/deephistopath/wsi/tiles_png'



#'C:/Users/USUARIO/AppData/Local/Programs/Python/Python36/Phenotype/NucleiMaskCNN/WholeSLide/deephistopath/wsi/tiles_png'
# ----- parse command line arguments
    parser = argparse.ArgumentParser(description='Make output for entire image using Unet')
    parser.add_argument('input_pattern',
                    help="input filename pattern. try: *.png, or tsv file containing list of files to analyze",
                    nargs="*")

    parser.add_argument('-p', '--patchsize', help="patchsize, default 256", default=256, type=int)
    parser.add_argument('-s', '--batchsize', help="batchsize for controlling GPU memory usage, default 10", default=10, type=int)
#parser.add_argument('-o', '--outdir', help="outputdir, default ./output_rgb/001", default="./output/", type=str)
#parser.add_argument('-r', '--resize', help="resize factor 1=1x, 2=2x, .5 = .5x", default=1, type=float)
    parser.add_argument('-m', '--model', help="model", default="epi_unet_best_model_rgb.pth", type=str)
    parser.add_argument('-i', '--gpuid', help="id of gpu to use", default=0, type=int)
    parser.add_argument('-f', '--force', help="force regeneration of output even if it exists", default=False,
                    action="store_true")
    parser.add_argument('-b', '--basepath',
                    help="base path to add to file names, helps when producing data using tsv file as input",
                    default="", type=str)
    OUTPUT_DIR_ = 'C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/output/output_rgb'
    args = parser.parse_args()
    batch_size = 10
    patch_size = 256
    stride_size = patch_size//2
    model = "epi_unet_best_model_rgb.pth"
    gpuid  = 0
    force = False
# ----- load network
    device = torch.device(args.gpuid if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage) #load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666
    model = UNet(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"],
             padding=checkpoint["padding"], depth=checkpoint["depth"], wf=checkpoint["wf"],
             up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"]).to(device)
    model.load_state_dict(checkpoint["model_dict"])
    model.eval()

    print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

# ----- get file list
#name_folder = 'data'
    
    target_img= 'C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/Unet_nuclei/codes/TCGA-001-tile-r5-c109-x110598-y4098-w1024-h1024.png'
    source_image= cv2.imread(target_img,-1)
    mean, std = lab_mean_std(source_image)

   
    OUTPUT_DIR = OUTPUT_DIR_ + '/' +suffix
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
#folder_dir = OUTPUT_DIR + '/' + name_folder
#if not os.path.exists(folder_dir):
 #   os.makedirs(folder_dir)
    files = []
#basepath = args.basepath
#print(basepath)#
#basepath = basepath + os.sep if len(
#    basepath) > 0 else ""  # if the user supplied a different basepath, make sure it ends with an os.sep
#print(len(args.input_pattern))
#if len(args.input_pattern) > 1:  # bash has sent us a list of files
#    files = args.input_pattern
#elif args.input_pattern[0].endswith("tsv"):  # user sent us an input file
    # load first column here and store into files
 #   with open(args.input_pattern[0], 'r') as f:
  #      for line in f:
   #         if line[0] == "#":
    #            continue
     #       files.append(basepath + line.strip().split("\t")[0])
#else:  # user sent us a wildcard, need to use glob to find files

    files = os.listdir(basepath + '/'+suffix +'/')
#print(files)
    
# ------ work on files
    for num, fname in (enumerate(os.listdir(basepath + '/'+suffix ))):
        print(fname)
        fname = fname.strip()
        newfname_class = "%s/%s_class.png" % (OUTPUT_DIR, os.path.basename(fname)[0:os.path.basename(fname).rfind(".")])

        print(f"working on file: \t {fname}")
        print(f"saving to : \t {newfname_class}")

        if not args.force and os.path.exists(newfname_class):
            print("Skipping as output file exists")
            continue

        cv2.imwrite(newfname_class, np.zeros(shape=(1, 1)))

        io=cv2.imread(basepath + '/' + suffix +'/'+fname)
        io = zoom(io, 2)
        io = reinhard (io,mean, std)
    #source = cv2.imread('TCGA-001-tile-r5-c109-x110598-y4098-w1024-h1024.png',-1)
    #io = color_transfer(source, io, clip=True, preserve_paper=True)
        io = cv2.cvtColor(io,cv2.COLOR_BGR2RGB)
    #io = cv2.resize(io, (0, 0), fx=args.resize, fy=args.resize)
    
   # io = cv2.cvtColor(cv2.imread(fname),cv2.COLOR_BGR2RGB)
   # io = cv2.resize(io, (0, 0), fx=args.resize, fy=args.resize)
        io_shape_orig = np.array(io.shape)
    
    #add half the stride as padding around the image, so that we can crop it away later
        io = np.pad(io, [(stride_size//2, stride_size//2), (stride_size//2, stride_size//2), (0, 0)], mode="reflect")
    
        io_shape_wpad = np.array(io.shape)
    
    #pad to match an exact multiple of unet patch size, otherwise last row/column are lost
        npad0 = int(np.ceil(io_shape_wpad[0] / patch_size) * patch_size - io_shape_wpad[0])
        npad1 = int(np.ceil(io_shape_wpad[1] / patch_size) * patch_size - io_shape_wpad[1])

        io = np.pad(io, [(0, npad0), (0, npad1), (0, 0)], mode="constant")

        arr_out = sklearn.feature_extraction.image._extract_patches(io,(patch_size,patch_size,3),stride_size)
        arr_out_shape = arr_out.shape
        arr_out = arr_out.reshape(-1,patch_size,patch_size,3)

    #in case we have a large network, lets cut the list of tiles into batches
        output = np.zeros((0,checkpoint["n_classes"],patch_size,patch_size))
        for batch_arr in divide_batch(arr_out,batch_size):
        
            arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)

        # ---- get results
            output_batch = model(arr_out_gpu)

        # --- pull from GPU and append to rest of output 
            output_batch = output_batch.detach().cpu().numpy()
        
            output = np.append(output,output_batch,axis=0)


        output = output.transpose((0, 2, 3, 1))
    
    #turn from a single list into a matrix of tiles
        output = output.reshape(arr_out_shape[0],arr_out_shape[1],patch_size,patch_size,output.shape[3])

    #remove the padding from each tile, we only keep the center
        output=output[:,:,stride_size//2:-stride_size//2,stride_size//2:-stride_size//2,:]

    #turn all the tiles into an image
        output=np.concatenate(np.concatenate(output,1),1)
    
    #incase there was extra padding to get a multiple of patch size, remove that as well
        output = output[0:io_shape_orig[0], 0:io_shape_orig[1], :] #remove paddind, crop back

    # --- save output

    # cv2.imwrite(newfname_class, (output.argmax(axis=2) * (256 / (output.shape[-1] - 1) - 1)).astype(np.uint8))
        cv2.imwrite(newfname_class, output.argmax(axis=2) * (256 / (output.shape[-1] - 1) - 1))
        