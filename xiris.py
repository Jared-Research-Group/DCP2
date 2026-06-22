import numpy as np
import os
from pathlib import Path
import cv2 as cv
from tqdm import tqdm
from PIL import Image
import shutil

import helper_functions

logger = helper_functions.setup_logger(__name__)

def getFrameList(dir):

    dir = Path(dir)
    logger.info('Reading and converting *.raw Xiris frames...')

    frames = [entry for entry in dir.iterdir() if entry.suffix == '.raw']
    frames.sort()

    return frames

def convertFrame(f):
    """Efficiently read raw frame data using NumPy bulk read instead of pixel-by-pixel loop"""

    with open(f, "rb") as frame:
        frame.read(16)  # Skip header
        width = int.from_bytes(frame.read(4), 'little')
        height = int.from_bytes(frame.read(4), 'little')
        frame.read(32)  # Skip remaining header
        
        img = Image.frombytes('I;16', [width, height], frame.read(2 * width * height))
        intensity = np.asarray(img) * 2**2
    
    return intensity

def saveSingleFrame(f, f_name, show=False):
    
    img = convertFrame(f)

    if show:
        cv.imshow('image', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    cv.imwrite(f_name, img)

def getFrameData(dir, frames, mustMakeFrames=False):
    
    if not os.access(dir, os.F_OK):
        os.mkdir(dir)
        mustMakeFrames = True
    elif not [entry for entry in dir.iterdir()]:
        mustMakeFrames = True
        
    pngList = []
    
    for f in tqdm(frames, disable= not (__name__ == '__main__')):
        filename = dir / (f.stem + '.png')
        pngList.append(filename)
        
        if mustMakeFrames:
            saveSingleFrame(f, filename)

    logger.info(f'Xiris frames saved to {dir}')

    return pngList

def buildVideoCV2(dir, **kwargs):
    dir = Path(dir)

    input_filename = dir / 'Xiris' / 'raw'
    output_filenames = ['Xiris/raw', 'Xiris.mp4', 'Xiris_Frames']
    [input_dir, [untouched_dir, output_video, output_dir]] = helper_functions.setup_directory_structure(dir, input_filename, output_filenames, **kwargs)

    if not os.access(untouched_dir, os.R_OK):
        shutil.copytree(input_dir, untouched_dir)
        shutil.rmtree(dir / 'Xiris')

    frame_paths = getFrameList(input_dir, **kwargs)
    frames = getFrameData(output_dir, frame_paths)

    logger.info('Constructing Xiris video...')
    
    shape = cv.imread(frames[0], flags=cv.IMREAD_UNCHANGED).shape
    shape = (shape[1], shape[0])

    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Use more efficient codec
    out = cv.VideoWriter(output_video, fourcc, 50., shape, False)
 
    for frame in tqdm(frames, disable= not (__name__ == '__main__')):
        im = cv.imread(frame, flags=cv.IMREAD_UNCHANGED)
        out.write((im >> 8).astype(np.uint8))

    out.release()
    cv.destroyAllWindows()

    logger.info(f'Xiris video saved to {output_video}')

    return

if __name__ == '__main__':

    [dir, kwargs] = helper_functions.setup_kwargs()

    buildVideoCV2(dir, **kwargs)