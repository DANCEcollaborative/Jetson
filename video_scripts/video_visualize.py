import os
import cv2
import glob
import numpy as np

OUT_NAME = 'video'
OUT_DIR = './output/'

out_path = os.path.join(OUT_DIR, OUT_NAME)

depth_dir = os.path.join(out_path, 'depth')
img_dir = os.path.join(out_path, 'img')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./test.mp4', fourcc, 15, (640*2,480))
filelist = list(glob.glob(os.path.join(img_dir, '*.png')))
filelist = sorted(filelist, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

print('Writing Video')
for file in filelist:
    img = cv2.imread(file)
    color_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    depth = cv2.imread(os.path.join(depth_dir, os.path.basename(file)))
    
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

    images = np.hstack((color_img, depth_colormap))

    out.write(images)
print('Finished Writing')
