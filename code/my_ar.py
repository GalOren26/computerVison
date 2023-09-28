import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
import my_homography as mh
#Add imports if needed:
from matplotlib.pyplot import plot, draw, show, close
#end imports

#Add functions here:
def im2im(scene_im,ref_im, pts = []):
    if len(pts) == 0:
        fig22 = plt.figure(figsize=(8, 8))
        plt.imshow(scene_im)
        plt.title("select book corners in the order: up_Left, up_Right, low_Left, low_Right", fontsize=12)
        pts = np.asarray(plt.ginput(4, timeout=-1)).T
        close(fig22)
    p2 = np.array([[0,ref_im.shape[1],0,ref_im.shape[1]],[0,0,ref_im.shape[0],ref_im.shape[0]]])
    H = mh.computeH(p2, pts)
    H = mh.adjustH(H)
    warp_ref = cv2.warpPerspective(ref_im,H,(scene_im.shape[1],scene_im.shape[0]),cv2.INTER_LINEAR)
    new_image = mh.imageStitching(scene_im, warp_ref)
    return new_image, pts
#Functions end

# HW functions:
def create_ref(im_path):
    im1 = cv2.imread(im_path)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    fig11 = plt.figure(figsize=(8, 8))
    plt.imshow(im1)
    plt.title("select book corners in the order: up_Left, up_Right, low_Left, low_Right", fontsize=12)
    pts = np.asarray(plt.ginput(4, timeout=-1)).T
    close(fig11)
    warp_width = pts[0,1]-pts[0,0]
    warp_height = pts[1,2]-pts[1,1]
    p2 = np.array([[0,warp_width,0,warp_width],[0,0,warp_height,warp_height]])
    H = mh.computeH(pts, p2)
    ref_image = mh.warpH(im1, H, (int(warp_width),int(warp_height)),cv2.INTER_LINEAR)
    return ref_image

if __name__ == '__main__':
    print('my_ar')
    
    # Section 2.1 #####
    ref_image = create_ref('my_data/Book.jpg')
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(ref_image)
    
    # Section 2.2 #####
    ref_image = cv2.imread(r'my_data/Book_A.jpg')
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    scene_im = cv2.imread(r'my_data/2.2_scene.jpg')
    scene_im = cv2.cvtColor(scene_im, cv2.COLOR_BGR2RGB)
    [new_image, pts] = im2im(scene_im,ref_image)
    fig2 = plt.figure(figsize=(8, 8))
    plt.imshow(new_image)
    ref_image = cv2.imread(r'my_data/elephant_B.jpg')
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    [new_image,_] = im2im(scene_im,ref_image,pts)
    fig3 = plt.figure(figsize=(8, 8))
    plt.imshow(new_image)
    ref_image = cv2.imread(r'my_data/New_year.png')
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    [new_image,_] = im2im(scene_im,ref_image,pts)
    fig4 = plt.figure(figsize=(8, 8))
    plt.imshow(new_image)
    show()