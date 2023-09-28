from contextlib import nullcontext
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
import os 
#Add imports if needed:
import scipy.io as sio #ohad add for debug
from matplotlib.pyplot import plot, draw, show, subplot
import pickle as pk 
#end imports


#Add extra functions here:

def displayResults(Imgslist,titles):
    fig = plt.figure(figsize=(15, 15))
    row_shape=np.ceil(len(Imgslist)/2)
    for idx,img in enumerate (Imgslist):
        plt.subplot(np.int16(row_shape),2,idx+1)
        plt.title(titles[idx])
        plt.imshow(cv2.cvtColor(Imgslist[idx],cv2.COLOR_BGR2RGB))
        _ = plt.axis('off')
def showTransformation(im1,im2,H2to1):
    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(im1)
    fig.add_subplot(1, 2, 2)
    plt.imshow(im2)
    for i in range(7):
        pts = np.asarray(plt.ginput(1, timeout=-1))
        pts = np.append(pts,1)
        pts1im = H2to1@pts
        fig.add_subplot(1, 2, 1)
        plt.plot(pts1im[0]/pts1im[2],pts1im[1]/pts1im[2],'ro') 
        fig.add_subplot(1, 2, 2)
        plt.plot(pts[0],pts[1],'ro') 
        

def ReadImagesDirectory(folder):

    images=[]
    for filename in sorted (os.listdir(folder)):
        path=os.path.join(folder,filename)
        isDirectory = os.path.isdir(path)
        if(not isDirectory):
            img = cv2.imread(path)
            images.append(img)
    return images

  
def PanoramaPair(imgSrc,imgDst,dist_ratio=0.3,manual=False,manPoints=[],k=0,dist_cond=True):

        if(not manual ):
            SrcPts,DestPts = getPoints_SIFT(imgSrc, imgDst,dist_ratio,k=k,dist_cond=dist_cond)  
            H= computeH(SrcPts,DestPts)
            imgDst,imgSrc=wrapH2Images(imgSrc,H,cv2.INTER_LINEAR,imgDst)
            imgPano=imageStitching(imgDst,imgSrc)
            return imgPano,0
        else :
            SrcPts,DestPts=manPoints
            H= computeH(SrcPts,DestPts)
            imgDst,imgSrc,translation=wrapH2Images(imgSrc,H,cv2.INTER_LINEAR,imgDst,return_translation=True)
            imgPano=imageStitching(imgDst,imgSrc)
            return imgPano,translation

def imageStitchingDir(path,dist_ratio=0.3,manual= False,manPoints=[],k=0,dist_cond=True):
    def HalfPano(images_list,rtl=True,manual= False,manPoints=[],dist_ratio=0.3):
        if( rtl):
            imgPano=images_list[0]
            Range= range(1,len(images_list)//2+1)
        else : 
            imgPano=images_list[-1]
            Range= range(len(images_list)-2,len(images_list)//2,-1)
        translation=[]
        Pointsidx=0
        for next_image in Range:
            JoinImg = images_list[next_image]
            if (manual):
                if(translation!=[]):
                    manPoints[Pointsidx]=cv2.perspectiveTransform(np.dstack(manPoints[Pointsidx]), translation).T.squeeze(2)
                imgPano,translation=PanoramaPair(imgPano,JoinImg,manual=True,manPoints=manPoints[Pointsidx:Pointsidx+2],k=k,dist_cond=dist_cond)  
                Pointsidx+=2
              
            else:
                imgPano,_=PanoramaPair(imgPano,JoinImg,manual=False,manPoints=[],dist_ratio=dist_ratio,k=k,dist_cond=dist_cond)
        if(manual):
            return imgPano,translation
        return imgPano

    images_list = ReadImagesDirectory(path)
    P=[]
    if(manual): 
        manualLeft= list(np.flip(manPoints['4-5'],axis=0))
        manualRight= list(manPoints['1-2'])+list(manPoints['2-3'])
        PanoRight,trans_R=HalfPano(images_list,rtl=True,manual=True,manPoints=manualRight)
        PanoLeft,trans_L=HalfPano(images_list,rtl=False,manual=True,manPoints=manualLeft)
        p3=cv2.perspectiveTransform(np.dstack(manPoints['3-4'][0]),trans_R ).T.squeeze(2)
        p4=cv2.perspectiveTransform(np.dstack(manPoints['3-4'][1]), trans_L).T.squeeze(2)
        P=[p4,p3]
    else:
        PanoRight=HalfPano(images_list,rtl=True,manual=False,dist_ratio=dist_ratio)
        PanoLeft=HalfPano(images_list,rtl=False,manual=False,dist_ratio=dist_ratio)
    imgPano,_=PanoramaPair(PanoLeft,PanoRight,manual=manual,manPoints=P,dist_ratio=dist_ratio)
        
    return imgPano
def findBordernonZero(im):
    positions = np.nonzero(im)
    top = positions[0].min()
    bottom = positions[0].max()
    left = positions[1].min()
    right = positions[1].max()
    border =np.float32([[left,top],[left,bottom],[right,bottom],[right,top]]).reshape(-1,1,2)
    return border

def findBorderTransform(im,H,im2=[]):
    border = findBordernonZero(im)
    border_transformed = cv2.perspectiveTransform(border, H)
    if im2!=[]:
        border2=findBordernonZero(im2)
        border_transformed=np.concatenate((border_transformed,border2))
    points=np.floor(border_transformed.min(axis=0).ravel())
    [top_left,bottom_right]=np.int16(np.stack((points,np.ceil(border_transformed.max(axis=0).ravel())),axis=0))

    return   [top_left,bottom_right]


def wrapH2Images(im2_wrap, H, flags,im1=[],return_translation=False):
    [top_left,bottom_right]=findBorderTransform(im2_wrap,H,im1)
    tranlsation =-top_left
    tranlsation=np.float64(tranlsation)
    t = np.array([
        [1,0,tranlsation[0]],
        [0,1,tranlsation[1]],
        [0,0,1]]) 
    im1Warp = cv2.warpPerspective(src=im1, M=t, dsize=(abs(top_left[0]-bottom_right[0]),abs(top_left[1]-bottom_right[1])))
    im2Warp=  cv2.warpPerspective(src=im2_wrap, M=t@H, dsize=(abs(top_left[0]-bottom_right[0]),abs(top_left[1]-bottom_right[1])), flags=flags) 
    if(return_translation):
        return im1Warp,im2Warp,t
    return im1Warp,im2Warp
#Extra functions end

# HW functions:
def getPoints(im1,im2,N):
    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(im1)
    fig.add_subplot(1, 2, 2)
    plt.imshow(im2)
    plt.title("Pick point from left image and then its correspondence point from the right")
    p1 = []
    p2 = []
    for i in range(N):  
        pts = np.asarray(plt.ginput(2, timeout=-1))
        p1.append(pts[0,:])
        p2.append(pts[1,:])
    p1 = np.array(p1).T
    p2 = np.array(p2).T
    plt.close(fig)
    return p1,p2


def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    ones_vector = np.ones([1,p1.shape[1]])
    p1 = np.vstack((p1,ones_vector)).T
    p2 = np.vstack((p2,ones_vector)).T
    A_help_mat_p1_row1 = np.array([[1, 0, 0, 0, 0, 0, -1, 0, 0],
                                   [0, 1, 0, 0, 0, 0, 0, -1, 0],
                                   [0, 0, 1, 0, 0, 0, 0, 0, 1]])
    A_help_mat_p1_row2 = np.array([[0, 0, 0, 1, 0, 0, -1, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, -1, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 0, 1]])
    A_p1_row1 = p1@A_help_mat_p1_row1
    A_p1_row2 = p1@A_help_mat_p1_row2
    A_help_mat_p2_row1 = np.array([[0, 0, 0, 0, 0, 0, 1, 1,-1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 1, 1, 0, 0, 0, 0, 0, 0]])
    A_help_mat_p2_row2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 1,-1],
                                   [0, 0, 0, 1, 1, 1, 0, 0, 0]])
    A_p2_row1 = p2@A_help_mat_p2_row1
    A_p2_row2 = p2@A_help_mat_p2_row2
    A_row1_helper = np.multiply(A_p1_row1,A_p2_row1)
    A_row2_helper = np.multiply(A_p1_row2,A_p2_row2)
    A = np.zeros([p1.shape[0]*2,9])
    A[0::2,:] = A_row1_helper
    A[1::2,:] = A_row2_helper
    U, sigma, V = np.linalg.svd(A) #the smallest eigenvalue is the last one in sigma
    H2to1 = V[-1,:]
    H2to1 = H2to1.reshape([3,3])
    H2to1 = H2to1/H2to1[-1,-1]
    return H2to1

def warpH(im2, H, flags): 
    [top_left,bottom_right]=findBorderTransform(im2,H)   
    tranlsation =-top_left
    tranlsation=np.float64(tranlsation)
    t = np.array([
        [1,0,tranlsation[0]],
        [0,1,tranlsation[1]],
        [0,0,1]])  
    im2Warp = cv2.warpPerspective(src=im2, M=t@H, dsize=(abs(top_left[0]-bottom_right[0]),abs(top_left[1]-bottom_right[1])), flags=flags)
    return im2Warp

def imageStitching(img1, wrap_img2):
    """
    Your code here
    wrap_img2 should be the same height as img1 at the intersection
    """
    panoImg = np.zeros((wrap_img2.shape[0], wrap_img2.shape[1], 3), np.uint8)
    panoImg[img1!=0]=img1[img1!=0]
    panoImg[wrap_img2!=0]=wrap_img2[wrap_img2!=0]
    return panoImg
def getPoints_SIFT(im1,im2,dist_ratio=0.3,k=0,dist_cond=True):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1,None)
    kp2, des2 = sift.detectAndCompute(im2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    p1=[]
    p2=[]
    for (m,n) in matches:
     
         if m.distance <dist_ratio*n.distance or k!=0 or not dist_cond :
            kp1_idx = m.queryIdx
            kp2_idx = m.trainIdx
            p1.append((kp1[kp1_idx].pt, m.distance))
            p2.append((kp2[kp2_idx].pt,m.distance))
    if( k!=0):
        p1.sort(key= lambda x :x[1])    
        p2.sort(key= lambda x :x[1])    
    p1=[p[0] for p in p1 ]
    p2=[p[0] for p in p2]

    if(k!=0):
        return np.array(p1[:k]).T,np.array(p2[:k]).T
    return np.array(p1).T,np.array(p2).T

    
def ransacH(p1, p2, nIter=1000, tol=1):
    N = p1.shape[1]
    best_inliers_n = 0
    best_inliers = []
    p1_T=np.dstack(p1)
    p2_T=np.dstack(p2)
    for iter in range(nIter):
        rand_idxs = np.random.choice(np.arange(N), 4, replace=False)
        chosen_p1 = p1[:, rand_idxs]
        chosen_p2 = p2[:, rand_idxs]
        H1to2 = computeH(chosen_p1, chosen_p2)
        p1_trans=cv2.perspectiveTransform(p1_T,H1to2)
        L2dists = np.sqrt(np.sum((p1_trans - p2_T) ** 2, 2))
        inliers = (p1_T[L2dists < tol], p2_T[L2dists < tol])
        n_inliers = np.sum(L2dists < tol)
        if n_inliers > best_inliers_n:
            best_inliers_n = n_inliers
            best_inliers = inliers
    bestH = computeH(best_inliers[0].T, best_inliers[1].T)
    return bestH
if __name__ == '__main__':
    print('my_homography')
    im1 = cv2.imread(r'./data/incline_L.png')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.imread(r'./data/incline_R.png')
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    
    chose_new_points=False

    if(chose_new_points):
    # # Section 1.1 +1.2 ####
        [p1,p2] = getPoints(im1,im2,4)
        H2to1 = computeH(p2, p1)
    else:
        H2to1_file = sio.loadmat('H2to1_debug.mat') # debug
        H2to1 = H2to1_file['H']

    # showTransformation(im1,im2,H2to1)

    # Section 1.3 #### 

    fig3 = plt.figure(figsize=(8, 8))
    ax1 = fig3.add_subplot(1, 3, 1)
    plt.imshow(im2)
    ax1.title.set_text('Original image')
    im2_to1 = warpH(im2, H2to1, cv2.INTER_CUBIC)
    ax2 = fig3.add_subplot(1, 3, 2)
    plt.imshow(im2_to1)
    ax2.title.set_text('Cubic warp')
    im2_to1 = warpH(im2, H2to1,cv2.INTER_LINEAR)
    ax3 = fig3.add_subplot(1, 3, 3)
    plt.imshow(im2_to1)
    ax3.title.set_text('Linear warp')

    # Section 1.4 ####
    # check for identity matrix :
    H_I = np.identity(3)
    im1test = warpH(im1, H_I,cv2.INTER_LINEAR)
    plt.figure()
    plt.imshow(im1test)
    im1_for_pano, im2_for_pano=wrapH2Images(im1,H2to1,cv2.INTER_LINEAR,im2)
    panoImg = imageStitching(im1_for_pano, im2_for_pano)
    fig_4 = plt.figure(figsize=(15, 15))
    plt.imshow(panoImg)  
    show()

# Section 1.5 ####

    p1,p2=getPoints_SIFT(im1,im2)
    #from img2 to img1 
    H_SIFT=computeH(p2,p1)
    im2SiftPano,im1SiftPano=wrapH2Images(im2,H_SIFT,cv2.INTER_LINEAR,im1)
    panoImgSIFT = imageStitching(im1SiftPano,im2SiftPano)
    # display result
    fig = plt.figure(figsize=(15,15))
    fig.suptitle("stitched images - SIFT")
    plt.imshow(panoImgSIFT)
    plt.axis('off')

 # Section 1.6
    # with sift 
    beachPano=imageStitchingDir('./data/beach',dist_ratio= 0.3)
    sintraPano=imageStitchingDir('./data/sintra',dist_ratio=0.15)
    images=[sintraPano,beachPano]
    titles=['images SIFT-sintra','images SIFT-beach']
    displayResults(images,titles)

    ### #check with the 7 most minmized distance points :
    sintraPanoKclosest=imageStitchingDir('./data/sintra',dist_ratio=0.15,k=7,dist_cond=True)
    ### #check with randomly selected points
    sintraPanoKclosest=imageStitchingDir('./data/sintra',dist_ratio=0.15,k=0,dist_cond=True)

    # # manual
    with open ('./data/points.pkl', 'rb') as file : 
        points=pk.load(file)
    
    beachPano=imageStitchingDir('./data/beach',manual=True,manPoints= points['beach'])
    intraPano=imageStitchingDir('./data/sintra',manual=True,manPoints= points['sintra'])
    images=[intraPano,beachPano]
    titles=['images SIFT-sintra','images SIFT-beach']
    displayResults(images,titles)


# section 1.7 -Copmare between using RANSAC vs. not using it for both manual and SIFT features
    im1 = cv2.imread(r'data/incline_L.png')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2_wrap = cv2.imread(r'data/incline_R.png')
    im2_wrap = cv2.cvtColor(im2_wrap, cv2.COLOR_BGR2RGB)
    p1,p2=getPoints_SIFT(im1,im2_wrap,dist_ratio=0.3)
    #test for sift with ransac 
    H_ransac=ransacH(p1,p2)
    imgDst,imgSrc,_=wrapH2Images(im1,H_ransac,cv2.INTER_LINEAR,im2_wrap,return_translation=True)
    imgPano=imageStitching(imgDst,imgSrc)
    ## test for sift without ransac
    sift_panorama_without_ransac,_=PanoramaPair(im1,im2_wrap)

    # #test for manual with ransac 

    p1,p2=getPoints(im1,im2_wrap,20)
    H_ransac=ransacH(p1,p2)
    imgDst,imgSrc,_=wrapH2Images(im1,H_ransac,cv2.INTER_LINEAR,im2_wrap,return_translation=True)
    imgPano=imageStitching(imgDst,imgSrc)
     # #test for manual without ransac 
    manual_panorama_without_ransac=PanoramaPair(im1,im2_wrap,manual=True,manPoints=(p1,p2))

    #part 1.8 creative part
    haifaPano=imageStitchingDir('./my_data/haifa',dist_ratio= 0.3)
    haifaPano=[haifaPano]
    titles=['haifa']
    displayResults(haifaPano,titles)


