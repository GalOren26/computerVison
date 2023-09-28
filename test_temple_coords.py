import cv2
import numpy as np
import scipy.optimize
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Helper Functions
# helper function 1: singualrizes F using SVD
def _singularize(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))

    return F

# helper function 2.1: defines an objective function using F and the epipolar constraint
def _objective_F(f, pts1, pts2):
    F = _singularize(f.reshape([3, 3]))
    num_points = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
    hpts2 = np.concatenate([pts2, np.ones([num_points, 1])], axis=1)
    Fp1 = F.dot(hpts1.T)
    FTp2 = F.T.dot(hpts2.T)

    r = 0
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        r += (hp2.dot(fp1.T))**2 * (1/(fp1[0,0]**2 + fp1[0,1]**2) + 1/(fp2[0,0]**2 + fp2[0,1]**2))

    return r

# helper function 2.2: refines F using the objective from above and local optimization
def refineF(F, pts1, pts2):
    f = scipy.optimize.fmin_powell(
        lambda x: _objective_F(x, pts1, pts2), F.reshape([-1]),
        maxiter=100000,
        maxfun=10000
    )

    return _singularize(f.reshape([3, 3]))

# helper function 5: returns the 4 options for camera matrix M2 given the essential matrix
def camera2(E):
    U,S,V = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    if np.linalg.det(U.dot(W).dot(V))<0:
        W = -W

    M2s = np.zeros([3,4,4])
    M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)

    return M2s
# helper function 5: check that the points are infront of camera 1 . 
def numValidZCam2(M2,points):
    pts2d=M2.dot(points.T).T
    validZ=len(pts2d[pts2d[:,2]>0])
    if validZ>0: # ceeck if there is point with postive Z cordiante . 
        return validZ 
    return 0 


## Our implementation
# Section 1.1
def eight_point(pts1, pts2, pmax):
    """
    Eight Point Algorithm
    [I] pts1, points in image 1 (Nx2 matrix)
        pts2, points in image 2 (Nx2 matrix)
        pmax, scalar value computed as max(H1,W1)
    [O] F, the fundamental matrix (3x3 matrix)
    """
    # stage  0 - normalize points
    ones_vector = np.ones([pts1.shape[0],1])
    pts1_hom = np.hstack((pts1,ones_vector))
    pts2_hom = np.hstack((pts2,ones_vector))
    T = np.matrix([[1/pmax, 0, 0], [0, 1/pmax, 0], [0, 0, 1]])
    pts1_norm = pts1_hom@T.T
    pts2_norm = pts2_hom@T.T
    #stage 1 - Construct NX9 matrix A
    A_help_mat_p1 = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 1, 1]])
    A_help_mat_p2 = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 1, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 1, 0, 0, 1]])
    A_p1_helper = pts1_norm@A_help_mat_p1
    A_p2_helper = pts2_norm@A_help_mat_p2
    A = np.multiply(A_p1_helper,A_p2_helper)
    
    # stage 2 - Find the SVD of A
    U, S, V = np.linalg.svd(A)
    F = V[-1,:]
    F = F.reshape([3,3])

    # Stage 4 - Enforce rank 2 constraint on F
    F = refineF(F, pts1_norm[:,0:2], pts2_norm[:,0:2]) # this functio includes the enforsment of rank 2
    
    # Stage 5 - Un-normalize F
    F = T.T@F@T
    return F

# Section 1.2
def epipolar_correspondences(I1, I2, F, pts1):
    """
    Epipolar Correspondences
    [I] I1, image 1 (H1xW1 matrix)
    I2, image 2 (H2xW2 matrix)
    F, fundamental matrix from image 1 to image 2 (3x3 matrix)
    pts1, points in image 1 (Nx2 matrix)
    [O] pts2, points in image 2 (Nx2 matrix)
    """
    window_size = 50
    points_in_line = 200
    num_points = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
    height, width,_ = I2.shape
    l = np.array(F @ hpts1.T)
    l = l.T

    s = np.sqrt(l[:,0]**2+l[:,1]**2)
    s = np.reshape(s,(s.shape[0],1))
    s = s @ np.ones([1,3])
    if 0 in s:
        print('Zero line vector in displayEpipolar')
        return None
    l = np.multiply(l,1/s)
    if 0 not in l[:,1]:
        xstart = window_size
        xend = width - window_size - 1
        x_vec = np.linspace(xstart, xend, num=points_in_line)
        x_vec = np.reshape(x_vec,(x_vec.shape[0],1))
        y_vec = -(l[:,0] * x_vec + l[:,2]) / l[:,1]
        #y_vec = y_vec.T
    else:
        ystart = window_size
        yend = height - window_size - 1
        y_vec = np.linspace(ystart, yend, num=points_in_line)
        y_vec = np.reshape(y_vec,(y_vec.shape[0],1))
        x_vec = -(l[:,1] * y_vec + l[:,2]) / l[:,0]
    
    pts2 = []
    for i in range(num_points):
        if not((window_size <= pts1[i,0] < width-window_size) or (window_size <= pts1[i,1] < height-window_size)):
            print('The point is too close to the edge of the image')
            break
        #I1_window = I1[pts1[i,0]-window_size:pts1[i,0]+window_size-1, pts1[i,1]-window_size:pts1[i,1]+window_size-1]
        I1_window = I1[pts1[i,1]-window_size:pts1[i,1]+window_size-1,pts1[i,0]-window_size:pts1[i,0]+window_size-1]
        I1_window = np.array(np.reshape(I1_window,(1,I1_window.shape[0]*I1_window.shape[1],3)))
        dist = []
        for j in range(points_in_line):
            if x_vec.shape[1] == 1:
                if not((window_size <= y_vec[j,i] < width-window_size) or (window_size <= y_vec[j,i] < height-window_size)):
                    print('y coordinate of epipolar line is too close the the edge of image 2')
                    break
                I2_window_j = I2[int(np.round(y_vec[j,i]))-window_size:int(np.round(y_vec[j,i]))+window_size-1,int(np.round(x_vec[j]))-window_size:int(np.round(x_vec[j]))+window_size-1]
            else:
                if not((window_size <= x_vec[j,i] < width-window_size) or (window_size <= x_vec[j,i] < height-window_size)):
                    print('x coordinate of epipolar line is too close the the edge of image 2')
                    break
                I2_window_j = I2[int(np.round(y_vec[j]))-window_size:int(np.round(y_vec[j]))+window_size-1,int(np.round(x_vec[j,i]))-window_size:int(np.round(x_vec[j,i]))+window_size-1]
            I2_window_j = np.array(np.reshape(I2_window_j,(1,I2_window_j.shape[0]*I2_window_j.shape[1],3)))
            dist_tmp = np.sum((I1_window.astype("float") - I2_window_j.astype("float")) ** 2)
            dist_tmp = dist_tmp / float(window_size**2)
            dist.append(dist_tmp)
        index_min = np.argmin(dist)
        if x_vec.shape[1] == 1:
            pts2.append([int(x_vec[index_min]),int(y_vec[index_min,i])])
        else:
            pts2.append([int(x_vec[index_min,i]),int(y_vec[index_min])])
    return np.array(pts2)

# Section 1.3
def essential_matrix(F, K1, K2):
    """
    Essential Matrix
    [I] F, the fundamental matrix (3x3 matrix)
        K1, camera matrix 1 (3x3 matrix)
        K2, camera matrix 2 (3x3 matrix)
    [O] E, the essential matrix (3x3 matrix)
    """
    E=K2.T*F*K1
    return E


# Section 1.4
def triangulate(M1, pts1, M2, pts2):
    """
    Triangulation
    [I] M1, camera projection matrix 1 (3x4 matrix)
        pts1, points in image 1 (Nx2 matrix)
        M2, camera projection matrix 2 (3x4 matrix)
        pts2, points in image 2 (Nx2 matrix)
    [O] pts3d, 3D points in space (Nx3 matrix)
    """
    def Create2Lines(M, pts):
        row1=pts[:,1]*M[2]-M[1]
        row2=M[0]- pts[:,0]*M[2]
        rows=np.stack((row1,row2),axis=1)
        return rows
    hpts1=np.hstack( ( pts1 ,np.ones((pts1.shape[0],1))))
    hpts2=np.hstack( ( pts2 ,np.ones((pts2.shape[0],1))))
    hpts1=hpts1.reshape(*hpts1.shape,1)
    hpts2=hpts2.reshape(*hpts2.shape,1)
    lines1=Create2Lines(M1,hpts1)
    lines2=Create2Lines(M2,hpts2)
    lines=np.hstack((lines1,lines2))
    U, S, V = np.linalg.svd(lines)
    hpts3d = V[:,-1,:]
    scale=hpts3d[:,-1]
    scale=scale.reshape(*scale.shape,1)
    hpts3d_normlize=np.divide(hpts3d,scale)
    return hpts3d_normlize


if __name__ == '__main__':
    I1 = cv2.imread(r'./data/im1.png')
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
    I2 = cv2.imread(r'./data/im2.png')
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)
    data = np.load("data/some_corresp.npz")
    sy, sx,_ = I1.shape
    pmax = np.maximum(sy,sx)
    F = eight_point(data['pts1'],data['pts2'],pmax)
    data2 = np.load("data/temple_coords.npz")
    pts1 = data2['pts1']
    pts2 = epipolar_correspondences(I1, I2, F, pts1)

    k=np.load('./data/intrinsics.npz')
    E=essential_matrix(F,k['K1'],k['K2'])
    zeros_vector = np.zeros([1,3])
    M1=np.identity(3)
    M1=np.hstack((M1,zeros_vector.T))
    M1=k['K1'].dot(M1)
    M2s=camera2(E)
    pts3d=0
    num_poisitve_z_cam1=0 
    num_poisitve_z_cam2=0
    selected_idx = 0
    for idx in range(M2s.shape[2]):
        M2=k['K2'].dot(M2s[:,:,idx])
        points=triangulate(M1,pts1,M2,pts2)
        validZcam1=len(points[points[:,2]>0])
        validZcam2=numValidZCam2(M2s[:,:,idx],points)
        if validZcam1>num_poisitve_z_cam1 and  validZcam2>num_poisitve_z_cam2:
              pts3d=points
              num_poisitve_z_cam1=validZcam1
              num_poisitve_z_cam2=validZcam2
              selected_idx = idx
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2])
    plt.show()
    print("M2 extrinsic matrix is: ")
    print(M2s[:,:,selected_idx])