import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Global Variables
row,col = 256,256
sigma_0 = np.sqrt(2)
octave = 3
level = 3
threshold = 0.1
r = 10
interval = level - 1


def read_image(image_path):
    '''
    Read the image from the given path and return the image in grayscale format
    
    Args:
        image_path: str: path to the image
    Returns:
        img: np.array: image in converted format for processing
    '''

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (row,col))
    img = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    return img


def scale_space_extrema_detection(img,vis=False):
    '''
    Implement the Scale-Space Extrema Detection step of the SIFT algorithm

    Args:
        img: np.array: image in grayscale format
    Returns:
        keypoints: list: list of keypoints
    '''
    # DoG Layered matrix
    D = [np.zeros((int(row * 2**(2-i)) + 2, int(col * 2**(2-i)) + 2, level)) for i in range(1,octave+1)]

    # Creating a temporary image by interpolation and making it of size 512 and then making border of one pixel all around thus size size becomes 514
    img_processed = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    img_processed = cv2.copyMakeBorder(img_processed, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    
    n = 0
    # Iterating thorugh all octaves and levels to create a DoG stack 
    for i in range(1,octave + 1):
        temp_D = D[i-1]
        for j in range(1,level + 1):
            # Create a Gaussian filter
            scale = sigma_0*(np.sqrt(2)**(1/level))** ((i-1) * level + j)
            kernel_size = int(np.floor(6*scale))
            
            # Get the kernel for gaussian filter 
            f = cv2.getGaussianKernel(kernel_size, scale) 
            
            # Difference of gaussian by applying the filter and pyramid of the various 
            L1 = img_processed
            
            # Convolving to apply gaussian filter
            if(i == 1 and j == 1):
                L2 = convolve2d(img_processed, f.reshape(1, kernel_size),mode='same') 
                L2 = convolve2d(L2, f.reshape(1, kernel_size),mode='same') 
                temp_D[:,:,j-1] = L2-L1
                L1 = L2

            else:
                L2 = convolve2d(img_processed, f.reshape(1, kernel_size),mode='same')
                L2 = convolve2d(L2, f.reshape(1, kernel_size),mode='same')                
                temp_D[:,:,j-1] = L2-L1
                L1 = L2

                if(j == level):
                    img_processed = L1[1:-2,1:-2]

            if vis == True:
                n += 1
                plot_img = 255 * temp_D[:,:,j-1]
                cv2.imshow('DoG_image_{0}'.format(n),plot_img)
                cv2.waitKey(0)

        D[i-1] = temp_D
        img_processed = img_processed[::2,::2]
        img_processed = cv2.copyMakeBorder(img_processed, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    return D


def keypoint_localization(D,img,vis=False):
    '''
    Implement the Keypoint Localization step of the SIFT algorithm 

    Args:
        D: list: list of DoG images
        img: np.array: image in grayscale format
    '''
    num = 0
    flag = 0

    for i in range(2, octave+2):
        num = num + (2**(i-octave) * col) * (2 * row) * interval

    extrema = np.zeros(int(4 * num))

    for i in range(1,octave + 1):
        m,n,_ = D[i-1].shape
        m = m - 2
        n = n - 2
        
        # Octave search space
        volume = int(m * n/4**(i - 1))   
        
        # Extract from 3 levels of octave for a pixel neighborhood
        for k in range(2,interval + 1):
            for j in range(1,volume + 1):
                x = int(((j-1)/n) + 1)
                y = np.remainder(j-1, m) + 1
                
                # Extract from 3 levels of octave for a pixel neighborhood
                sub = D[i - 1][x:x+3, y:y+3, k-2:k+1]

                # Find the max and min value in the neighborhood
                max_val = np.max(sub)
                min_val = np.min(sub)

                # Find the maxima and minima in the neighborhood
                if (max_val == D[i - 1][x, y, k-1]):
                    temp = np.array([i, k-1, j, 1])
                    extrema[flag:flag + 4] = temp
                    flag += 4

                if (min_val == D[i - 1][x, y, k-1]):
                    temp = np.array([i, k-1, j, -1])
                    extrema[flag:flag + 4] = temp
                    flag += 4

    # Eliminate the points which are not maxima or minima
    extrema = extrema[extrema != 0]
    extrema_val = extrema[2::4] 
    extrema_octave = extrema[0::4]

    # reconstructing the x and y from the given volume location and octave
    x = np.floor((extrema_val - 1) / ((col) / (2 ** (extrema_octave-2)))) +1
    y = np.remainder((extrema_val - 1), ((row) / (2 ** (extrema_octave-2))))+1
    ry = y / (2 ** (octave - 1 - extrema_octave))
    rx = x / (2 ** (octave - 1 - extrema_octave))

    if vis == True:
        plt.figure(1)
        plt.imshow(img, cmap='gray')
        plt.scatter(ry, rx, marker='+', color='green')
        plt.savefig('results/interest_keypoint_image.png')
        plt.show()

    # Accurate key point localisation
    extrema_volume = len(extrema)/4
    m,n = img.shape
    y_ddot = convolve2d([[-1,-1],[1,1]],[[-1,-1],[1,1]])            

    # Octaves and levels convolving the DoG images
    for i in range(1,octave+1):
        for j in range(1,level+1):
            test = D[i-1][:,:,j-1]
            temp = -1/convolve2d(test, y_ddot, mode='same') * convolve2d(test, [[-1,-1],[1,1]], mode='same')
            D[i-1][:,:,j-1] = temp * convolve2d(test, [[-1,-1],[1,1]], mode='same') * 0.5 + test

    # locating the extrema and selecting them based on threshold
    c = 0
    for i in range(1,int(extrema_volume+1)):
        # reconstructing x and y from the octave
        x = np.floor((extrema[4*(i-1)+2] - 1) / (n / (2 ** (extrema[4*(i-1)] - 2)))) +1
        y = np.remainder((extrema[4*(i-1)+2] - 1) ,(m / (2 ** (extrema[4*(i-1)] - 2)))) +1
        rx = int(x+1)
        ry = int(y+1)
        rz = int(extrema[4*(i-1) + 1])

        # Getting the value of the extrema and comapring it with the threshold
        z = D[int(extrema[4*(i-1)])-1][rx-1,ry-1,rz]
        if( np.abs(z) < threshold ):
            extrema[4 * (i-1) + 3] = 0
            c += 1

    # Keeping the best extremas
    index = np.where(extrema == 0)[0]
    index = np.concatenate([index,index-1,index-2,index-3])
    
    # Eliminate the points which are not maxima or minima
    extrema = np.delete(extrema,index)
    extrema_volume = len(extrema)/4
    extrema_val = extrema[2::4] 
    extrema_octave = extrema[0::4]

    # reconstructing the x and y from the given volume location and octave
    x = np.floor((extrema_val - 1) / ((col) / (2 ** (extrema_octave-2)))) +1
    y = np.remainder((extrema_val - 1), ((row) / (2 ** (extrema_octave-2))))+1
    ry = y / (2 ** (octave - 1 - extrema_octave))
    rx = x / (2 ** (octave - 1 - extrema_octave))

    if vis == True:
        plt.figure(1)
        plt.imshow(img,cmap='gray')
        plt.scatter(ry,rx,marker='+',color='red')
        plt.savefig('results/filtered_keypoint_image.png')
        plt.show()

    # re-iterating through the filtered extremas and then find the double derivatives
    d = 0
    for i in range(1,int(extrema_volume+1)):
        x = np.floor((extrema[4*(i-1)+2] - 1) / (col / (2 ** (extrema[4*(i-1)] - 2)))) +1
        y = np.remainder((extrema[4*(i-1)+2] - 1) ,(row / (2 ** (extrema[4*(i-1)] - 2)))) +1
        rx = int(x+1)
        ry = int(y+1)
        rz = extrema[4*(i-1 )+1]
        rz = int(rz)

        # Calculating the doube derivatives 
        Dxx = D[int(extrema[4*(i-1)])-1][rx-2,ry-1,rz]+D[int(extrema[4*(i-1)])-1][rx,ry-1,rz]-2*D[int(extrema[4*(i-1)])-1][rx-1,ry-1,rz]
        Dyy = D[int(extrema[4*(i-1)])-1][rx-1,ry-2,rz]+D[int(extrema[4*(i-1)])-1][rx-1,ry,rz]-2*D[int(extrema[4*(i-1)])-1][rx-1,ry-1,rz]
        Dxy = D[int(extrema[4*(i-1)])-1][rx-2,ry-2,rz]+D[int(extrema[4*(i-1)])-1][rx,ry,rz]*D[int(extrema[4*(i-1)])-1][rx-2,ry,rz]*D[int(extrema[4*(i-1)])-1][rx,ry-2,rz]

        # Calculating the Hessian matrix
        Det = Dxx*Dyy - Dxy*Dxy
        R = (Dxx + Dyy)/Det
        R_thres = (r+1)**2/r
        
        if(Det < 0 or R_thres < R):
            extrema[4*(i-1) + 3] = 0
            d += 1

    # Keeping the best extremas
    index = np.where(extrema == 0)[0]
    index = np.concatenate([index,index - 1,index - 2,index - 3])
    
    # Eliminate the points which are not maxima or minima
    extrema = np.delete(extrema,index)
    extrema_volume = len(extrema)/4
    extrema_val = extrema[2::4] 
    extrema_octave = extrema[0::4]

    # reconstructing the x and y from the given volume location and octave
    x = np.floor((extrema_val - 1) / ((col) / (2 ** (extrema_octave-2)))) +1
    y = np.remainder((extrema_val - 1), ((row) / (2 ** (extrema_octave-2))))+1
    ry = y / (2 ** (octave - 1 - extrema_octave))
    rx = x / (2 ** (octave - 1 - extrema_octave))
    
    return [rx,ry]


def custom_SIFT(): 
    vis = True

    # Read the image
    lenna_image = read_image('lenna.png')

    # Scale-Space Extrema Detection
    D = scale_space_extrema_detection(lenna_image, vis)

    # Accurate Keypoint localization
    keypoints = keypoint_localization(D, lenna_image, vis)

    # Plot the final keypoints
    plt.figure(1)
    plt.imshow(lenna_image,cmap='gray')
    plt.scatter(keypoints[0],keypoints[1],marker='+',color='blue')
    plt.savefig('results/final_keypoint_image.png')
    plt.show()


if __name__ == '__main__':
    custom_SIFT()


