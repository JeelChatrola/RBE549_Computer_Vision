import cv2
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
import os 
import glob
import open3d as o3d

# Part 2: Feature Extraction and Matching
def extract_and_match_features(img1, img2):
    # Create SIFT object
    sift = cv2.SIFT_create()
    
    # Detect and compute keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    
    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Perform matching
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.69 * n.distance:
            good.append(m)
    
    return kp1, kp2, good

# Part 3: Fundamental Matrix Calculation
def calculate_fundamental_matrix(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    
    # Verify the epipolar constraint
    for i in range(len(matches)):
        pt1 = np.array([pts1[i][0][0], pts1[i][0][1], 1])
        pt2 = np.array([pts2[i][0][0], pts2[i][0][1], 1])
        error = np.abs(np.dot(np.dot(pt2.T, F), pt1))
        print(f"Epipolar constraint error for point {i}: {error}")

    return F

# Part 4: Essential Matrix Calculation
def calculate_essential_matrix(F, K):
    E = np.dot(np.dot(K.T, F), K)
    
    # Enforce the constraint that E should have rank 2
    U, S, Vt = np.linalg.svd(E)
    S = [1, 1, 0]  # Force the singular values
    E = np.dot(U, np.dot(np.diag(S), Vt))
    
    print(f"Determinant of E: {np.linalg.det(E)}")
    return E

# Part 5: Decompose Essential Matrix
def decompose_essential_matrix(E):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    R1 = np.dot(U, np.dot(W, Vt))
    R2 = np.dot(U, np.dot(W.T, Vt))
    t = U[:, 2]
    
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    
    return R1, R2, t

# Part 6: Create Projection Matrices
def create_projection_matrices(K, R1, R2, t):
    P0 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P1_1 = np.dot(K, np.hstack((R1, t.reshape(3, 1))))
    P1_2 = np.dot(K, np.hstack((R1, -t.reshape(3, 1))))
    P1_3 = np.dot(K, np.hstack((R2, t.reshape(3, 1))))
    P1_4 = np.dot(K, np.hstack((R2, -t.reshape(3, 1))))
    
    return P0, [P1_1, P1_2, P1_3, P1_4]

# Part 7: Triangulation
def linear_ls_triangulation(u1, P1, u2, P2):
    A = np.zeros((4, 4))
    A[0] = u1[0] * P1[2] - P1[0]
    A[1] = u1[1] * P1[2] - P1[1]
    A[2] = u2[0] * P2[2] - P2[0]
    A[3] = u2[1] * P2[2] - P2[1]
    
    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X / X[3]

# Updated Part 8: Reprojection Error
def calculate_reprojection_error(points_3d, points_2d, P):
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    reprojected_points = np.dot(P, points_3d_homogeneous.T).T
    reprojected_points = reprojected_points[:, :2] / reprojected_points[:, 2:]
    error = np.mean(np.sqrt(np.sum((reprojected_points - points_2d)**2, axis=1)))
    return error

# Part 9: Save PCD File
def save_pcd_to_file(points_3d, colors, filename):
    with open(filename, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write(f"FIELDS x y z rgb\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F U\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points_3d)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points_3d)}\n")
        f.write("DATA ascii\n")
        
        for point, color in zip(points_3d, colors):
            rgb = color[0] << 16 | color[1] << 8 | color[2]
            f.write(f"{point[0]} {point[1]} {point[2]} {rgb}\n")

def visualize_matches(img1, img2, kp1, kp2, matches, num_matches=100):
    # Create a new output image that concatenates the two images
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    out = np.zeros((max([rows1,rows2]), cols1+cols2, 3), dtype='uint8')
    
    # Place the first image to the left
    out[:rows1,:cols1,:] = img1
    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = img2
    
    # Use only the top matches
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:num_matches]
    
    # Generate random colors for each match
    colors = np.random.randint(0, 255, (len(matches), 3))
    
    # For each pair of points we have between both images
    for i, mat in enumerate(matches):
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns, y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1),int(y1)), 4, colors[i].tolist(), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, colors[i].tolist(), 1)

        # Draw a line in between the two points
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), colors[i].tolist(), 1)
    
    # Show the image
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.title('Feature Matching')
    plt.axis('off')
    plt.show()

def visualize_pcd_with_open3d(pcd_file_path):
    # Read the point cloud
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    
    # Print some basic information about the point cloud
    print(f"Point cloud contains {len(pcd.points)} points.")
    print(f"Point cloud has colors: {pcd.has_colors()}")
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


# Main execution
if __name__ == "__main__":
    # img1 = cv2.imread('image1.jpg')
    # img2 = cv2.imread('image2.jpg')
    # K = np.array([[4000, -8.5, width/2],
            #  [0, 4000, height/2],
            #  [0, 0, 1]])

    img1 = cv2.imread('a.jpg')
    img2 = cv2.imread('b.jpg')

    # load K and dist from folder
    K = np.load('camera_matrix.npy')
    dist = np.load('camera_dist_coeff.npy')
    
    height, width = img1.shape[:2]

    K = np.array([[3900, -9.5, width/2],
                 [0, 3600, height/2],
                 [0, 0, 1]])
    
    print("Camera matrix:")
    print(K)
    print("Distortion coefficients:")
    print(dist)

    # undistort images
    img1 = cv2.undistort(img1, K, dist)
    img2 = cv2.undistort(img2, K, dist)
    
    # Extract and match features
    kp1, kp2, matches = extract_and_match_features(img1, img2)
    
    visualize_matches(img1, img2, kp1, kp2, matches)

    # Calculate Fundamental Matrix
    F = calculate_fundamental_matrix(kp1, kp2, matches)
    
    # Calculate Essential Matrix
    E = calculate_essential_matrix(F, K)
    
    # Decompose Essential Matrix
    R1, R2, t = decompose_essential_matrix(E)
    
    # Create Projection Matrices
    P0, P1_list = create_projection_matrices(K, R1, R2, t)
    
    # Triangulate points
    points_3d = []
    valid_P1 = None
    for P1 in P1_list:
        points_3d_temp = []
        for match in matches:
            u1 = np.array([kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1], 1])
            u2 = np.array([kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1], 1])
            X = linear_ls_triangulation(u1, P0, u2, P1)
            points_3d_temp.append(X[:3])
        
        # Check if points are in front of both cameras
        if all(X[2] > 0 for X in points_3d_temp):
            points_3d = points_3d_temp
            valid_P1 = P1
            break
    
    if not points_3d:
        print("No valid reconstruction found. All points are behind at least one camera.")
        exit()

    points_3d = np.array(points_3d)
    
    # Calculate reprojection error
    points_2d = np.array([kp1[m.queryIdx].pt for m in matches])
    error_cam1 = calculate_reprojection_error(points_3d, points_2d, P0)
    error_cam2 = calculate_reprojection_error(points_3d, points_2d, valid_P1)
    print(f"Reprojection error for camera 1: {error_cam1}")
    print(f"Reprojection error for camera 2: {error_cam2}")
    
    # Save PCD file
    colors = [img1[int(kp1[m.queryIdx].pt[1]), int(kp1[m.queryIdx].pt[0])] for m in matches]
    save_pcd_to_file(points_3d, colors, 'results/output.pcd')

    # Open the PCD file with Open3D
    print("Output PCD file saved. Opening the file with Open3D...")
    visualize_pcd_with_open3d('results/output.pcd')
