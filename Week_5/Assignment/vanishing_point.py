import numpy as np
import cv2

def canny_edge_detection(image, low_threshold, high_threshold):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the Canny Edge detection algorithm to extract the edges of the image
    edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=3)

    return edges

def hough_line_transform(image, rho, theta, threshold):
    # Apply the Hough Transform method to detect lines
    lines = cv2.HoughLines(image, rho, theta, threshold)
    
    return lines

def draw_lines(img, lines):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw the lines on the image
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img

def least_square_solution(lines):
    A = []
    b = []

    for line in lines:
        for rho, theta in line:
            A.append([np.cos(theta), np.sin(theta)])
            b.append(rho)

    A = np.array(A)
    b = np.array(b)

    # Solve for t using the least square solution
    A_transpose = np.transpose(A) # A^T
    A_transpose_A = np.dot(A_transpose, A) # A^T * A
    A_transpose_A_inv = np.linalg.inv(A_transpose_A) # (A^T * A)^-1
    A_transpose_b = np.dot(A_transpose, b) # A^T * b

    t = np.dot(A_transpose_A_inv, A_transpose_b) # (A^T * A)^-1 * A^T * b

    # Calculate the vanishing point (u, v)
    u = t[0]
    v = t[1]

    return u, v


def main():# Read the image from Images folder 
    vis = True
    save = True
    img = cv2.imread('Images/texas.png')

    # Apply the Canny Edge detection algorithm to extract the edges of the image
    edge_image = canny_edge_detection(img, 290, 350)

    # Employ the Hough Transform method to detect lines
    lines = hough_line_transform(edge_image, 1, np.pi / 180, 200)

    # Calculate the vanishing point (u, v) using the least square solution
    u,v = least_square_solution(lines)

    # Visually mark the (u, v) position with a distinctive red circle on the image plane
    cv2.circle(img, (int(u), int(v)), 15, (0, 0, 255), -1)

    # Display the image
    if vis:
        cv2.imshow('Original Image', img)
        cv2.imshow('Detected Edges', edge_image)
        cv2.imshow('Detected Lines', draw_lines(edge_image, lines))
        cv2.imshow('Vanishing_Point', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save the images
    if save:
        cv2.imwrite('Images/vanishing_point.png', img)
        cv2.imwrite('Images/detected_edges.png', edge_image)
        cv2.imwrite('Images/detected_lines.png', draw_lines(edge_image, lines))

if __name__ == '__main__':
    main()
