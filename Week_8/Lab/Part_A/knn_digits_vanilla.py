import numpy as np
import cv2 as cv

img = cv.imread('../Data/digits.png')
if img is None:
    raise ValueError("Failed to load the image. Please check the file path.")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# Make it into a Numpy array: its size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare the training data and test data
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

print(f"Train data shape: {train.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test data shape: {test.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Initiate kNN, train it on the training data, then test it with the test data with k=5
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5)

# Now we check the accuracy of classification - For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print(f"Accuracy: {accuracy}")