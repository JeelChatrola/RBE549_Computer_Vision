import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('../Data/digits.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
x = np.array(cells)

# Now we prepare the training data and test data
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

k_list=np.arange(1,10,1)
accuracies=[]
for k in range(1,10,1):
    knn = cv.ml.KNearest_create()
    knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
    ret,result,neighbours,dist = knn.findNearest(test,k=k)
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    accuracies.append(accuracy)
    print( accuracy )


# Increase the figure size
plt.figure(figsize=(10, 6))

plt.plot(k_list, accuracies, marker='o')

# Adding grid lines
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Enhancing the title and labels
plt.title('KNN Accuracy vs. K Value for Digit Recognition', fontsize=14)
plt.xlabel('K Values', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)

# Enhancing the tick labels size
plt.xticks(k_list, fontsize=10)
plt.yticks(fontsize=10)

# Save and show the plot with enhancements
plt.savefig('results/K_Value_graph.png')
plt.show()