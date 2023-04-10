import os
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from joblib import dump

# Define the path to the MALL dataset
MALL_DIR = 'MALL/'

# Load the labels
labels_df = pd.read_csv(os.path.join(MALL_DIR, 'labels.csv'))

# Load the pre-extracted image features
image_features = np.load(os.path.join(MALL_DIR, 'images.npy'))

# Load labels
labels = np.load(os.path.join(MALL_DIR, 'labels.npy'))

# Extract SIFT features from the images
sift = cv2.SIFT_create()
descriptors = []

for i in range(image_features.shape[0]):
    img_path = os.path.join(MALL_DIR, 'frames', 'frames', 'seq_{:06d}.jpg'.format(i+1))
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    descriptors.append(des)
descriptors = np.vstack(descriptors)

# Cluster the SIFT features using k-means
kmeans = KMeans(n_clusters=100, random_state=42)
kmeans.fit(descriptors)

# Construct the feature vectors for the images
image_vectors = np.zeros((image_features.shape[0], 100))
for i in range(image_features.shape[0]):
    img_path = os.path.join(MALL_DIR, 'frames', 'frames', 'seq_{:06d}.jpg'.format(i+1))
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    if des is not None:
        cluster_labels = kmeans.predict(des)
        for label in cluster_labels:
            image_vectors[i, label] += 1
    image_vectors[i, :] /= np.sum(image_vectors[i, :])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_vectors, labels, test_size=0.2, random_state=42)

# Train SVR
svr = SVR(kernel='rbf', epsilon=0.1, C=100)
svr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svr.predict(X_test)

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print('Mean absolute error: {:.2f}'.format(mae))

# Save the trained model
dump(svr, 'svr_model.joblib')

# Display an image from the test set with the predicted and actual counts
img_path = os.path.join(MALL_DIR, 'frames', 'frames', 'seq_000001.jpg')
img = cv2.imread(img_path)
actual_count = labels_df[labels_df['id'] == 1]['count'].values
predicted_count = svr.predict(X_test[0].reshape(1,-1))[0]
ax = plt.subplot(1, 1, 1)
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax.axis('off')
ax.text(10, 30, 'Actual count: {}'.format(actual_count[0]), color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
ax.text(10, 60, 'Predicted count: {:.2f}'.format(predicted_count), color='white', fontsize=10, bbox=dict(facecolor='blue', alpha=0.5))
plt.show()