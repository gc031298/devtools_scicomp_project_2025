from pyclassify.utils import read_config, read_file
from pyclassify import kNN
import random

# Read the configuration file
kwargs = read_config('experiments/config')
dataset = kwargs['dataset']
k = kwargs['k']

# Read the dataset
features, labels = read_file(dataset)

# Generate random indices
indices = list(range(len(features)))
random.shuffle(indices)
features = [features[i] for i in indices]
labels = [labels[i] for i in indices]

# Split the dataset into test 80% and train 20%
test_size = int(len(features) * .8)
features_test = features[:test_size]
features_train = features[test_size:]
labels_test = labels[:test_size]
labels_train = labels[test_size:]

# Train the model
knn = kNN(k)
knn((features_train, labels_train), features_test)

# Compute the accuracy
score = sum([i == j for i, j in zip(labels_test, knn.prediction)])
acc = score / len(labels_test)
print(f"Accuracy: {acc:.2f}")
