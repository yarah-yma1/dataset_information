# FASHION MNIST DATASET MODEL ACCURACY

import numpy as np
import time
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras import layers, models

# logging
results = []

def log_result(dataset, model_name, accuracy, runtime):
    results.append({
        "Dataset": dataset,
        "Model": model_name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Runtime (s)": round(runtime, 2)
    })

print("\n--- FASHION MNIST ---")

# loading dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# normalizing dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

# flatten for classical ML models
x_train_flat = x_train.reshape(len(x_train), -1)
x_test_flat = x_test.reshape(len(x_test), -1)

# traditonal ML models
for name, model in [
    ("SVM", SVC(max_iter=200)),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier(n_estimators=50))
]:
    start = time.time()
    model.fit(x_train_flat[:5000], y_train[:5000])
    acc = model.score(x_test_flat[:1000], y_test[:1000])
    log_result("Fashion-MNIST", name, acc, time.time() - start)

# shallow neural network
start = time.time()
shallow_nn = MLPClassifier(hidden_layer_sizes=(128,), max_iter=3)
shallow_nn.fit(x_train_flat[:5000], y_train[:5000])
acc = shallow_nn.score(x_test_flat[:1000], y_test[:1000])
log_result("Fashion-MNIST", "Shallow NN", acc, time.time() - start)

# deep neural network 
start = time.time()
deep_nn = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=3)
deep_nn.fit(x_train_flat[:5000], y_train[:5000])
acc = deep_nn.score(x_test_flat[:1000], y_test[:1000])
log_result("Fashion-MNIST", "Deep NN", acc, time.time() - start)

# CNN
start = time.time()
cnn = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(x_train[..., np.newaxis], y_train, epochs=1, batch_size=128, verbose=0)
acc = cnn.evaluate(x_test[..., np.newaxis], y_test, verbose=0)[1]
log_result("Fashion-MNIST", "CNN", acc, time.time() - start)

# print
print("\n=== MODEL RESULTS (Fashion MNIST) ===")
for r in results:
    print(f"{r['Model']:<15}  Accuracy: {r['Accuracy (%)']}%   Runtime: {r['Runtime (s)']}s")
