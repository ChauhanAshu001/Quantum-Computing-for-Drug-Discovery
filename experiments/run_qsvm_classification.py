import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Qiskit imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# 1. Load Real Dataset
print("Loading and shuffling real QSAR biodegradation data...")
df = pd.read_csv('qsar-biodeg.csv')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. PCA for Dimensionality Reduction (41 features -> 2 features)
print("Applying PCA to compress features...")
X_all = df.iloc[:100, :-1].values 
pca = PCA(n_components=2)
X_raw = pca.fit_transform(X_all)

# Use LabelEncoder for targets
y_raw = df.iloc[:100, -1].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Scale data for the quantum feature map
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Setup Quantum Kernel
print("Setting up Quantum Feature Map...")
feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

# 4. Train QSVM
print("Training QSVM... (Please wait, quantum simulation takes a minute)")
qsvm = SVC(kernel=quantum_kernel.evaluate)
qsvm.fit(X_train, y_train)

# 5. Evaluate and Save Accuracy
print("Evaluating model...")
accuracy = accuracy_score(y_test, qsvm.predict(X_test))
print(f"Model Accuracy: {accuracy * 100:.2f}%")

os.makedirs('../results', exist_ok=True)
with open('../results/qsvm_output.txt', 'w') as f:
    f.write(f"QSVM Classification Accuracy: {accuracy * 100:.2f}%\n")

# 6. Plot and Save Graph
print("Generating and saving graph...")
y_pred = qsvm.predict(X_test)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

classes = label_encoder.classes_
ax1.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='red', label=f'{classes[0]} (0)', edgecolors='k')
ax1.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='blue', label=f'{classes[1]} (1)', edgecolors='k')
ax1.set_title("Actual Classes (PCA Features)")
ax1.set_xlabel("Principal Component 1")
ax1.set_ylabel("Principal Component 2")
ax1.legend()

ax2.scatter(X_test[y_pred == 0, 0], X_test[y_pred == 0, 1], color='red', label=f'Pred {classes[0]}', edgecolors='k')
ax2.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], color='blue', label=f'Pred {classes[1]}', edgecolors='k')
ax2.set_title(f"QSVM Predictions (Accuracy: {accuracy * 100:.2f}%)")
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
ax2.legend()

plt.savefig('../results/qsvm_classification_results.png', dpi=300, bbox_inches='tight')
print("Graph successfully saved to results/qsvm_classification_results.png")