# MyDeepLib: Neural Network from Scratch

**MyDeepLib** is a lightweight, modular Deep Learning library built entirely from scratch using **Python** and **NumPy**.

This project demonstrates the core mechanics of neural networks—forward propagation, backpropagation (derivatives), and parameter optimization—without relying on high-level frameworks like TensorFlow or PyTorch.

YOU CAN READ nn_full_code.ipynb (full code and example) instead of my_deep_lib folder 

## Features

This library implements the full Deep Learning pipeline manually:

1.  **Dense Layers**:
    * **Forward**: Dot product + Bias calculation.
    * **Backward**: Partial derivatives calculation w.r.t weights, biases, and inputs ($dWeights, dBiases, dInputs$).
    * **Regularization**: L1 & L2 support to prevent overfitting.
2.  **Activation Functions**:
    * **ReLU**: Rectified Linear Unit for hidden layers (includes derivative logic).
    * **Softmax**: Probability distribution for output layers.
3.  **Loss Function**:
    * **Categorical Cross-Entropy**: Standard loss for multi-class classification (e.g., 3 classes).
4.  **Optimization Engine**:
    * **Combined Softmax + Cross-Entropy**: Optimized backward step for numerical stability and speed.
    * **Adam Optimizer**: The default recommendation. Implements Learning Rate Decay, Momentum ($\beta_1$), and RMSProp Cache ($\beta_2$).
    * **Other Optimizers**: SGD, SGD with Momentum, Adagrad, RMSProp.
5.  **Model Management**:
    * High-level API: `add()`, `compile()`, `fit()`, `predict()`.
    * Save & Load model capabilities (`pickle`).

---
## Architecture & Data Flow
The library architecture strictly follows the Forward and Backward pass logic.
1. Code  layer dense (forward + backward)
2. Code activation function
  	relu: neuron move through layer (forward + backward )
  	softmax: output last neuron 
3. Code loss function, classification 3 classes
4. Combined Softmax activation and Cross-Entropy loss 
5. Optimizer (adam) (lr rate, rate decay, momentum, RMSProp, sgd)
6. Build lib 
7. Use 
dempty: d = derivative

model: X, y = spiral_data(samples=100, classes=3)
```mermaid
graph LR
Forward pass 
	
	Input (2) 
	  → Dense Layer (foward) (2 → 64) 
	  → ReLU (forward)
	  → Dense Layer (foward) (64 → 3) 
	  → Softmax 
	  → Loss Function (Categorical Cross-Entropy)
	  
	  
Backward Pass 
   Loss Gradient 
	  → Softmax + Cross-Entropy Backward 
	  → Dense Layer (backward) (64 → 3) 
	  → ReLU (backward)
	  → Dense Layer (backward) (2 → 64) 
	  → Adam Optimizer (update weights + biases) 
	  → Update Parameters (dense 1 and dense 2)
	  
<b>Installation & Usage</b>

<b> 1. Installation	 </b> 
pip install -r requirements.txt

<b> 2. Usage Example </b>
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Import custom library
from my_deep_lib import NeuralNetwork_Model, Layer_Dense, Activation_ReLU

# 1. Prepare Data
nnfs.init()
X, y = spiral_data(samples=100, classes=3)

# 2. Initialize Model
model = NeuralNetwork_Model()

# 3. Build Architecture (Matches Diagram)
# Input: 2 features -> Hidden: 64 neurons
model.add(Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())

# Hidden: 64 -> Output: 3 classes
model.add(Layer_Dense(64, 3))

# 4. Compile (Adam + Categorical Cross-Entropy)
model.compile(optimizer='adam', loss='categorical_crossentropy', learning_rate=0.05, decay=5e-5)

# 5. Train
print("--- Starting Training ---")
model.fit(X, y, epochs=10001, print_every=1000)

# 6. Validate
print("\n--- Validation ---")
X_test, y_test = spiral_data(samples=100, classes=3)
probs = model.predict(X_test)
accuracy = np.mean(np.argmax(probs, axis=1) == y_test)
print(f"Validation Accuracy: {accuracy:.3f}")