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
	  → Adam Optimizer (update weights + biasé) 
	  → Update Parameters (dense 1 and dense 2)
	  
	  
	  
