# MAKELESS: Ascending from Noobgrad to Tensor Lord

## Context
This is my rigorous, daily checklist for Andrej Karpathy's `makemore` series (Videos 2 through 5). I am building a character-level autoregressive language model from scratch.

## Week 1: The Bigram Baseline & Tensor Basics (Makemore Part 1)
**Goal:** Build a literal counting model, then recreate it using a 1-layer Neural Network. Learn PyTorch tensor semantics.

- [ ] **Day 1: Data & Counting**
  - [ ] Load the `names.txt` dataset.
  - [ ] Create a character-to-integer (`stoi`) and integer-to-character (`itos`) mapping.
  - [ ] Build a 2D integer tensor `N` of shape `[27, 27]` to count bigram frequencies.
  - [ ] Visualize the count matrix using `matplotlib.pyplot.imshow`.
- [ ] **Day 2: Probability & Broadcasting**
  - [ ] Convert the `N` matrix into a probability matrix `P`.
  - [ ] **Checkpoint:** Did I divide by the sum across the correct dimension? Did `keepdim=True` save my life?
  - [ ] Implement model evaluation: Calculate the Negative Log-Likelihood (NLL) of the dataset.
  - [ ] Implement "model smoothing" (adding a fake count of 1 to everything to prevent `-inf` loss).
- [ ] **Day 3: The Simplest Neural Network**
  - [ ] Create a training set of inputs `xs` and targets `ys`.
  - [ ] Implement one-hot encoding for the inputs: `F.one_hot`.
  - [ ] Initialize a single weight matrix `W` of shape `[27, 27]`.
  - [ ] Write the forward pass: `logits = xenc @ W`. Apply softmax (exp, then normalize).
  - [ ] Write the training loop (forward, zero_grad, backward, update W). 
  - [ ] **Checkpoint:** Verify the NN loss converges to the exact same loss as the counting method.

## Week 2: The Bengio 2003 MLP (Makemore Part 2)
**Goal:** Implement "A Neural Probabilistic Language Model". Fix the curse of dimensionality using embeddings.

- [ ] **Day 4: The Embedding Table**
  - [ ] Define the embedding dimensionality (e.g., 2D for visualization).
  - [ ] Initialize embedding table `C` of shape `[27, 2]`.
  - [ ] Implement the context window (e.g., block size 3).
  - [ ] Build the new dataset `X` (context) and `Y` (target).
  - [ ] Look up embeddings for `X` using PyTorch's advanced indexing: `emb = C[X]`.
- [ ] **Day 5: The Hidden Layer**
  - [ ] Initialize `W1` and `b1`.
  - [ ] **Checkpoint:** Conquer tensor reshaping. Use `.view()` to flatten the `emb` tensor from `[B, block_size, C]` to `[B, block_size * C]`.
  - [ ] Implement the hidden layer forward pass with a `tanh` activation.
- [ ] **Day 6: Output & Training Loop**
  - [ ] Initialize `W2` and `b2`.
  - [ ] Compute logits and use `F.cross_entropy` (understand why it's numerically stable compared to manual softmax).
  - [ ] Set up the mini-batch training loop.
- [ ] **Day 7: The Learning Rate (LR) Finder**
  - [ ] Implement a learning rate finder. Step the LR exponentially from `0.001` to `1.0`.
  - [ ] Plot the loss vs. learning rate to find the "valley" of optimal learning.
  - [ ] Train the model and split data into Train (80%), Val (10%), Test (10%).

## Week 3: The Guts - Initialization & BatchNorm (Makemore Part 3)
**Goal:** Look inside the network. Understand vanishing gradients, dead neurons, and how to fix them.

- [ ] **Day 8: Diagnosing the Patient**
  - [ ] Look at the initialization loss. Is it way higher than $-log(1/27)$? Fix the scale of `W2` and `b2`.
  - [ ] Plot a histogram of the `tanh` hidden layer activations `h`.
  - [ ] **Checkpoint:** Observe the "dead neurons" (activations squashed at -1 or 1). Realize that during backprop, the gradient is multiplied by `(1 - tanh(x)^2)`. If `tanh(x)` is 1, gradient is 0. Network stops learning.
- [ ] **Day 9: Kaiming Initialization**
  - [ ] Read up on Kaiming He's initialization.
  - [ ] Scale `W1` by the fan-in (e.g., `(5/3) / sqrt(fan_in)` for tanh).
  - [ ] Re-plot the histograms. Marvel at the beautifully distributed activations.
- [ ] **Day 10: Batch Normalization**
  - [ ] Rip out Kaiming init. We are going to brute force it with BatchNorm.
  - [ ] Compute the batch mean and batch variance of the hidden states *before* the `tanh`.
  - [ ] Normalize the hidden states to mean 0, variance 1.
  - [ ] Introduce the learned scale `gamma` and shift `beta` parameters.
- [ ] **Day 11: BatchNorm Inference (Folding)**
  - [ ] Track the running mean and running variance during training using exponential moving average (EMA).
  - [ ] Update the forward pass to use running stats during inference.
  - [ ] Understand why we can mathematically fold the BatchNorm parameters into the Linear layer weights for zero-overhead inference.

## Week 4: The Backprop Ninja (Makemore Part 4)
**Goal:** Take off the PyTorch training wheels. Calculate the analytical gradients by hand.

- [ ] **Day 12: Manual Cross Entropy & Softmax**
  - [ ] Set `requires_grad=True` on parameters, but DO NOT use `.backward()`.
  - [ ] Derive and implement the backward pass for `F.cross_entropy`. (Hint: It's surprisingly elegant: `probs - 1` for the true label).
  - [ ] Compare `dlogits` manual vs `dlogits` PyTorch using `torch.allclose`.
- [ ] **Day 13: Manual Linear and Tanh**
  - [ ] Write the backward pass for `tanh`.
  - [ ] Write the backward pass for the linear layer (`dgamma`, `dbeta`, `dW1`, `db1`).
  - [ ] Master matrix calculus transpose rules. If `Y = X @ W`, then `dW = X.T @ dY`.
- [ ] **Day 14: The Final Boss (Manual BatchNorm)**
  - [ ] Cry. 
  - [ ] Look up the analytical derivative of Batch Normalization.
  - [ ] Implement the sprawling, multi-line backward pass for BatchNorm.
  - [ ] Run `torch.allclose`. If it passes, you are officially an ML Engineer.
