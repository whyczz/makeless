# MAKELESS: Ascending from Noobgrad to Tensor Lord

## Context
This is my rigorous, daily checklist for Andrej Karpathy's `makemore` series (Videos 1 through 4). I am building a character-level autoregressive language model from scratch.

---

## Week 1: The Bigram Baseline & Tensor Basics (Makemore Part 1)
**Video:** [The spelled-out intro to language modeling: building makemore](https://youtu.be/PaCmpygFfXo)
**Goal:** Build a literal counting model, then recreate it using a 1-layer Neural Network. Learn PyTorch tensor semantics.

- [x] **Day 1: Data & Counting**
  - [x] Load the `names.txt` dataset and explore it [[00:03:04]](https://youtu.be/PaCmpygFfXo?t=3m4s).
  - [x] Create character-to-integer (`stoi`) and integer-to-character (`itos`) mappings.
  - [x] Build a 2D integer tensor `N` of shape `[27, 27]` to count bigram frequencies [[00:09:26]](https://youtu.be/PaCmpygFfXo?t=9m26s).
  - [x] Visualize the count matrix using `matplotlib.pyplot.imshow`.
- [ ] **Day 2: Probability, Broadcasting, & Loss**
  - [ ] Convert the `N` matrix into a probability matrix `P` [[00:25:31]](https://youtu.be/PaCmpygFfXo?t=25m31s).
  - [ ] **Checkpoint:** Understand PyTorch Broadcasting semantics. Watch Karpathy demonstrate the `keepdim=True` bug [[00:41:44]](https://youtu.be/PaCmpygFfXo?t=41m44s). Do not skip this.
  - [ ] Evaluate the model using Negative Log-Likelihood (NLL) [[00:50:47]](https://youtu.be/PaCmpygFfXo?t=50m47s).
  - [ ] Implement "model smoothing" (adding fake counts) to prevent `-inf` loss.
- [ ] **Day 3: The Simplest Neural Network**
  - [ ] Cast the bigram problem into a neural network framework [[01:04:07]](https://youtu.be/PaCmpygFfXo?t=1h4m7s).
  - [ ] Implement one-hot encoding for the integer inputs: `F.one_hot` [[01:10:04]](https://youtu.be/PaCmpygFfXo?t=1h10m4s).
  - [ ] Write the forward pass using matrix multiplication: `logits = xenc @ W` [[01:13:52]](https://youtu.be/PaCmpygFfXo?t=1h13m52s).
  - [ ] Write the training loop (forward, zero_grad, backward, update) [[01:32:00]](https://youtu.be/PaCmpygFfXo?t=1h32m0s). Verify it matches the counting method's loss.

---

## Week 2: The Bengio 2003 MLP (Makemore Part 2)
**Video:** [Building makemore Part 2: MLP](https://youtu.be/TCH_1BHY58I)
**Goal:** Implement "A Neural Probabilistic Language Model". Fix the curse of dimensionality using embeddings.

- [ ] **Day 4: The Embedding Table**
  - [ ] Review the Bengio 2003 paper architecture [[00:01:51]](https://youtu.be/TCH_1BHY58I?t=1m51s).
  - [ ] Initialize embedding lookup table `C` of shape `[27, 2]` [[00:12:16]](https://youtu.be/TCH_1BHY58I?t=12m16s).
  - [ ] Build dataset `X` (context window, e.g., block size 3) and `Y` (target).
  - [ ] Use PyTorch multi-dimensional indexing to look up embeddings: `emb = C[X]`.
- [ ] **Day 5: The Hidden Layer & Tensor Reshaping**
  - [ ] Initialize `W1` and `b1` for the hidden layer [[00:18:39]](https://youtu.be/TCH_1BHY58I?t=18m39s).
  - [ ] **Checkpoint:** Conquer tensor reshaping. Learn why `.view()` is insanely fast due to underlying storage rules and use it to flatten the `emb` tensor from `[B, block_size, C]` to `[B, block_size * C]` [[00:23:40]](https://youtu.be/TCH_1BHY58I?t=23m40s).
  - [ ] Implement the forward pass with `tanh` activation: `h = torch.tanh(emb.view(...) @ W1 + b1)`.
- [ ] **Day 6: Output & Training Loop**
  - [ ] Initialize `W2` and `b2` and compute logits [[00:29:13]](https://youtu.be/TCH_1BHY58I?t=29m13s).
  - [ ] Replace manual softmax with `F.cross_entropy`. Understand *why* it's numerically stable (subtracting the max logit internally) [[00:32:52]](https://youtu.be/TCH_1BHY58I?t=32m52s).
  - [ ] Set up the mini-batch training loop [[00:37:56]](https://youtu.be/TCH_1BHY58I?t=37m56s).
- [ ] **Day 7: The Learning Rate (LR) Finder**
  - [ ] Implement a learning rate finder by stepping exponentially from `0.001` to `1.0` [[00:45:29]](https://youtu.be/TCH_1BHY58I?t=45m29s).
  - [ ] Plot loss vs. learning rate to find the optimal valley. 
  - [ ] Train the model proper (splitting into Train/Val/Test).

---

## Week 3: The Guts - Initialization & BatchNorm (Makemore Part 3)
**Video:** [Building makemore Part 3: Activations & Gradients, BatchNorm](https://youtu.be/P6sfmUTpUmc)
**Goal:** Look inside the network. Understand vanishing gradients, dead neurons, and how to fix them.

- [ ] **Day 8: Diagnosing the Patient (Dead Neurons)**
  - [ ] Look at the initialization loss. Why is it way higher than expected? Fix the initial softmax overconfidence by shrinking `W2` and `b2` [[00:04:19]](https://youtu.be/P6sfmUTpUmc?t=4m19s).
  - [ ] Plot a histogram of the `tanh` hidden layer activations `h`. Observe the saturated/dead neurons [[00:13:14]](https://youtu.be/P6sfmUTpUmc?t=13m14s).
  - [ ] **Checkpoint:** Understand *why* a saturated `tanh` kills gradients during backprop (local derivative `1 - t^2` becomes 0).
- [ ] **Day 9: Kaiming Initialization**
  - [ ] Learn the math behind shrinking `W1` to keep pre-activations Gaussian [[00:28:10]](https://youtu.be/P6sfmUTpUmc?t=28m10s).
  - [ ] Implement Kaiming initialization (gain / sqrt(fan_in)) [[00:31:20]](https://youtu.be/P6sfmUTpUmc?t=31m20s).
  - [ ] Re-plot the histograms and marvel at the beautifully distributed activations.
- [ ] **Day 10: Batch Normalization**
  - [ ] Rip out Kaiming init. We are going to brute force it with BatchNorm [[00:40:49]](https://youtu.be/P6sfmUTpUmc?t=40m49s).
  - [ ] Compute the batch mean and batch variance of the hidden states *before* the `tanh`.
  - [ ] Normalize the hidden states to mean 0, variance 1.
  - [ ] Introduce the learned scale `gamma` and shift `beta` parameters.
- [ ] **Day 11: BatchNorm Inference (Running Stats)**
  - [ ] Understand why BatchNorm couples examples in a batch (it acts as a weird regularizer) [[00:54:01]](https://youtu.be/P6sfmUTpUmc?t=54m1s).
  - [ ] Track the running mean and running variance during training using exponential moving average (EMA) [[00:56:26]](https://youtu.be/P6sfmUTpUmc?t=56m26s).
  - [ ] Update the forward pass to use running stats during inference so we can forward a single example at a time.

---

## Week 4: The Backprop Ninja (Makemore Part 4)
**Video:** [Building makemore Part 4: Becoming a Backprop Ninja](https://youtu.be/q8SA3rM6ckI)
**Goal:** Take off the PyTorch training wheels. Calculate the analytical gradients by hand on the tensor level. 

- [ ] **Day 12: Manual Cross Entropy & Softmax**
  - [ ] Turn off `requires_grad=True` tracking. We do this live.
  - [ ] Calculate `dlogprobs` manually [[00:13:05]](https://youtu.be/q8SA3rM6ckI?t=13m5s).
  - [ ] Skip the step-by-step unrolled Softmax/CrossEntropy backward pass and derive the fast, analytical gradient for `F.cross_entropy` [[01:26:30]](https://youtu.be/q8SA3rM6ckI?t=1h26m30s). (Hint: It's beautifully simple: `probs - 1` at the true label index).
- [ ] **Day 13: Manual Linear and Tanh**
  - [ ] Master matrix calculus transpose rules. If `D = H @ W`, then `dW = H.T @ dD` [[00:41:48]](https://youtu.be/q8SA3rM6ckI?t=41m48s).
  - [ ] Write the backward pass for the linear layer (`dW2`, `db2`, `dH`).
  - [ ] Write the backward pass for `tanh` (`(1 - t**2) * out_grad`) [[00:53:39]](https://youtu.be/q8SA3rM6ckI?t=53m39s).
- [ ] **Day 14: The Final Boss (Manual BatchNorm)**
  - [ ] Trace the backward pass for `gamma` and `beta` [[00:55:16]](https://youtu.be/q8SA3rM6ckI?t=55m16s).
  - [ ] Get your pencil and paper. Watch Karpathy derive the full analytical gradient of Batch Normalization [[01:36:38]](https://youtu.be/q8SA3rM6ckI?t=1h36m38s).
  - [ ] Implement the sprawling, multi-line backward pass for `d_batchnorm`.
  - [ ] Run `torch.allclose` to compare your manual gradients against PyTorch. If it passes, you ascend to Tensor Lord.
