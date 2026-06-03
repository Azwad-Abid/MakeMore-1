"""
Makemore Part 1: Making a name generator

The idea: download 32k names, teach a computer to make up new ones that sound similar.
bruh ;-;
"""

import torch
import torch.nn as nn


# ==============================================================================
# PART 1: LOAD THE NAMES
# ==============================================================================

print("Loading 32k English names...")
words = open('names.txt', 'r').read().splitlines()
print(f"Got {len(words)} names. Nice.\n")


# ==============================================================================
# PART 2: SETUP - BUILD LOOKUP TABLES
# ==============================================================================

# We need to convert letters to numbers (computers only understand numbers).
# So we create two lookup tables:
#   stoi: "string to integer" — convert 'a' to 0, 'b' to 1, etc.
#   itos: "integer to string" — convert 0 back to 'a', 1 back to 'b', etc.

print("Building character lookup tables...")
chars = ['.'] + sorted(list(set(''.join(words))))
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}

print(f"Vocab size: {len(chars)} characters")
print(f"Examples: {list(stoi.items())[:5]}\n")


# ==============================================================================
# PART 3: BUILD THE BASELINE (JUST COUNTING)
# ==============================================================================

print("Building baseline model (just counting character pairs)...")

# Create a 27x27 matrix to count bigrams.
# Each cell [i, j] means: "how many times does character j follow character i?"
N = torch.zeros((27, 27), dtype=torch.int32)

# Go through every name and count every character pair
for w in words:
    # Add dots at the start and end so we know when words start/end
    chs = ['.'] + list(w) + ['.']
    
    # Look at each pair of consecutive characters
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]  # convert first char to number
        ix2 = stoi[ch2]  # convert second char to number
        N[ix1, ix2] += 1  # increment the count

print(f"Counted {N.sum()} total character pairs\n")


# ==============================================================================
# PART 4: TURN COUNTS INTO PROBABILITIES
# ==============================================================================

print("Converting counts to probabilities...")

# Take the counts and divide by row totals so each row sums to 1.0
# This way we know: "what's the probability that 'x' follows 'a'?"
P_baseline = (N + 1).float()  # add 1 to avoid division by zero (smoothing)
P_baseline = P_baseline / P_baseline.sum(dim=1, keepdim=True)

# Now P_baseline[i, j] = probability that character j follows character i
print("Done. Now we can sample from the baseline.\n")


# ==============================================================================
# PART 5: EVALUATE THE BASELINE
# ==============================================================================

print("How good is the baseline? Let's measure it...")

# Calculate the loss (negative log likelihood)
# For every character pair in the training data, we take -log(probability)
# If the model is confident, loss is low. If uncertain, loss is high.
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P_baseline[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

baseline_nll = -log_likelihood / n
print(f"Baseline model NLL: {baseline_nll:.4f}")
print("(Lower is better. We'll try to beat this with a neural network.)\n")


# ==============================================================================
# PART 6: SAMPLE FROM THE BASELINE
# ==============================================================================

print("Sample names from the baseline (just counting):")
g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0  # start with '.'
    while True:
        # Look up the probabilities for what comes next
        p = P_baseline[ix]
        # Pick a random character based on those probabilities
        ix = torch.multinomial(p, num_samples=1, generator=g).item()
        out.append(itos[ix])
        if ix == 0:  # stop if we hit '.' again
            break
    print(f"  {''.join(out)}")
print()


# ==============================================================================
# PART 7: BUILD THE NEURAL NETWORK
# ==============================================================================

print("Now let's build an actual neural network...")

class BigramNeuralNet(nn.Module):
    """
    A tiny neural network that learns to predict the next character.
    
    It has:
    - An embedding layer (learns a 10-dimensional vector for each character)
    - A linear layer (learns to predict the next character from that vector)
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # Embedding: converts character indices into dense vectors
        # This is where the network LEARNS what each character means
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Linear layer: takes the embedding and outputs logits
        # Logits are raw scores for each possible next character
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x):
        # x is a tensor of character indices (like [0, 5, 12, ...])
        embedded = self.embedding(x)  # convert to 10-dim vectors
        logits = self.linear(embedded)  # predict next character
        return logits


vocab_size = 27
embedding_dim = 10  # how many dimensions for each character vector

model = BigramNeuralNet(vocab_size, embedding_dim)

print(f"Neural network created!")
print(f"  Input: character index (0-26)")
print(f"  Embedding: 27 chars → {embedding_dim}-dim vectors")
print(f"  Linear layer: {embedding_dim} → 27 (one score per possible char)")
print()


# ==============================================================================
# PART 8: PREPARE TRAINING DATA
# ==============================================================================

print("Preparing training data (228k character pairs)...")

# Convert all bigrams to tensors
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)  # input: previous character
        ys.append(ix2)  # output: next character

xs = torch.tensor(xs)
ys = torch.tensor(ys)

print(f"Total training examples: {len(xs)}")
print()


# ==============================================================================
# PART 9: TRAINING LOOP
# ==============================================================================

print("Training the neural network (this might take a few seconds)...")
print("(We're doing 1000 epochs, printing every 100)\n")

# Setup optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# CrossEntropyLoss does: softmax → log → mean
# It's the standard loss for classification tasks
loss_fn = nn.CrossEntropyLoss()

num_epochs = 1000
for epoch in range(num_epochs):
    # FORWARD PASS: feed all 228k examples through the network
    logits = model(xs)  # get raw predictions
    
    # COMPUTE LOSS: how wrong are we?
    loss = loss_fn(logits, ys)
    
    # BACKWARD PASS: compute gradients using backpropagation
    optimizer.zero_grad()  # clear old gradients (important!)
    loss.backward()  # compute new gradients via chain rule
    
    # UPDATE: nudge weights in the direction of lower loss
    optimizer.step()
    
    # Print progress
    if epoch % 100 == 0:
        print(f"  Epoch {epoch:4d} | Loss: {loss.item():.4f}")

print()

# Check final performance
with torch.no_grad():
    logits = model(xs)
    final_loss = loss_fn(logits, ys)

print(f"Final NLL: {final_loss.item():.4f}")
print(f"Baseline NLL: {baseline_nll:.4f}")
if final_loss < baseline_nll:
    print(f"✓ We beat the baseline!\n")
else:
    print(f"Close though! (And embeddings can be improved.)\n")


# ==============================================================================
# PART 10: GENERATE NAMES FROM THE TRAINED MODEL
# ==============================================================================

print("Generating names from the trained neural network:\n")

g = torch.Generator().manual_seed(42)
for i in range(10):
    out = []
    ix = 0  # start with '.'
    
    while True:
        # Wrap the character index in a tensor (the model expects a tensor)
        xenc = torch.tensor([ix])
        
        # Get predictions from the model
        logits = model(xenc)  # raw scores
        
        # Convert logits to probabilities using softmax
        probs = torch.softmax(logits, dim=1)
        
        # Sample a character randomly based on those probabilities
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        
        out.append(itos[ix])
        
        # Stop if we hit the end-of-word marker
        if ix == 0:
            break
    
    print(f"  {''.join(out)}")

print("\n✓ All done!")
print("\nWhat you just did:")
print("  1. Loaded 32k names")
print("  2. Built a baseline by counting character pairs")
print("  3. Created a neural network with embeddings")
print("  4. Trained it on 228k examples")
print("  5. Generated new names from it")
print("\nThat's the whole pipeline. Next: deeper networks (part 2)!")
