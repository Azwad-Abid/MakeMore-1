# Makemore Part 1: Building a Name Generator

So I built a neural network that learns to generate English names. Sounds fancy, but it's actually just learning which characters tend to follow other characters.

## The Setup

I took 32k English names and trained two models on them:

**The Simple Baseline:** Just counting which letters follow which letters. Surprisingly, this works okay. NLL: 2.454

**The Neural Network:** Embedding layer + linear layer = learns actual patterns instead of just counting. It's slower but smarter. Final NLL: 2.558 (close!)

## What Actually Happens

1. Load names
2. Count bigrams (pairs of characters)
3. Convert counts to probabilities
4. Build a tiny neural net (27 vocab → 10-dim embedding → 27 output)
5. Train for 1000 epochs watching loss go down
6. Sample names by picking characters based on what the model learned

## Generated Names (after training)


fabiqu
dxfmubn
uguonvtpe
msilaynglaojmahwla

They're not perfect, but you can kinda see it learning English-like phonetics.

## What I Actually Learned

- Forward pass, loss, backward pass, update — this is literally how all neural nets train
- Embeddings aren't magic, they just learn to cluster similar things
- Softmax turns unbounded numbers into valid probabilities
- 100 epochs isn't enough, 1000 is better
- A 2D embedding can't hold much — bigger is better
- Negative log likelihood is the real measure of model quality
- NN outputs are called logits
- One Hot encodings
- LL is always negative
- Better model has lower NLL meaning lower loss
- One Hot encodings
- Overfitting
  
## Code

Everything's in the  notebook. It's messy but it works.
