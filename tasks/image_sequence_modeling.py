# train_step():
# 1. Sample between [0, N-1] observed patches, where N is total number of patches
# 2. Sample 1 unmasking location among the unobserved patches
# 3. Linearly project the observed patches to the model's hidden size
# 4. Add the positional embeddings to the observed patches
# 5. Add the "unmask" embedding to the unmasking location's positional embedding
# 6. Pass to the model the set of encoded patches
# 7. Readout the final representation of the unmasking location
# 8. Linearly project unmasking location's representation to the patch size and reshape to the target patch shape
# 9. Compute the loss between the readout at the unmasking location and the target patch
