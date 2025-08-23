from compressor import PromptCompressor

# Initialize the compressor (supports Hugging Face models)
model_name = "Qwen/Qwen2-0.5B-Instruct"
compressor = PromptCompressor(model_name)

# Long input context (e.g., retrieved documents, conversation history)
context = """
Artificial intelligence is a branch of computer science aimed at creating systems capable of performing tasks that typically require human intelligence...
"""

# Perform compression
result = compressor.compress(
    context=context,
    compress_ratio=0.9,                     # Keep only 10% of tokens (10x compression)
    method="dynamic_attn_ppl",              # Compression method
    fusion="additive",                      # Fusion strategy
    alpha=0.8,                              # Attention weight in additive fusion
    dyn_time=10,                            # Number of dynamic iterations
    preserve_punct=False,                   # Preserve punctuation and special tokens or not
    return_info=True                        # Return detailed info
)

# Output results
print("Compressed text:", result["compressed_text"])
print("Original tokens:", result["original_tokens"])
print("Compressed tokens:", result["compressed_tokens"])
print("Actual compression ratio:", result["actual_ratio"])