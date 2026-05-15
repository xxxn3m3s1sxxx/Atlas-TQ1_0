#!/usr/bin/env python3
"""Save reference model intermediate activations for calibration."""
import torch
torch._dynamo.config.suppress_errors = True
from transformers import AutoModelForCausalLM, AutoConfig
import numpy as np
import os, pickle

model_path = r'C:\models\Falcon3-7B-Instruct-1.58bit'

# Load config to understand structure
config = AutoConfig.from_pretrained(model_path, local_files_only=True)
print(f"Config: {config.num_hidden_layers} layers, {config.hidden_size} hidden")

# Load reference model
ref = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16)
ref.eval()

# Forward BOS with hooks
activations = {}
def make_hook(name):
    def hook(m, inp, out):
        if isinstance(out, tuple):
            activations[name] = inp[0].detach().float().numpy()
        else:
            activations[name] = out.detach().float().numpy()
    return hook

# Register hooks on key sublayers of each layer
for i in range(config.num_hidden_layers):
    layer = ref.model.layers[i]
    layer.mlp.gate_proj.register_forward_hook(make_hook(f'L{i}_gate_out'))
    layer.mlp.up_proj.register_forward_hook(make_hook(f'L{i}_up_out'))
    layer.mlp.down_proj.register_forward_hook(make_hook(f'L{i}_down_inp'))
    layer.self_attn.q_proj.register_forward_hook(make_hook(f'L{i}_q_out'))
    layer.self_attn.k_proj.register_forward_hook(make_hook(f'L{i}_k_out'))
    layer.self_attn.v_proj.register_forward_hook(make_hook(f'L{i}_v_out'))
    layer.self_attn.o_proj.register_forward_hook(make_hook(f'L{i}_o_inp'))
    layer.input_layernorm.register_forward_hook(make_hook(f'L{i}_norm_in'))
    layer.post_attention_layernorm.register_forward_hook(make_hook(f'L{i}_post_norm_in'))

# Also hook into embed
ref.model.embed_tokens.register_forward_hook(make_hook('embed_out'))

# Also hook into final norm + lm head
ref.model.norm.register_forward_hook(make_hook('final_norm_in'))
# Hook before and after lm_head
def lm_head_hook(m, inp, out):
    activations['lm_head_in'] = inp[0].detach().float().numpy()
    activations['lm_head_out'] = out.detach().float().numpy()
ref.lm_head.register_forward_hook(lm_head_hook)

# Run forward
print("Running reference forward...")
with torch.no_grad():
    out = ref(torch.tensor([[1]]), use_cache=False, output_hidden_states=True)

logits = out.logits[0, -1].float().numpy()
top5 = np.argsort(-logits)[:5]
print(f"Ref top-5: {top5}, predicted: {int(np.argmax(logits))}")
print(f"Logits RMS: {np.sqrt(np.mean(logits**2)):.4f}")

# Save all activations
del ref  # free memory
save_path = r'C:\atlas\ref_activations.pkl'
with open(save_path, 'wb') as f:
    pickle.dump({
        'activations': activations,
        'hidden_states': [h.float().numpy() for h in out.hidden_states],
        'logits': logits,
        'top5': top5,
    }, f)
print(f"\nSaved {len(activations)} activations to {save_path}")
print(f"Size: {os.path.getsize(save_path) / 1e6:.1f} MB")
