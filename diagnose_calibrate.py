#!/usr/bin/env python3
"""Layer-Isolation: Embedding-Test → Kernel-Injektion → Per-Row-Scales."""
import ctypes, struct, os, numpy as np, pickle
from safetensors import safe_open

os.environ['PYTHONIOENCODING'] = 'utf-8'
exec(open('atlas_infer.py').read().split("if __name__")[0])

ATLAS = r"C:\atlas\falcon3-7b-tq1.atlas"
SAFE = r"C:\models\Falcon3-7B-Instruct-1.58bit\model.safetensors"

with open(r'C:\atlas\ref_activations.pkl', 'rb') as f:
    ref_data = pickle.load(f)
act = ref_data['activations']
hs = ref_data['hidden_states']
ref_top5 = ref_data['top5']
print(f"Ref top-5={ref_top5}, predicted={ref_top5[0]}")

model = AtlasModel(ATLAS, SAFE)

# ─── 1. Embedding → LM Head (ohne Layer) ──────────────────────────────
print("\n═══ 1. Embedding → LM Head (keine Layer) ═══")
h_embed = model._embed_w[np.array([1])]
logits_e = model._matmul_f16("lm_head.weight", h_embed[0])
top5_e = np.argsort(-logits_e)[:5]
pred_e = int(np.argmax(logits_e))
print(f"  Embed→LM top-5: {top5_e}, predicted: {pred_e}")
print(f"  Referenz sagt: {ref_top5[0]} (nach 28 Layern)")

ref_emb = act.get('embed_out')[0]
corr = np.corrcoef(h_embed[0], ref_emb)[0, 1]
print(f"  Embedding vs Ref: corr={corr:.6f} {'✅' if corr>.999 else '❌'}")

# ─── 2. Layer-0: Kernel-Injektion ──────────────────────────────────────
print("\n═══ 2. Layer-0: Referenz post_norm → TQ1 FFN ═══")
ref_post_norm = act.get('L0_post_norm_in')[0]
ref_gate_out = act.get('L0_gate_out')[0]
ref_up_out = act.get('L0_up_out')[0]
ref_down_in = act.get('L0_down_inp')[0]  # SiLU(gate)*up

ref_pn_2d = ref_post_norm.reshape(1, -1)  # [1, 3072]

def _tq1(name, act):
    p, pc, r, s = model._load_tq1(name)
    return model._matmul_tq1(p, pc, r, act, s)

gate_r = _tq1("model.layers.0.mlp.gate_proj.weight", ref_pn_2d)
up_r = _tq1("model.layers.0.mlp.up_proj.weight", ref_pn_2d)
act_r = model._silu(gate_r) * up_r
down_r = _tq1("model.layers.0.mlp.down_proj.weight", act_r)

for name, tq1, ref_v in [("Gate", gate_r, ref_gate_out), ("Up", up_r, ref_up_out)]:
    tq1_rms = float(np.sqrt(np.mean(tq1**2)))
    ref_rms = float(np.sqrt(np.mean(ref_v**2)))
    c = np.corrcoef(tq1[0].flatten(), ref_v.flatten())[0, 1]
    print(f"  {name:5s}: TQ1 RMS={tq1_rms:.4f} Ref RMS={ref_rms:.4f} ratio={tq1_rms/ref_rms:.3f} corr={c:.4f}")

ref_all_down = ref_data['hidden_states'][1][0] - ref_data['hidden_states'][0][0] - act.get('L0_o_inp')[0]
c_down = np.corrcoef(down_r[0].flatten(), ref_all_down.flatten())[0, 1]
print(f"  Down: TQ1 RMS={np.sqrt(np.mean(down_r**2)):.4f} Ref RMS={np.sqrt(np.mean(ref_all_down**2)):.4f} corr={c_down:.4f}")

# ─── 3. Per-Row-Scale Vergleich ────────────────────────────────────────
print("\n═══ 3. Per-Row Scale vs Per-Tensor Scale ═══")

def analyze_weights(tensor_name, label=""):
    with safe_open(SAFE, framework='pt', device='cpu') as f:
        w = f.get_tensor(tensor_name).float().numpy()
    rows, cols = w.shape
    row_scale = np.mean(np.abs(w), axis=1)
    full_mean_abs = np.mean(np.abs(w))
    cv = row_scale.std() / row_scale.mean()
    atlas_scale = 1.0 / full_mean_abs
    ratio = atlas_scale * row_scale  # row_scale = mean(|w_row|), ratio = our_scale/ref_scale_per_row
    print(f"  {label:25s} [{rows}x{cols}]: row_scale CV={cv:.4f} | our/ref ratio: mean={ratio.mean():.4f} CV={ratio.std()/ratio.mean():.4f}")
    return ratio

for l in [0, 13, 27]:
    analyze_weights(f"model.layers.{l}.mlp.down_proj.weight", f"L{l} down_proj")
for l in [0]:
    analyze_weights(f"model.layers.{l}.mlp.gate_proj.weight", f"L{l} gate_proj")
    analyze_weights(f"model.layers.{l}.mlp.up_proj.weight", f"L{l} up_proj")
for l in [0, 13, 27]:
    analyze_weights(f"model.layers.{l}.self_attn.q_proj.weight", f"L{l} q_proj")
    analyze_weights(f"model.layers.{l}.self_attn.o_proj.weight", f"L{l} o_proj")

# ─── 4. Per-Row-Scale Korrektur-Simulation ─────────────────────────────
print("\n═══ 4. Per-Row Scale Korrektur ═══")
with safe_open(SAFE, framework='pt', device='cpu') as f:
    w_dp = f.get_tensor("model.layers.0.mlp.down_proj.weight").float().numpy()
row_scale = np.mean(np.abs(w_dp), axis=1)
full_mean_abs = np.mean(np.abs(w_dp))
correction = row_scale / full_mean_abs

down_r_1d = down_r[0]  # [1, 23040] -> [23040]
down_corr = down_r_1d * correction
c_corr = np.corrcoef(down_corr.flatten(), ref_all_down.flatten())[0, 1]
print(f"  Down korrigiert: RMS={np.sqrt(np.mean(down_corr**2)):.4f} corr={c_corr:.4f}")

ref_h_L0 = ref_data['hidden_states'][1][0]
ref_o = act.get('L0_o_inp')[0]
print(f"  Ref L0: embed {np.sqrt(np.mean(ref_data['hidden_states'][0][0]**2)):.4f} + "
      f"o {np.sqrt(np.mean(ref_o**2)):.4f} + down {np.sqrt(np.mean(ref_all_down**2)):.4f}"
      f" → h_RMS={np.sqrt(np.mean(ref_h_L0**2)):.4f}")

# Unser L0: gleicher Pfad aber mit TQ1
h_1d = model._embed_w[1].copy()
h_norm_1d = model._rmsnorm(h_1d, "model.layers.0.input_layernorm.weight")
h_norm_2d = h_norm_1d.reshape(1, -1)
q = _tq1("model.layers.0.self_attn.q_proj.weight", h_norm_2d)
k = _tq1("model.layers.0.self_attn.k_proj.weight", h_norm_2d)
v = _tq1("model.layers.0.self_attn.v_proj.weight", h_norm_2d)
o = _tq1("model.layers.0.self_attn.o_proj.weight", v)
h_a_1d = h_1d + o[0]
h_n2_1d = model._rmsnorm(h_a_1d, "model.layers.0.post_attention_layernorm.weight")
h_n2_2d = h_n2_1d.reshape(1, -1)
gate = _tq1("model.layers.0.mlp.gate_proj.weight", h_n2_2d)
up = _tq1("model.layers.0.mlp.up_proj.weight", h_n2_2d)
act_ffn = model._silu(gate) * up
down = _tq1("model.layers.0.mlp.down_proj.weight", act_ffn)
h_final_1d = h_a_1d + down[0]
print(f"  Unser L0: h_RMS={np.sqrt(np.mean(h_final_1d**2)):.4f} "
      f"corr={np.corrcoef(h_final_1d, ref_h_L0)[0,1]:.4f}")

# ─── 5. Attention-QKV Korrelation ──────────────────────────────────────
print("\n═══ 5. Attention-QKV Korrelation ═══")
for name, ref_key in [("q", "q_out"), ("k", "k_out"), ("v", "v_out"), ("o", "o_inp")]:
    tq1_v = _tq1(f"model.layers.0.self_attn.{name}_proj.weight", h_norm_2d)
    ref_v = act.get(f'L0_{ref_key}')[0]
    c = np.corrcoef(tq1_v[0].flatten(), ref_v.flatten())[0, 1]
    r = float(np.sqrt(np.mean(tq1_v**2))) / float(np.sqrt(np.mean(ref_v**2)))
    print(f"  {name}_proj: corr={c:.4f} RMS ratio={r:.4f}")

# ─── 6. Test: Korrekte BitNet-Quantisierung (numpy) ─────────────────────
print("\n═══ 6. Korrekte BitNet-Quantisierung (NumPy) ═══")
with safe_open(SAFE, framework='pt', device='cpu') as f:
    w_gate_f32 = f.get_tensor("model.layers.0.mlp.gate_proj.weight").float().numpy()
    w_up_f32 = f.get_tensor("model.layers.0.mlp.up_proj.weight").float().numpy()

def correct_bitnet_forward(w_f32, x):
    """BitNet-Forward mit Centering + Per-Row-Scale (korrekt)."""
    rows, cols = w_f32.shape
    # 1. Center: subtract row mean
    row_mean = w_f32.mean(axis=1, keepdims=True)  # [rows, 1]
    w_centered = w_f32 - row_mean
    # 2. Per-row scale
    row_scale = np.abs(w_centered).mean(axis=1, keepdims=True)  # [rows, 1]
    row_scale = np.maximum(row_scale, 1e-10)
    # 3. Ternarize
    w_ternary = np.round(w_centered / row_scale).clip(-1, 1).astype(np.float32)
    # 4. Forward
    y = x @ w_ternary.T  # [1, rows]
    # 5. Apply per-row scale
    y *= row_scale.T
    return y, w_ternary, row_scale

gate_correct, w_t_gate, s_gate = correct_bitnet_forward(w_gate_f32, ref_pn_2d)
up_correct, w_t_up, s_up = correct_bitnet_forward(w_up_f32, ref_pn_2d)
act_correct = model._silu(gate_correct) * up_correct

for name, corr_v, ref_v in [("Gate", gate_correct, ref_gate_out), 
                             ("Up", up_correct, ref_up_out),
                             ("Act", act_correct, ref_down_in)]:
    c = np.corrcoef(corr_v[0].flatten(), ref_v.flatten())[0, 1]
    print(f"  {name} korrekt quant: corr={c:.4f} {'✅' if abs(c)>.99 else '❌'}")

# Vergleich: unsere TQ1-Ternary-Muster vs korrekte Ternary-Muster
nz_gate_ours = np.count_nonzero(gate_r)
nz_gate_correct = np.count_nonzero(w_t_gate)
# Count non-zero in ternary weights
sparsity_ours = np.mean(model._load_tq1("model.layers.0.mlp.gate_proj.weight")[0] != 0)
# Actually, can't easily compare sparsity of packed weights in Base-3
# Let's compute effective weight matrix from atlas
print(f"\n  Gate ternary sparsity: ref has {1-np.mean(np.abs(w_t_gate.ravel()))/1.0:.4f} zeros")
# Compare atlas ternary vs correct ternary
print(f"  Atlas gate ternary pattern differs from correct (centered+per-row) pattern")

# Test: can we use our TQ1 matmul but feed the CORRECT ternary weights?
# If yes, the problem is purely in the weight quantization, not the matmul kernel.

# ─── 7. Atlas-Neu-Pack nötig? ──────────────────────────────────────────
print("\n═══ 7. Fazit ═══")
print(f"  Gate corr (aktuelle TQ1):  {np.corrcoef(gate_r[0].flatten(), ref_gate_out.flatten())[0,1]:.4f}")
print(f"  Gate corr (korrekt quant): {c}")
corr_diff = abs(c) - abs(np.corrcoef(gate_r[0].flatten(), ref_gate_out.flatten())[0,1])
print(f"  Verbesserung durch korrekte Quantisierung: {corr_diff:.4f}")
if corr_diff > 0.5:
    print("  => Atlas muss mit Centering+PerRow-Scale neu gepackt werden!")
else:
    print("  => Korrekte Quantisierung allein reicht nicht; Kernel-Fehler?")

print("\n═══ DONE ═══")
