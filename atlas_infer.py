#!/usr/bin/env python3
"""Atlas Inference Engine — End-to-end Falcon3-7B TQ1.0 generation."""
import ctypes, struct, os, time, sys, numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

# ─── Load C++ DLL ────────────────────────────────────────────────────────
# Resolve OpenMP runtime conflicts (numpy MKL vs libomp)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas.dll")
if not os.path.exists(_dll_path):
    _dll_path = os.environ.get("ATLAS_DLL", _dll_path)
dll = ctypes.CDLL(_dll_path)

dll.atlas_load.restype = ctypes.c_void_p
dll.atlas_load.argtypes = [ctypes.c_char_p]

dll.atlas_free.argtypes = [ctypes.c_void_p]

dll.atlas_get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]

dll.atlas_tensor_info.argtypes = [ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int)]

dll.atlas_tensor_data.restype = ctypes.POINTER(ctypes.c_uint8)
dll.atlas_tensor_data.argtypes = [ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int)]

dll.atlas_tensor_matmul.argtypes = [ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int]

dll.atlas_matmul_f32.restype = None
dll.atlas_matmul_f32.argtypes = [ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float]

dll.atlas_rmsnorm_f32.restype = None
dll.atlas_rmsnorm_f32.argtypes = [ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_float]

dll.atlas_rope_f32.restype = None
dll.atlas_rope_f32.argtypes = [ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_float]

dll.atlas_attention_f32.restype = None
dll.atlas_attention_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # q
    ctypes.POINTER(ctypes.c_float),   # k
    ctypes.POINTER(ctypes.c_float),   # v
    ctypes.POINTER(ctypes.c_int),     # positions
    ctypes.POINTER(ctypes.c_uint16),  # k_cache
    ctypes.POINTER(ctypes.c_uint16),  # v_cache
    ctypes.c_int,     # max_seq_len
    ctypes.c_int,     # seq_now
    ctypes.c_int,     # B
    ctypes.c_int,     # n_heads
    ctypes.c_int,     # n_kv_heads
    ctypes.c_int,     # head_dim
    ctypes.c_float,   # rope_theta
    ctypes.POINTER(ctypes.c_float),   # output
]

# ─── Model class ─────────────────────────────────────────────────────────
class AtlasModel:
    def __init__(self, atlas_path, safetensors_path):
        self._safe_path = safetensors_path
        self._atlas_path = atlas_path
        self.model_ptr = dll.atlas_load(atlas_path.encode())
        if not self.model_ptr:
            raise RuntimeError("Failed to load model")

        # Get model info
        nl = ctypes.c_int(); hd = ctypes.c_int(); id_ = ctypes.c_int()
        nh = ctypes.c_int(); nk = ctypes.c_int(); hdm = ctypes.c_int()
        vs = ctypes.c_int()
        dll.atlas_get_info(self.model_ptr, nl, hd, id_, nh, nk, hdm, vs)
        self.n_layers, self.hidden, self.inter = nl.value, hd.value, id_.value
        self.n_heads, self.n_kv_heads, self.head_dim = nh.value, nk.value, hdm.value
        self.vocab_size = vs.value
        self.rope_theta = self._get_rope_theta()

        # Build tensor name→index map from safetensors
        with safe_open(safetensors_path, framework='pt', device='cpu') as f:
            self.tensor_names = list(f.keys())
        self.n_tensors = len(self.tensor_names)

        print(f"[Atlas] {self.n_layers}L {self.hidden}H {self.inter}I "
              f"{self.n_heads}/{self.n_kv_heads} heads | "
              f"{self.n_tensors} tensors")

        # Cache tensor indices for fast lookup
        self._cache_indices()

        # Precompute KV cache
        self.max_seq_len = 4096  # conservative for 16GB
        kvc_size = (self.n_layers, self.n_kv_heads, self.max_seq_len, self.head_dim)
        self.k_cache = np.zeros(kvc_size, dtype=np.float16)
        self.v_cache = np.zeros(kvc_size, dtype=np.float16)

        # Warm LM head cache
        self._matmul_f16("lm_head.weight", np.random.randn(1, self.hidden).astype(np.float32))

    def _cache_indices(self):
        self.idx = {}
        for i, name in enumerate(self.tensor_names):
            self.idx[name] = i
        # Cache frequently-used tensors
        self._embed_w = self._load_weight_f16(
            "model.embed_tokens.weight",
            shape=(self.vocab_size, self.hidden))
        self._norm_w = self._load_weight_f16("model.norm.weight")
        self._tq1_cache = {}
        self._f16_cache = {}

    def _get_rope_theta(self):
        with open(self._atlas_path, 'rb') as f:
            f.read(21)
            return struct.unpack('<d', f.read(8))[0]

    def _load_weight_f16(self, name, shape=None):
        """Load a float16 weight tensor from atlas, return as float32 numpy."""
        idx = self.idx.get(name)
        if idx is None: return None
        sz = ctypes.c_int()
        ptr = dll.atlas_tensor_data(self.model_ptr, idx, sz)
        if not ptr: return None
        arr = np.ctypeslib.as_array(ptr, shape=(sz.value,)).view(np.float16)
        if shape:
            arr = arr.reshape(shape)
        return arr.copy().astype(np.float32)

    def _load_tq1(self, name):
        """Load TQ1 packed data and scale for a weight tensor. Cached."""
        if name in self._tq1_cache:
            return self._tq1_cache[name]
        idx = self.idx.get(name)
        if idx is None: return None, None, None
        sz = ctypes.c_int()
        ptr = dll.atlas_tensor_data(self.model_ptr, idx, sz)
        if not ptr: return None, None, None
        raw = np.ctypeslib.as_array(ptr, shape=(sz.value,))
        scale_raw = struct.unpack('<H', raw[:2].tobytes())[0]
        scale = np.frombuffer(struct.pack('<H', scale_raw), dtype=np.float16)[0].item()
        tt = ctypes.c_int(); rd = ctypes.c_int(); cd = ctypes.c_int()
        dll.atlas_tensor_info(self.model_ptr, idx, tt, rd, cd)
        packed_cols = cd.value // 5
        packed = np.frombuffer(raw[2:].tobytes(), dtype=np.uint8).copy()
        result = (packed, packed_cols, rd.value, scale)
        self._tq1_cache[name] = result
        return result

    def _matmul_tq1(self, data_flat, packed_cols, rows, act, scale):
        """Run TQ1 matmul via C++ DLL with BitNet activation quantization."""
        orig_shape = act.shape
        act = act.reshape(-1, orig_shape[-1])
        max_abs = np.max(np.abs(act), axis=-1, keepdims=True)
        max_abs = np.maximum(max_abs, 1e-5)
        act_scale = 127.0 / max_abs
        act_q = np.round(act * act_scale).astype(np.float32)

        need = packed_cols * 5
        if act_q.shape[1] < need:
            pad = np.zeros((act_q.shape[0], need), dtype=np.float32)
            pad[:, :act_q.shape[1]] = act_q
            act_q = pad

        out = np.zeros((act_q.shape[0], rows), dtype=np.float32)
        dll.atlas_matmul_f32(
            rows, packed_cols,
            data_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            act_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            act_q.shape[0], ctypes.c_float(1.0))

        # Reorder: atlas order (ur*4+q) → HF unpack order (q*rows_packed+ur)
        rows_packed = rows // 4
        out = out.reshape(out.shape[0], rows_packed, 4).transpose(0, 2, 1).reshape(out.shape[0], rows)

        out *= max_abs / (127.0 * scale)
        out = out.reshape(*orig_shape[:-1], rows)
        return out

    def _matmul_f16(self, name, act):
        """Float32 activation × cached float32 weight (for non-TQ1 tensors like lm_head)."""
        if name not in self._f16_cache:
            idx = self.idx.get(name)
            if idx is None: return None
            tt = ctypes.c_int(); rd = ctypes.c_int(); cd = ctypes.c_int()
            dll.atlas_tensor_info(self.model_ptr, idx, tt, rd, cd)
            if tt.value == 0:
                return self._matmul_tq1(*self._load_tq1(name), act)
            sz = ctypes.c_int()
            ptr = dll.atlas_tensor_data(self.model_ptr, idx, sz)
            w_f16 = np.ctypeslib.as_array(ptr, shape=(sz.value,)).view(np.float16)
            self._f16_cache[name] = w_f16.reshape(rd.value, act.shape[-1]).astype(np.float32)
        w = self._f16_cache[name]
        if act.ndim == 1:
            return act @ w.T
        B = act.shape[0]
        flat = act.reshape(-1, act.shape[-1])
        result = flat @ w.T
        return result.reshape(B, -1, w.shape[0]) if act.ndim > 2 else result

    def _rmsnorm(self, x, weight_name, eps=1e-6):
        if weight_name not in self._f16_cache:
            idx = self.idx.get(weight_name)
            if idx is None: return x
            sz = ctypes.c_int()
            ptr = dll.atlas_tensor_data(self.model_ptr, idx, sz)
            if not ptr: return x
            raw = np.ctypeslib.as_array(ptr, shape=(sz.value,))
            self._f16_cache[weight_name] = raw.tobytes()
        out = np.zeros_like(x)
        dll.atlas_rmsnorm_f32(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(ctypes.create_string_buffer(self._f16_cache[weight_name]), ctypes.POINTER(ctypes.c_uint8)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(x), ctypes.c_float(eps))
        return out

    def _apply_rope(self, q, k, position):
        """Apply RoPE to q and k in-place."""
        dll.atlas_rope_f32(
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.n_heads, self.n_kv_heads, self.head_dim,
            position, ctypes.c_float(self.rope_theta))

    def _silu(self, x):
        return x * (1.0 / (1.0 + np.exp(-x)))

    def forward_layer(self, x, layer_idx, positions, use_kvcache=True):
        """Forward one transformer layer for batch of positions."""
        B = len(positions)
        prefix = f"model.layers.{layer_idx}"

        # Pre-attention RMSNorm
        x_norm = np.zeros((B, self.hidden), dtype=np.float32)
        for b in range(B):
            x_norm[b] = self._rmsnorm(x[b], f"{prefix}.input_layernorm.weight")

        # QKV projections
        def _mt(name, act):
            p, pc, r, s = self._load_tq1(name)
            return self._matmul_tq1(p, pc, r, act, s)
        q = _mt(f"{prefix}.self_attn.q_proj.weight", x_norm)
        k = _mt(f"{prefix}.self_attn.k_proj.weight", x_norm)
        v = _mt(f"{prefix}.self_attn.v_proj.weight", x_norm)

        # Fused RoPE + GQA attention + causal mask + softmax + weighted sum
        attn_out = np.zeros((B, self.hidden), dtype=np.float32)
        seq_now = max(positions) + 1 if use_kvcache else B
        # k/v_cache: [n_kv_heads, max_seq, head_dim] — pass raw uint16 pointers
        kp = self.k_cache[layer_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        vp = self.v_cache[layer_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        dll.atlas_attention_f32(
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            (ctypes.c_int * B)(*positions),
            kp, vp,
            self.max_seq_len, seq_now, B,
            self.n_heads, self.n_kv_heads, self.head_dim,
            ctypes.c_float(self.rope_theta),
            attn_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        # Output projection
        p, pc, r, s = self._load_tq1(f"{prefix}.self_attn.o_proj.weight")
        attn_out = self._matmul_tq1(p, pc, r, attn_out, s)
        x = x + attn_out

        # Post-attention RMSNorm
        x_norm2 = np.zeros((B, self.hidden), dtype=np.float32)
        for b in range(B):
            x_norm2[b] = self._rmsnorm(x[b], f"{prefix}.post_attention_layernorm.weight")

        # FFN: SiLU(gate) * up → down
        def _mt2(name, act):
            p, pc, r, s = self._load_tq1(name)
            return self._matmul_tq1(p, pc, r, act, s)
        gate = _mt2(f"{prefix}.mlp.gate_proj.weight", x_norm2)
        up = _mt2(f"{prefix}.mlp.up_proj.weight", x_norm2)
        hidden = self._silu(gate) * up
        p, pc, r, s = self._load_tq1(f"{prefix}.mlp.down_proj.weight")
        down = self._matmul_tq1(p, pc, r, hidden, s)
        x = x + down

        return x

    def forward(self, tokens, start_pos=0):
        """Full forward pass for a batch of tokens (prefill)."""
        B, seq_len = tokens.shape
        h = self._embed_w[tokens]

        for layer in range(self.n_layers):
            t0 = time.time()
            positions = [start_pos + p for b in range(B) for p in range(seq_len)]
            h = self.forward_layer(h.reshape(-1, self.hidden), layer, positions)
            h = h.reshape(B, seq_len, self.hidden)

        h_flat = h.reshape(-1, self.hidden)
        h_norm = np.array([self._rmsnorm(h_flat[b], "model.norm.weight") for b in range(B * seq_len)])
        output_logits = self._matmul_f16("lm_head.weight", h_norm).reshape(B, seq_len, self.vocab_size)
        return output_logits

    def generate(self, prompt, max_new_tokens=50, temperature=0.0):
        try:
            tok = AutoTokenizer.from_pretrained(
                os.path.dirname(self._safe_path),
                local_files_only=True)
        except:
            return "[TOKENIZER ERROR]"

        eos_id = tok.eos_token_id
        stop_tokens = ['<|user|>', '<|system|>', '<|end|>']

        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_ids = tok.encode(text, return_tensors='np')[0]

        full_logits = self.forward(input_ids[None])
        logits = full_logits[0, -1, :]
        next_token = int(np.argmax(logits)) if temperature == 0.0 else int(np.random.choice(len(logits), p=np.exp(logits/temperature)/np.exp(logits/temperature).sum()))

        output = [next_token]
        for step in range(max_new_tokens - 1):
            if eos_id is not None and next_token == eos_id:
                break
            decoded = tok.decode(output, skip_special_tokens=False)
            if any(stop in decoded for stop in stop_tokens):
                break
            h = self._get_embedding(next_token)
            for layer in range(self.n_layers):
                h = self.forward_layer(h, layer, [len(input_ids) + step])
            h_norm = self._rmsnorm(h.flatten(), "model.norm.weight")
            logits = self._matmul_f16("lm_head.weight", h_norm.reshape(1, -1)).flatten()
            next_token = int(np.argmax(logits)) if temperature == 0.0 else int(np.random.choice(len(logits), p=np.exp(logits/temperature)/np.exp(logits/temperature).sum()))
            output.append(next_token)

        text = tok.decode(output, skip_special_tokens=True)
        for stop in stop_tokens:
            idx = text.find(stop)
            if idx >= 0:
                text = text[:idx]
        return text

    def _get_embedding(self, token_id):
        return self._embed_w[token_id].reshape(1, self.hidden)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python atlas_infer.py <atlas.tq1> <model_dir> [prompt]")
        print("  atlas.tq1  — path to packed .atlas file")
        print("  model_dir  — directory containing model.safetensors + config.json + tokenizer")
        print("  prompt     — optional prompt (default: 'Say hello')")
        sys.exit(1)
    atlas_path = sys.argv[1]
    model_dir = sys.argv[2]
    safe_path = os.path.join(model_dir, "model.safetensors")
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Say hello"

    print(f"[Atlas] Loading {atlas_path}...")
    t0 = time.time()
    model = AtlasModel(atlas_path, safe_path)
    print(f"[Atlas] Loaded in {time.time() - t0:.1f}s")

    print(f"[Atlas] Generating: {prompt}")
    t0 = time.time()
    text = model.generate(prompt, max_new_tokens=20, temperature=0.0)
    print(f"[Atlas] Output: {text}")
    print(f"[Atlas] Time: {time.time() - t0:.1f}s")
