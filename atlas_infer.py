#!/usr/bin/env python3
"""Atlas Inference Engine — End-to-end Falcon3-7B TQ1.0 generation."""
import ctypes, struct, os, sys, time, numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

# ─── Load C++ DLL ────────────────────────────────────────────────────────
# Resolve OpenMP runtime conflicts (numpy MKL vs libomp)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_dll_name = "atlas.dll" if sys.platform == "win32" else "libatlas.so"
_dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _dll_name)
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

dll.atlas_matmul_i8_f32.restype = None
dll.atlas_matmul_i8_f32.argtypes = [ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int]

dll.atlas_decompress_all.restype = None
dll.atlas_decompress_all.argtypes = [ctypes.c_void_p]

dll.atlas_save_cache.restype = None
dll.atlas_save_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

dll.atlas_load_cache.restype = ctypes.c_int
dll.atlas_load_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

dll.atlas_prefetch_int8.restype = None
dll.atlas_prefetch_int8.argtypes = [ctypes.c_void_p]

dll.atlas_forward.restype = None
dll.atlas_forward.argtypes = [
    ctypes.c_void_p,                              # model
    ctypes.POINTER(ctypes.c_float),               # hidden_states (in-place)
    ctypes.c_int,                                  # B
    ctypes.POINTER(ctypes.c_int),                  # positions
    ctypes.POINTER(ctypes.c_uint16),               # k_cache (flat)
    ctypes.POINTER(ctypes.c_uint16),               # v_cache (flat)
    ctypes.c_int,                                  # max_seq_len
    ctypes.c_int,                                  # seq_now
    ctypes.POINTER(ctypes.c_int),                  # layer_idx [n_layers * 9]
    ctypes.c_int,                                  # n_layers
]

dll.atlas_get_int8.restype = ctypes.POINTER(ctypes.c_int8)
dll.atlas_get_int8.argtypes = [ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))]

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
        with safe_open(safetensors_path, framework='np', device='cpu') as f:
            self.tensor_names = list(f.keys())
        self.n_tensors = len(self.tensor_names)

        print(f"[Atlas] {self.n_layers}L {self.hidden}H {self.inter}I "
              f"{self.n_heads}/{self.n_kv_heads} heads | "
              f"{self.n_tensors} tensors")

        # Cache tensor indices for fast lookup
        self._cache_indices()

        # Try loading int8 cache (mmap'd, instant). If not found, decompress + save.
        cache_loaded = dll.atlas_load_cache(self.model_ptr, self._atlas_path.encode())
        if cache_loaded:
            print("[Atlas] Loaded int8 weights from cache (mmap)")
        else:
            dll.atlas_decompress_all(self.model_ptr)
            print("[Atlas] TQ1 tensors decoded to int8")
            dll.atlas_save_cache(self.model_ptr, self._atlas_path.encode())
        # Prefetch int8 data into physical RAM (page-in mmap or fresh decompress)
        dll.atlas_prefetch_int8(self.model_ptr)

        # Precompute KV cache
        self.max_seq_len = 4096  # conservative for 16GB
        kvc_size = (self.n_layers, self.n_kv_heads, self.max_seq_len, self.head_dim)
        self.k_cache = np.zeros(kvc_size, dtype=np.float16)
        self.v_cache = np.zeros(kvc_size, dtype=np.float16)

        # No warmup needed — C++ lazily converts lm_head to fp32 on first atlas_forward call

        # Pre-convert lm_head to fp32 at load time to avoid 1240ms first-token spike
        # Adds 1.5 GB persistent RAM but saves ~1.2s warmup on first decode
        print("[Atlas] Converting lm_head to fp32...")
        t0 = time.time()
        dummy = np.empty((1, self.hidden), dtype=np.float32)
        self._matmul_f16("lm_head.weight", dummy)
        print(f"[Atlas] lm_head ready ({time.time()-t0:.1f}s)")


    def _cache_indices(self):
        self.idx = {}
        for i, name in enumerate(self.tensor_names):
            self.idx[name] = i
        # Cache frequently-used tensors
        self._embed_w = self._load_embed("model.embed_tokens.weight")
        self._norm_w = self._load_weight_f16("model.norm.weight")
        self._tq1_cache = {}
        self._f16_cache = {}
        self._i8_cache = {}
        # Build flat index array for atlas_forward (fused C++ layer loop)
        idx = self.idx
        per_layer = ['input_layernorm.weight',
            'self_attn.q_proj.weight', 'self_attn.k_proj.weight',
            'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
            'post_attention_layernorm.weight',
            'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']
        arrs = []
        for L in range(self.n_layers):
            for n in per_layer:
                arrs.append(idx[f'model.layers.{L}.{n}'])
        self._layer_idx_arr = np.array(arrs, dtype=np.int32)

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

    def _load_embed(self, name):
        """Load embedding as fp16 (not fp32) to save 1.6GB RAM."""
        idx = self.idx.get(name)
        if idx is None: return None
        sz = ctypes.c_int()
        ptr = dll.atlas_tensor_data(self.model_ptr, idx, sz)
        if not ptr: return None
        flat = np.ctypeslib.as_array(ptr, shape=(sz.value,)).view(np.float16).copy()
        return flat.reshape(self.vocab_size, self.hidden)

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

    def _load_int8(self, name):
        """Load int8 data from C++ (decompressed at load time)."""
        if name in self._i8_cache:
            return self._i8_cache[name]
        idx = self.idx.get(name)
        if idx is None:
            self._i8_cache[name] = (None, None, None, None, None)
            return self._i8_cache[name]
        rows = ctypes.c_int()
        input_dim = ctypes.c_int()
        scale = ctypes.c_float()
        rs_ptr = ctypes.POINTER(ctypes.c_int32)()
        i8_ptr = dll.atlas_get_int8(
            self.model_ptr, idx, rows, input_dim, scale, rs_ptr)
        if not i8_ptr:
            self._i8_cache[name] = (None, None, None, None, None)
            return self._i8_cache[name]
        r, d = rows.value, input_dim.value
        # Create numpy views into DLL memory (no copy)
        i8 = np.ctypeslib.as_array(i8_ptr, shape=(r, d))
        row_sums = np.ctypeslib.as_array(rs_ptr, shape=(r,))
        result = (i8, r, d, scale.value, row_sums)
        self._i8_cache[name] = result
        return result

    def _matmul_int8(self, i8_data, rows, input_dim, scale, row_sums, act):
        """Int8 matmul via C++ AVX2 maddubs kernel."""
        orig_shape = act.shape
        act = act.reshape(-1, orig_shape[-1])

        # Pad activation to packed_cols*5 (TQ1 packer pads to multiple of 5)
        if act.shape[1] < input_dim:
            pad = np.zeros((act.shape[0], input_dim), dtype=np.float32)
            pad[:, :act.shape[1]] = act
            act = pad

        # Quantize activations to int8 [-127, 127]
        max_abs = np.max(np.abs(act), axis=-1, keepdims=True)
        max_abs = np.maximum(max_abs, 1e-5)
        act_scale = 127.0 / max_abs
        act_q = np.round(act * act_scale).astype(np.int8)

        # Convert to uint8 with +128 offset (for maddubs unsigned operand)
        act_u8 = (act_q.astype(np.int32) + 128).clip(0, 255).astype(np.uint8)

        out = np.zeros((act.shape[0], rows), dtype=np.float32)
        dll.atlas_matmul_i8_f32(
            rows, input_dim,
            i8_data.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            act_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            row_sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            act.shape[0])

        # Reorder: atlas order → HF order
        rows_packed = rows // 4
        out = out.reshape(out.shape[0], rows_packed, 4).transpose(0, 2, 1).reshape(out.shape[0], rows)

        # Dequantize
        out *= max_abs / (127.0 * scale)
        out = out.reshape(*orig_shape[:-1], rows)
        return out

    def _matmul_f16(self, name, act):
        """Float32 activation × cached weight (kept fp16 on init, convert to fp32 lazily for lm_head)."""
        if name not in self._f16_cache:
            idx = self.idx.get(name)
            if idx is None: return None
            tt = ctypes.c_int(); rd = ctypes.c_int(); cd = ctypes.c_int()
            dll.atlas_tensor_info(self.model_ptr, idx, tt, rd, cd)
            if tt.value == 0:
                return self._matmul_tq1(*self._load_tq1(name), act)
            sz = ctypes.c_int()
            ptr = dll.atlas_tensor_data(self.model_ptr, idx, sz)
            # Small tensors (norms) stay as fp16 view. lm_head converts to fp32 for fast matmul.
            w = np.ctypeslib.as_array(ptr, shape=(sz.value,)).view(np.float16).reshape(rd.value, act.shape[-1])
            if w.shape[0] * w.shape[1] > 100000:  # lm_head: 131072x3072 -> convert to fp32
                self._f16_cache[name] = np.ascontiguousarray(w, dtype=np.float32)
            else:
                self._f16_cache[name] = w
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
            self._f16_cache[weight_name] = ptr
        out = np.zeros_like(x)
        dll.atlas_rmsnorm_f32(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._f16_cache[weight_name],
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            x.shape[-1], ctypes.c_float(eps))
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
        """Forward one transformer layer via atlas_forward (fused, n_layers=1)."""
        B = len(positions) if isinstance(positions, (list, np.ndarray)) else positions.shape[0]
        positions_arr = np.array(positions, dtype=np.int32)
        seq_now = int(positions_arr.max()) + 1 if use_kvcache else B

        out = x.copy()
        idx_slice = self._layer_idx_arr[layer_idx * 9 : (layer_idx + 1) * 9].copy()
        # Offset K/V cache to this specific layer (atlas_forward loop starts at L=0)
        kc = self.k_cache[layer_idx].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        vc = self.v_cache[layer_idx].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        dll.atlas_forward(
            self.model_ptr,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B,
            positions_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            kc, vc,
            self.max_seq_len, seq_now,
            idx_slice.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            1)
        return out

    def forward(self, tokens, start_pos=0):
        """Full forward pass — all layers fused in C++, final RMSNorm + LM head in Python."""
        B, seq_len = tokens.shape
        h = self._embed_w[tokens].astype(np.float32)
        h = h.reshape(-1, self.hidden)  # [B*seq_len, H]
        n = B * seq_len

        positions = np.array([start_pos + p for b in range(B) for p in range(seq_len)], dtype=np.int32)
        seq_now = start_pos + seq_len

        dll.atlas_forward(
            self.model_ptr,
            h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.k_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            self.v_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            self.max_seq_len, seq_now,
            self._layer_idx_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.n_layers)

        h_norm = np.array([self._rmsnorm(h[b], "model.norm.weight") for b in range(n)])
        output_logits = self._matmul_f16("lm_head.weight", h_norm).reshape(B, seq_len, self.vocab_size)
        return output_logits

    @staticmethod
    def _sample(logits, temperature=1.0, top_k=0, top_p=0.0):
        """Sample next token from logits with temperature, top-k, top-p."""
        if temperature == 0.0:
            return int(np.argmax(logits))

        probs = np.exp(logits / temperature)
        probs /= probs.sum()

        if top_k > 0:
            idx = np.argpartition(probs, -top_k)[-top_k:]
            mask = np.zeros_like(probs)
            mask[idx] = 1.0
            probs *= mask
            probs /= probs.sum()

        if top_p > 0.0:
            idx = np.argsort(probs)[::-1]
            cum = np.cumsum(probs[idx])
            cutoff = np.searchsorted(cum, top_p, side='right') + 1
            mask = np.zeros_like(probs)
            mask[idx[:cutoff]] = 1.0
            probs *= mask
            probs /= probs.sum()

        return int(np.random.choice(len(probs), p=probs))

    def generate(self, prompt, max_new_tokens=50, temperature=1.0,
                 top_k=40, top_p=0.9):
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

        output = []
        for step in range(max_new_tokens):
            if step == 0:
                current_logits = logits
            else:
                h = self._get_embedding(next_token)
                pos = np.array([len(input_ids) + step - 1], dtype=np.int32)
                dll.atlas_forward(
                    self.model_ptr,
                    h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    1, pos.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    self.k_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                    self.v_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                    self.max_seq_len, len(input_ids) + step,
                    self._layer_idx_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    self.n_layers)
                h_norm = self._rmsnorm(h.flatten(), "model.norm.weight")
                current_logits = self._matmul_f16("lm_head.weight", h_norm.reshape(1, -1)).flatten()

            next_token = self._sample(current_logits, temperature, top_k, top_p)
            output.append(next_token)

            if eos_id is not None and next_token == eos_id:
                break
            decoded = tok.decode(output, skip_special_tokens=False)
            if any(stop in decoded for stop in stop_tokens):
                break

        text = tok.decode(output, skip_special_tokens=True)
        for stop in stop_tokens:
            idx = text.find(stop)
            if idx >= 0:
                text = text[:idx]
        return text

    def _get_embedding(self, token_id):
        return self._embed_w[token_id].astype(np.float32).reshape(1, self.hidden)


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
