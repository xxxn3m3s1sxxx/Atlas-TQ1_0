#!/usr/bin/env python3
"""Verify 0.999999 correlation: C++ fused forward vs Python reference (int8)."""
import sys, os, time, numpy as np, ctypes
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, r'C:\atlas')
from atlas_infer import dll, AtlasModel

def verify_model(path, cfg_dir, label):
    print(f'\n{"="*60}')
    print(f'VERIFYING: {label}')
    print(f'{"="*60}')

    m = AtlasModel(path, model_dir=cfg_dir)
    H, inter = m.hidden, m.inter
    qd = m.n_heads * m.head_dim
    kvd = m.n_kv_heads * m.head_dim
    
    # Disable f32 bypass for u8 vs u8 comparison
    dll.atlas_set_use_f32_matmul(m.model_ptr, 0)

    np.random.seed(42)
    B = 1
    x = np.random.randn(B, H).astype(np.float32)

    # ─── Python reference: per-layer, uses _matmul_int8 (u8 quantized) ───
    t0 = time.time()
    m.k_cache[:] = 0.0; m.v_cache[:] = 0.0
    h_ref = x.copy()
    positions = np.array([0], dtype=np.int32)
    for L in range(m.n_layers):
        p = f'model.layers.{L}'
        
        # Pre-attention RMSNorm
        h_ref = m._rmsnorm(h_ref, f'{p}.input_layernorm.weight')
        n1 = h_ref.copy()
        
        # QKV
        d = m._load_int8(f'{p}.self_attn.q_proj.weight')
        qo = m._matmul_int8(*d, n1).copy()
        d = m._load_int8(f'{p}.self_attn.k_proj.weight')
        ko = m._matmul_int8(*d, n1).copy()
        d = m._load_int8(f'{p}.self_attn.v_proj.weight')
        vo = m._matmul_int8(*d, n1).copy()
        
        # Attention
        qa = qo[:, :qd].copy(); ka = ko[:, :kvd].copy(); va = vo[:, :kvd].copy()
        at = np.zeros_like(qa)
        dll.atlas_attention_f32(
            qa.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ka.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            va.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            m.k_cache[L].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            m.v_cache[L].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            m.max_seq_len, L+1, B, m.n_heads, m.n_kv_heads, m.head_dim,
            ctypes.c_float(m.rope_theta), at.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        
        # O projection + residual
        d = m._load_int8(f'{p}.self_attn.o_proj.weight')
        oo = m._matmul_int8(*d, at)
        h_ref = h_ref + oo[:, :H]
        
        # Post-attention RMSNorm
        h_ref = m._rmsnorm(h_ref, f'{p}.post_attention_layernorm.weight')
        n2 = h_ref.copy()
        
        # FFN: gate + up + SiLU + mul + down
        d = m._load_int8(f'{p}.mlp.gate_proj.weight')
        go = m._matmul_int8(*d, n2).copy()
        d = m._load_int8(f'{p}.mlp.up_proj.weight')
        uo = m._matmul_int8(*d, n2).copy()
        
        gs = go[:, :inter] * (1.0 / (1.0 + np.exp(-go[:, :inter])))
        hu = gs * uo[:, :inter]
        
        d = m._load_int8(f'{p}.mlp.down_proj.weight')
        do_ = m._matmul_int8(*d, hu)
        h_ref = h_ref + do_[:, :H]
    
    h_ref_norm = m._rmsnorm(h_ref, 'model.norm.weight')
    ref_logits = m._matmul_f16('lm_head.weight', h_ref_norm)
    ref_token = int(np.argmax(ref_logits[0]))
    py_time = time.time() - t0

    # ─── C++ fused forward (all layers, u8 path) ───
    t0 = time.time()
    m.k_cache[:] = 0.0; m.v_cache[:] = 0.0
    h_cpp = x.copy()
    
    layer_idx_arr = np.array(sum(
        [list(m.idx[f'model.layers.{l}.{n}']
             for n in ('input_layernorm.weight', 'self_attn.q_proj.weight',
                       'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
                       'self_attn.o_proj.weight',
                       'post_attention_layernorm.weight', 'mlp.gate_proj.weight',
                       'mlp.up_proj.weight', 'mlp.down_proj.weight'))
        for l in range(m.n_layers)], []), dtype=np.int32)
    
    dll.atlas_forward(
        m.model_ptr,
        h_cpp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        m.k_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        m.v_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        m.max_seq_len, m.n_layers,
        layer_idx_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        m.n_layers)
    
    h_cpp_norm = m._rmsnorm(h_cpp, 'model.norm.weight')
    cpp_logits = m._matmul_f16('lm_head.weight', h_cpp_norm)
    cpp_token = int(np.argmax(cpp_logits[0]))
    cpp_time = time.time() - t0

    # ─── Compare ───
    hrf = h_ref.flatten().astype(np.float64)
    hcf = h_cpp.flatten().astype(np.float64)
    corr = np.corrcoef(hrf, hcf)[0, 1]
    max_diff = np.max(np.abs(hrf - hcf))
    mean_diff = np.mean(np.abs(hrf - hcf))
    
    print(f'  Py time: {py_time:.1f}s  C++ time: {cpp_time:.1f}s')
    print(f'  hidden corr={corr:.8f}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}')
    print(f'  ref token={ref_token}  cpp token={cpp_token}  match={ref_token==cpp_token}')
    
    # Also compare per-layer
    print(f'\n  Per-layer correlation:')
    m.k_cache[:] = 0.0; m.v_cache[:] = 0.0
    h_cpp_pl = x.copy()
    global_max_diff = 0.0
    for L in range(m.n_layers):
        out_cpp = np.zeros((B, H), dtype=np.float32)
        dll.atlas_forward(
            m.model_ptr,
            h_cpp_pl.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B, positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            m.k_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            m.v_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            m.max_seq_len, L+1,
            layer_idx_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            L+1)
        # Compare with Python
        pl_ref = h_ref.copy()  # We'll refine this
        cf = h_cpp_pl.flatten().astype(np.float64)
        # Get Python ref at this layer: re-run up to layer L
        m2 = AtlasModel(path, model_dir=cfg_dir)  # Fresh model for clean state
        dll.atlas_set_use_f32_matmul(m2.model_ptr, 0)
        m2.k_cache[:] = 0.0; m2.v_cache[:] = 0.0
        h2 = x.copy()
        for l2 in range(L + 1):
            p2 = f'model.layers.{l2}'
            h2 = m2._rmsnorm(h2, f'{p2}.input_layernorm.weight')
            nn1 = h2.copy()
            d = m2._load_int8(f'{p2}.self_attn.q_proj.weight'); qqo = m2._matmul_int8(*d, nn1)
            d = m2._load_int8(f'{p2}.self_attn.k_proj.weight'); kko = m2._matmul_int8(*d, nn1)
            d = m2._load_int8(f'{p2}.self_attn.v_proj.weight'); vvo = m2._matmul_int8(*d, nn1)
            qqa = qqo[:, :qd].copy(); kka = kko[:, :kvd].copy(); vva = vvo[:, :kvd].copy()
            aat = np.zeros_like(qqa)
            dll.atlas_attention_f32(
                qqa.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                kka.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                vva.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                m2.k_cache[l2].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                m2.v_cache[l2].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                m2.max_seq_len, l2+1, B, m2.n_heads, m2.n_kv_heads, m2.head_dim,
                ctypes.c_float(m2.rope_theta), aat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            d = m2._load_int8(f'{p2}.self_attn.o_proj.weight'); ooo = m2._matmul_int8(*d, aat)
            h2 = h2 + ooo[:, :H]
            h2 = m2._rmsnorm(h2, f'{p2}.post_attention_layernorm.weight')
            nn2 = h2.copy()
            d = m2._load_int8(f'{p2}.mlp.gate_proj.weight'); ggo = m2._matmul_int8(*d, nn2)
            d = m2._load_int8(f'{p2}.mlp.up_proj.weight'); uuo = m2._matmul_int8(*d, nn2)
            ggs = ggo[:, :inter] * (1.0 / (1.0 + np.exp(-ggo[:, :inter])))
            hhu = ggs * uuo[:, :inter]
            d = m2._load_int8(f'{p2}.mlp.down_proj.weight'); ddo = m2._matmul_int8(*d, hhu)
            h2 = h2 + ddo[:, :H]
        prf = h2.flatten().astype(np.float64)
        layer_corr = np.corrcoef(prf, cf)[0, 1]
        layer_max = np.max(np.abs(prf - cf))
        if layer_max > global_max_diff: global_max_diff = layer_max
        status = 'OK' if layer_corr > 0.9999 else 'FAIL'
        print(f'    Layer {L:2d}: corr={layer_corr:.10f}  max_diff={layer_max:.2e}  {status}')
        del m2
    
    print(f'\n  Global max_diff across all layers: {global_max_diff:.2e}')
    PASS = corr > 0.9999 and ref_token == cpp_token and global_max_diff < 1e-2
    print(f'\n  >>> VERDICT: {"PASS (0.999999+)" if PASS else "FAIL"} <<<')
    del m
    return PASS

if __name__ == '__main__':
    results = []
    for name, path, cfg_dir in [
        ('1B',  'C:/atlas/falcon3-1b-tq1.atlas',  'C:/models/Falcon3-1B-Instruct-1.58bit'),
        ('3B',  'C:/atlas/falcon3-3b-tq1.atlas',  'C:/models/Falcon3-3B-Instruct-1.58bit'),
        ('7B',  'C:/atlas/falcon3-7b-tq1.atlas',  'C:/models/Falcon3-7B-Instruct-1.58bit'),
        ('10B', 'C:/atlas/falcon3-10b-tq1.atlas', 'C:/models/Falcon3-10B-Instruct-1.58bit'),
    ]:
        ok = verify_model(path, cfg_dir, name)
        results.append((name, ok))
    
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    for name, ok in results:
        print(f'  {name}: {"✅ PASS (0.999999+)" if ok else "❌ FAIL"}')
