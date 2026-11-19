import os

# Remove the deployment secret before any model installer starts subprocesses.
_DEPLOY_PASSWORD = os.environ.pop("EASYCOLAB_ADMIN_PASSWORD", "") if __name__ == "__main__" else ""
import io
import gc
import re
import sys
import uuid
import time
import glob
import queue
import atexit
import socket
import base64
import contextlib
import secrets
import signal
try:
    from runtime_auth import authenticated_ingress, COMFY_BACKEND, COMFY_INGRESS, GRADIO_BACKEND, GRADIO_INGRESS
except ImportError:
    authenticated_ingress = None
    GRADIO_INGRESS = 7860
    GRADIO_BACKEND = 27860
    COMFY_INGRESS = 8188
    COMFY_BACKEND = 28188
if __name__ == "__main__" and not _DEPLOY_PASSWORD and (authenticated_ingress is None or not sys.stdin.isatty()):
    GRADIO_BACKEND = GRADIO_INGRESS
    COMFY_BACKEND = COMFY_INGRESS
import subprocess
import threading
from collections import deque
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

# ==============================================================================
# 实时终端进度与 tqdm 捕获器 (将 Colab 网页端的实时下载进度同步给 macOS App)
# ==============================================================================
class TerminalProgressMirror:
    def __init__(self, orig_stream, buffer_deque: deque, active_bars: dict, lock: threading.Lock):
        self.orig_stream = orig_stream
        self.buffer_deque = buffer_deque
        self.active_bars = active_bars
        self.lock = lock
        self._tls = threading.local()
        self._ansi_re = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

    def write(self, data: str):
        try:
            self.orig_stream.write(data)
        except Exception:
            pass
        if not data:
            return
        clean = self._ansi_re.sub("", data)
        partial = getattr(self._tls, "partial", "")
        with self.lock:
            for ch in clean:
                if ch in ("\r", "\n"):
                    line = partial.strip()
                    if line:
                        # 识别类似 "model-00001-of-00014.safetensors: 22%..." 或无前缀采样/VAE进度条 "54%|#####4 | 19/35 [08:31<07:11...]"
                        if "%" in line and ("it/s" in line or "B/s" in line or "step" in line or "|" in line):
                            pct_idx = line.find("%")
                            colon_idx = line.find(":")
                            if 0 < colon_idx < pct_idx:
                                key = line[:colon_idx].strip()
                                self.active_bars[key] = line
                            else:
                                stage_lbl = CURRENT_STAGE_BAR_LABEL
                                if stage_lbl:
                                    # 若在 VAE 解码等子阶段内部触发 tqdm，标注阶段名避免子阶段 100% 被误认为整条视频已生成完毕
                                    if line.startswith("100%"):
                                        self.active_bars["sampling"] = f"{stage_lbl}: 子阶段已完成，正在继续后续合成封装..."
                                    else:
                                        self.active_bars["sampling"] = f"{stage_lbl}: {line}"
                                else:
                                    self.active_bars["sampling"] = line
                        else:
                            self.buffer_deque.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
                    partial = ""
                else:
                    partial += ch
        self._tls.partial = partial

    def flush(self):
        try:
            self.orig_stream.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.orig_stream, "isatty", lambda: False)()


CONSOLE_LOCK = threading.Lock()
CONSOLE_LINES: deque = deque(maxlen=120)
ACTIVE_TQDM_BARS: Dict[str, str] = {}
CURRENT_STAGE_BAR_LABEL: Optional[str] = None

if not isinstance(sys.stdout, TerminalProgressMirror):
    sys.stdout = TerminalProgressMirror(sys.__stdout__, CONSOLE_LINES, ACTIVE_TQDM_BARS, CONSOLE_LOCK)
if not isinstance(sys.stderr, TerminalProgressMirror):
    sys.stderr = TerminalProgressMirror(sys.__stderr__, CONSOLE_LINES, ACTIVE_TQDM_BARS, CONSOLE_LOCK)


def get_live_console_Lines() -> List[str]:
    with CONSOLE_LOCK:
        bars = list(ACTIVE_TQDM_BARS.values())[-10:]
        logs = list(CONSOLE_LINES)[-60:]
        return logs + (["─── 实时下载/采样进度流 ───"] + bars if bars else [])


# ==============================================================================
# 自动补齐音视频依赖 (防止出现 No module named 'av' 等错误)
# ==============================================================================
def ensure_runtime_dependencies():
    missing = []
    for pkg, mod in [
        ("av", "av"),
        ("hf_transfer", "hf_transfer"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("sentencepiece", "sentencepiece"),
        ("tiktoken", "tiktoken"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"正在自动安装缺失的运行期依赖: {', '.join(missing)} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("缺失依赖自动安装完成。")


ensure_runtime_dependencies()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# 强制 DiffSynth-Studio 所有内置下载走 HuggingFace 国际高速源 (彻底杜绝回退到国内 ModelScope 导致 1~2MB/s)
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
# 将 DiskMap 的句柄缓冲阈值扩大至 1TB，避免加载 33B 模型时频繁关闭/重开 safetensors 映射
os.environ["DIFFSYNTH_DISK_MAP_BUFFER_SIZE"] = str(10**12)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import shutil
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import gradio as gr
from PIL import Image


def _ensure_torchaudio_compatible():
    """防止 pip -U 升级覆盖 Colab 预装的 CUDA 定制版 torchaudio 导致 libtorchaudio.so ABI 不兼容崩溃。"""
    try:
        import torchaudio  # noqa: F401
        return
    except (OSError, ImportError) as e:
        for k in [m for m in list(sys.modules.keys()) if m == "torchaudio" or m.startswith("torchaudio.")]:
            sys.modules.pop(k, None)
        ver_raw = getattr(torch, "__version__", "")
        base_ver, plus, local_tag = ver_raw.partition("+")
        cu_tag = local_tag if plus and local_tag.startswith("cu") else ""
        print(f"检测到 torchaudio 动态库异常 ({e})，正在自动对齐 PyTorch ({ver_raw}) ABI...")
        repaired = False
        if base_ver and cu_tag:
            rc = subprocess.call([
                sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
                "--index-url", f"https://download.pytorch.org/whl/{cu_tag}",
                f"torchaudio=={base_ver}",
            ])
            if rc == 0:
                try:
                    import torchaudio  # noqa: F401
                    print(f"torchaudio 已自动修复对齐至 {torchaudio.__version__}！")
                    repaired = True
                except Exception:
                    for k in [m for m in list(sys.modules.keys()) if m == "torchaudio" or m.startswith("torchaudio.")]:
                        sys.modules.pop(k, None)
        if not repaired:
            try:
                import transformers.utils.import_utils as _iu
                _iu._torchaudio_available = False
            except Exception:
                pass


_ensure_torchaudio_compatible()


def _register_triton_causal_conv1d_shim():
    """当环境无预编译 causal_conv1d wheel (如 Python 3.13 + CUDA 13) 时，注册原生 Triton 融合算子以消除逐 token 多算子回退开销。"""
    if "causal_conv1d" in sys.modules:
        return
    try:
        import types
        import torch.nn.functional as F
        import triton
        import triton.language as tl

        @triton.jit
        def _triton_conv1d_update_k4(
            x_ptr, state_ptr, w_ptr, b_ptr, out_ptr,
            D,
            STATE_LEN: tl.constexpr,
            HAS_BIAS: tl.constexpr,
            APPLY_SILU: tl.constexpr,
            BLOCK_D: tl.constexpr,
        ):
            pid_b = tl.program_id(0)
            pid_d = tl.program_id(1)
            offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
            mask = offs_d < D

            x_idx = pid_b * D + offs_d
            x_raw = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
            x0 = x_raw.to(tl.float32)

            st_base = (pid_b * D + offs_d) * STATE_LEN
            st_proto = tl.load(state_ptr + st_base, mask=mask, other=0.0)
            s0 = tl.load(state_ptr + st_base + (STATE_LEN - 3), mask=mask, other=0.0).to(tl.float32)
            s1 = tl.load(state_ptr + st_base + (STATE_LEN - 2), mask=mask, other=0.0).to(tl.float32)
            s2 = tl.load(state_ptr + st_base + (STATE_LEN - 1), mask=mask, other=0.0).to(tl.float32)

            if STATE_LEN == 4:
                tl.store(state_ptr + st_base + 0, s0.to(st_proto.dtype), mask=mask)
                tl.store(state_ptr + st_base + 1, s1.to(st_proto.dtype), mask=mask)
                tl.store(state_ptr + st_base + 2, s2.to(st_proto.dtype), mask=mask)
                tl.store(state_ptr + st_base + 3, x0.to(st_proto.dtype), mask=mask)
            elif STATE_LEN == 3:
                tl.store(state_ptr + st_base + 0, s1.to(st_proto.dtype), mask=mask)
                tl.store(state_ptr + st_base + 1, s2.to(st_proto.dtype), mask=mask)
                tl.store(state_ptr + st_base + 2, x0.to(st_proto.dtype), mask=mask)

            w_base = offs_d * 4
            w0 = tl.load(w_ptr + w_base + 0, mask=mask, other=0.0).to(tl.float32)
            w1 = tl.load(w_ptr + w_base + 1, mask=mask, other=0.0).to(tl.float32)
            w2 = tl.load(w_ptr + w_base + 2, mask=mask, other=0.0).to(tl.float32)
            w3 = tl.load(w_ptr + w_base + 3, mask=mask, other=0.0).to(tl.float32)

            acc = s0 * w0 + s1 * w1 + s2 * w2 + x0 * w3
            if HAS_BIAS:
                b = tl.load(b_ptr + offs_d, mask=mask, other=0.0).to(tl.float32)
                acc += b
            if APPLY_SILU:
                acc = acc * tl.sigmoid(acc)
            tl.store(out_ptr + x_idx, acc.to(x_raw.dtype), mask=mask)

        def _shim_causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None, conv_state_indices=None):
            act_silu = activation in ("silu", "swish")
            if (
                x.is_cuda
                and cache_seqlens is None
                and conv_state_indices is None
                and x.ndim == 3
                and x.shape[2] == 1
                and weight.ndim == 2
                and weight.shape[1] == 4
                and conv_state.ndim == 3
                and conv_state.shape[2] in (3, 4)
                and conv_state.is_contiguous()
                and (activation is None or act_silu)
            ):
                x_c = x if x.is_contiguous() else x.contiguous()
                w_c = weight if weight.is_contiguous() else weight.contiguous()
                b_c = (bias if bias.is_contiguous() else bias.contiguous()) if bias is not None else w_c
                out = torch.empty_like(x_c)
                B, D, _ = x_c.shape
                grid = (B, triton.cdiv(D, 256))
                _triton_conv1d_update_k4[grid](
                    x_c, conv_state, w_c, b_c, out,
                    D,
                    STATE_LEN=conv_state.shape[2],
                    HAS_BIAS=bias is not None,
                    APPLY_SILU=act_silu,
                    BLOCK_D=256,
                )
                return out

            _, hidden_size, seq_len = x.shape
            state_len = conv_state.shape[-1]
            h_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
            conv_state.copy_(h_new[:, :, -state_len:])
            out = F.conv1d(h_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)[:, :, -seq_len:]
            if act_silu:
                out = F.silu(out)
            return out.to(x.dtype)

        def _shim_causal_conv1d_fn(x, weight, bias=None, activation=None, **kwargs):
            _, hidden_size, seq_len = x.shape
            padding = weight.shape[-1] - 1
            out = F.conv1d(
                x.to(weight.dtype),
                weight=weight.unsqueeze(1),
                bias=bias,
                padding=padding,
                groups=hidden_size,
            )[:, :, :seq_len]
            if activation in ("silu", "swish"):
                out = F.silu(out)
            return out.to(x.dtype)

        mod = types.ModuleType("causal_conv1d")
        mod.__version__ = "1.5.0+triton"
        mod.causal_conv1d_update = _shim_causal_conv1d_update
        mod.causal_conv1d_fn = _shim_causal_conv1d_fn
        sys.modules["causal_conv1d"] = mod
    except Exception as e:
        print(f"[加速引擎] Triton causal_conv1d 注册跳过: {e}")


def _ensure_qwen_fast_kernels():
    """确保 Qwen3.8 线性注意力 Triton 算子 (fla-core) 与融合 causal_conv1d_update 算子真正生效。"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    need_rebind = False
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule  # noqa: F401
    except Exception:
        print("正在安装 Qwen3.8 GatedDeltaNet 原生 Triton 算子 (fla-core + flash-linear-attention)...")
        for k in [m for m in list(sys.modules.keys()) if m == "fla" or m.startswith("fla.")]:
            sys.modules.pop(k, None)
        subprocess.call([sys.executable, "-m", "pip", "install", "-q", "einops"])
        subprocess.call([
            sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
            "fla-core", "flash-linear-attention",
        ])
        for k in [m for m in list(sys.modules.keys()) if m == "fla" or m.startswith("fla.")]:
            sys.modules.pop(k, None)
        need_rebind = True

    try:
        import causal_conv1d  # noqa: F401
    except ImportError:
        if sys.version_info >= (3, 13):
            _register_triton_causal_conv1d_shim()
            need_rebind = True
        else:
            rc = subprocess.call([
                sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                "--only-binary=:all:", "causal-conv1d",
            ])
            if rc != 0:
                _register_triton_causal_conv1d_shim()
                need_rebind = True

    if need_rebind and "transformers.models.qwen3_5.modeling_qwen3_5" in sys.modules:
        import importlib
        try:
            if "transformers.integrations.hub_kernels" in sys.modules:
                importlib.reload(sys.modules["transformers.integrations.hub_kernels"])
            importlib.reload(sys.modules["transformers.models.qwen3_5.modeling_qwen3_5"])
        except Exception:
            pass


def _optimize_qwen35_for_fast_inference(mdl, max_context_len: int = 32768):
    """
    针对 Qwen3.8-27B BF16 (64层混合 GatedDeltaNet + Attention) 的深度算子融合与全模型 CUDAGraph 解码引擎：
    1. 融合 Qwen3_5RMSNorm -> 原生 C++ F.rms_norm (消除每 token 1,700+ 次 Python 标量/张量调度)
    2. 融合 Qwen3_5MLP [gate_proj; up_proj] 单次 GEMV + FLA Triton SwiGLU
    3. 融合 Qwen3_5Attention [q_proj; k_proj; v_proj] 单次 GEMV + PyTorch GQA SDPA
    4. 融合 Qwen3_5GatedDeltaNet [qkv; z; b; a] 单次 GEMV + Triton g/beta + 原生 GQA FLA + Triton RMSNormGated
    5. 预分配持久化 StaticCache 捕获 CUDAGraph 极速解码，同时兼容超长上下文 (最高 256K) 动态混合缓存
    """
    try:
        import torch.nn as nn
        import torch.nn.functional as F
        import triton
        import triton.language as tl
        import transformers.models.qwen3_5.modeling_qwen3_5 as mq
        from transformers import StaticCache
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        from fla.modules.activations import swiglu

        text_model = getattr(getattr(mdl, "model", None), "language_model", None)
        if text_model is None:
            return

        # 单卡权重已经在 GPU 上。device_map 留下的 AlignDevicesHook 会在每层
        # forward 里递归扫描混合缓存，NF4 比 BF16 慢的一大截来自这里。
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(mdl)
        except Exception as hook_err:
            print(f"[加速引擎] accelerate hook 拆除跳过: {hook_err}")

        @triton.jit
        def _qwen35_rmsnorm_gated_silu(x_ptr, z_ptr, w_ptr, out_ptr, D, BLOCK_D: tl.constexpr, eps):
            row = tl.program_id(0)
            offs = tl.arange(0, BLOCK_D)
            mask = offs < D
            x_raw = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
            x = x_raw.to(tl.float32)
            z = tl.load(z_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            var = tl.sum(x * x, axis=0) / D
            rstd = tl.rsqrt(var + eps)
            y = (x * rstd * w) * (z * tl.sigmoid(z))
            tl.store(out_ptr + row * D + offs, y.to(x_raw.dtype), mask=mask)

        @triton.jit
        def _qwen35_g_beta_fused(ba_ptr, neg_exp_A_ptr, dt_bias_ptr, g_ptr, beta_ptr, H, BLOCK_H: tl.constexpr):
            row = tl.program_id(0)
            offs = tl.arange(0, BLOCK_H)
            mask = offs < H
            b_raw = tl.load(ba_ptr + row * (2 * H) + offs, mask=mask, other=0.0)
            b = b_raw.to(tl.float32)
            a = tl.load(ba_ptr + row * (2 * H) + H + offs, mask=mask, other=0.0).to(tl.float32)
            neg_exp_A = tl.load(neg_exp_A_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            dt_b = tl.load(dt_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            beta = tl.sigmoid(b)
            x = a + dt_b
            sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
            g = neg_exp_A * sp
            tl.store(beta_ptr + row * H + offs, beta.to(b_raw.dtype), mask=mask)
            tl.store(g_ptr + row * H + offs, g, mask=mask)

        # head_dim=256 时把整段 D 放进寄存器会溢出，解码被延迟绑死。
        # 按 BLOCK_D=32 分块，fp32 累加写到捕获前就分配好的全局缓冲。
        _decode_acc_holder = [None]

        def _decode_acc_buf(q):
            hq, hd = q.shape[1], q.shape[-1]
            buf = _decode_acc_holder[0]
            if buf is None or buf.shape != (hq, hd) or buf.device != q.device:
                buf = torch.empty((hq, hd), device=q.device, dtype=torch.float32)
                _decode_acc_holder[0] = buf
            return buf

        @triton.jit
        def _qwen35_gqa_decode_attn(
            q_ptr, k_ptr, v_ptr, cum_ptr, acc_ptr, out_ptr, scale, max_seq,
            D: tl.constexpr, GQA_GROUPS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        ):
            pid_hq = tl.program_id(0)
            pid_hkv = pid_hq // GQA_GROUPS
            seq_len = tl.load(cum_ptr).to(tl.int32)
            acc_base = acc_ptr + pid_hq * D
            q_base = q_ptr + pid_hq * D
            k_base = k_ptr + pid_hkv * max_seq * D
            v_base = v_ptr + pid_hkv * max_seq * D
            q_dtype = tl.load(q_base).dtype
            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                tl.store(acc_base + offs_d, tl.zeros([BLOCK_D], dtype=tl.float32), mask=offs_d < D)
            m_i = -float("inf")
            l_i = 0.0
            for start_n in range(0, seq_len, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < seq_len
                qk = tl.zeros([BLOCK_N], dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < D
                    q_tile = tl.load(q_base + offs_d, mask=mask_d, other=0.0).to(tl.float32)
                    ptrs = offs_n[:, None] * D + offs_d[None, :]
                    mask_2d = mask_n[:, None] & mask_d[None, :]
                    k_blk = tl.load(k_base + ptrs, mask=mask_2d, other=0.0).to(tl.float32)
                    qk += tl.sum(k_blk * q_tile[None, :], axis=1)
                qk = qk * scale
                qk = tl.where(mask_n, qk, -float("inf"))
                m_ij = tl.max(qk, axis=0)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(qk - m_new)
                l_i = l_i * alpha + tl.sum(tl.where(mask_n, p, 0.0), axis=0)
                for d0 in range(0, D, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < D
                    ptrs = offs_n[:, None] * D + offs_d[None, :]
                    mask_2d = mask_n[:, None] & mask_d[None, :]
                    v_blk = tl.load(v_base + ptrs, mask=mask_2d, other=0.0).to(tl.float32)
                    acc_tile = tl.load(acc_base + offs_d, mask=mask_d, other=0.0)
                    acc_tile = acc_tile * alpha + tl.sum(p[:, None] * v_blk, axis=0)
                    tl.store(acc_base + offs_d, acc_tile, mask=mask_d)
                m_i = m_new
            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                acc_tile = tl.load(acc_base + offs_d, mask=mask_d, other=0.0)
                tl.store(out_ptr + pid_hq * D + offs_d, (acc_tile / l_i).to(q_dtype), mask=mask_d)

        for mod in text_model.modules():
            if isinstance(mod, mq.Qwen3_5RMSNorm):
                mod._w1 = (1.0 + mod.weight.detach().float()).to(mod.weight.dtype).contiguous()
                mod._norm_shape = (mod.weight.shape[0],)

                def _fast_rmsnorm(self, x):
                    return F.rms_norm(x, self._norm_shape, self._w1, self.eps)

                mod.forward = _fast_rmsnorm.__get__(mod, type(mod))

            elif isinstance(mod, mq.Qwen3_5MLP):
                if (
                    isinstance(mod.gate_proj, nn.Linear)
                    and mod.gate_proj.bias is None
                    and mod.up_proj.bias is None
                    and mod.gate_proj.weight.dtype in (torch.bfloat16, torch.float16)
                ):
                    inter = mod.gate_proj.out_features
                    mod._gate_up_w = torch.cat([mod.gate_proj.weight.detach(), mod.up_proj.weight.detach()], dim=0).contiguous()
                    mod.gate_proj = None
                    mod.up_proj = None
                    mod._inter = inter

                    def _fast_mlp(self, x):
                        gu = F.linear(x, self._gate_up_w)
                        g, u = gu.split(self._inter, dim=-1)
                        return self.down_proj(swiglu(g, u))

                    mod.forward = _fast_mlp.__get__(mod, type(mod))

            elif isinstance(mod, mq.Qwen3_5Attention):
                # NF4 权重是 uint8，拼不成一份 BF16 QKV，但解码仍要走静态 Triton 核。
                # 预填继续用 SDPA，因为静态缓冲里有 padding，必须靠官方 mask。
                can_fuse_qkv = (
                    isinstance(mod.q_proj, nn.Linear)
                    and mod.q_proj.bias is None
                    and mod.k_proj.bias is None
                    and mod.v_proj.bias is None
                    and getattr(getattr(mod.q_proj, "weight", None), "dtype", None) in (torch.bfloat16, torch.float16)
                )
                if can_fuse_qkv:
                    qd, kd, vd = mod.q_proj.out_features, mod.k_proj.out_features, mod.v_proj.out_features
                    mod._qkv_w = torch.cat(
                        [mod.q_proj.weight.detach(), mod.k_proj.weight.detach(), mod.v_proj.weight.detach()], dim=0
                    ).contiguous()
                    mod._qkv_splits = (qd, kd, vd)
                    mod.q_proj = mod.k_proj = mod.v_proj = None

                def _fast_attn(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, self.head_dim)
                    if self.q_proj is None:
                        qkv = F.linear(hidden_states, self._qkv_w)
                        q_raw, k_raw, v_raw = torch.split(qkv, self._qkv_splits, dim=-1)
                    else:
                        q_raw = self.q_proj(hidden_states)
                        k_raw = self.k_proj(hidden_states)
                        v_raw = self.v_proj(hidden_states)
                    query_states, gate = torch.chunk(q_raw.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
                    gate = gate.reshape(*input_shape, -1)
                    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                    key_states = self.k_norm(k_raw.view(hidden_shape)).transpose(1, 2)
                    value_states = v_raw.view(hidden_shape).transpose(1, 2)
                    cos, sin = position_embeddings
                    query_states, key_states = mq.apply_rotary_pos_emb(query_states, key_states, cos, sin)
                    if past_key_values is not None:
                        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
                    cache_lyr = (
                        past_key_values.layers[self.layer_idx]
                        if (past_key_values is not None and hasattr(past_key_values, "layers"))
                        else None
                    )
                    use_static = (
                        input_shape == (1, 1)
                        and cache_lyr is not None
                        and hasattr(cache_lyr, "cumulative_length")
                        and torch.is_tensor(cache_lyr.cumulative_length)
                        and getattr(cache_lyr, "keys", None) is not None
                        and getattr(past_key_values, "_no_left_pad", True)
                    )
                    if use_static:
                        q_c = query_states.contiguous()
                        attn_output = torch.empty_like(q_c)
                        k_buf = cache_lyr.keys
                        v_buf = cache_lyr.values
                        hq = q_c.shape[1]
                        hkv = k_buf.shape[1]
                        _qwen35_gqa_decode_attn[(hq,)](
                            q_c,
                            k_buf,
                            v_buf,
                            cache_lyr.cumulative_length,
                            _decode_acc_buf(q_c),
                            attn_output,
                            self.scaling,
                            k_buf.shape[2],
                            D=self.head_dim,
                            GQA_GROUPS=hq // hkv,
                            BLOCK_N=32,
                            BLOCK_D=32,
                            num_warps=4,
                        )
                    else:
                        attn_output = F.scaled_dot_product_attention(
                            query_states,
                            key_states,
                            value_states,
                            attn_mask=attention_mask,
                            dropout_p=0.0,
                            scale=self.scaling,
                            enable_gqa=True,
                        )
                    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
                    return self.o_proj(attn_output * torch.sigmoid(gate)), None

                mod.forward = _fast_attn.__get__(mod, type(mod))

            elif isinstance(mod, mq.Qwen3_5GatedDeltaNet):
                if (
                    isinstance(mod.in_proj_qkv, nn.Linear)
                    and mod.in_proj_qkv.bias is None
                    and mod.in_proj_z.bias is None
                    and mod.in_proj_b.bias is None
                    and mod.in_proj_a.bias is None
                    and mod.in_proj_qkv.weight.dtype in (torch.bfloat16, torch.float16)
                ):
                    qkv_d = mod.in_proj_qkv.out_features
                    z_d = mod.in_proj_z.out_features
                    b_d = mod.in_proj_b.out_features
                    a_d = mod.in_proj_a.out_features
                    mod._qkvzba_w = torch.cat(
                        [
                            mod.in_proj_qkv.weight.detach(),
                            mod.in_proj_z.weight.detach(),
                            mod.in_proj_b.weight.detach(),
                            mod.in_proj_a.weight.detach(),
                        ],
                        dim=0,
                    ).contiguous()
                    mod.in_proj_qkv = mod.in_proj_z = mod.in_proj_b = mod.in_proj_a = None
                    mod._splits = (qkv_d, z_d, b_d + a_d)
                    mod._neg_exp_A = (-mod.A_log.detach().float().exp()).contiguous()
                    mod._dt_bias_f32 = mod.dt_bias.detach().float().contiguous()
                    mod._block_h = triton.next_power_of_2(b_d)
                    mod._block_d = triton.next_power_of_2(mod.head_v_dim)

                    def _fast_gdn(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
                        if attention_mask is not None:
                            hidden_states = mq.apply_mask_to_padding_states(hidden_states, attention_mask)
                        B, L, _ = hidden_states.shape
                        conv_state = None
                        rec_state = None
                        recording_past = False
                        if cache_params is not None:
                            if hasattr(cache_params, "get_conv_state"):
                                conv_state = cache_params.get_conv_state(self.layer_idx, state_idx=0)
                                rec_state = cache_params.get_recurrent_state(self.layer_idx, state_idx=0)
                                recording_past = cache_params.is_recording_past(self.layer_idx)
                            elif getattr(cache_params, "has_previous_state", False):
                                cs_list = getattr(cache_params, "conv_states", None)
                                rs_list = getattr(cache_params, "recurrent_states", None)
                                if isinstance(cs_list, (list, tuple)) and self.layer_idx < len(cs_list):
                                    conv_state = cs_list[self.layer_idx]
                                if isinstance(rs_list, (list, tuple)) and self.layer_idx < len(rs_list):
                                    rec_state = rs_list[self.layer_idx]

                        proj = F.linear(hidden_states, self._qkvzba_w)
                        mixed_qkv, z, ba = torch.split(proj, self._splits, dim=-1)
                        mixed_qkv = mixed_qkv.transpose(1, 2)

                        if conv_state is not None and L == 1 and not recording_past:
                            mixed_qkv = mq.causal_conv1d_update(
                                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
                            )
                        else:
                            if cache_params is not None:
                                if hasattr(cache_params, "update_conv_state"):
                                    mixed_qkv = cache_params.update_conv_state(
                                        mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                                    )
                                else:
                                    cs_list = getattr(cache_params, "conv_states", None)
                                    if isinstance(cs_list, list) and self.layer_idx < len(cs_list):
                                        ksize = self.conv_kernel_size
                                        padded = F.pad(mixed_qkv, (ksize - mixed_qkv.shape[-1], 0)) if mixed_qkv.shape[-1] < ksize else mixed_qkv
                                        cs_list[self.layer_idx] = padded[..., -ksize:].contiguous()
                            mixed_qkv = mq.causal_conv1d_fn(
                                mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation=self.activation
                            )
                            if cache_params is not None:
                                mixed_qkv = mixed_qkv[:, :, -L:]

                        mixed_qkv = mixed_qkv.transpose(1, 2)
                        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
                        q = q.view(B, L, -1, self.head_k_dim)
                        k = k.view(B, L, -1, self.head_k_dim)
                        v = v.view(B, L, -1, self.head_v_dim)

                        H = self.num_v_heads
                        g_t = torch.empty((B, L, H), device=hidden_states.device, dtype=torch.float32)
                        beta_t = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)
                        _qwen35_g_beta_fused[(B * L,)](
                            ba.contiguous(), self._neg_exp_A, self._dt_bias_f32, g_t, beta_t, H, BLOCK_H=self._block_h
                        )

                        if self.num_v_heads // self.num_k_heads > 1:
                            rep = self.num_v_heads // self.num_k_heads
                            q = q.repeat_interleave(rep, dim=2)
                            k = k.repeat_interleave(rep, dim=2)
                        if rec_state is not None and L == 1:
                            core_out, last_state = fused_recurrent_gated_delta_rule(
                                q, k, v, g=g_t, beta=beta_t, initial_state=rec_state, output_final_state=True, use_qk_l2norm_in_kernel=True
                            )
                        else:
                            core_out, last_state = chunk_gated_delta_rule(
                                q, k, v, g=g_t, beta=beta_t, initial_state=rec_state, output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True
                            )
                        if cache_params is not None:
                            if hasattr(cache_params, "update_recurrent_state"):
                                cache_params.update_recurrent_state(last_state, self.layer_idx)
                            else:
                                rs_list = getattr(cache_params, "recurrent_states", None)
                                if isinstance(rs_list, list) and self.layer_idx < len(rs_list):
                                    rs_list[self.layer_idx] = last_state

                        rows = B * L * H
                        normed = torch.empty((rows, self.head_v_dim), device=hidden_states.device, dtype=hidden_states.dtype)
                        _qwen35_rmsnorm_gated_silu[(rows,)](
                            core_out.contiguous(),
                            z.contiguous(),
                            self.norm.weight,
                            normed,
                            self.head_v_dim,
                            BLOCK_D=self._block_d,
                            eps=self.norm.variance_epsilon,
                        )
                        return self.out_proj(normed.view(B, L, -1))

                    mod.forward = _fast_gdn.__get__(mod, type(mod))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as fuse_err:
        print(f"[加速引擎] 算子融合未完成 ({type(fuse_err).__name__}: {fuse_err})")
        return

    # 图捕获失败不能把已经装上的融合层一起丢掉。
    try:
        from transformers.modeling_outputs import BaseModelOutputWithPast

        free_gb_now = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
        static_cap = 32768 if free_gb_now >= 12.0 else (16384 if free_gb_now >= 8.0 else 4096)
        max_static_seq = max(2048, min(int(max_context_len or 32768), static_cap))
        t_cfg = text_model.config
        embed_dtype = text_model.embed_tokens.weight.dtype
        dev = text_model.embed_tokens.weight.device
        try:
            sc = StaticCache(config=text_model.config, max_cache_len=max_static_seq)
        except Exception:
            sc = StaticCache(config=mdl.config, max_cache_len=max_static_seq)
        for _lyr in sc.layers:
            _lyr.is_compileable = False
        try:
            sc.is_compileable = False
        except Exception:
            pass
        sc._no_left_pad = True

        # 旧版 StaticCache 把 conv 状态长度锁成第一次 prefill 的长度，而且没有 get_conv_state。
        # 这里改成按卷积核宽度保存，这样不同长度的 prefill 都能对上已捕获的 decode 图。
        def _stabilize_linear_cache(cache):
            for lyr in cache.layers:
                if isinstance(getattr(lyr, "conv_states", None), dict) or isinstance(getattr(lyr, "has_previous_state", None), dict):
                    continue
                if not hasattr(lyr, "update_conv_state"):
                    continue
                box = {"conv": None, "rec": None, "has_prev": False, "ksize": None, "batch_shape": None}
                lyr._fast_conv = box
                orig_reset = lyr.reset

                def update_conv_state(conv_states, state_idx=0, conv_kernel_size=None, _box=box, **kwargs):
                    ksize = int(conv_kernel_size or conv_states.shape[-1])
                    batch_shape = tuple(conv_states.shape[:-1])
                    if (
                        _box["conv"] is None
                        or _box["ksize"] != ksize
                        or _box["batch_shape"] != batch_shape
                        or _box["conv"].device != conv_states.device
                    ):
                        _box["ksize"] = ksize
                        _box["batch_shape"] = batch_shape
                        _box["conv"] = torch.zeros(
                            *batch_shape, ksize, dtype=conv_states.dtype, device=conv_states.device
                        )
                        _box["has_prev"] = False
                    if not _box["has_prev"]:
                        full = conv_states
                        if full.shape[-1] < ksize:
                            full = F.pad(full, (ksize - full.shape[-1], 0))
                        _box["has_prev"] = True
                    else:
                        full = torch.cat([_box["conv"], conv_states], dim=-1)
                    _box["conv"].copy_(full[..., -ksize:])
                    return full

                def update_recurrent_state(recurrent_states, state_idx=0, _box=box, **kwargs):
                    if (
                        _box["rec"] is None
                        or _box["rec"].shape != recurrent_states.shape
                        or _box["rec"].device != recurrent_states.device
                    ):
                        _box["rec"] = torch.empty_like(recurrent_states)
                    _box["rec"].copy_(recurrent_states)
                    return _box["rec"]

                def reset(_box=box, _orig=orig_reset):
                    try:
                        _orig()
                    except Exception:
                        pass
                    _box["has_prev"] = False
                    if _box["conv"] is not None:
                        _box["conv"].zero_()
                    if _box["rec"] is not None:
                        _box["rec"].zero_()

                lyr.update_conv_state = update_conv_state
                lyr.update_recurrent_state = update_recurrent_state
                lyr.reset = reset

        def _install_cache_api(cache):
            if not hasattr(cache, "get_conv_state"):
                def get_conv_state(layer_idx, state_idx=0):
                    lyr = cache.layers[layer_idx]
                    box = getattr(lyr, "_fast_conv", None)
                    if box is not None:
                        if not box["has_prev"] or box["conv"] is None:
                            return None
                        return box["conv"]
                    hps = getattr(lyr, "has_previous_state", False)
                    if isinstance(hps, dict):
                        if not hps.get(state_idx, False):
                            return None
                    elif not hps:
                        return None
                    st = getattr(lyr, "conv_states", None)
                    if isinstance(st, dict):
                        return st.get(state_idx)
                    if st is None or not getattr(lyr, "is_conv_states_initialized", False):
                        return None
                    return st

                cache.get_conv_state = get_conv_state
            if not hasattr(cache, "get_recurrent_state"):
                def get_recurrent_state(layer_idx, state_idx=0):
                    lyr = cache.layers[layer_idx]
                    box = getattr(lyr, "_fast_conv", None)
                    if box is not None:
                        if not box["has_prev"] or box["rec"] is None:
                            return None
                        return box["rec"]
                    hps = getattr(lyr, "has_previous_state", False)
                    if isinstance(hps, dict):
                        if not hps.get(state_idx, False):
                            return None
                    elif not hps:
                        return None
                    st = getattr(lyr, "recurrent_states", None)
                    if isinstance(st, dict):
                        return st.get(state_idx)
                    if st is None or not getattr(lyr, "is_recurrent_states_initialized", False):
                        return None
                    return st

                cache.get_recurrent_state = get_recurrent_state
            if not hasattr(cache, "is_recording_past"):
                def is_recording_past(layer_idx):
                    return bool(getattr(cache.layers[layer_idx], "record_past", False))

                cache.is_recording_past = is_recording_past
            orig_len = cache.get_seq_length

            def get_seq_length(layer_idx=0):
                n = orig_len(layer_idx)
                if torch.is_tensor(n):
                    return int(n.item())
                return n

            cache.get_seq_length = get_seq_length

        _stabilize_linear_cache(sc)
        _install_cache_api(sc)

        min_val = torch.finfo(embed_dtype).min
        mask_buf = torch.full((1, 1, 1, max_static_seq), min_val, device=dev, dtype=embed_dtype)
        fast_out_cls = getattr(mq, "Qwen3_5ModelOutputWithPast", None) or BaseModelOutputWithPast

        def _reset_sc_inplace():
            try:
                sc.reset()
                return
            except Exception:
                pass
            for lyr in sc.layers:
                box = getattr(lyr, "_fast_conv", None)
                if box is not None:
                    box["has_prev"] = False
                    if box["conv"] is not None:
                        box["conv"].zero_()
                    if box["rec"] is not None:
                        box["rec"].zero_()
                conv = getattr(lyr, "conv_states", None)
                if isinstance(conv, dict):
                    for st in conv.values():
                        if torch.is_tensor(st):
                            st.zero_()
                    rec = getattr(lyr, "recurrent_states", None)
                    if isinstance(rec, dict):
                        for st in rec.values():
                            if torch.is_tensor(st):
                                st.zero_()
                    hps = getattr(lyr, "has_previous_state", None)
                    if isinstance(hps, dict):
                        for k in list(hps):
                            hps[k] = False
                elif torch.is_tensor(conv):
                    conv.zero_()
                    rec = getattr(lyr, "recurrent_states", None)
                    if torch.is_tensor(rec):
                        rec.zero_()
                    if isinstance(getattr(lyr, "has_previous_state", None), bool):
                        lyr.has_previous_state = False
                if getattr(lyr, "is_initialized", False) and torch.is_tensor(getattr(lyr, "keys", None)):
                    lyr.keys.zero_()
                    lyr.values.zero_()
                cum = getattr(lyr, "cumulative_length", None)
                if torch.is_tensor(cum):
                    cum.zero_()
                elif isinstance(cum, int):
                    lyr.cumulative_length = 0

        orig_tm_forward = text_model.forward
        orig_lm_head_forward = mdl.lm_head.forward

        with torch.inference_mode():
            dummy_emb = torch.zeros((1, 4, t_cfg.hidden_size), device=dev, dtype=embed_dtype)
            orig_tm_forward(inputs_embeds=dummy_emb, past_key_values=sc, use_cache=True)

            static_attn_lyr = next(lyr for lyr in sc.layers if hasattr(lyr, "cumulative_length"))
            s_emb = torch.zeros((1, 1, t_cfg.hidden_size), device=dev, dtype=embed_dtype)
            s_pos3 = torch.zeros((3, 1, 1), device=dev, dtype=torch.long)
            mask_dict = {"full_attention": mask_buf, "linear_attention": None}

            def _decode_graph_step():
                pos_emb = text_model.rotary_emb(s_emb, s_pos3)
                h = s_emb
                for i, dec in enumerate(text_model.layers[: t_cfg.num_hidden_layers]):
                    h = dec(
                        h,
                        position_embeddings=pos_emb,
                        attention_mask=mask_dict[t_cfg.layer_types[i]],
                        past_key_values=sc,
                        use_cache=True,
                    )
                h_norm = text_model.norm(h)
                logits_out = orig_lm_head_forward(h_norm)
                return h_norm, logits_out

            side_stream = torch.cuda.Stream(device=dev)
            side_stream.wait_stream(torch.cuda.current_stream(device=dev))
            with torch.cuda.stream(side_stream):
                for _ in range(3):
                    s_out, s_logits = _decode_graph_step()
            torch.cuda.current_stream(device=dev).wait_stream(side_stream)

            cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg, stream=side_stream):
                s_out, s_logits = _decode_graph_step()
            torch.cuda.synchronize(device=dev)
            _reset_sc_inplace()

        s_out_ptr = s_out.data_ptr()
        s_out_numel = s_out.numel()

        def _fast_lm_head_forward(x):
            if x.is_cuda and x.data_ptr() == s_out_ptr and x.numel() == s_out_numel:
                return s_logits
            return orig_lm_head_forward(x)

        def _hooked_tm_forward(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            **kwargs,
        ):
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = text_model.embed_tokens(input_ids)
                input_ids = None
            if past_key_values is sc and use_cache and inputs_embeds is not None and inputs_embeds.shape[0] == 1:
                L = inputs_embeds.shape[1]
                if L > 1:
                    mask_buf.fill_(min_val)
                    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2 and attention_mask.shape[1] == L:
                        mask_buf[:, :, :, :L] = torch.where(attention_mask.view(1, 1, 1, L).bool(), 0.0, min_val)
                    else:
                        mask_buf[:, :, :, :L] = 0.0
                    return orig_tm_forward(
                        input_ids=None,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=sc,
                        inputs_embeds=inputs_embeds,
                        use_cache=True,
                        **kwargs,
                    )
                elif L == 1:
                    cur_len = static_attn_lyr.cumulative_length
                    if position_ids is None:
                        p3 = cur_len.reshape(1, 1, 1).expand(3, 1, 1)
                    else:
                        pos = position_ids
                        if pos.ndim == 1:
                            pos = pos.reshape(1, 1, -1)
                        elif pos.ndim == 2:
                            pos = pos.reshape(1, pos.shape[0], pos.shape[-1])
                        if pos.shape[0] == 4:
                            pos = pos[1:]
                        elif pos.shape[0] == 1:
                            pos = pos.expand(3, -1, -1)
                        elif pos.shape[0] != 3:
                            pos = pos.reshape(3, 1, -1)
                        p3 = pos[:, :1, -1:].reshape(3, 1, 1)
                    if p3.dtype != s_pos3.dtype or p3.device != s_pos3.device:
                        p3 = p3.to(device=s_pos3.device, dtype=s_pos3.dtype)
                    s_emb.copy_(inputs_embeds)
                    s_pos3.copy_(p3)
                    mask_buf.scatter_(3, cur_len.reshape(1, 1, 1, 1), 0.0)
                    cg.replay()
                    return fast_out_cls(last_hidden_state=s_out, past_key_values=sc)

            return orig_tm_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )

        text_model.forward = _hooked_tm_forward
        mdl.lm_head.forward = _fast_lm_head_forward

        mdl._cg_refs = (cg, sc, side_stream, s_emb, s_pos3, mask_buf, s_out, s_logits)
        if getattr(mdl, "generation_config", None) is not None:
            mdl.generation_config.disable_compile = True

        orig_generate = mdl.generate

        def _fast_generate(*args, **kwargs):
            kwargs.setdefault("disable_compile", True)
            inp_ids = kwargs.get("input_ids")
            if inp_ids is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
                inp_ids = args[0]
            amask = kwargs.get("attention_mask")
            no_pad = True if not isinstance(amask, torch.Tensor) else bool(amask.all().item())
            max_new = int(kwargs.get("max_new_tokens") or 2048)
            if (
                no_pad
                and kwargs.get("past_key_values") is None
                and inp_ids is not None
                and inp_ids.ndim == 2
                and inp_ids.shape[0] == 1
                and (inp_ids.shape[1] + max_new) <= (max_static_seq - 8)
            ):
                _reset_sc_inplace()
                kwargs["past_key_values"] = sc
            return orig_generate(*args, **kwargs)

        mdl.generate = _fast_generate
        print(
            f"[加速引擎] 全 64 层融合算子 + CUDAGraph 静态图引擎已就绪 "
            f"(CUDAGraph 静态加速 {max_static_seq} tokens / 动态最大上下文 {int(max_context_len or 32768)} tokens)！"
        )
    except Exception as opt_err:
        print(f"[加速引擎] CUDAGraph 未捕获，融合算子仍保留 ({type(opt_err).__name__}: {opt_err})")
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ==============================================================================
# 全局状态、算力计费账本与 /dev/shm 内存盘加速 (借鉴 comfy-colab-kit)
# ==============================================================================
BOOT_TIME = time.time()
LAST_ACTIVE_TIME = time.time()
try:
    IDLE_TIMEOUT_MINUTES = int(os.environ.get("EASYCOLAB_IDLE_TIMEOUT_MINUTES", "30").strip())
except Exception:
    IDLE_TIMEOUT_MINUTES = 30  # 默认闲置 30 分钟自动释放 Colab 虚拟机，可在 Mac 端调节

import gc

PIPELINE = None
LLM_BUNDLE = None  # {"backend": "transformers"|"gguf", ...}
LOADED_RUNTIMES: Dict[str, Dict[str, Any]] = {}  # 多模型显存共驻表: model_id -> runtime dict
SELECTED_MODEL_ID = "minimax_h3"
LOADED_MODEL_ID = None
CURRENT_MODEL_MODE = None
QWEN_CONTEXT_LENGTH = 32768  # 用户可自定义上下文长度 (4K ~ 256K = 262144 tokens)
INITIAL_API_KEY = secrets.token_urlsafe(32)
API_AUTH_KEY = INITIAL_API_KEY
API_KEY_LOCK = threading.Lock()
API_REQUEST_COUNT = 0
GPU_LOCK = threading.Lock()
_ACTIVE_API_STREAMS = 0
_ACTIVE_API_STREAMS_LOCK = threading.Lock()
ABORT_GENERATION_EVENT = threading.Event()
JOBS: Dict[str, Dict[str, Any]] = {}
_JOB_CANCEL_EVENTS: Dict[str, threading.Event] = {}


class GenerationAbortedError(RuntimeError):
    """用户主动点击停止推理时立即中断 DiffSynth / Transformers GPU 循环的受控异常"""
    pass


def _clamp_context_length(val: Optional[int]) -> int:
    try:
        v = int(val) if val is not None else QWEN_CONTEXT_LENGTH
    except Exception:
        v = QWEN_CONTEXT_LENGTH
    if v <= 0:
        v = QWEN_CONTEXT_LENGTH
    return max(4096, min(262144, v))


def _is_llm_model_id(model_id: Optional[str]) -> bool:
    if not model_id:
        return False
    return MODEL_CATALOG.get(model_id, {}).get("modality") == "llm_vlm"


def _stop_gguf_server_bundle(bundle: Optional[Dict[str, Any]]):
    if not bundle or bundle.get("backend") != "gguf":
        return
    proc = bundle.pop("server_proc", None)
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        except Exception:
            pass
        time.sleep(0.25)


def _cleanup_all_gguf_servers():
    for rt in list(LOADED_RUNTIMES.values()):
        _stop_gguf_server_bundle(rt.get("llm_bundle"))
        _stop_h3_image_server(rt)
    _stop_gguf_server_bundle(LLM_BUNDLE)


atexit.register(_cleanup_all_gguf_servers)
if os.path.exists("/content"):
    subprocess.run(["pkill", "-f", "llama-server"], check=False)


def _make_abort_stopping_criteria(local_cancel: Optional[threading.Event] = None):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _AbortCriteria(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs) -> bool:
            if ABORT_GENERATION_EVENT.is_set():
                return True
            if local_cancel is not None and local_cancel.is_set():
                return True
            return False

    return StoppingCriteriaList([_AbortCriteria()])


def _make_timed_token_streamer(tokenizer, timeout: float = 120.0):
    from transformers import TextIteratorStreamer

    class _TimedStreamer(TextIteratorStreamer):
        def __init__(self, tok, **kw):
            super().__init__(tok, **kw)
            self.generated_tokens = 0
            self.first_token_ts: Optional[float] = None
            self._saw_prompt = False

        def put(self, value):
            now = time.time()
            if self.skip_prompt and not self._saw_prompt:
                self._saw_prompt = True
            else:
                n = int(value.numel()) if hasattr(value, "numel") else 1
                self.generated_tokens += max(1, n)
                if self.first_token_ts is None:
                    self.first_token_ts = now
            super().put(value)

        @property
        def tok_per_sec(self) -> float:
            if self.first_token_ts is None or self.generated_tokens <= 1:
                return 0.0
            dt = max(0.05, time.time() - self.first_token_ts)
            return round((self.generated_tokens - 1) / dt, 1)

    return _TimedStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=timeout,
    )

MODEL_DOWNLOAD_STATE: Dict[str, Any] = {
    "is_downloading": False,
    "is_loading_vram": False,
    "progress": 0.0,
    "message": "等待检查模型状态...",
}


def _get_system_ram_gb() -> Tuple[float, float]:
    """返回 (系统总内存 GB, 当前可用内存 GB)"""
    try:
        mem_total, mem_avail = 0.0, 0.0
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) / (1024 ** 2)
                elif line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1]) / (1024 ** 2)
        return round(mem_total, 1), round(mem_avail, 1)
    except Exception:
        return 0.0, 0.0


def setup_colab_ramdisk() -> Optional[str]:
    """
    借鉴 comfy-colab-kit/packages/colab/setup.py：
    在 Colab 大内存实例 (G4 176GB / A100 83GB / L4 53GB RAM) 上自动扩容 /dev/shm (tmpfs 内存盘，读写 2.4GB/s+)，
    为适配容量的模型提供零延迟 mmap 预热加速。
    """
    if not os.path.exists("/dev/shm"):
        return None
    total_ram_gb, avail_ram_gb = _get_system_ram_gb()
    if total_ram_gb >= 48.0:
        # 避免 DiffSynth DiskMap 默认每 2GB (10^9 参数) 反复关闭并重开全部 14 个 safetensors 分片句柄
        os.environ.setdefault("DIFFSYNTH_DISK_MAP_BUFFER_SIZE", str(80 * 10**9))
    if total_ram_gb < 40.0 or avail_ram_gb < 25.0:
        return None
    try:
        cur_shm_total_gb = shutil.disk_usage("/dev/shm").total / (1024 ** 3)
        target_gb = min(88, max(30, int(total_ram_gb * 0.50)))
        if cur_shm_total_gb + 1.0 < target_gb:
            subprocess.run(
                ["mount", "-o", f"remount,size={target_gb}G", "/dev/shm"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        shm_stat = shutil.disk_usage("/dev/shm")
        shm_free_gb = shm_stat.free / (1024 ** 3)
        if shm_free_gb >= 22.5:
            ram_root = "/dev/shm/EasyColab_Models"
            os.makedirs(ram_root, exist_ok=True)
            print(f"⚡ [RAMDisk 内存盘加速已开启] /dev/shm 可用空间 {shm_free_gb:.1f} GB (2.4GB/s 零延迟读写)")
            return ram_root
    except Exception:
        pass
    return None


RAM_ROOT_DIR = setup_colab_ramdisk()

def _native_drive_mounted() -> bool:
    """Colab 原生挂载的 Google Drive 才算「已挂载云盘」。唤醒脚本有时会用 rclone mount 把云盘挂到同一位置，
    往这种挂载里写几十 GB 的权重要先在本地写缓存里存满一份，所以按没挂载处理，改走本地镜像 + 定时同步"""
    if not os.path.exists("/content/drive/MyDrive"):
        return False
    if not (os.path.ismount("/content/drive") or os.path.ismount("/content/drive/MyDrive")):
        return False
    try:
        with open("/proc/mounts", "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                parts = ln.split()
                if len(parts) >= 3 and "rclone" in parts[2] and parts[1].startswith("/content/drive"):
                    return False
    except Exception:
        pass
    return True


# 如果挂载了 Google Drive，优先将模型持久化保存在云盘 EasyColab_Models（并自动兼容读取历史 Aevix_Models / MiniMax_H3_Models 目录）
DRIVE_MOUNTED = _native_drive_mounted()


def _has_rclone_drive_remote() -> bool:
    rc_path = os.path.expanduser("~/.config/rclone/rclone.conf")
    if not os.path.exists(rc_path) or not shutil.which("rclone"):
        return False
    try:
        with open(rc_path, "r", encoding="utf-8", errors="ignore") as f:
            return "[easycolab_gdrive]" in f.read()
    except Exception:
        return False


COMFYUI_LOCK = threading.Lock()
COMFYUI_STATE: Dict[str, Any] = {
    "is_running": False,
    "is_starting": False,
    "is_syncing_drive": False,
    "is_downloading_weight": False,
    "installing_pack_id": "",
    "progress": 0.0,
    "status_message": "尚未启动（支持一键拉起 ComfyUI + Google Drive 自动软链持久化）",
    "public_url": "",
    "local_port": COMFY_BACKEND,
    "queue_running": 0,
    "queue_pending": 0,
    "persistence_mode": "drive_symlink" if DRIVE_MOUNTED else ("rclone_drive_sync" if _has_rclone_drive_remote() else "local_only"),
    "persistence_label": "Google Drive 实时软链挂载" if DRIVE_MOUNTED else ("Google Drive 云盘自动持久化" if _has_rclone_drive_remote() else "实例本地高速存储"),
    "drive_persisted": bool(DRIVE_MOUNTED or _has_rclone_drive_remote()),
    "custom_nodes_count": 0,
    "models_count": 0,
    "workflows_count": 0,
    "last_sync_ts": 0.0,
    "last_history_count": 0,
    "comfy_proc": None,
    "tunnel_proc": None,
}
MODEL_BASE_ROOT = (
    "/content/drive/MyDrive/EasyColab_Models"
    if DRIVE_MOUNTED
    else os.path.join(os.getcwd(), "models")
)
LEGACY_DRIVE_MODEL_ROOT = "/content/drive/MyDrive/MiniMax_H3_Models"
LEGACY_DRIVE_MODEL_ROOTS = [
    "/content/drive/MyDrive/Aevix_Models",
    "/content/drive/MyDrive/MiniMax_H3_Models",
]
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = MODEL_BASE_ROOT

MINIMAX_H3_BF16_FILES = (
    [
        (f"FL2VA/transformer/model-{i:05d}-of-00013.safetensors", f"DiT 音视频主干 BF16 {i}/13")
        for i in range(1, 14)
    ]
    + [
        (f"FL2VA/text_encoder/model-{i:05d}-of-00014.safetensors", f"多模态文本编码器 BF16 {i}/14")
        for i in range(1, 15)
    ]
    + [
        ("FL2VA/video_vae/source/model.safetensors", "3D 视频 VAE 解码器 BF16 (约 9.7 GB)"),
        ("FL2VA/audio_vae/model.safetensors", "32kHz 立体声 VAE 解码器 BF16 (约 0.56 GB)"),
    ]
)

# MiniMax-H3 的两个官方分区只差 DiT 主干：Ref2VA（参考图/视频/音频生音视频）的文本编码器、Processor、视频与音频 VAE
# 和 FL2VA 逐字节相同，直接复用 FL2VA 目录里的那份。Ref2VA 主干（13 分片约 66 GB）不算进必备文件，第一次用参考生成时才下载。
MINIMAX_H3_DIT_FILES = {
    variant: [f"{variant}/transformer/model-{i:05d}-of-00013.safetensors" for i in range(1, 14)]
    for variant in ("FL2VA", "Ref2VA")
}
# 官方 README 的 Ref2VA 输入上限：图 9 张、视频 3 段、音频 3 段，总数 12；每段视频/音频 2~15 秒，同类合计不超过 15 秒
H3_REF2VA_LIMITS = {"image": 9, "video": 3, "audio": 3, "total": 12, "min_s": 2.0, "max_s": 15.0}

# MiniMax-H3-Image（官方 BF16 图像模型）：DiffSynth 暂无对应管线，由独立虚拟环境里的 SGLang 服务加载（见 _start_h3_image_server）。
# 文本编码器的 14 个分片与 MiniMax-H3 逐字节相同，两个模型谁先下载都只存一份（见 _find_existing_model_file）。
MINIMAX_H3_IMAGE_FILES = (
    [
        (f"transformer/model-{i:05d}-of-00012.safetensors", f"图像 DiT 主干 BF16 {i}/12")
        for i in range(1, 13)
    ]
    + [
        (f"text_encoder/model-{i:05d}-of-00014.safetensors", f"多模态文本编码器 BF16 {i}/14 (与 MiniMax-H3 共用)")
        for i in range(1, 15)
    ]
    + [
        ("vae/diffusion_pytorch_model.safetensors", "图像 VAE (约 2.2 GB)"),
        ("pdd_fused_heads.safetensors", "6 步蒸馏输出头 (12 MB)"),
    ]
)
H3_IMAGE_SGLANG_COMMIT = "a93e89fd4a4eefb52c3ce73a609317d716d2a6ab"  # 已在 G4 上实测可用的 SGLang 提交
H3_IMAGE_SGLANG_DIR = os.path.join(os.getcwd(), "sglang_h3image")
H3_IMAGE_SGLANG_PORT = 30117
H3_IMAGE_MAX_REFERENCES = 16  # 模型卡写明的参考图上限

# ==============================================================================
# 开源旗舰多模型注册中心 (Hugging Face Top Models 2026 · 全量免审核 Gated=False)
# ==============================================================================
# Qwen3.8-27B GGUF 对话模型改用 huihui-ai 的 abliterated 版（源自同一份 Unsloth UD-Q4_K_M 量化）；MTP 头未被改动，仍取自 Unsloth。
QWEN_GGUF_MAIN_FILE = "Huihui-Qwen3.8-27B-abliterated-UD-DW-Q4_K_M.gguf"
QWEN_GGUF_MMPROJ_FILE = "mmproj-model-bf16.gguf"

MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "minimax_h3": {
        "id": "minimax_h3",
        "name": "MiniMax-H3 (33B)",
        "repo_id": "MiniMaxAI/MiniMax-H3",
        "modality": "video_audio",
        "category": "全模态音视频生成 · BF16 原生满血 · 32kHz 立体声",
        "approx_size_gb": 134.1,
        "files": MINIMAX_H3_BF16_FILES,
    },
    "minimax_h3_image": {
        "id": "minimax_h3_image",
        "name": "MiniMax-H3-Image",
        "repo_id": "MiniMaxAI/MiniMax-H3-Image",
        "modality": "image",
        "pipeline_arch": "minimax_h3_image",
        "category": "官方 BF16 · 6 步 4K 文生图与参考图编辑 · SGLang 引擎",
        "approx_size_gb": 126.0,
        "files": MINIMAX_H3_IMAGE_FILES,
    },
    "qwen38_27b": {
        "id": "qwen38_27b",
        "name": "Qwen3.8-27B",
        "repo_id": "Qwen/Qwen3.8-27B",
        "modality": "llm_vlm",
        "category": "视觉理解与深度推理大模型 · BF16 原生满血",
        "approx_size_gb": 51.8,
        "files": [
            (f"model-{i:05d}-of-00018.safetensors", f"Qwen3.8-27B BF16 权重分片 {i}/18")
            for i in range(1, 19)
        ],
    },
    "qwen38_27b_gguf": {
        "id": "qwen38_27b_gguf",
        "name": "Qwen3.8-27B-GGUF",
        "repo_id": "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF",
        "modality": "llm_vlm",
        "category": "Huihui abliterated UD-DW-Q4_K_M + MTP 投机加速 · 支持 256K 上下文",
        "approx_size_gb": 17.5,
        "files": [
            (QWEN_GGUF_MAIN_FILE, "Huihui abliterated UD-DW-Q4_K_M 主模型权重 (约 16.6 GB)"),
            (QWEN_GGUF_MMPROJ_FILE, "多模态视觉投影器 mmproj (约 0.93 GB)"),
        ],
    },
    "qwen_image_21": {
        "id": "qwen_image_21",
        "name": "Qwen-Image-2.1",
        "repo_id": "Qwen/Qwen-Image-2.1",
        "modality": "image",
        "category": "旗舰文生图与多图指令修图",
        "approx_size_gb": 30.9,
        "files": [
            ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "DiT 图像主干 1/2"),
            ("transformer/diffusion_pytorch_model-00002-of-00002.safetensors", "DiT 图像主干 2/2"),
            ("text_encoder/model-00001-of-00004.safetensors", "多模态编码器 1/4"),
            ("text_encoder/model-00002-of-00004.safetensors", "多模态编码器 2/4"),
            ("text_encoder/model-00003-of-00004.safetensors", "多模态编码器 3/4"),
            ("text_encoder/model-00004-of-00004.safetensors", "多模态编码器 4/4"),
            ("vae/diffusion_pytorch_model.safetensors", "高保真图像 VAE"),
        ],
    },
    "z_image_turbo": {
        "id": "z_image_turbo",
        "name": "Z-Image-Turbo",
        "repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "modality": "image",
        "pipeline_arch": "z_image_turbo",
        "category": "8 步极速超清写实与中英文字生图",
        "approx_size_gb": 30.6,
        "files": [
            ("transformer/diffusion_pytorch_model-00001-of-00003.safetensors", "Z-Image DiT 主干 1/3"),
            ("transformer/diffusion_pytorch_model-00002-of-00003.safetensors", "Z-Image DiT 主干 2/3"),
            ("transformer/diffusion_pytorch_model-00003-of-00003.safetensors", "Z-Image DiT 主干 3/3"),
            ("text_encoder/model-00001-of-00003.safetensors", "文本编码器 1/3"),
            ("text_encoder/model-00002-of-00003.safetensors", "文本编码器 2/3"),
            ("text_encoder/model-00003-of-00003.safetensors", "文本编码器 3/3"),
            ("vae/diffusion_pytorch_model.safetensors", "图像 VAE 解码器"),
        ],
    },
}

BUILTIN_MODEL_IDS = set(MODEL_CATALOG.keys())
CUSTOM_MODELS_LOCK = threading.Lock()


def _get_custom_models_json_paths() -> List[str]:
    paths = [os.path.join(MODEL_BASE_ROOT, "custom_models.json")]
    local_p = os.path.join(os.getcwd(), "models", "custom_models.json")
    if os.path.abspath(local_p) != os.path.abspath(paths[0]):
        paths.append(local_p)
    return paths


def _make_custom_model_id(repo_id: str, include_pattern: str = "") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", repo_id.strip().lower()).strip("_")
    if include_pattern.strip():
        pat_slug = re.sub(r"[^a-zA-Z0-9]+", "_", include_pattern.strip().lower()).strip("_")
        if pat_slug and pat_slug not in ("gguf", "safetensors"):
            slug = f"{slug}_{pat_slug[:24]}"
    return f"custom_{slug[:64]}"


def _normalize_hf_repo_and_pattern(raw_repo: str, raw_pattern: str = "") -> Tuple[str, str]:
    s = (raw_repo or "").strip()
    pat = (raw_pattern or "").strip()
    s = re.sub(r"^https?://(?:www\.)?(?:huggingface\.co|hf\.co|hf-mirror\.com)/", "", s, flags=re.IGNORECASE)
    s = s.strip("/")
    for marker in ("/blob/", "/resolve/", "/raw/"):
        if marker in s:
            left, right = s.split(marker, 1)
            parts = right.split("/", 1)
            if len(parts) == 2 and parts[1] and not pat:
                pat = parts[1].strip()
            s = left.strip("/")
            break
    if "/tree/" in s:
        s = s.split("/tree/", 1)[0].strip("/")
    tokens = [p for p in s.split("/") if p]
    if len(tokens) >= 2:
        repo_id = f"{tokens[0]}/{tokens[1]}"
        if len(tokens) > 2 and not pat:
            pat = "/".join(tokens[2:])
        return repo_id, pat
    return s, pat


def _load_persisted_custom_models():
    for path in _get_custom_models_json_paths():
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or not item.get("id") or not item.get("repo_id"):
                    continue
                mid = str(item["id"])
                if mid in BUILTIN_MODEL_IDS:
                    continue
                raw_files = item.get("files") or []
                norm_files: List[Tuple[str, str]] = []
                for rf in raw_files:
                    if isinstance(rf, (list, tuple)) and len(rf) >= 2:
                        norm_files.append((str(rf[0]), str(rf[1])))
                    elif isinstance(rf, dict) and rf.get("name"):
                        norm_files.append((str(rf["name"]), str(rf.get("desc") or rf["name"])))
                if not norm_files:
                    continue
                MODEL_CATALOG[mid] = {
                    "id": mid,
                    "name": str(item.get("name") or item["repo_id"].split("/")[-1]),
                    "repo_id": str(item["repo_id"]),
                    "include_pattern": str(item.get("include_pattern") or ""),
                    "modality": str(item.get("modality") or "llm_vlm"),
                    "pipeline_arch": str(item.get("pipeline_arch") or "gguf"),
                    "category": str(item.get("category") or "自定义 HuggingFace 模型"),
                    "approx_size_gb": float(item.get("approx_size_gb") or 0.0),
                    "files": norm_files,
                    "is_custom": True,
                }
            break
        except Exception as e:
            print(f"[自定义模型] 读取 {path} 跳过: {e}")


def _save_persisted_custom_models():
    custom_list = []
    for mid, spec in MODEL_CATALOG.items():
        if mid in BUILTIN_MODEL_IDS or not spec.get("is_custom"):
            continue
        custom_list.append({
            "id": spec["id"],
            "name": spec["name"],
            "repo_id": spec["repo_id"],
            "include_pattern": spec.get("include_pattern", ""),
            "modality": spec.get("modality", "llm_vlm"),
            "pipeline_arch": spec.get("pipeline_arch", "gguf"),
            "category": spec.get("category", "自定义 HuggingFace 模型"),
            "approx_size_gb": round(float(spec.get("approx_size_gb") or 0.0), 2),
            "files": [[fn, desc] for fn, desc in (spec.get("files") or [])],
            "is_custom": True,
        })
    for path in _get_custom_models_json_paths():
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_p = path + ".tmp"
            with open(tmp_p, "w", encoding="utf-8") as wf:
                json.dump(custom_list, wf, ensure_ascii=False, indent=2)
            os.replace(tmp_p, path)
        except Exception as e:
            print(f"[自定义模型] 写入 {path} 跳过: {e}")


def _get_custom_models_public_list() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for mid, spec in MODEL_CATALOG.items():
        if mid in BUILTIN_MODEL_IDS or not spec.get("is_custom"):
            continue
        out.append({
            "id": spec["id"],
            "name": spec["name"],
            "repo_id": spec["repo_id"],
            "include_pattern": spec.get("include_pattern", ""),
            "modality": spec.get("modality", "llm_vlm"),
            "pipeline_arch": spec.get("pipeline_arch", "gguf"),
            "category": spec.get("category", "自定义 HuggingFace 模型"),
            "approx_size_gb": round(float(spec.get("approx_size_gb") or 0.0), 2),
            "files": [
                {"name": fn, "desc": desc, "exists": False, "size_gb": 0.0}
                for fn, desc in (spec.get("files") or [])
            ],
            "is_custom": True,
        })
    return out


_HF_TRENDING_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_HF_TRENDING_LOCK = threading.Lock()


def _clean_hf_readme_excerpt(raw_md: str) -> str:
    body = raw_md or ""
    if body.startswith("---"):
        end_idx = body.find("\n---", 3)
        if end_idx != -1:
            body = body[end_idx + 4 :]
    body = re.sub(r"<!--[\s\S]*?-->", "", body)
    body = re.sub(r"<a[^>]*>\s*<img[^>]*>\s*</a>", "", body, flags=re.IGNORECASE)
    body = re.sub(r"<img[^>]*>", "", body, flags=re.IGNORECASE)
    body = re.sub(r"<h[1-4][^>]*>(.*?)</h[1-4]>", r"\n### \1\n", body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"<[^>]+>", "", body)
    body = re.sub(r"\[!\[[^\]]*\]\([^\)]*\)\]\([^\)]*\)", "", body)
    body = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", body)
    lines_clean = []
    total_chars = 0
    for ln in body.splitlines():
        s_ln = ln.strip()
        if not s_ln or s_ln.startswith(("```", "|---", "| ---")):
            continue
        lines_clean.append(s_ln)
        total_chars += len(s_ln) + 1
        if total_chars >= 1400:
            break
    out = "\n".join(lines_clean).strip()
    # Clean any trailing incomplete markdown link if truncated
    out = re.sub(r"\[([^\]]+)\]\([^\)\n]*$", r"\1", out)
    return out


_UNSUPPORTED_HF_PIPELINE_TAGS = {
    "image-to-3d",
    "text-to-3d",
    "image-segmentation",
    "object-detection",
    "depth-estimation",
    "mask-generation",
    "keypoint-detection",
    "robotics",
    "reinforcement-learning",
    "tabular-classification",
    "tabular-regression",
    "time-series-forecasting",
    "audio-classification",
    "automatic-speech-recognition",
    "text-to-speech",
    "audio-to-audio",
    "feature-extraction",
    "sentence-similarity",
    "token-classification",
    "text-classification",
    "zero-shot-image-classification",
    "zero-shot-classification",
    "fill-mask",
    "question-answering",
    "table-question-answering",
    "video-classification",
}

_SUPPORTED_TRENDING_PIPELINE_TAGS = {
    "text-generation",
    "image-text-to-text",
    "visual-question-answering",
    "any-to-any",
    "text-to-image",
    "image-to-image",
    "text-to-video",
    "image-to-video",
}


def _analyze_hf_model_sizing(
    item: Dict[str, Any], is_gguf: bool, fallback_storage_gb: float = 0.0
) -> Dict[str, Any]:
    """
    提取 HuggingFace 模型参数量 (safetensors.total / gguf.total / 名称正则回退)，
    并根据精度分布与量化状态估算 Colab 单卡显存占用及是否适合部署。
    """
    rid = str(item.get("id") or item.get("modelId") or "")
    ptag = str(item.get("pipeline_tag") or "").strip().lower()
    st = item.get("safetensors") if isinstance(item.get("safetensors"), dict) else {}
    gg = item.get("gguf") if isinstance(item.get("gguf"), dict) else {}
    total_params = int(st.get("total") or gg.get("total") or 0)
    if total_params <= 0:
        m = re.search(r"(?:^|[-_/])(\d+(?:\.\d+)?)\s*([bBmMtT])(?:[-_/\b]|$)", rid)
        if m:
            val = float(m.group(1))
            unit = m.group(2).upper()
            if unit == "T":
                total_params = int(val * 1e12)
            elif unit == "B":
                total_params = int(val * 1e9)
            elif unit == "M":
                total_params = int(val * 1e6)

    if total_params >= 1e12:
        param_label = f"{total_params / 1e12:.1f}T"
    elif total_params >= 1e9:
        b = total_params / 1e9
        param_label = f"{b:.0f}B" if b >= 20 else f"{b:.1f}B".replace(".0B", "B")
    elif total_params >= 1e6:
        param_label = f"{total_params / 1e6:.0f}M"
    else:
        param_label = ""

    params_b = total_params / 1e9 if total_params > 0 else 0.0
    if is_gguf:
        est_vram_gb = round(params_b * 0.62 + 1.2, 1) if params_b > 0 else 0.0
    else:
        pmap = st.get("parameters") if isinstance(st.get("parameters"), dict) else {}
        if pmap:
            bytes_total = 0.0
            for dtype, cnt in pmap.items():
                dt = str(dtype).upper()
                c = float(cnt or 0)
                if "32" in dt:
                    bytes_total += c * 4.0
                elif "16" in dt:
                    bytes_total += c * 2.0
                elif "8" in dt:
                    bytes_total += c * 1.0
                elif "4" in dt:
                    bytes_total += c * 0.55
                else:
                    bytes_total += c * 2.0
            est_vram_gb = round((bytes_total / 1e9) * 1.12 + 1.2, 1)
        elif params_b > 0:
            est_vram_gb = round(params_b * 2.1 + 1.2, 1)
        elif fallback_storage_gb > 0:
            est_vram_gb = round(fallback_storage_gb * 1.12 + 1.0, 1)
        else:
            est_vram_gb = 0.0

    vram_disp = max(1, int(round(est_vram_gb)))
    if not is_gguf and ptag in _UNSUPPORTED_HF_PIPELINE_TAGS:
        status = "too_large"
        zh = f"非内置管线 ({ptag} · 需专用SDK)"
        en = f"Custom Pipeline ({ptag})"
    elif est_vram_gb <= 0:
        status = "unknown"
        zh = "显存待探测"
        en = "VRAM TBD"
    elif est_vram_gb <= 14.2:
        status = "great"
        zh = f"适合 · T4(15G)可跑 (~{vram_disp}G)"
        en = f"Fit: T4 15G (~{vram_disp}G)"
    elif est_vram_gb <= 22.5:
        status = "good"
        zh = f"适合 · 需L4(24G) (~{vram_disp}G)"
        en = f"Fit: L4 24G (~{vram_disp}G)"
    elif est_vram_gb <= 39.0:
        status = "a100"
        zh = f"可部署 · 需A100(40G) (~{vram_disp}G)"
        en = f"Fit: A100 40G (~{vram_disp}G)"
    elif est_vram_gb <= 78.0:
        status = "heavy"
        zh = f"需A100(80G) (~{vram_disp}G)"
        en = f"Needs A100 80G (~{vram_disp}G)"
    elif is_gguf and params_b <= 240:
        status = "heavy"
        zh = f"需A100(80G)+CPU卸载 (~{vram_disp}G)"
        en = f"A100 80G + Offload (~{vram_disp}G)"
    else:
        status = "too_large"
        zh = f"超单卡上限 (~{vram_disp}G · 建议搜GGUF)"
        en = f"Too Large (~{vram_disp}G · Try GGUF)"

    return {
        "param_count": total_params,
        "param_label": param_label,
        "est_vram_gb": est_vram_gb,
        "deploy_status": status,
        "deploy_label_zh": zh,
        "deploy_label_en": en,
    }


def fetch_hf_trending_models_core(
    category: str = "all",
    limit: int = 12,
    hf_token: str = "",
    force_refresh: bool = False,
    search_query: str = "",
) -> List[Dict[str, Any]]:
    """
    实时从 HuggingFace Hub API (带 hf-mirror.com 自动容灾回退与内存缓存) 拉取最新热门趋势或关键词搜索模型，
    同时返回模型参数量 (param_label) 与 Colab 部署适用性评估 (deploy_status / deploy_label_zh / est_vram_gb)。
    """
    cat_key = (category or "all").strip().lower()
    sq = (search_query or "").strip()
    lim = max(4, min(24, int(limit or 12)))
    cache_key = f"{cat_key}:{lim}:{sq.lower()}"
    now = time.time()
    if not force_refresh:
        with _HF_TRENDING_LOCK:
            cached = _HF_TRENDING_CACHE.get(cache_key)
            if cached and (now - cached[0]) < 180.0 and cached[1]:
                return list(cached[1])

    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    headers = {"User-Agent": "EasyColab/2.0", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    expand_qs = (
        "&expand[]=safetensors&expand[]=gguf&expand[]=likes"
        "&expand[]=downloads&expand[]=pipeline_tag&expand[]=tags&expand[]=gated"
    )
    search_qs = f"&search={urllib.parse.quote(sq)}" if sq else ""

    query_paths: List[str] = []
    if cat_key == "gguf":
        query_paths.append(f"/api/models?sort=trendingScore&direction=-1&filter=gguf&limit={lim * 2}{search_qs}{expand_qs}")
    elif cat_key in ("llm", "llm_vlm"):
        query_paths.append(f"/api/models?sort=trendingScore&direction=-1&pipeline_tag=text-generation&limit={lim * 2}{search_qs}{expand_qs}")
        if not sq:
            query_paths.append(f"/api/models?sort=trendingScore&direction=-1&pipeline_tag=image-text-to-text&limit={lim}{expand_qs}")
    elif cat_key in ("image", "video", "image_video"):
        query_paths.append(f"/api/models?sort=trendingScore&direction=-1&pipeline_tag=text-to-image&limit={lim}{search_qs}{expand_qs}")
        query_paths.append(f"/api/models?sort=trendingScore&direction=-1&pipeline_tag=image-to-video&limit={lim}{search_qs}{expand_qs}")
        if not sq:
            query_paths.append(f"/api/models?sort=trendingScore&direction=-1&pipeline_tag=text-to-video&limit={lim}{expand_qs}")
    else:
        query_paths.append(f"/api/models?sort=trendingScore&direction=-1&limit={lim * 2}{search_qs}{expand_qs}")

    raw_items: List[Dict[str, Any]] = []
    seen_ids = set()

    for qpath in query_paths:
        fetched = None
        for host in ("https://huggingface.co", "https://hf-mirror.com"):
            try:
                req = urllib.request.Request(f"{host}{qpath}", headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if isinstance(data, list):
                        fetched = data
                        break
            except Exception:
                continue
        if fetched:
            for item in fetched:
                if not isinstance(item, dict):
                    continue
                rid = str(item.get("id") or item.get("modelId") or "").strip()
                if not rid or "/" not in rid or rid in seen_ids:
                    continue
                seen_ids.add(rid)
                raw_items.append(item)

    results: List[Dict[str, Any]] = []
    for item in raw_items:
        rid = str(item.get("id") or item.get("modelId") or "").strip()
        rid_low = rid.lower()
        ptag = str(item.get("pipeline_tag") or "").strip().lower()
        tags = [str(t).lower() for t in (item.get("tags") or []) if isinstance(t, str)]
        is_gguf = ("gguf" in rid_low) or ("gguf" in tags)
        gated_raw = item.get("gated")
        is_gated = bool(gated_raw) and str(gated_raw).lower() not in ("false", "0", "none")

        if "video" in ptag or "text-to-video" in tags or "image-to-video" in tags:
            modality = "video_audio"
        elif "text-to-image" in ptag or "image-to-image" in ptag or ("diffusers" in tags and not is_gguf):
            modality = "image"
        else:
            modality = "llm_vlm"

        if not sq and cat_key == "all" and not is_gguf:
            if ptag in _UNSUPPORTED_HF_PIPELINE_TAGS:
                continue
            if ptag and ptag not in _SUPPORTED_TRENDING_PIPELINE_TAGS:
                continue
            if not ptag and not any(t in tags for t in ("transformers", "diffusers", "text-generation", "conversational")):
                continue

        sizing = _analyze_hf_model_sizing(item, is_gguf)
        suggested_pat = "*Q4_K_M*.gguf" if is_gguf else ""
        results.append({
            "repo_id": rid,
            "name": rid.split("/")[-1],
            "pipeline_tag": ptag or ("gguf" if is_gguf else "text-generation"),
            "modality": modality,
            "is_gguf": is_gguf,
            "is_gated": is_gated,
            "suggested_pattern": suggested_pat,
            "likes": int(item.get("likes") or 0),
            "downloads": int(item.get("downloads") or 0),
            "param_count": sizing["param_count"],
            "param_label": sizing["param_label"],
            "est_vram_gb": sizing["est_vram_gb"],
            "deploy_status": sizing["deploy_status"],
            "deploy_label_zh": sizing["deploy_label_zh"],
            "deploy_label_en": sizing["deploy_label_en"],
        })
        if len(results) >= lim:
            break

    if results:
        with _HF_TRENDING_LOCK:
            _HF_TRENDING_CACHE[cache_key] = (now, list(results))
    return results


def fetch_hf_model_detail_core(repo_id: str, hf_token: str = "") -> Dict[str, Any]:
    """
    获取单个 HuggingFace 模型仓库的完整详情：
    参数量、架构、上下文长度、开源协议、基座模型、各权重/量化分片精确体积与单文件 Colab 卡型适配评估，以及 README 摘要。
    """
    clean_id = _normalize_hf_repo_id(repo_id)
    if not clean_id or "/" not in clean_id:
        return {"ok": False, "error": "Invalid HuggingFace Repo ID"}

    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    headers = {"User-Agent": "EasyColab/2.0", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    meta: Optional[Dict[str, Any]] = None
    for host in ("https://huggingface.co", "https://hf-mirror.com"):
        try:
            url = f"{host}/api/models/{clean_id}?blobs=true"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, dict) and (data.get("id") or data.get("modelId")):
                    meta = data
                    break
        except Exception:
            continue

    if not meta:
        return {"ok": False, "repo_id": clean_id, "error": f"无法获取仓库 {clean_id} 详情，请检查名称或填写 HF Token"}

    rid = str(meta.get("id") or meta.get("modelId") or clean_id).strip()
    rid_low = rid.lower()
    ptag = str(meta.get("pipeline_tag") or "").strip().lower()
    tags = [str(t) for t in (meta.get("tags") or []) if isinstance(t, str)]
    tags_low = [t.lower() for t in tags]
    is_gguf = ("gguf" in rid_low) or ("gguf" in tags_low) or bool(meta.get("gguf"))
    gated_raw = meta.get("gated")
    is_gated = bool(gated_raw) and str(gated_raw).lower() not in ("false", "0", "none")

    if "video" in ptag or "text-to-video" in tags_low or "image-to-video" in tags_low:
        modality = "video_audio"
    elif "text-to-image" in ptag or "image-to-image" in ptag or ("diffusers" in tags_low and not is_gguf):
        modality = "image"
    else:
        modality = "llm_vlm"

    card_data = meta.get("cardData") if isinstance(meta.get("cardData"), dict) else {}
    config_data = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    gguf_data = meta.get("gguf") if isinstance(meta.get("gguf"), dict) else {}

    lic = card_data.get("license") or ""
    if isinstance(lic, list):
        lic = ", ".join(str(x) for x in lic)
    lic = str(lic).strip() or "未标注 (See Model Card)"

    base_model = card_data.get("base_model") or ""
    if isinstance(base_model, list):
        base_model = ", ".join(str(x) for x in base_model[:2])
    base_model = str(base_model).strip()

    archs = config_data.get("architectures")
    arch_str = ""
    if isinstance(archs, list) and archs:
        arch_str = str(archs[0])
    elif gguf_data.get("architecture"):
        arch_str = str(gguf_data.get("architecture"))
    elif config_data.get("model_type"):
        arch_str = str(config_data.get("model_type"))

    ctx_len = int(gguf_data.get("context_length") or 0)
    langs = card_data.get("language") or []
    if isinstance(langs, str):
        langs = [langs]
    langs = [str(x) for x in langs if isinstance(x, str)][:8]

    used_storage_bytes = float(meta.get("usedStorage") or 0.0)
    used_storage_gb = round(used_storage_bytes / (1024.0 ** 3), 2) if used_storage_bytes > 0 else 0.0

    weight_files: List[Dict[str, Any]] = []
    total_weight_files_gb = 0.0
    for sib in meta.get("siblings") or []:
        if not isinstance(sib, dict):
            continue
        fname = str(sib.get("rfilename") or "").strip()
        f_low = fname.lower()
        if not fname or not f_low.endswith((".gguf", ".safetensors", ".bin", ".pth", ".ckpt")):
            continue
        lfs = sib.get("lfs") if isinstance(sib.get("lfs"), dict) else {}
        f_bytes = float(sib.get("size") or lfs.get("size") or 0.0)
        size_gb = round(f_bytes / (1024.0 ** 3), 2) if f_bytes > 0 else 0.0
        total_weight_files_gb += size_gb
        vram_need = round(size_gb * 1.12 + 1.0, 1) if size_gb > 0 else 0.0
        if vram_need <= 0:
            f_status, f_zh, f_en = "unknown", "体积未知", "Size TBD"
        elif vram_need <= 14.5:
            f_status, f_zh, f_en = "great", "T4(15G)可跑", "T4 15G Ready"
        elif vram_need <= 22.5:
            f_status, f_zh, f_en = "good", "L4(24G)可跑", "L4 24G Ready"
        elif vram_need <= 39.0:
            f_status, f_zh, f_en = "a100", "需A100(40G)", "A100 40G"
        elif vram_need <= 78.0:
            f_status, f_zh, f_en = "heavy", "需A100(80G)", "A100 80G"
        else:
            f_status, f_zh, f_en = "too_large", "超单卡显存", ">80G VRAM"
        weight_files.append({
            "filename": fname,
            "size_gb": size_gb,
            "fit_status": f_status,
            "fit_zh": f_zh,
            "fit_en": f_en,
        })

    fallback_gb = total_weight_files_gb if (not is_gguf and total_weight_files_gb > 0) else used_storage_gb
    sizing = _analyze_hf_model_sizing(meta, is_gguf, fallback_storage_gb=fallback_gb)

    readme_excerpt = ""
    for host in ("https://huggingface.co", "https://hf-mirror.com"):
        try:
            r_url = f"{host}/{rid}/raw/main/README.md"
            r_req = urllib.request.Request(r_url, headers={"User-Agent": "EasyColab/2.0"})
            if token:
                r_req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(r_req, timeout=6) as r_resp:
                raw_md = r_resp.read(32768).decode("utf-8", errors="ignore")
                readme_excerpt = _clean_hf_readme_excerpt(raw_md)
                if readme_excerpt:
                    break
        except Exception:
            continue

    return {
        "ok": True,
        "repo_id": rid,
        "name": rid.split("/")[-1],
        "author": str(meta.get("author") or rid.split("/")[0]),
        "pipeline_tag": ptag or ("gguf" if is_gguf else "text-generation"),
        "modality": modality,
        "is_gguf": is_gguf,
        "is_gated": is_gated,
        "license": lic,
        "base_model": base_model,
        "architecture": arch_str,
        "context_length": ctx_len,
        "languages": langs,
        "likes": int(meta.get("likes") or 0),
        "downloads": int(meta.get("downloads") or 0),
        "last_modified": str(meta.get("lastModified") or "")[:10],
        "used_storage_gb": used_storage_gb,
        "param_count": sizing["param_count"],
        "param_label": sizing["param_label"],
        "est_vram_gb": sizing["est_vram_gb"],
        "deploy_status": sizing["deploy_status"],
        "deploy_label_zh": sizing["deploy_label_zh"],
        "deploy_label_en": sizing["deploy_label_en"],
        "weight_files": weight_files[:36],
        "readme_excerpt": readme_excerpt,
    }


def inspect_and_resolve_hf_repo(
    repo_id: str,
    include_pattern: str = "",
    preferred_modality: str = "auto",
    hf_token: str = "",
) -> Dict[str, Any]:
    """
    通过 HuggingFace Hub API 实时扫描任意自定义模型仓库，支持通配符过滤 (include_pattern)、
    多档位 GGUF 智能优选、分片自动聚合与架构识别。
    """
    import fnmatch

    repo_clean, pat_clean = _normalize_hf_repo_and_pattern(repo_id, include_pattern)
    if "/" not in repo_clean or len(repo_clean.split("/")) != 2:
        raise ValueError(f"无效的 HuggingFace 仓库地址: '{repo_id}'（格式应为 Owner/Repo）")

    repo_low = repo_clean.lower()
    # 兼容已知特定多模态/图像生成仓库结构（如用户在客户端手动导入）
    if not pat_clean:
        if repo_low == "minimaxai/minimax-h3" or "minimax-h3" in repo_low and "gguf" not in repo_low:
            return {
                "repo_id": repo_clean,
                "include_pattern": "",
                "modality": "video_audio",
                "pipeline_arch": "minimax_h3",
                "category": "自定义导入 · 全模态音视频生成 · BF16 · 32kHz 立体声",
                "approx_size_gb": 134.1,
                "files": list(MINIMAX_H3_BF16_FILES),
            }
        if repo_low == "qwen/qwen-image-2.1" or ("qwen-image-2.1" in repo_low and "gguf" not in repo_low):
            q_spec = MODEL_CATALOG["qwen_image_21"]
            return {
                "repo_id": repo_clean,
                "include_pattern": "",
                "modality": "image",
                "pipeline_arch": "qwen_image_21",
                "category": "自定义导入 · 旗舰文生图与多图指令修图",
                "approx_size_gb": q_spec["approx_size_gb"],
                "files": list(q_spec["files"]),
            }
        if repo_low == "tongyi-mai/z-image-turbo" or ("z-image-turbo" in repo_low and "gguf" not in repo_low):
            z_spec = MODEL_CATALOG["z_image_turbo"]
            return {
                "repo_id": repo_clean,
                "include_pattern": "",
                "modality": "image",
                "pipeline_arch": "z_image_turbo",
                "category": "自定义导入 · 8 步极速超清写实生图",
                "approx_size_gb": z_spec["approx_size_gb"],
                "files": list(z_spec["files"]),
            }

    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
    headers = {"User-Agent": "EasyColab/2.0", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    meta = None
    last_http_code = 0
    last_err = None
    for host in ("https://huggingface.co", "https://hf-mirror.com"):
        api_url = f"{host}/api/models/{repo_clean}?blobs=true"
        req = urllib.request.Request(api_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                meta = json.loads(resp.read().decode("utf-8"))
                break
        except urllib.error.HTTPError as he:
            last_http_code = he.code
            last_err = he
            break
        except Exception as ex:
            last_err = ex
            continue

    if meta is None:
        if last_http_code in (401, 403):
            if not token:
                raise RuntimeError(
                    f"访问受限/私有仓库 {repo_clean} 需要登录鉴权 (HTTP {last_http_code})："
                    f"请先在弹窗填写 HuggingFace Access Token (hf_...) 后重试（获取地址: https://huggingface.co/settings/tokens）。"
                )
            raise RuntimeError(
                f"访问受限仓库 {repo_clean} 被拒绝 (HTTP {last_http_code})："
                f"请先在浏览器打开 https://huggingface.co/{repo_clean} 登录并点击「Agree」同意该模型许可协议，同时确认 HF Token 具备 Read 权限。"
            )
        elif last_http_code == 404:
            if not token:
                raise RuntimeError(
                    f"未找到仓库 {repo_clean} (HTTP 404)：请检查 Repo ID 拼写；若该仓库为私有仓库，请填写有效的 HF Token (hf_...)。"
                )
            raise RuntimeError(f"在 HuggingFace 上未找到仓库: {repo_clean} (HTTP 404)")
        raise RuntimeError(f"查询 HuggingFace 仓库 {repo_clean} 失败: {last_err}")

    gated_raw = meta.get("gated")
    is_gated = bool(gated_raw) and str(gated_raw).lower() not in ("false", "0", "none")
    if is_gated and not token:
        raise RuntimeError(
            f"模型 {repo_clean} 为受限授权模型 (Gated Repo 🔒)："
            f"请先在网页端 https://huggingface.co/{repo_clean} 点击 Agree 同意协议，并在下方填写您的 HuggingFace Token (hf_...) 后再导入。"
        )

    siblings = meta.get("siblings") or []
    file_entries: List[Tuple[str, int]] = []
    all_rfilenames: List[str] = []
    for s in siblings:
        rfn = str(s.get("rfilename") or "").strip()
        if not rfn:
            continue
        all_rfilenames.append(rfn.lower())
        sz = int(s.get("size") or (s.get("lfs") or {}).get("size") or 0)
        file_entries.append((rfn, sz))

    if not file_entries:
        raise RuntimeError(f"仓库 {repo_clean} 中未包含任何可用文件")

    def _fmt_sz(b: int) -> str:
        gb = b / (1024 ** 3)
        if gb >= 0.1:
            return f"约 {gb:.2f} GB"
        mb = b / (1024 ** 2)
        return f"约 {mb:.1f} MB" if mb >= 1.0 else "权重文件"

    raw_patterns = [p.strip() for p in re.split(r"[,;\s]+", pat_clean) if p.strip()] if pat_clean else []
    selected_entries: List[Tuple[str, int]] = []

    if raw_patterns:
        for rfn, sz in file_entries:
            rfn_low = rfn.lower()
            base_low = os.path.basename(rfn_low)
            matched = False
            for pat in raw_patterns:
                pl = pat.lower()
                glob_pat = pl if ("*" in pl or "?" in pl or "[" in pl) else f"*{pl}*"
                if fnmatch.fnmatch(rfn_low, glob_pat) or fnmatch.fnmatch(base_low, glob_pat) or pl in rfn_low:
                    matched = True
                    break
            if matched and not rfn_low.endswith((".md", ".txt", ".gitattributes", ".png", ".jpg", ".gif")):
                selected_entries.append((rfn, sz))
        if not selected_entries:
            raise RuntimeError(
                f"在仓库 {repo_clean} 中未找到匹配 '{pat_clean}' 的权重文件，请检查通配符或文件名是否正确"
            )
        # 若匹配的是 GGUF 主模型且仓库内含有 mmproj 多模态投影器，自动附带最佳 mmproj
        if any(rfn.lower().endswith(".gguf") for rfn, _ in selected_entries) and not any(
            "mmproj" in rfn.lower() for rfn, _ in selected_entries
        ):
            mmprojs = [(r, s) for r, s in file_entries if r.lower().endswith(".gguf") and "mmproj" in r.lower()]
            if mmprojs:
                mmprojs.sort(key=lambda x: (0 if "f16" in x[0].lower() else (1 if "bf16" in x[0].lower() else 2), x[0]))
                selected_entries.append(mmprojs[0])
    else:
        gguf_all = [(r, s) for r, s in file_entries if r.lower().endswith(".gguf")]
        if gguf_all:
            mmprojs = [(r, s) for r, s in gguf_all if "mmproj" in r.lower()]
            mtps = [(r, s) for r, s in gguf_all if "mtp" in r.lower() and "mmproj" not in r.lower()]
            mains = [
                (r, s)
                for r, s in gguf_all
                if "mmproj" not in r.lower()
                and "mtp" not in r.lower()
                and "imatrix" not in r.lower()
                and not os.path.basename(r.lower()).startswith("ggml-vocab")
            ]
            if not mains:
                mains = gguf_all

            # 将分片文件 (如 *-00001-of-00003.gguf) 归并为同一量化组
            groups: Dict[str, List[Tuple[str, int]]] = {}
            for r, s in mains:
                key = re.sub(r"-\d{5}-of-\d{5}\.gguf$", ".gguf", r, flags=re.IGNORECASE)
                groups.setdefault(key, []).append((r, s))

            quant_priority = [
                "ud-q4_k_m", "q4_k_m", "q4_k_l", "iq4_xs", "q4_k_s",
                "ud-q5_k_m", "q5_k_m", "q6_k", "q8_0", "q4_0",
                "ud-q3_k_m", "q3_k_m", "iq3_m", "iq2_xs", "bf16", "f16",
            ]
            best_key = next(iter(groups.keys()))
            best_rank = 999
            for k in groups.keys():
                kl = k.lower()
                for idx_p, qp in enumerate(quant_priority):
                    if qp in kl:
                        if idx_p < best_rank:
                            best_rank = idx_p
                            best_key = k
                        break

            selected_entries.extend(sorted(groups[best_key], key=lambda x: x[0]))
            if mmprojs:
                mmprojs.sort(key=lambda x: (0 if "f16" in x[0].lower() else (1 if "bf16" in x[0].lower() else 2), x[0]))
                selected_entries.append(mmprojs[0])
            if mtps:
                mtps.sort(key=lambda x: (0 if "q4_0" in x[0].lower() else 1, x[0]))
                selected_entries.append(mtps[0])
        else:
            st_files = [
                (r, s)
                for r, s in file_entries
                if r.lower().endswith(".safetensors") and not os.path.basename(r).startswith(".")
            ]
            if st_files:
                selected_entries.extend(sorted(st_files, key=lambda x: x[0]))
            else:
                bin_files = [
                    (r, s)
                    for r, s in file_entries
                    if r.lower().endswith((".bin", ".pt", ".pth"))
                    and "training_args" not in r.lower()
                    and "optimizer" not in r.lower()
                ]
                if bin_files:
                    selected_entries.extend(sorted(bin_files, key=lambda x: x[0]))

    if not selected_entries:
        raise RuntimeError(f"在仓库 {repo_clean} 中未检测到 .gguf 或 .safetensors 模型权重文件")

    has_gguf = any(r.lower().endswith(".gguf") for r, _ in selected_entries)
    total_bytes = sum(max(0, s) for _, s in selected_entries)
    approx_gb = round(total_bytes / (1024 ** 3), 2) if total_bytes > 0 else 8.0

    pipeline_tag = str(meta.get("pipeline_tag") or "").lower()
    tags_low = [str(t).lower() for t in (meta.get("tags") or [])]
    has_config_json = any(r == "config.json" or r.endswith("/config.json") for r in all_rfilenames)
    has_model_index = any(r == "model_index.json" or r.endswith("/model_index.json") for r in all_rfilenames)

    if not has_gguf and pipeline_tag in _UNSUPPORTED_HF_PIPELINE_TAGS:
        raise RuntimeError(
            f"仓库 {repo_clean} 为「{pipeline_tag}」专用算法管线，仅包含自定义组件权重，"
            f"无法作为标准对话/视觉/生图大模型直接装载。请选用 GGUF 量化模型、标准 Transformers 对话/多模态模型或 Diffusers 图像模型。"
        )

    if preferred_modality in ("llm_vlm", "image", "video_audio"):
        modality = preferred_modality
    elif has_gguf:
        modality = "llm_vlm"
    elif "text-to-video" in pipeline_tag or "image-to-video" in pipeline_tag or "video" in pipeline_tag:
        modality = "video_audio"
    elif "text-to-image" in pipeline_tag or "image-to-image" in pipeline_tag or "diffusers" in tags_low or (has_model_index and not has_config_json):
        modality = "image"
    else:
        modality = "llm_vlm"

    if modality == "llm_vlm" and not has_gguf and not has_config_json:
        raise RuntimeError(
            f"仓库 {repo_clean} 未包含标准 Transformers 配置文件 (config.json) 或 GGUF 权重，"
            f"仅包含独立权重分片（如 3D 重建 / 专用 SDK / ComfyUI 单体组件），无法作为对话或视觉大模型直接加载。"
        )

    if has_gguf:
        pipeline_arch = "gguf"
    elif modality == "video_audio":
        pipeline_arch = "minimax_h3"
    elif modality == "image":
        if "z-image" in repo_low:
            pipeline_arch = "z_image_turbo"
        elif "qwen-image" in repo_low:
            pipeline_arch = "qwen_image_21"
        else:
            pipeline_arch = "diffusers_image"
    else:
        pipeline_arch = "transformers"

    files_spec: List[Tuple[str, str]] = []
    total_cnt = len(selected_entries)
    for i, (rfn, sz) in enumerate(selected_entries, start=1):
        rl = rfn.lower()
        if "mmproj" in rl:
            desc = f"多模态视觉投影器 ({_fmt_sz(sz)})"
        elif "mtp" in rl and rl.endswith(".gguf"):
            desc = f"MTP 多词元投机解码头 ({_fmt_sz(sz)})"
        elif rl.endswith(".gguf"):
            desc = f"GGUF 量化权重 {i}/{total_cnt} ({_fmt_sz(sz)})" if total_cnt > 1 else f"GGUF 主模型权重 ({_fmt_sz(sz)})"
        else:
            desc = f"模型权重分片 {i}/{total_cnt} ({_fmt_sz(sz)})"
        files_spec.append((rfn, desc))

    arch_badge = "GGUF (llama.cpp)" if has_gguf else ("Safetensors (BF16)" if modality == "llm_vlm" else "DiffSynth / Diffusers")
    pat_tag = f" · 过滤: {pat_clean}" if pat_clean else ""
    category = f"自定义 HF 模型 · {arch_badge} · {approx_gb:.1f} GB{pat_tag}"

    return {
        "repo_id": repo_clean,
        "include_pattern": pat_clean,
        "modality": modality,
        "pipeline_arch": pipeline_arch,
        "category": category,
        "approx_size_gb": approx_gb,
        "files": files_spec,
    }


def register_custom_hf_model_core(
    repo_id: str,
    custom_name: str = "",
    include_pattern: str = "",
    modality: str = "auto",
    hf_token: str = "",
    model_id_override: str = "",
    modality_hint: str = "",
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    global SELECTED_MODEL_ID
    touch_activity()
    try:
        effective_modality = (modality_hint or modality or "auto").strip()
        if hf_token and hf_token.strip():
            os.environ["HF_TOKEN"] = hf_token.strip()
        resolved = inspect_and_resolve_hf_repo(
            repo_id=repo_id,
            include_pattern=include_pattern,
            preferred_modality=effective_modality,
            hf_token=hf_token,
        )
        repo_clean = resolved["repo_id"]
        pat_clean = resolved["include_pattern"]
        mid = (model_id_override or "").strip() or _make_custom_model_id(repo_clean, pat_clean)
        if mid in BUILTIN_MODEL_IDS:
            mid = f"custom_{mid}"

        default_title = repo_clean.split("/")[-1]
        if pat_clean and len(pat_clean) <= 24 and "*" in pat_clean:
            clean_pat_lbl = pat_clean.replace("*", "").replace(".gguf", "").strip("-_.")
            if clean_pat_lbl and clean_pat_lbl.lower() not in default_title.lower():
                default_title = f"{default_title} ({clean_pat_lbl})"
        display_name = (custom_name or "").strip() or default_title

        spec = {
            "id": mid,
            "name": display_name,
            "repo_id": repo_clean,
            "include_pattern": pat_clean,
            "modality": resolved["modality"],
            "pipeline_arch": resolved["pipeline_arch"],
            "category": resolved["category"],
            "approx_size_gb": resolved["approx_size_gb"],
            "files": resolved["files"],
            "is_custom": True,
        }
        with CUSTOM_MODELS_LOCK:
            MODEL_CATALOG[mid] = spec
            _save_persisted_custom_models()
        with _TELEMETRY_LOCK:
            _CACHED_FILE_STATUS_MAP.pop(mid, None)

        SELECTED_MODEL_ID = mid
        msg = (
            f"已成功导入自定义模型「{display_name}」({repo_clean} · 共 {len(spec['files'])} 个文件 / "
            f"约 {spec['approx_size_gb']:.1f} GB)，可立即下载权重并加载到显存！"
        )
        MODEL_DOWNLOAD_STATE["message"] = msg
        print(f"[自定义模型] {msg}")
        return True, msg, spec
    except Exception as e:
        err = f"导入自定义 HuggingFace 仓库失败: {str(e)}"
        print(f"[自定义模型] {err}")
        return False, err, None


def remove_custom_hf_model_core(model_id: str, delete_files: bool = False) -> Tuple[bool, str]:
    global SELECTED_MODEL_ID
    touch_activity()
    mid = (model_id or "").strip()
    if not mid or mid in BUILTIN_MODEL_IDS or mid not in MODEL_CATALOG:
        return False, "只能移除用户自定义添加的模型"
    name = MODEL_CATALOG[mid].get("name", mid)
    if mid in LOADED_RUNTIMES:
        unload_model_from_vram_core(mid)
    with CUSTOM_MODELS_LOCK:
        MODEL_CATALOG.pop(mid, None)
        _save_persisted_custom_models()
    with _TELEMETRY_LOCK:
        _CACHED_FILE_STATUS_MAP.pop(mid, None)
    if SELECTED_MODEL_ID == mid:
        SELECTED_MODEL_ID = "z_image_turbo" if "z_image_turbo" in MODEL_CATALOG else next(iter(MODEL_CATALOG.keys()))
    msg = f"已从模型中心移除自定义模型「{name}」"
    MODEL_DOWNLOAD_STATE["message"] = msg
    print(f"[自定义模型] {msg}")
    return True, msg


_load_persisted_custom_models()

DEFAULT_OUTPUT_DIR = (
    "/content/drive/MyDrive/EasyColab_Outputs"
    if DRIVE_MOUNTED
    else os.path.join(os.getcwd(), "outputs")
)
LEGACY_DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/MiniMax_H3_Outputs"
LEGACY_DRIVE_OUTPUT_DIRS = [
    "/content/drive/MyDrive/Aevix_Outputs",
    "/content/drive/MyDrive/MiniMax_H3_Outputs",
]
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), "models"), exist_ok=True)


# ==============================================================================
# 借鉴 comfy-colab-kit 的 8 线程 HTTP Range 并行极速下载器 (400~700 MB/s)
# ==============================================================================
class _StripAuthRedirectHandler(urllib.request.HTTPRedirectHandler):
    """重定向到 HuggingFace CDN (CloudFront/Xet) 时剥离 Authorization 头，防止 400/403 拒绝"""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None and "Authorization" in new_req.unredirected_hdrs:
            del new_req.unredirected_hdrs["Authorization"]
        if new_req is not None and "Authorization" in new_req.headers:
            del new_req.headers["Authorization"]
        return new_req


def fast_parallel_hf_download(
    repo_id: str,
    filename: str,
    dest_dir: str,
    hf_token: str = "",
    num_connections: int = 8,
) -> str:
    """
    使用 8 路并发 HTTP Range 连接直接从 HuggingFace CDN 下载大权重文件至内存盘/本地盘，
    规避单连接 ~60MB/s 限速，在 Colab 上可达 400~700 MB/s。
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".part"

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    opener = urllib.request.build_opener(_StripAuthRedirectHandler())
    headers = {"User-Agent": "EasyColab/2.0"}
    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        # Probe final CDN URL & Content-Length via Range: bytes=0-0
        probe_req = urllib.request.Request(url, headers={**headers, "Range": "bytes=0-0"})
        with opener.open(probe_req, timeout=20) as resp:
            cdn_url = resp.geturl()
            content_range = resp.headers.get("Content-Range", "")
            if "/" in content_range:
                total_size = int(content_range.split("/")[-1])
            else:
                total_size = int(resp.headers.get("Content-Length", 0))

        # 若文件较小 (<32MB) 或未返回支持分片的大小，直接走 hf_hub_download
        if total_size < 32 * 1024 * 1024:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir, token=token or None)

        with open(temp_path, "wb") as f:
            f.truncate(total_size)

        downloaded_bytes = 0
        dl_lock = threading.Lock()
        start_ts = time.time()
        last_print_ts = [0.0]

        part_size = (total_size + num_connections - 1) // num_connections
        ranges = [
            (i * part_size, min((i + 1) * part_size - 1, total_size - 1))
            for i in range(num_connections)
            if i * part_size < total_size
        ]

        def _download_chunk(byte_range: Tuple[int, int]):
            nonlocal downloaded_bytes
            start_b, end_b = byte_range
            chunk_req = urllib.request.Request(
                cdn_url,
                headers={"User-Agent": "EasyColab/2.0", "Range": f"bytes={start_b}-{end_b}"},
            )
            with urllib.request.urlopen(chunk_req, timeout=60) as r, open(temp_path, "r+b") as wf:
                wf.seek(start_b)
                while True:
                    buf = r.read(4 * 1024 * 1024)  # 4MB buffer
                    if not buf:
                        break
                    wf.write(buf)
                    now = time.time()
                    with dl_lock:
                        downloaded_bytes += len(buf)
                        if now - last_print_ts[0] >= 0.35 or downloaded_bytes >= total_size:
                            last_print_ts[0] = now
                            elapsed = max(0.001, now - start_ts)
                            speed_mbs = (downloaded_bytes / (1024 ** 2)) / elapsed
                            pct = int(downloaded_bytes * 100 / total_size)
                            cur_gb = downloaded_bytes / (1024 ** 3)
                            tot_gb = total_size / (1024 ** 3)
                            eta_s = int(((total_size - downloaded_bytes) / (1024 ** 2)) / max(0.1, speed_mbs))
                            sys.stdout.write(
                                f"\r{filename}: {pct}%|{'█' * (pct // 10):<10}| "
                                f"{cur_gb:.2f}G/{tot_gb:.2f}G [{int(elapsed):02d}s<{eta_s:02d}s, {speed_mbs:.1f}MB/s]\n"
                            )
                            sys.stdout.flush()

        with ThreadPoolExecutor(max_workers=num_connections) as pool:
            futures = [pool.submit(_download_chunk, r) for r in ranges]
            for fut in as_completed(futures):
                fut.result()

        os.replace(temp_path, dest_path)
        return dest_path

    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "403" in err_str:
            if not token:
                raise RuntimeError(
                    f"下载受限仓库 {repo_id} 需要鉴权 (HTTP 401/403)：请先填写 HuggingFace Token (hf_...) 后重新下载。"
                )
            raise RuntimeError(
                f"下载受限仓库 {repo_id} 被拒绝 (HTTP 403 Gated Repo)：请先用浏览器打开 https://huggingface.co/{repo_id} 点击 Agree 同意协议后再试。"
            )
        print(f"ℹ️ 多线程分片直连回退至标准 HF 下载 ({e})...")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir, token=token or None)



_LAST_JUPYTER_HEARTBEAT_TS: float = 0.0
_JUPYTER_HB_WAKE_EVENT = threading.Event()
_STOP_JUPYTER_KEEPALIVE = False


def _build_ws_masked_text_frame(text: str) -> bytes:
    """构建标准 RFC 6455 客户端掩码文本帧 (opcode=0x1, FIN=1)，零第三方依赖连接本地 Jupyter WebSocket"""
    payload = text.encode("utf-8")
    length = len(payload)
    mask_key = os.urandom(4)
    header = bytearray([0x81])
    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.append(0x80 | 126)
        header.extend(length.to_bytes(2, "big"))
    else:
        header.append(0x80 | 127)
        header.extend(length.to_bytes(8, "big"))
    header.extend(mask_key)
    masked = bytearray(length)
    for i in range(length):
        masked[i] = payload[i] ^ mask_key[i & 3]
    return bytes(header) + bytes(masked)


def _build_ws_masked_pong_frame(payload: bytes = b"") -> bytes:
    mask_key = os.urandom(4)
    length = min(len(payload), 125)
    header = bytearray([0x8A, 0x80 | length])
    header.extend(mask_key)
    masked = bytes(payload[i] ^ mask_key[i & 3] for i in range(length))
    return bytes(header) + masked


def _ensure_local_colab_kernel_id() -> Optional[str]:
    """查询或启动本地 127.0.0.1:8080 上的 Jupyter 内核并返回 kernel_id"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Colab-Client-Agent": "vscode",
    }
    try:
        req = urllib.request.Request("http://127.0.0.1:8080/api/kernels", headers=headers)
        with opener.open(req, timeout=3.0) as resp:
            kernels = json.loads(resp.read().decode("utf-8", errors="replace"))
        if isinstance(kernels, list) and kernels:
            for k in kernels:
                kid = k.get("id")
                if kid:
                    return str(kid)
    except Exception:
        return None

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8080/api/kernels",
            data=b'{"name":"python3"}',
            headers=headers,
            method="POST",
        )
        with opener.open(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        kid = data.get("id")
        if kid:
            return str(kid)
    except Exception:
        pass
    return None


def _open_local_jupyter_ws_channel(kernel_id: str, session_id: str) -> Optional[socket.socket]:
    """建立到 ws://127.0.0.1:8080/api/kernels/<id>/channels 的常驻 WebSocket 连接，使内核 connections 始终 >= 1"""
    sock = None
    try:
        sock = socket.create_connection(("127.0.0.1", 8080), timeout=5.0)
        ws_key = base64.b64encode(os.urandom(16)).decode("ascii")
        path = f"/api/kernels/{kernel_id}/channels?session_id={session_id}"
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:8080\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {ws_key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"X-Colab-Client-Agent: vscode\r\n"
            f"Origin: http://127.0.0.1:8080\r\n\r\n"
        )
        sock.sendall(handshake.encode("ascii"))
        buf = b""
        while b"\r\n\r\n" not in buf and len(buf) < 8192:
            chunk = sock.recv(1024)
            if not chunk:
                break
            buf += chunk
        status_line = buf.split(b"\r\n", 1)[0]
        if b" 101 " not in status_line:
            sock.close()
            return None
        sock.settimeout(0.2)
        return sock
    except Exception:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        return None


def _recv_ws_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("WebSocket EOF mid-frame")
        buf.extend(chunk)
    return bytes(buf)


def _recv_ws_frame(sock: socket.socket, poll_timeout: float = 0.2) -> Optional[Tuple[int, bytes]]:
    """
    严格按 RFC 6455 帧边界解析单帧，返回 (opcode, payload)。
    首个字节到达前若超时无数据返回 None；对端断开连接则抛出 ConnectionError。
    杜绝直接用 sock.recv(4096)[0] & 0x0F 将分片 JSON 载荷字节 ('h'/'x'/'8'/'(') 误判为 Close 帧。
    """
    prev_timeout = sock.gettimeout()
    try:
        sock.settimeout(poll_timeout)
        try:
            first_two = sock.recv(2)
        except socket.timeout:
            return None
        if not first_two:
            raise ConnectionError("WebSocket closed by peer")
        sock.settimeout(2.5)
        if len(first_two) == 1:
            first_two += _recv_ws_exact(sock, 1)
        b0, b1 = first_two[0], first_two[1]
        opcode = b0 & 0x0F
        masked = (b1 & 0x80) != 0
        payload_len = b1 & 0x7F
        if payload_len == 126:
            payload_len = int.from_bytes(_recv_ws_exact(sock, 2), "big")
        elif payload_len == 127:
            payload_len = int.from_bytes(_recv_ws_exact(sock, 8), "big")
        if payload_len > 32 * 1024 * 1024:
            raise ConnectionError(f"Oversized WebSocket frame: {payload_len}")
        mask_key = _recv_ws_exact(sock, 4) if masked else b""
        payload = _recv_ws_exact(sock, payload_len) if payload_len > 0 else b""
        if masked and payload:
            payload = bytes(payload[i] ^ mask_key[i & 3] for i in range(payload_len))
        return opcode, payload
    finally:
        try:
            sock.settimeout(prev_timeout)
        except Exception:
            pass


def _drain_local_ws_frames(sock: socket.socket, max_frames: int = 24, poll_timeout: float = 0.15) -> bool:
    """按完整 RFC 6455 帧排空本地 Jupyter WebSocket 缓冲区：收到 Ping 回显原 payload Pong，收到 Close 返回 False"""
    try:
        for idx in range(max_frames):
            frame = _recv_ws_frame(sock, poll_timeout=poll_timeout if idx == 0 else 0.05)
            if frame is None:
                break
            opcode, payload = frame
            if opcode == 0x08:
                return False
            if opcode == 0x09:
                sock.sendall(_build_ws_masked_pong_frame(payload))
        return True
    except Exception:
        return False


def _send_jupyter_ws_execute_heartbeat(sock: socket.socket, session_id: str) -> bool:
    """在已连接的本地 Jupyter WebSocket 通道上发送轻量 execute_request，真实刷新内核 last_activity 与 execution_state"""
    msg = {
        "channel": "shell",
        "header": {
            "msg_id": uuid.uuid4().hex,
            "username": "easycolab",
            "session": session_id,
            "date": datetime.utcnow().isoformat() + "Z",
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": "_easycolab_hb = 1",
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": False,
        },
    }
    try:
        frame = _build_ws_masked_text_frame(json.dumps(msg))
        sock.sendall(frame)
        return _drain_local_ws_frames(sock, max_frames=32, poll_timeout=0.25)
    except Exception:
        return False


def _local_jupyter_ws_keepalive_loop():
    """
    后台守护线程：只要当前处于活跃保活窗口内，始终保持一条连接到本地 Colab Jupyter Server
    (ws://127.0.0.1:8080/api/kernels/<id>/channels) 的常驻 WebSocket，确保内核 connections >= 1，
    并每 30 秒发送一次 execute_request 脉冲刷新 last_activity，彻底杜绝无网页模式下被 Colab 误判闲置回收。
    """
    session_id = uuid.uuid4().hex
    ws_sock: Optional[socket.socket] = None
    active_kid: Optional[str] = None
    last_exec_ts = 0.0

    while not _STOP_JUPYTER_KEEPALIVE:
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for ep in ("http://127.0.0.1:8080/api/kernels", "http://127.0.0.1:8080/api/sessions", "http://127.0.0.1:8080/api/status"):
                try:
                    req = urllib.request.Request(ep, headers={"X-Colab-Client-Agent": "vscode"})
                    with opener.open(req, timeout=2.0) as resp:
                        resp.read(256)
                except Exception:
                    pass

            if ws_sock is None:
                active_kid = _ensure_local_colab_kernel_id()
                if active_kid:
                    ws_sock = _open_local_jupyter_ws_channel(active_kid, session_id)
                    last_exec_ts = 0.0

            now = time.time()
            if ws_sock is not None and (now - last_exec_ts >= 30.0 or _JUPYTER_HB_WAKE_EVENT.is_set()):
                if _send_jupyter_ws_execute_heartbeat(ws_sock, session_id):
                    last_exec_ts = now
                else:
                    try:
                        ws_sock.close()
                    except Exception:
                        pass
                    ws_sock = None
            elif ws_sock is not None:
                # 非发送周期也按 RFC 6455 帧边界排空控制帧，保持连接健康
                if not _drain_local_ws_frames(ws_sock, max_frames=16, poll_timeout=0.15):
                    try:
                        ws_sock.close()
                    except Exception:
                        pass
                    ws_sock = None
        except Exception:
            pass

        _JUPYTER_HB_WAKE_EVENT.wait(timeout=5.0)
        _JUPYTER_HB_WAKE_EVENT.clear()

    if ws_sock is not None:
        try:
            ws_sock.close()
        except Exception:
            pass


threading.Thread(target=_local_jupyter_ws_keepalive_loop, daemon=True).start()


def _ping_local_colab_jupyter(force: bool = False):
    """触发本地常驻 Jupyter WebSocket 保活线程立即执行一次脉冲"""
    global _LAST_JUPYTER_HEARTBEAT_TS
    now = time.time()
    if not force and (now - _LAST_JUPYTER_HEARTBEAT_TS) < 25.0:
        return
    _LAST_JUPYTER_HEARTBEAT_TS = now
    _JUPYTER_HB_WAKE_EVENT.set()


def touch_activity():
    global LAST_ACTIVE_TIME
    LAST_ACTIVE_TIME = time.time()
    _ping_local_colab_jupyter(force=False)


class _ActiveApiStreamGuard:
    """标记当前正在进行的流式/长耗时网关请求，避免长思维链生成期间被闲置看门狗误判"""

    def __enter__(self):
        global _ACTIVE_API_STREAMS
        with _ACTIVE_API_STREAMS_LOCK:
            _ACTIVE_API_STREAMS += 1
        touch_activity()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _ACTIVE_API_STREAMS
        with _ACTIVE_API_STREAMS_LOCK:
            _ACTIVE_API_STREAMS = max(0, _ACTIVE_API_STREAMS - 1)
        touch_activity()
        return False


def estimate_cu_burn_rate(gpu_name: str) -> float:
    """根据 Colab GPU 型号估算每小时消耗的计算单元 (CU/hr)"""
    name_up = (gpu_name or "").upper()
    if "H100" in name_up:
        return 11.78
    elif "PRO 6000" in name_up or "BLACKWELL" in name_up or "G4" in name_up:
        return 10.05
    elif "A100" in name_up:
        return 5.37
    elif "L4" in name_up:
        return 1.71
    elif "T4" in name_up:
        return 1.17
    elif "V100" in name_up:
        return 3.80
    return 1.50


def _idle_watchdog_loop():
    """借鉴 comfy-colab-kit 的 stopper.py：当超过设定闲置时间无任务时自动释放 Colab 虚拟机"""
    global _STOP_JUPYTER_KEEPALIVE
    while True:
        time.sleep(15)
        # 检查客户端桥接层 (cmd_keepalive) 写入的保活时间戳文件
        try:
            if os.path.exists("/tmp/.easycolab_client_hb"):
                hb_mtime = os.path.getmtime("/tmp/.easycolab_client_hb")
                if hb_mtime > LAST_ACTIVE_TIME:
                    touch_activity()
        except Exception:
            pass

        # 若正在下载、加载模型、生成视频/LLM推理或 ComfyUI 正在运行/安装/执行节点队列，持续刷新活跃时间并向本地 Jupyter 发送保活脉冲
        if (
            MODEL_DOWNLOAD_STATE["is_downloading"]
            or MODEL_DOWNLOAD_STATE["is_loading_vram"]
            or GPU_LOCK.locked()
            or _ACTIVE_API_STREAMS > 0
            # ComfyUI 只是开着不算活跃，队列里有任务才算（队列每 2 秒由遥测线程刷新），否则开着画布的实例永远不会按闲置时长释放
            or COMFYUI_STATE["is_starting"]
            or COMFYUI_STATE["is_downloading_weight"]
            or COMFYUI_STATE["queue_running"] > 0
            or COMFYUI_STATE["queue_pending"] > 0
        ):
            touch_activity()
            continue

        if IDLE_TIMEOUT_MINUTES <= 0:
            _ping_local_colab_jupyter(force=False)
            continue

        idle_sec = time.time() - LAST_ACTIVE_TIME
        if idle_sec >= IDLE_TIMEOUT_MINUTES * 60:
            _STOP_JUPYTER_KEEPALIVE = True
            _JUPYTER_HB_WAKE_EVENT.set()
            print(f"🛑 [自动省额度看门狗] 已连续闲置 {IDLE_TIMEOUT_MINUTES} 分钟，正在自动释放 Colab 虚拟机...")
            disconnect_colab_runtime()
            break
        else:
            # 尚在用户设定的活跃保活窗口内，轻触本地 Jupyter 防止 Colab 提前强杀
            _ping_local_colab_jupyter(force=False)


threading.Thread(target=_idle_watchdog_loop, daemon=True).start()


def _safe_shm_available_bytes(reserve_ram_gb: float = 34.0) -> int:
    """
    计算 /dev/shm 内存盘真正可安全写入的字节数：
    Colab 容器中 /dev/shm (tmpfs) 与进程匿名内存共享同一个 Cgroup 物理内存上限（如 G4 176GB）。
    若 MiniMax-H3 已锁入 ~128GB 系统内存热备，shutil.disk_usage('/dev/shm').free 仍会虚报 ~88GB 空闲，
    此时继续向 /dev/shm 写入大权重会直接击穿容器 Cgroup 触发 OOM 强杀。
    因此必须同时受限于 /dev/shm 配额剩余空间与 /proc/meminfo MemAvailable 安全余量。
    """
    if not os.path.exists("/dev/shm"):
        return 0
    try:
        shm_free = int(shutil.disk_usage("/dev/shm").free)
    except Exception:
        return 0
    total_ram_gb, avail_ram_gb = _get_system_ram_gb()
    if total_ram_gb > 0:
        eff_reserve = reserve_ram_gb
        try:
            if any("内存热备" in str(rt.get("mode_label", "")) for rt in LOADED_RUNTIMES.values()):
                eff_reserve = max(eff_reserve, 40.0)
        except Exception:
            pass
        ram_budget = max(0, int((avail_ram_gb - eff_reserve) * (1024 ** 3)))
        return min(shm_free, ram_budget)
    return shm_free


def _get_model_dirs(model_id: str) -> Tuple[Optional[str], str, str]:
    """返回指定模型的 (RAM内存盘目录或None, 本地高速NVMe目录, Google Drive/持久盘目录)"""
    spec = MODEL_CATALOG.get(model_id, MODEL_CATALOG["minimax_h3"])
    repo_sub = spec["repo_id"]
    ram_dir = None
    if os.path.exists("/dev/shm"):
        try:
            shm_free_gb = _safe_shm_available_bytes(reserve_ram_gb=34.0) / (1024 ** 3)
            # 仅当 /dev/shm 配额与容器真实可用物理内存均足以容纳模型时才把新模型放入内存盘
            if shm_free_gb >= spec["approx_size_gb"] + 3.0:
                ram_dir = f"/dev/shm/EasyColab_Models/{model_id}"
            elif spec["approx_size_gb"] <= 60.0 and (
                os.path.exists(f"/dev/shm/EasyColab_Models/{model_id}")
                or os.path.exists(f"/dev/shm/Aevix_Models/{model_id}")
                or os.path.exists(f"/dev/shm/OpenModels/{model_id}")
            ):
                ram_dir = f"/dev/shm/EasyColab_Models/{model_id}"
        except Exception:
            pass
    nvme_dir = os.path.join(os.getcwd(), "models", repo_sub)
    drive_dir = os.path.join(MODEL_BASE_ROOT, repo_sub)
    return ram_dir, nvme_dir, drive_dir


# 模型 ID -> (对方模型 ID, 自己的分片目录前缀, 对方的分片目录前缀)
_H3_SHARED_TEXT_ENCODER = {
    "minimax_h3": ("minimax_h3_image", "FL2VA/text_encoder/", "text_encoder/"),
    "minimax_h3_image": ("minimax_h3", "text_encoder/", "FL2VA/text_encoder/"),
}


def _find_existing_model_file(model_id: str, fname: str, _follow_shared: bool = True) -> Optional[str]:
    """按速度优先级查找已存在的模型权重文件：1. /dev/shm 内存盘 -> 2. 本地 NVMe -> 3. Google Drive (含历史目录与 HuggingFace 缓存兼容)"""
    ram_dir, nvme_dir, drive_dir = _get_model_dirs(model_id)
    spec = MODEL_CATALOG.get(model_id, MODEL_CATALOG["minimax_h3"])
    repo_sub = spec["repo_id"]
    candidates = []
    if ram_dir:
        candidates.append(os.path.join(ram_dir, fname))
    candidates.append(os.path.join(f"/dev/shm/EasyColab_Models/{model_id}", fname))
    candidates.append(os.path.join(f"/dev/shm/Aevix_Models/{model_id}", fname))
    candidates.append(os.path.join(f"/dev/shm/OpenModels/{model_id}", fname))
    candidates.append(os.path.join(nvme_dir, fname))
    candidates.append(os.path.join(drive_dir, fname))
    if DRIVE_MOUNTED:
        for leg_root in LEGACY_DRIVE_MODEL_ROOTS:
            if os.path.isdir(leg_root):
                candidates.append(os.path.join(leg_root, repo_sub, fname))
    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 64 * 1024:
            return p
    # MiniMax-H3 与 MiniMax-H3-Image 的文本编码器分片逐字节相同：自己目录里没有时去对方目录找，谁先下载都只存一份
    shared = _H3_SHARED_TEXT_ENCODER.get(model_id)
    if _follow_shared and shared and fname.startswith(shared[1]):
        return _find_existing_model_file(shared[0], shared[2] + fname[len(shared[1]):], _follow_shared=False)
    return None


def check_model_files_status(model_id: Optional[str] = None) -> Dict[str, Any]:
    """检查指定开源模型在内存盘/磁盘/云盘上的权重文件完整性与大小"""
    target_id = model_id if (model_id and model_id in MODEL_CATALOG) else SELECTED_MODEL_ID
    spec = MODEL_CATALOG[target_id]
    req_files = spec["files"]
    ram_dir, nvme_dir, drive_dir = _get_model_dirs(target_id)

    files_info = []
    downloaded_count = 0
    total_size_gb = 0.0
    ram_cached_count = 0

    for fname, desc in req_files:
        actual_path = _find_existing_model_file(target_id, fname)
        if actual_path:
            size_gb = round(os.path.getsize(actual_path) / (1024 ** 3), 2)
            total_size_gb += size_gb
            downloaded_count += 1
            in_ram = actual_path.startswith("/dev/shm/")
            if in_ram:
                ram_cached_count += 1
            files_info.append({
                "name": fname,
                "desc": desc + (" [RAM内存盘]" if in_ram else ""),
                "exists": True,
                "size_gb": size_gb,
            })
        else:
            files_info.append({
                "name": fname,
                "desc": desc,
                "exists": False,
                "size_gb": 0.0,
            })

    all_downloaded = downloaded_count == len(req_files) and len(req_files) > 0
    return {
        "selected_model_id": target_id,
        "selected_model_name": spec["name"],
        "selected_model_repo": spec["repo_id"],
        "selected_model_modality": spec["modality"],
        "files": files_info,
        "downloaded_count": downloaded_count,
        "total_count": len(req_files),
        "all_downloaded": all_downloaded,
        "total_size_gb": round(total_size_gb, 2),
        "drive_persisted": DRIVE_MOUNTED,
        "ramdisk_enabled": ram_dir is not None,
        "ram_cached_count": ram_cached_count,
        "model_storage_path": ram_dir or drive_dir,
        "bf16_shards_count": downloaded_count if target_id == "minimax_h3" else 0,
    }


_TELEMETRY_LOCK = threading.Lock()
_CACHED_DRIVE_METRICS: Dict[str, Any] = {
    "ts": 0.0,
    "drive_total_gb": 0.0,
    "drive_used_gb": 0.0,
    "drive_free_gb": 0.0,
}
_CACHED_FILE_STATUS_MAP: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHED_LIVE_TELEMETRY: Dict[str, Any] = {}


def _get_cached_model_files_status(target_id: str) -> Dict[str, Any]:
    """对磁盘/云盘 FUSE 权重扫描做轻量缓存，避免高频状态轮询在视频生成期间阻塞 I/O"""
    now = time.time()
    ttl = 1.8 if MODEL_DOWNLOAD_STATE["is_downloading"] else 12.0
    with _TELEMETRY_LOCK:
        cached = _CACHED_FILE_STATUS_MAP.get(target_id)
        if cached and (now - cached[0]) < ttl:
            return dict(cached[1])
    fresh = check_model_files_status(target_id)
    with _TELEMETRY_LOCK:
        _CACHED_FILE_STATUS_MAP[target_id] = (now, dict(fresh))
    return fresh


def _get_storage_metrics_dict(refresh_drive: bool = False) -> Dict[str, Any]:
    """采集 Colab 实例系统内存 (RAM)、本地磁盘、/dev/shm 内存加速盘及 Google Drive 云盘指标 (单位 GB)"""
    ram_total_gb, ram_free_gb = _get_system_ram_gb()
    ram_used_gb = round(max(0.0, ram_total_gb - ram_free_gb), 1) if ram_total_gb > 0 else 0.0

    disk_path = "/content" if os.path.exists("/content") else os.getcwd()
    disk_total_gb, disk_used_gb, disk_free_gb = 0.0, 0.0, 0.0
    try:
        du = shutil.disk_usage(disk_path)
        disk_total_gb = round(du.total / (1024 ** 3), 1)
        disk_free_gb = round(du.free / (1024 ** 3), 1)
        disk_used_gb = round(max(0.0, (du.total - du.free) / (1024 ** 3)), 1)
    except Exception:
        pass

    shm_total_gb, shm_used_gb, shm_free_gb = 0.0, 0.0, 0.0
    if os.path.exists("/dev/shm"):
        try:
            su = shutil.disk_usage("/dev/shm")
            shm_total_gb = round(su.total / (1024 ** 3), 1)
            shm_free_gb = round(su.free / (1024 ** 3), 1)
            shm_used_gb = round(max(0.0, (su.total - su.free) / (1024 ** 3)), 1)
        except Exception:
            pass

    now = time.time()
    if DRIVE_MOUNTED and (refresh_drive or (now - _CACHED_DRIVE_METRICS["ts"]) >= 15.0):
        if os.path.exists("/content/drive/MyDrive"):
            try:
                gu = shutil.disk_usage("/content/drive/MyDrive")
                _CACHED_DRIVE_METRICS["drive_total_gb"] = round(gu.total / (1024 ** 3), 1)
                _CACHED_DRIVE_METRICS["drive_free_gb"] = round(gu.free / (1024 ** 3), 1)
                _CACHED_DRIVE_METRICS["drive_used_gb"] = round(max(0.0, (gu.total - gu.free) / (1024 ** 3)), 1)
                _CACHED_DRIVE_METRICS["ts"] = now
            except Exception:
                pass

    return {
        "ram_total_gb": ram_total_gb,
        "ram_used_gb": ram_used_gb,
        "ram_free_gb": ram_free_gb,
        "disk_total_gb": disk_total_gb,
        "disk_used_gb": disk_used_gb,
        "disk_free_gb": disk_free_gb,
        "shm_total_gb": shm_total_gb,
        "shm_used_gb": shm_used_gb,
        "shm_free_gb": shm_free_gb,
        "drive_total_gb": _CACHED_DRIVE_METRICS["drive_total_gb"],
        "drive_used_gb": _CACHED_DRIVE_METRICS["drive_used_gb"],
        "drive_free_gb": _CACHED_DRIVE_METRICS["drive_free_gb"],
    }


def _sample_live_hardware_telemetry_once() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        gpu_name = "CPU Mode (未检测到 GPU)"
        cuda_avail = False
        free_gb, total_gb, used_gb = 0.0, 0.0, 0.0
        cu_rate = 0.10
    else:
        gpu_name = torch.cuda.get_device_name(0)
        cuda_avail = True
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        free_gb = round(free_mem / (1024 ** 3), 2)
        total_gb = round(total_mem / (1024 ** 3), 2)
        used_gb = round(total_gb - free_gb, 2)
        cu_rate = estimate_cu_burn_rate(gpu_name)

    snap = {
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "vram_free_gb": free_gb,
        "vram_total_gb": total_gb,
        "vram_used_gb": used_gb,
        "cu_burn_rate_per_hour": cu_rate,
    }
    snap.update(_get_storage_metrics_dict(refresh_drive=False))
    if "_poll_local_comfyui_runtime_quiet" in globals():
        _poll_local_comfyui_runtime_quiet()
    with _TELEMETRY_LOCK:
        _CACHED_LIVE_TELEMETRY.clear()
        _CACHED_LIVE_TELEMETRY.update(snap)
    return snap


def _telemetry_sampler_loop():
    """独立后台线程以 2s 周期采样 GPU 显存/内存/磁盘指标，保证 HTTP 轮询非阻塞秒回"""
    while True:
        try:
            _sample_live_hardware_telemetry_once()
        except Exception:
            pass
        time.sleep(2.0)


threading.Thread(target=_telemetry_sampler_loop, daemon=True).start()


def get_live_telemetry_snapshot() -> Dict[str, Any]:
    with _TELEMETRY_LOCK:
        if _CACHED_LIVE_TELEMETRY:
            return dict(_CACHED_LIVE_TELEMETRY)
    return _sample_live_hardware_telemetry_once()


def get_gpu_status_dict(model_id: Optional[str] = None) -> Dict[str, Any]:
    """返回结构化的 GPU 状态、实例磁盘空间、多模型注册信息、CU 算力消耗账本、闲置倒计时与实时终端进度流"""
    global SELECTED_MODEL_ID
    if model_id and model_id in MODEL_CATALOG:
        SELECTED_MODEL_ID = model_id
    target_id = SELECTED_MODEL_ID

    file_status = _get_cached_model_files_status(target_id)
    hw = get_live_telemetry_snapshot()
    uptime_seconds = int(time.time() - BOOT_TIME)
    idle_seconds = int(max(0.0, time.time() - LAST_ACTIVE_TIME))
    idle_remaining_seconds = (
        max(0, IDLE_TIMEOUT_MINUTES * 60 - idle_seconds) if IDLE_TIMEOUT_MINUTES > 0 else -1
    )

    gpu_name = hw["gpu_name"]
    cuda_avail = hw["cuda_available"]
    free_gb = hw["vram_free_gb"]
    total_gb = hw["vram_total_gb"]
    used_gb = hw["vram_used_gb"]
    cu_rate = hw["cu_burn_rate_per_hour"]

    session_cu_used = round((uptime_seconds / 3600.0) * cu_rate, 3)
    loaded_ids = list(LOADED_RUNTIMES.keys())
    loaded_details = [
        {
            "id": mid,
            "name": rt.get("name", mid),
            "mode": rt.get("mode_label", ""),
            "vram_gb": rt.get("vram_gb", 0.0),
            "context_length": int((rt.get("llm_bundle") or {}).get("context_length") or 0),
        }
        for mid, rt in LOADED_RUNTIMES.items()
    ]
    loaded_ctx = 0
    if target_id in LOADED_RUNTIMES and LOADED_RUNTIMES[target_id].get("llm_bundle"):
        loaded_ctx = int(LOADED_RUNTIMES[target_id]["llm_bundle"].get("context_length") or 0)
    else:
        for rt in LOADED_RUNTIMES.values():
            if rt.get("llm_bundle") and rt["llm_bundle"].get("context_length"):
                loaded_ctx = int(rt["llm_bundle"]["context_length"])
                break

    any_loaded = bool(LOADED_RUNTIMES) or (PIPELINE is not None) or (LLM_BUNDLE is not None)
    loaded_for_selected = (target_id in LOADED_RUNTIMES) or (any_loaded and LOADED_MODEL_ID == target_id)
    summary_mode = (
        LOADED_RUNTIMES[target_id]["mode_label"]
        if target_id in LOADED_RUNTIMES
        else (" + ".join(rt["name"] for rt in LOADED_RUNTIMES.values()) if LOADED_RUNTIMES else (CURRENT_MODEL_MODE or "未加载"))
    )

    base = {
        "ok": True,
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "vram_free_gb": free_gb,
        "vram_total_gb": total_gb,
        "vram_used_gb": used_gb,
        "model_loaded": loaded_for_selected,
        "any_model_loaded": any_loaded,
        "loaded_model_id": target_id if target_id in LOADED_RUNTIMES else LOADED_MODEL_ID,
        "loaded_model_ids": loaded_ids,
        "loaded_models_detail": loaded_details,
        "current_model": summary_mode,
        "context_length": QWEN_CONTEXT_LENGTH,
        "loaded_context_length": loaded_ctx,
        "is_downloading": MODEL_DOWNLOAD_STATE["is_downloading"],
        "is_loading_vram": MODEL_DOWNLOAD_STATE["is_loading_vram"],
        "task_progress": MODEL_DOWNLOAD_STATE["progress"],
        "task_message": MODEL_DOWNLOAD_STATE["message"],
        "uptime_seconds": uptime_seconds,
        "cu_burn_rate_per_hour": cu_rate,
        "session_cu_used": session_cu_used,
        "idle_timeout_minutes": IDLE_TIMEOUT_MINUTES,
        "idle_remaining_seconds": idle_remaining_seconds,
        "api_key_enabled": True,
        "api_key_configured": API_AUTH_KEY != INITIAL_API_KEY,
        "api_request_count": API_REQUEST_COUNT,
        "console_lines": get_live_console_Lines(),
        "custom_models": _get_custom_models_public_list(),
    }
    for k in (
        "ram_total_gb", "ram_used_gb", "ram_free_gb",
        "disk_total_gb", "disk_used_gb", "disk_free_gb",
        "shm_total_gb", "shm_used_gb", "shm_free_gb",
        "drive_total_gb", "drive_used_gb", "drive_free_gb",
    ):
        base[k] = hw.get(k, 0.0)
    base.update(file_status)
    if _has_rclone_drive_remote():
        base["drive_persisted"] = True
    base["comfyui"] = get_comfyui_status_dict()
    return base


def get_gpu_and_model_summary_text() -> str:
    info = get_gpu_status_dict()
    lines = []
    if not info["cuda_available"]:
        lines.append("GPU 状态: 未检测到 GPU（请在 Colab 中切换为 A100 / L4 GPU）")
    else:
        lines.append(
            f"GPU: {info['gpu_name']} | 显存可用: {info['vram_free_gb']:.1f} / {info['vram_total_gb']:.1f} GB"
        )
    if info.get("ram_total_gb", 0) > 0:
        lines.append(
            f"实例内存 (RAM): 已用 {info['ram_used_gb']:.1f} GB | 可用 {info['ram_free_gb']:.1f} / {info['ram_total_gb']:.1f} GB"
        )
    if info.get("disk_total_gb", 0) > 0:
        shm_part = (
            f" | /dev/shm 内存盘可用: {info['shm_free_gb']:.1f} / {info['shm_total_gb']:.1f} GB"
            if info.get("shm_total_gb", 0) >= 1.0
            else ""
        )
        lines.append(
            f"实例磁盘: 已用 {info['disk_used_gb']:.1f} GB | 可用 {info['disk_free_gb']:.1f} / {info['disk_total_gb']:.1f} GB{shm_part}"
        )
    lines.append(
        f"算力计费: 费率 ≈ {info['cu_burn_rate_per_hour']:.2f} CU/小时 | 本次开机已用: {info['session_cu_used']:.2f} CU (运行 {info['uptime_seconds']//60} 分钟)"
    )
    ram_tag = " + /dev/shm 内存盘(2.4GB/s)" if info.get("ramdisk_enabled") else ""
    storage_tag = ("Google Drive 云盘持久化" if info["drive_persisted"] else "Colab 高速磁盘") + ram_tag
    lines.append(
        f"当前模型 [{info['selected_model_name']}] 权重 [{storage_tag}]: 已就绪 {info['downloaded_count']}/{info['total_count']} 个文件 ({info['total_size_gb']:.2f} GB)"
    )
    for f in info["files"][:8]:
        mark = "[已就绪]" if f["exists"] else "[未下载]"
        sz = f"({f['size_gb']:.2f} GB)" if f["exists"] else ""
        lines.append(f"   {mark} {f['name']} {sz}")
    if len(info["files"]) > 8:
        lines.append(f"   ... 其余 {len(info['files']) - 8} 个分片文件")

    vram_state = f"已加载进显存 ({info['current_model']})" if info["any_model_loaded"] else "尚未加载进显存"
    lines.append(f"显存引擎状态: {vram_state}")
    return "\n".join(lines)


def build_structured_prompt(
    visual_prompt: str,
    soundscape_prompt: str,
    music_prompt: str,
) -> str:
    visual_prompt = (visual_prompt or "").strip()
    soundscape_prompt = (soundscape_prompt or "").strip()
    music_prompt = (music_prompt or "").strip()

    if not soundscape_prompt and not music_prompt:
        return visual_prompt

    parts = [visual_prompt]
    if soundscape_prompt:
        parts.append(f"Soundscape & Dialogue: {soundscape_prompt}")
    if music_prompt:
        parts.append(f"Background Music: {music_prompt}")
    return "\n".join(parts)


# ==============================================================================
# 独立功能 1：多模型通用极速下载器 (8路并发 HTTP Range + 后台 Google Drive 备份)
# ==============================================================================
def _async_backup_to_drive(src_file: str, dst_dir: str, fname: str):
    """在后台线程中将内存盘/NVMe下载好的权重同步到 Google Drive，不阻塞用户加载模型"""
    if not DRIVE_MOUNTED:
        return
    try:
        dst_file = os.path.join(dst_dir, fname)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        if os.path.exists(dst_file) and os.path.getsize(dst_file) == os.path.getsize(src_file):
            return
        tmp_dst = dst_file + ".syncing"
        with open(src_file, "rb") as rf, open(tmp_dst, "wb") as wf:
            shutil.copyfileobj(rf, wf, length=32 * 1024 * 1024)
        os.replace(tmp_dst, dst_file)
        print(f"[后台云盘同步完成] {fname} 已持久化保存至 Google Drive")
    except Exception as e:
        print(f"[后台云盘同步跳过] {fname}: {e}")


def download_model_weights_core(model_id: Optional[str] = None, hf_token: str = "") -> Tuple[bool, str]:
    global SELECTED_MODEL_ID
    touch_activity()
    if MODEL_DOWNLOAD_STATE["is_downloading"]:
        return False, "模型正在下载中，请稍候..."

    target_id = model_id if (model_id and model_id in MODEL_CATALOG) else SELECTED_MODEL_ID
    SELECTED_MODEL_ID = target_id
    spec = MODEL_CATALOG[target_id]
    repo_id = spec["repo_id"]
    req_files = spec["files"]

    MODEL_DOWNLOAD_STATE["is_downloading"] = True
    MODEL_DOWNLOAD_STATE["progress"] = 0.02
    MODEL_DOWNLOAD_STATE["message"] = f"正在为 {spec['name']} 启动 8 路并发 HTTP Range 极速下载通道..."
    with CONSOLE_LOCK:
        ACTIVE_TQDM_BARS.clear()

    try:
        if hf_token and hf_token.strip():
            os.environ["HF_TOKEN"] = hf_token.strip()
            from huggingface_hub import login
            login(token=hf_token.strip(), add_to_git_credential=False)

        ram_dir, nvme_dir, drive_dir = _get_model_dirs(target_id)
        drive_has_space = False
        if DRIVE_MOUNTED and os.path.exists("/content/drive/MyDrive"):
            try:
                drive_free_gb = shutil.disk_usage("/content/drive/MyDrive").free / (1024 ** 3)
                drive_has_space = drive_free_gb >= spec["approx_size_gb"] + 2.0
                if not drive_has_space:
                    print(
                        f"ℹ️ Google Drive 剩余空间 ({drive_free_gb:.1f} GB) 小于 {spec['name']} 所需容量 "
                        f"({spec['approx_size_gb']:.1f} GB)，自动改用实例本地高速 NVMe 磁盘 ({nvme_dir}) 存储！"
                    )
            except Exception:
                drive_has_space = False

        primary_dl_dir = ram_dir if ram_dir else (drive_dir if (DRIVE_MOUNTED and drive_has_space) else nvme_dir)
        os.makedirs(primary_dl_dir, exist_ok=True)
        total_files = len(req_files)

        for idx, (fname, desc) in enumerate(req_files):
            touch_activity()
            existing = _find_existing_model_file(target_id, fname)
            base_prog = idx / float(total_files + 1)
            MODEL_DOWNLOAD_STATE["progress"] = round(base_prog + 0.04, 2)

            if existing:
                # 若使用 Transformers / Diffusers 目录加载或分片 GGUF，确保同一目录内有软链接以便按目录直接加载
                pipeline_arch = spec.get("pipeline_arch") or target_id
                if pipeline_arch in ("qwen38_27b", "transformers", "diffusers_image", "gguf"):
                    target_link = os.path.join(primary_dl_dir, fname)
                    os.makedirs(os.path.dirname(target_link), exist_ok=True)
                    if os.path.abspath(existing) != os.path.abspath(target_link) and not os.path.exists(target_link):
                        try:
                            os.symlink(existing, target_link)
                        except Exception:
                            pass
                size_gb = os.path.getsize(existing) / (1024 ** 3)
                MODEL_DOWNLOAD_STATE["message"] = f"[{idx+1}/{total_files}] {fname} 已就绪 ({size_gb:.2f} GB)，跳过下载"
                print(MODEL_DOWNLOAD_STATE["message"])
                continue

            MODEL_DOWNLOAD_STATE["message"] = f"[{idx+1}/{total_files}] 正在 8 线程并发下载 {fname} ({desc})..."
            print(MODEL_DOWNLOAD_STATE["message"])
            dl_path = fast_parallel_hf_download(
                repo_id=repo_id,
                filename=fname,
                dest_dir=primary_dl_dir,
                hf_token=hf_token,
                num_connections=8,
            )
            if DRIVE_MOUNTED and drive_has_space and primary_dl_dir != drive_dir:
                threading.Thread(
                    target=_async_backup_to_drive,
                    args=(dl_path, drive_dir, fname),
                    daemon=True,
                ).start()

        MODEL_DOWNLOAD_STATE["progress"] = 0.92
        pipeline_arch = spec.get("pipeline_arch") or target_id
        if pipeline_arch in ("qwen38_27b_gguf", "gguf"):
            MODEL_DOWNLOAD_STATE["message"] = "正在预装并校验 llama.cpp CUDA 高速推理引擎..."
            _ensure_llama_server_binary()
        elif pipeline_arch == "minimax_h3_image":
            MODEL_DOWNLOAD_STATE["message"] = "正在预装 MiniMax-H3-Image 的 SGLang 推理环境（首次约 5 分钟）..."
            _ensure_h3_image_sglang_env()
        else:
            MODEL_DOWNLOAD_STATE["message"] = f"正在获取 {spec['name']} 的分词器与 Processor 配置文件..."
            from huggingface_hub import snapshot_download

            if pipeline_arch == "minimax_h3":
                from diffsynth.pipelines.minimax_h3_audio_video import ModelConfig
                proc_cfg = ModelConfig(
                    model_id=repo_id or "MiniMaxAI/MiniMax-H3",
                    origin_file_pattern="FL2VA/processor/",
                    download_source="huggingface",
                )
                proc_cfg.download_if_necessary()
            elif pipeline_arch in ("qwen38_27b", "transformers", "diffusers_image"):
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=primary_dl_dir,
                    ignore_patterns=["*.safetensors", "*.bin", "*.pth", "*.gguf", "*.pt"],
                    token=(hf_token or os.environ.get("HF_TOKEN") or "").strip() or None,
                )
            elif pipeline_arch == "qwen_image_21":
                from diffsynth.pipelines.qwen_image_21 import ModelConfig
                proc_cfg = ModelConfig(
                    model_id=repo_id or "Qwen/Qwen-Image-2.1",
                    origin_file_pattern="processor/",
                    download_source="huggingface",
                )
                proc_cfg.download_if_necessary()
            elif pipeline_arch == "z_image_turbo":
                from diffsynth.pipelines.z_image import ModelConfig
                tok_cfg = ModelConfig(
                    model_id=repo_id or "Tongyi-MAI/Z-Image-Turbo",
                    origin_file_pattern="tokenizer/",
                    download_source="huggingface",
                )
                tok_cfg.download_if_necessary()

        MODEL_DOWNLOAD_STATE["progress"] = 1.0
        status = check_model_files_status(target_id)
        msg = f"{spec['name']} 全部 {status['downloaded_count']}/{status['total_count']} 个权重文件已就绪（共 {status['total_size_gb']:.2f} GB），现在可点击「加载到显存」！"
        MODEL_DOWNLOAD_STATE["message"] = msg
        print(msg)
        return True, msg

    except Exception as e:
        err_msg = f"下载 {spec['name']} 权重出错: {str(e)}"
        MODEL_DOWNLOAD_STATE["message"] = err_msg
        print(err_msg)
        return False, err_msg
    finally:
        with _TELEMETRY_LOCK:
            _CACHED_FILE_STATUS_MAP.clear()
        MODEL_DOWNLOAD_STATE["is_downloading"] = False


def download_minimax_h3_weights_core(hf_token: str = "") -> Tuple[bool, str]:
    return download_model_weights_core("minimax_h3", hf_token)


# ==============================================================================
# 独立功能 2：多模型通用显存加载引擎 (自动卸载旧模型释放显存 + 零拷贝极速加载)
# ==============================================================================
def _resolve_and_stage_model_file(model_id: str, fname: str) -> str:
    """解析模型权重路径；若位于较慢的云盘且 /dev/shm 内存盘充裕，自动预热至内存盘"""
    ram_dir, nvme_dir, drive_dir = _get_model_dirs(model_id)
    existing = _find_existing_model_file(model_id, fname)
    if not existing:
        return os.path.join(drive_dir, fname)

    if ram_dir and not existing.startswith("/dev/shm/"):
        ram_target = os.path.join(ram_dir, fname)
        if os.path.exists(ram_target) and os.path.getsize(ram_target) == os.path.getsize(existing):
            return ram_target
        try:
            os.makedirs(os.path.dirname(ram_target), exist_ok=True)
            free_shm = _safe_shm_available_bytes(reserve_ram_gb=34.0)
            file_sz = os.path.getsize(existing)
            if free_shm > file_sz + (2 * 1024 ** 3):
                t0 = time.time()
                print(f"正在将 {fname} ({file_sz/(1024**3):.2f} GB) 高速预热至 /dev/shm 内存盘...")
                tmp_ram = ram_target + ".staging"
                with open(existing, "rb") as rf, open(tmp_ram, "wb") as wf:
                    shutil.copyfileobj(rf, wf, length=32 * 1024 * 1024)
                os.replace(tmp_ram, ram_target)
                dt = max(0.01, time.time() - t0)
                print(f"{fname} 内存盘预热完成 ({file_sz/(1024**2)/dt:.0f} MB/s)")
                return ram_target
        except Exception as e:
            print(f"内存盘预热跳过 ({e})，直接读取原路径")
    return existing


def _stage_model_files_to_local_nvme(
    model_id: str,
    paths: List[str],
    rel_names: List[str],
    group_label: str,
) -> List[str]:
    """当分片仍需走磁盘映射且原文件位于慢速云盘 FUSE (/content/drive) 时，若本地 NVMe 空间充裕则整组顺序缓存至本地高速 SSD"""
    if not paths or len(paths) != len(rel_names):
        return paths
    _, nvme_dir, _ = _get_model_dirs(model_id)
    if not any(p.startswith("/content/drive/") for p in paths):
        return paths

    try:
        os.makedirs(nvme_dir, exist_ok=True)
        free_nvme_bytes = shutil.disk_usage(nvme_dir).free
        needed_bytes = 0
        stage_plan: List[Tuple[int, str, str, int]] = []
        result_paths = list(paths)

        for idx, (src_p, rel_name) in enumerate(zip(paths, rel_names)):
            if not src_p.startswith("/content/drive/") or not os.path.exists(src_p):
                continue
            src_sz = os.path.getsize(src_p)
            dst_p = os.path.join(nvme_dir, rel_name)
            if os.path.exists(dst_p) and os.path.getsize(dst_p) == src_sz:
                result_paths[idx] = dst_p
                continue
            needed_bytes += src_sz
            stage_plan.append((idx, src_p, dst_p, src_sz))

        if not stage_plan:
            return result_paths

        # 保留至少 22GB 本地磁盘安全余量，避免挤占视频导出与系统空间
        if free_nvme_bytes <= needed_bytes + (22 * 1024 ** 3):
            return result_paths

        total_gb = needed_bytes / (1024 ** 3)
        print(f"⚡ [NVMe 高速缓存] 正在将 {group_label} ({len(stage_plan)} 个分片 / {total_gb:.1f} GB) 从云盘顺序预热至本地 NVMe SSD...")
        t_all = time.time()
        for step_i, (idx, src_p, dst_p, src_sz) in enumerate(stage_plan, start=1):
            os.makedirs(os.path.dirname(dst_p), exist_ok=True)
            MODEL_DOWNLOAD_STATE["message"] = (
                f"正在将 {group_label} 高速缓存至本地 SSD ({step_i}/{len(stage_plan)})，消除后续换入延迟..."
            )
            tmp_dst = dst_p + ".staging"
            with open(src_p, "rb") as rf, open(tmp_dst, "wb") as wf:
                shutil.copyfileobj(rf, wf, length=32 * 1024 * 1024)
            os.replace(tmp_dst, dst_p)
            result_paths[idx] = dst_p
        dt_all = max(0.05, time.time() - t_all)
        print(f"⚡ [NVMe 高速缓存] {group_label} 本地 SSD 预热完成 ({total_gb / dt_all * 1024:.0f} MB/s)")
        return result_paths
    except Exception as e:
        print(f"ℹ️ 本地 NVMe 预热跳过 ({e})，继续读取原路径")
        return paths


def _make_noproxy_opener():
    """本地回环请求 (127.0.0.1 llama-server) 强制绕过环境变量代理"""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _build_llama_ld_env(install_dir: str) -> Dict[str, str]:
    """收集 llama.cpp 动态库目录 + PyTorch 内置 NVIDIA CUDA Runtime/cuBLAS 目录"""
    env = os.environ.copy()
    ld_dirs: List[str] = []
    if os.path.isdir(install_dir):
        for root, _, files in os.walk(install_dir):
            if any(f == "llama-server" or ".so" in f for f in files):
                if root not in ld_dirs:
                    ld_dirs.append(root)

    try:
        import site
        site_roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    except Exception:
        site_roots = []
    site_roots.append(os.path.dirname(os.path.dirname(torch.__file__)))

    for sp in site_roots:
        nv_root = os.path.join(sp, "nvidia")
        if os.path.isdir(nv_root):
            for sub in os.listdir(nv_root):
                lib_d = os.path.join(nv_root, sub, "lib")
                if os.path.isdir(lib_d) and lib_d not in ld_dirs:
                    ld_dirs.append(lib_d)

    for sys_lib in ("/usr/lib64-nvidia", "/usr/local/nvidia/lib64", "/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"):
        if os.path.isdir(sys_lib) and sys_lib not in ld_dirs:
            ld_dirs.append(sys_lib)

    prev_ld = env.get("LD_LIBRARY_PATH", "")
    if prev_ld:
        ld_dirs.append(prev_ld)
    env["LD_LIBRARY_PATH"] = ":".join(ld_dirs)
    return env


def _download_file_to_path(url: str, dst_path: str, desc: str = ""):
    req = urllib.request.Request(url, headers={"User-Agent": "EasyColab-Engine/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dst_path, "wb") as wf:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            wf.write(chunk)
            done += len(chunk)
            if desc and total > 0:
                pct = int(done * 100 / total)
                MODEL_DOWNLOAD_STATE["message"] = f"{desc} ({done/(1024**2):.0f}/{total/(1024**2):.0f} MB · {pct}%)"


def _ensure_llama_server_binary() -> Tuple[str, Dict[str, str]]:
    """
    自动安装/校验官方 ggml-org/llama.cpp Linux x64 CUDA 12.8 预编译 llama-server + cudart 二进制。
    固定使用 ubuntu-cuda-12.x（NVIDIA 驱动向下兼容 CUDA 12.x/13.0/13.5+ 全系驱动，避免 CUDA 13.5 PTX 在 Colab 13.0 驱动上报 unsupported toolchain）。
    """
    install_dir = (
        "/content/llama_cpp_cuda"
        if os.path.exists("/content")
        else os.path.join(os.getcwd(), "llama_cpp_cuda")
    )
    os.makedirs(install_dir, exist_ok=True)

    def _find_server_bin() -> Optional[str]:
        for root, _, files in os.walk(install_dir):
            if "llama-server" in files:
                cand = os.path.join(root, "llama-server")
                try:
                    os.chmod(cand, 0o755)
                except Exception:
                    pass
                return cand
        return None

    def _has_cudart12() -> bool:
        for _, _, files in os.walk(install_dir):
            if any(f.startswith("libcudart.so.12") or f.startswith("libcublas.so.12") for f in files):
                return True
        return False

    def _verify_server_bin(bin_p: str, run_env: Dict[str, str]) -> Tuple[bool, str]:
        try:
            proc = subprocess.run(
                [bin_p, "--version"],
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=15,
            )
            if proc.returncode != 0:
                return False, (proc.stdout or "").strip()
            ver_txt = (proc.stdout or "").strip()
            if torch.cuda.is_available():
                dev_proc = subprocess.run(
                    [bin_p, "--list-devices"],
                    env=run_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=15,
                )
                dev_txt = (dev_proc.stdout or "").strip()
                if dev_proc.returncode != 0 or "CUDA0" not in dev_txt:
                    return False, f"CUDA 设备未挂载 ({dev_txt[:200]})"
            return True, ver_txt
        except Exception as ex:
            return False, str(ex)

    existing_bin = _find_server_bin()
    if existing_bin and _has_cudart12():
        env = _build_llama_ld_env(install_dir)
        ok, _ = _verify_server_bin(existing_bin, env)
        if ok:
            return existing_bin, env

    shutil.rmtree(install_dir, ignore_errors=True)
    os.makedirs(install_dir, exist_ok=True)

    cuda_prefix = "ubuntu-cuda-12."
    fallback_tag = "b12180"
    main_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/llama-{fallback_tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"
    cudart_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/cudart-llama-{fallback_tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"

    try:
        api_req = urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "EasyColab-Engine/1.0"},
        )
        with urllib.request.urlopen(api_req, timeout=10) as resp:
            releases = json.loads(resp.read().decode("utf-8"))
        for rel in releases:
            tag = str(rel.get("tag_name") or "")
            if not tag.startswith("b"):
                continue
            found_main, found_rt = None, None
            for asset in rel.get("assets") or []:
                aname = str(asset.get("name") or "")
                aurl = str(asset.get("browser_download_url") or "")
                if not aurl or not aname.endswith("-x64.tar.gz"):
                    continue
                if aname.startswith("llama-") and cuda_prefix in aname:
                    found_main = aurl
                elif aname.startswith("cudart-llama-") and cuda_prefix in aname:
                    found_rt = aurl
            if found_main:
                main_url = found_main
                if found_rt:
                    cudart_url = found_rt
                break
    except Exception as api_err:
        print(f"[GGUF 引擎] GitHub Releases 查询跳过，使用稳定镜像版本 {fallback_tag} ({api_err})")

    print(f"[GGUF 引擎] 正在下载官方预编译 llama.cpp CUDA 12 推理后端: {os.path.basename(main_url)} ...")
    tar_path = os.path.join(install_dir, "llama_cuda_bin.tar.gz")
    _download_file_to_path(main_url, tar_path, desc="正在下载 llama.cpp CUDA 引擎")
    subprocess.check_call(["tar", "-xzf", tar_path, "-C", install_dir])
    try:
        os.remove(tar_path)
    except Exception:
        pass

    if cudart_url:
        print(f"[GGUF 引擎] 正在下载配套 CUDA 12 Runtime 动态库 ({os.path.basename(cudart_url)})...")
        rt_tar = os.path.join(install_dir, "cudart_bin.tar.gz")
        _download_file_to_path(cudart_url, rt_tar, desc="正在下载 CUDA Runtime 动态库")
        subprocess.check_call(["tar", "-xzf", rt_tar, "-C", install_dir])
        try:
            os.remove(rt_tar)
        except Exception:
            pass

    bin_path = _find_server_bin()
    if not bin_path:
        raise RuntimeError("解压 llama.cpp CUDA 归档后未找到 llama-server 可执行文件")

    env = _build_llama_ld_env(install_dir)
    ok, ver_out = _verify_server_bin(bin_path, env)
    if not ok:
        raise RuntimeError(f"llama-server CUDA 启动自检失败: {ver_out[:300]}")

    first_line = ver_out.splitlines()[0] if ver_out else "ready"
    print(f"[GGUF 引擎] llama-server CUDA 引擎校验通过 ({first_line})")
    return bin_path, env


def _pick_free_local_port() -> int:
    """在非临时端口段 (18080~18999) 选取空闲本地监听端口，规避 Colab 容器 ephemeral 端口段 (32768+) 抢占冲突。"""
    for cand in range(18080, 18999):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", cand))
                return cand
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _start_qwen_gguf_server(
    gguf_path: str,
    mmproj_path: Optional[str],
    ctx_len: int,
    cur_free_gb: float,
    mtp_path: Optional[str] = None,
    model_id: str = "qwen38_27b_gguf",
    model_name: str = "Qwen3.8-27B",
    total_weights_gb: Optional[float] = None,
) -> Dict[str, Any]:
    """
    启动驻留 GPU 的 llama-server 实例加载 GGUF 主模型 + 可选 mmproj 视觉投影器 + 可选 MTP 投机解码头。
    集成 Unsloth Native MTP 投机解码 (--spec-type draft-mtp)、GPU 端采样 (--backend-sampling)、
    超大 MoE 专家权重自动异构卸载 (-ot ".ffn_.*_exps.=CPU") 与 q8_0/q4_0 高带宽 KV Cache。
    """
    bin_path, env = _ensure_llama_server_binary()

    try:
        weights_gb = float(total_weights_gb) if (total_weights_gb and total_weights_gb > 0) else (os.path.getsize(gguf_path) / (1024 ** 3))
    except Exception:
        weights_gb = 18.0
    if mmproj_path and os.path.exists(mmproj_path):
        try:
            weights_gb += os.path.getsize(mmproj_path) / (1024 ** 3)
        except Exception:
            pass

    q8_kv_gb = (ctx_len / 262144.0) * 8.8
    q4_kv_gb = (ctx_len / 262144.0) * 4.6

    if cur_free_gb >= weights_gb + 1.5 + q8_kv_gb:
        primary_kv = "q8_0"
        primary_kv_gb = q8_kv_gb
    else:
        primary_kv = "q4_0"
        primary_kv_gb = q4_kv_gb

    has_mtp_file = bool(mtp_path and os.path.exists(mtp_path))

    # 计算精确 -ngl 层数并显式 --fit off，支持单卡装下全量 GPU 驻留，或超大 MoE / 大模型自动异构卸载
    if cur_free_gb >= (weights_gb + 1.2 + primary_kv_gb):
        primary_ngl = 999
        primary_moe_ot = False
    else:
        avail_for_weights = max(4.0, cur_free_gb - primary_kv_gb - 2.4)
        layer_unit_gb = max(0.20, weights_gb / 64.0)
        primary_ngl = max(8, min(64, int(avail_for_weights / layer_unit_gb)))
        primary_moe_ot = weights_gb > (cur_free_gb - 1.5)

    primary_mtp = has_mtp_file and primary_ngl == 999 and (cur_free_gb >= weights_gb + 2.8 + primary_kv_gb)
    # attempts 元素: (kv_type, ngl_val, use_mtp, moe_cpu_ot)
    attempts: List[Tuple[str, int, bool, bool]] = []
    if primary_moe_ot:
        # 优先尝试将 Attention 常驻 GPU (-ngl 999) + MoE 路由专家驻留大内存 (-ot .ffn_.*_exps.=CPU)
        attempts.append((primary_kv, 999, False, True))
    attempts.append((primary_kv, primary_ngl, primary_mtp, False))

    if has_mtp_file and not primary_mtp and not primary_moe_ot and (cur_free_gb >= weights_gb + 2.8 + q4_kv_gb):
        attempts.insert(0, ("q4_0", 999, True, False))

    if primary_ngl == 999:
        if primary_mtp:
            attempts.append((primary_kv, 999, False, False))
        fallback_ngl = max(12, min(64, int(max(4.0, cur_free_gb - q4_kv_gb - 2.8) / max(0.20, weights_gb / 64.0))))
        attempts.append(("q4_0", fallback_ngl, False, False))

    opener = _make_noproxy_opener()
    last_err = ""

    for attempt_idx, (kv_type, ngl_val, use_mtp, moe_ot) in enumerate(attempts):
        port = _pick_free_local_port()
        cmd = [
            bin_path,
            "-m", gguf_path,
            "--fit", "off",
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(ctx_len),
            "--kv-unified",
            "--parallel", "1",
            "--flash-attn", "on",
            "--cache-type-k", kv_type,
            "--cache-type-v", kv_type,
            "-b", "2048",
            "-ub", "1024",
            "--backend-sampling",
            "--samplers", "top_k;top_p;min_p;temperature",
            "--temp", "1.0",
            "--top-p", "0.95",
            "--top-k", "20",
            "--min-p", "0.0",
            "--repeat-penalty", "1.0",
            "--presence-penalty", "0.0",
            "--jinja",
            "--reasoning-format", "deepseek",
            "-ngl", str(ngl_val),
        ]
        if moe_ot:
            cmd.extend(["-ot", ".ffn_.*_exps.=CPU"])
        if use_mtp and mtp_path:
            cmd.extend([
                "--model-draft", mtp_path,
                "-ngld", "999",
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", "4",
                "--spec-draft-p-min", "0.45",
                "--cache-type-k-draft", kv_type,
                "--cache-type-v-draft", kv_type,
            ])
        if mmproj_path and os.path.exists(mmproj_path):
            cmd.extend(["--mmproj", mmproj_path, "--image-min-tokens", "1024"])

        if moe_ot:
            fit_desc = "MoE 异构驻留 (Attention GPU + Experts CPU)"
        elif ngl_val >= 99:
            fit_desc = "全量 GPU 驻留"
        else:
            fit_desc = f"混合驻留 ({ngl_val}/65 层 GPU)"
        mtp_desc = " + MTP 多词元投机解码" if use_mtp else ""
        print(
            f"[GGUF 引擎] 正在启动 {model_name}{mtp_desc} 服务 "
            f"(上下文 {ctx_len // 1024}K={ctx_len}, KV={kv_type}, GPU采样, {fit_desc}, port={port})..."
        )
        MODEL_DOWNLOAD_STATE["progress"] = 0.55
        MODEL_DOWNLOAD_STATE["message"] = (
            f"正在将 {model_name}{mtp_desc} 装载至 GPU ({ctx_len // 1024}K 上下文 · KV {kv_type} · {fit_desc})..."
        )

        recent_logs: List[str] = []
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _drain_stdout(p: subprocess.Popen, log_buf: List[str]):
            try:
                assert p.stdout is not None
                for raw_line in p.stdout:
                    ln = raw_line.rstrip("\r\n")
                    if not ln:
                        continue
                    log_buf.append(ln)
                    if len(log_buf) > 100:
                        del log_buf[:25]
            except Exception:
                pass

        threading.Thread(target=_drain_stdout, args=(proc, recent_logs), daemon=True).start()

        health_url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + 240.0
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                with opener.open(health_url, timeout=2.0) as h_resp:
                    if h_resp.status == 200:
                        ready = True
                        break
            except urllib.error.HTTPError as he:
                # 503 表示 llama-server 正在加载权重到显存
                if he.code == 503:
                    MODEL_DOWNLOAD_STATE["progress"] = min(0.95, round(MODEL_DOWNLOAD_STATE["progress"] + 0.01, 2))
            except Exception:
                pass
            time.sleep(0.5)

        if ready:
            ctx_k_str = f"{ctx_len // 1024}K"
            if model_id == "qwen38_27b_gguf":
                quant_tag = "UD-Q4_K_M+MTP" if use_mtp else "UD-Q4_K_M"
                mode_label = f"Qwen3.8-27B · {quant_tag} · {ctx_k_str} (KV:{kv_type})"
            else:
                extra_tag = " · MTP" if use_mtp else (" · MoE异构" if moe_ot else "")
                mode_label = f"{model_name} · GGUF{extra_tag} · {ctx_k_str} (KV:{kv_type})"
            print(f"[GGUF 引擎] {model_name} 已就绪 ({mode_label})")
            return {
                "backend": "gguf",
                "model_id": model_id,
                "model_name": model_name,
                "port": port,
                "base_url": f"http://127.0.0.1:{port}",
                "server_proc": proc,
                "context_length": ctx_len,
                "kv_type": kv_type,
                "mtp_enabled": use_mtp,
                "mode_label": mode_label,
                "gguf_path": gguf_path,
                "mmproj_path": mmproj_path,
                "mtp_path": mtp_path if use_mtp else None,
            }

        # 未就绪：清理子进程并记录日志
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        tail_txt = "\n".join(recent_logs[-12:])
        last_err = tail_txt or "llama-server 启动超时"
        if attempt_idx + 1 < len(attempts):
            print(f"[GGUF 引擎] 当前配置 (KV={kv_type}, ngl={ngl_val}, MTP={use_mtp}, MoE_OT={moe_ot}) 显存触顶，自动降级重试...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.5)

    raise RuntimeError(f"{model_name} GGUF 服务启动失败: {last_err[-400:]}")


def _sync_legacy_globals():
    """将多模型显存驻留表 LOADED_RUNTIMES 同步至兼容全局变量 PIPELINE / LLM_BUNDLE"""
    global PIPELINE, LLM_BUNDLE, LOADED_MODEL_ID, CURRENT_MODEL_MODE
    LLM_BUNDLE = None
    if _is_llm_model_id(SELECTED_MODEL_ID) and SELECTED_MODEL_ID in LOADED_RUNTIMES:
        LLM_BUNDLE = LOADED_RUNTIMES[SELECTED_MODEL_ID].get("llm_bundle")
    if LLM_BUNDLE is None:
        for rt in reversed(list(LOADED_RUNTIMES.values())):
            if rt.get("llm_bundle") is not None:
                LLM_BUNDLE = rt["llm_bundle"]
                break

    sel_rt = LOADED_RUNTIMES.get(SELECTED_MODEL_ID)
    if sel_rt and sel_rt.get("pipeline") is not None:
        PIPELINE = sel_rt["pipeline"]
    else:
        PIPELINE = None
        for rt in reversed(list(LOADED_RUNTIMES.values())):
            if rt.get("pipeline") is not None:
                PIPELINE = rt["pipeline"]
                break

    if SELECTED_MODEL_ID in LOADED_RUNTIMES:
        LOADED_MODEL_ID = SELECTED_MODEL_ID
    elif LOADED_RUNTIMES:
        LOADED_MODEL_ID = next(reversed(LOADED_RUNTIMES))
    else:
        LOADED_MODEL_ID = None

    if LOADED_RUNTIMES:
        CURRENT_MODEL_MODE = " + ".join(rt["mode_label"] for rt in LOADED_RUNTIMES.values())
    else:
        CURRENT_MODEL_MODE = None


def _trim_linux_cpu_heap():
    """在释放超大规模 CPU BF16 权重张量后主动触发 glibc malloc_trim(0)，将物理页立即归还给系统 MemAvailable"""
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def _unload_model_locked(model_id: Optional[str] = None) -> List[str]:
    """在已持有 GPU_LOCK 的前提下卸载指定模型（或全部模型），返回被卸载的模型名称列表"""
    global PIPELINE, LLM_BUNDLE, LOADED_MODEL_ID, CURRENT_MODEL_MODE
    removed_names: List[str] = []
    if not model_id or model_id == "all":
        for mid in list(LOADED_RUNTIMES.keys()):
            entry = LOADED_RUNTIMES.pop(mid, None)
            if entry:
                _stop_gguf_server_bundle(entry.get("llm_bundle"))
                _stop_h3_image_server(entry)
                removed_names.append(entry.get("name", mid))
                entry.clear()
        _stop_gguf_server_bundle(LLM_BUNDLE)
        PIPELINE = None
        LLM_BUNDLE = None
        LOADED_MODEL_ID = None
        CURRENT_MODEL_MODE = None
    elif model_id in LOADED_RUNTIMES:
        entry = LOADED_RUNTIMES.pop(model_id, None)
        if entry:
            _stop_gguf_server_bundle(entry.get("llm_bundle"))
            _stop_h3_image_server(entry)
            removed_names.append(entry.get("name", model_id))
            if PIPELINE is entry.get("pipeline"):
                PIPELINE = None
            if LLM_BUNDLE is entry.get("llm_bundle"):
                LLM_BUNDLE = None
            entry.clear()
        _sync_legacy_globals()

    _trim_linux_cpu_heap()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return removed_names


def _unload_all_gpu_models():
    """彻底释放全部已加载模型的 GPU 显存"""
    _unload_model_locked("all")


def unload_model_from_vram_core(model_id: Optional[str] = None) -> Tuple[bool, str]:
    """供客户端按需独立卸载单个或全部显存驻留模型（立即中断正在进行的推理任务以避免死锁）"""
    touch_activity()
    ABORT_GENERATION_EVENT.set()
    for jid, j in list(JOBS.items()):
        if j.get("status") in ("queued", "running") and (not model_id or model_id == "all" or j.get("model_id") == model_id):
            ev = _JOB_CANCEL_EVENTS.get(jid)
            if ev is not None:
                ev.set()
            j["status"] = "failed"
            j["progress"] = 0.0
            j["message"] = "已终止推理并从显存卸载该模型"
    with CONSOLE_LOCK:
        ACTIVE_TQDM_BARS.clear()
    with GPU_LOCK:
        try:
            removed = _unload_model_locked(model_id)
            free_gb = (torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)) if torch.cuda.is_available() else 0.0
            if not removed:
                msg = "当前指定模型未驻留显存，无需卸载。"
            else:
                remain_cnt = len(LOADED_RUNTIMES)
                remain_str = f"（显存中仍保留 {remain_cnt} 个常驻模型）" if remain_cnt > 0 else "（显存已完全清空）"
                msg = f"已从显存卸载 {', '.join(removed)}，当前可用显存 {free_gb:.1f} GB {remain_str}"
            MODEL_DOWNLOAD_STATE["message"] = msg
            print(f"[显存管理] {msg}")
            return True, msg
        finally:
            ABORT_GENERATION_EVENT.clear()


def _ensure_vram_for_new_model_locked(target_id: str, required_free_gb: float) -> List[str]:
    """
    多模型共驻智能显存守卫 (LRU Eviction)：
    - 若目标模型自身已在显存中，先卸载旧实例以便重载；
    - 若剩余可用显存 >= required_free_gb，保留所有已加载模型实现多模型同时常驻！
    - 仅当剩余显存不足以容纳新模型时，按最早加载顺序 (LRU) 自动腾退旧模型，防止 CUDA OOM。
    """
    evicted: List[str] = []
    if target_id in LOADED_RUNTIMES:
        _unload_model_locked(target_id)

    if not torch.cuda.is_available():
        return evicted

    # MiniMax-H3-Image 空闲时显存只占几 GB，出图时却要换入约 66GB：新模型留不出这块空间就先卸载它
    total_gb = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)
    for mid, rt in list(LOADED_RUNTIMES.items()):
        if rt.get("server_proc") is not None and required_free_gb > total_gb - 68.0:
            evicted.extend(_unload_model_locked(mid))

    free_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
    while free_gb < required_free_gb and LOADED_RUNTIMES:
        oldest_id = next(iter(LOADED_RUNTIMES))
        names = _unload_model_locked(oldest_id)
        evicted.extend(names)
        free_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
    return evicted


class _CausalLMProcessorAdapter:
    """将纯文本 AutoTokenizer 适配为与多模态 AutoProcessor 统一接口，兼容自定义 CausalLM 仓库"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        flat_msgs: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            c = m.get("content", "")
            if isinstance(c, list):
                txt = "\n".join(
                    str(b.get("text", ""))
                    for b in c
                    if isinstance(b, dict) and b.get("type") == "text"
                )
                flat_msgs.append({"role": role, "content": txt})
            else:
                flat_msgs.append({"role": role, "content": str(c)})
        try:
            return self.tokenizer.apply_chat_template(flat_msgs, **kwargs)
        except Exception:
            parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in flat_msgs]
            parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)

    def __call__(self, text=None, images=None, padding=True, return_tensors="pt", **kwargs):
        return self.tokenizer(text, padding=padding, return_tensors=return_tensors)


def _ensure_h3_image_sglang_env() -> str:
    """在独立虚拟环境里安装固定提交的 SGLang，返回其可执行文件路径；不改动主环境的 torch / transformers。"""
    root = H3_IMAGE_SGLANG_DIR
    sglang_bin = os.path.join(root, "venv", "bin", "sglang")
    if os.path.exists(sglang_bin):
        return sglang_bin
    os.makedirs(root, exist_ok=True)
    src = os.path.join(root, "sglang")
    env = dict(
        os.environ,
        PATH=os.path.expanduser("~/.cargo/bin") + os.pathsep + os.environ.get("PATH", ""),
        SGLANG_BUILD_RUST_EXTS="none",
    )
    steps = [
        # 依赖 outlines-core 在 Python 3.13 上没有预编译包，需要 Rust 工具链；SGLang 首个请求还会用 ninja 编译算子
        "command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
        "command -v uv >/dev/null || pip install -q uv",
        "command -v ninja >/dev/null || pip install -q ninja",
        f"rm -rf '{src}' && git init -q '{src}' && git -C '{src}' fetch -q --depth 1 "
        f"https://github.com/sgl-project/sglang.git {H3_IMAGE_SGLANG_COMMIT} && git -C '{src}' checkout -q FETCH_HEAD",
        f"rm -rf '{root}/venv' && uv venv -q --system-site-packages '{root}/venv'",
        f"uv pip install -q --python '{root}/venv/bin/python' -e '{src}/python[diffusion]'",
    ]
    for cmd in steps:
        touch_activity()
        r = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"SGLang 推理环境安装失败: {(r.stderr or r.stdout)[-600:]}")
    return sglang_bin


def _build_h3_image_model_dir() -> str:
    """把分散在内存盘 / 本地盘 / 云盘的权重软链接成 SGLang 需要的标准目录结构，并补齐配置与分词器小文件。"""
    from huggingface_hub import snapshot_download

    view = os.path.join(H3_IMAGE_SGLANG_DIR, "model")
    snapshot_download(
        repo_id=MODEL_CATALOG["minimax_h3_image"]["repo_id"],
        local_dir=view,
        allow_patterns=["*.json", "*.txt"],
        ignore_patterns=["examples/*", "*_diffusers/*"],
    )
    for fname, _ in MINIMAX_H3_IMAGE_FILES:
        link = os.path.join(view, fname)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(_find_existing_model_file("minimax_h3_image", fname), link)
    return view


def _start_h3_image_server() -> subprocess.Popen:
    """启动本机 SGLang 服务并等待其就绪。不指定驻留方式：96GB 显卡上 SGLang 默认按组件换入换出（峰值约 66GB）。"""
    sglang_bin = _ensure_h3_image_sglang_env()
    model_dir = _build_h3_image_model_dir()
    log_path = os.path.join(H3_IMAGE_SGLANG_DIR, "serve.log")
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            [
                sglang_bin, "serve",
                "--model-path", model_dir,
                # model_index.json 把文本编码器指向 MiniMaxAI/MiniMax-H3 仓库；不显式给路径会重新下载 62GB
                "--text-encoder-path", os.path.join(model_dir, "text_encoder"),
                "--backend", "sglang",
                "--host", "127.0.0.1",
                "--port", str(H3_IMAGE_SGLANG_PORT),
            ],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=H3_IMAGE_SGLANG_DIR,
            start_new_session=True,
        )

    def _log_tail() -> str:
        with open(log_path, "r", errors="ignore") as f:
            return f.read()[-600:]

    deadline = time.time() + 1800
    while time.time() < deadline:
        touch_activity()
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang 服务启动失败: {_log_tail()}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{H3_IMAGE_SGLANG_PORT}/health", timeout=5)
            return proc
        except Exception:
            time.sleep(5)
    _stop_h3_image_server({"server_proc": proc})
    raise RuntimeError(f"SGLang 服务启动超时: {_log_tail()}")


def _stop_h3_image_server(entry: Optional[Dict[str, Any]]):
    proc = entry.pop("server_proc", None) if entry else None
    if proc is None or proc.poll() is not None:
        return
    # SGLang 会派生工作子进程，按进程组结束才能把显存全部放掉
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5.0)
    except Exception:
        pass


def load_model_to_vram_core(
    model_id: Optional[str] = None,
    precision_mode: str = "bf16",
    offload_mode: str = "disk",
    hf_token: str = "",
    context_length: Optional[int] = None,
    h3_dit_variant: str = "FL2VA",
) -> Tuple[bool, str]:
    global SELECTED_MODEL_ID, QWEN_CONTEXT_LENGTH
    touch_activity()

    if context_length is not None:
        QWEN_CONTEXT_LENGTH = _clamp_context_length(context_length)
    ctx_len = QWEN_CONTEXT_LENGTH

    if MODEL_DOWNLOAD_STATE["is_loading_vram"]:
        return False, "模型正在加载进显存中，请稍候..."

    if not torch.cuda.is_available():
        return False, "未检测到 CUDA GPU！请在 Colab 中切换为 G4 / A100 / L4 GPU。"

    target_id = model_id if (model_id and model_id in MODEL_CATALOG) else SELECTED_MODEL_ID
    SELECTED_MODEL_ID = target_id
    spec = MODEL_CATALOG[target_id]
    pipeline_arch = spec.get("pipeline_arch") or target_id
    file_status = check_model_files_status(target_id)
    auto_switched_notice = ""

    if target_id == "qwen38_27b_gguf":
        if not (
            _find_existing_model_file("qwen38_27b_gguf", QWEN_GGUF_MAIN_FILE)
            and _find_existing_model_file("qwen38_27b_gguf", QWEN_GGUF_MMPROJ_FILE)
        ):
            return False, f"{spec['name']} 权重尚未下载完整 ({file_status['downloaded_count']}/{file_status['total_count']})！请先点击「下载权重」。"
    elif pipeline_arch == "gguf":
        main_ggufs = [
            fn for fn, _ in spec.get("files", [])
            if fn.lower().endswith(".gguf") and "mmproj" not in fn.lower() and "/mtp" not in fn.lower() and not os.path.basename(fn.lower()).startswith("mtp")
        ]
        if not main_ggufs or not all(_find_existing_model_file(target_id, fn) for fn in main_ggufs):
            return False, f"{spec['name']} GGUF 主权重尚未下载完整 ({file_status['downloaded_count']}/{file_status['total_count']})！请先点击「下载权重」。"
    else:
        if not file_status["all_downloaded"]:
            return False, f"{spec['name']} 权重尚未下载完整 ({file_status['downloaded_count']}/{file_status['total_count']})！请先点击「下载权重」。"

    t_start = time.time()
    MODEL_DOWNLOAD_STATE["is_loading_vram"] = True
    MODEL_DOWNLOAD_STATE["progress"] = 0.12
    MODEL_DOWNLOAD_STATE["message"] = f"正在评估显存余量并装载 {spec['name']} 至 GPU..."
    with CONSOLE_LOCK:
        ACTIVE_TQDM_BARS.clear()

    with GPU_LOCK:
        try:
            total_vram_gb = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)
            vram_limit = max(8.0, total_vram_gb - 2.5)
            approx_gb = float(spec.get("approx_size_gb") or 20.0)

            # 计算新模型所需最小空闲显存阈值：余量充足时自动与已加载模型共驻显存！
            if target_id == "qwen38_27b":
                required_free_gb = 53.0
            elif pipeline_arch in ("qwen38_27b_gguf", "gguf"):
                q4_kv_gb = (ctx_len / 262144.0) * 4.6
                required_free_gb = min(max(12.0, approx_gb + 2.0 + q4_kv_gb), max(12.0, total_vram_gb - 1.2))
            elif pipeline_arch == "minimax_h3":
                # BF16 单阶段峰值约 62.1GB；在 G4 (96GB) 上可与 Qwen3.8-GGUF (~21GB) 同时共驻显存
                required_free_gb = min(64.5, max(24.0, total_vram_gb - 5.0))
            elif pipeline_arch == "minimax_h3_image":
                # SGLang 在 96GB 显卡上按组件换入换出，官方实测 4K 峰值约 66GB
                required_free_gb = min(68.0, max(24.0, total_vram_gb - 2.5))
            elif pipeline_arch == "transformers":
                required_free_gb = min(max(12.0, approx_gb + 3.0), max(12.0, total_vram_gb - 1.5))
            else:
                required_free_gb = 24.0

            if torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3) < required_free_gb and COMFYUI_STATE.get("is_running"):
                free_comfyui_vram_core()
            evicted_names = _ensure_vram_for_new_model_locked(target_id, required_free_gb)
            free_before_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)

            vram_config = {
                "offload_dtype": "disk",
                "offload_device": "disk",
                "onload_dtype": "disk",
                "onload_device": "disk",
                "preparing_dtype": torch.bfloat16,
                "preparing_device": "cuda",
                "computation_dtype": torch.bfloat16,
                "computation_device": "cuda",
            }

            new_pipe = None
            new_llm = None
            h3_image_proc = None
            mode_label = spec["name"]
            repo_id = spec.get("repo_id", "")

            if pipeline_arch == "minimax_h3_image":
                MODEL_DOWNLOAD_STATE["progress"] = 0.25
                MODEL_DOWNLOAD_STATE["message"] = "正在启动 MiniMax-H3-Image 的 SGLang 推理服务并装载 BF16 权重（约 5 分钟）..."
                h3_image_proc = _start_h3_image_server()
                mode_label = f"{spec['name']} · BF16 原生 · SGLang"

            elif pipeline_arch == "minimax_h3":
                from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

                _trim_linux_cpu_heap()
                total_ram_gb, avail_ram_gb = _get_system_ram_gb()
                cpu_fast_vram_config = {
                    "offload_dtype": torch.bfloat16,
                    "offload_device": "cpu",
                    "onload_dtype": torch.bfloat16,
                    "onload_device": "cpu",
                    "preparing_dtype": torch.bfloat16,
                    "preparing_device": "cuda",
                    "computation_dtype": torch.bfloat16,
                    "computation_device": "cuda",
                }

                te_rel_names = [f"FL2VA/text_encoder/model-{i:05d}-of-00014.safetensors" for i in range(1, 15)]
                dit_rel_names = MINIMAX_H3_DIT_FILES.get(h3_dit_variant, MINIMAX_H3_DIT_FILES["FL2VA"])
                missing_dit = [fn for fn in dit_rel_names if not _find_existing_model_file(target_id, fn)]
                if missing_dit:
                    raise RuntimeError(f"{h3_dit_variant} DiT 主干还缺 {len(missing_dit)} 个分片，请先下载")

                MODEL_DOWNLOAD_STATE["progress"] = 0.22
                MODEL_DOWNLOAD_STATE["message"] = "正在映射 MiniMax-H3 BF16 多模态编码器权重 (14 分片)..."
                te_paths = [_resolve_and_stage_model_file(target_id, fn) for fn in te_rel_names]

                MODEL_DOWNLOAD_STATE["progress"] = 0.42
                MODEL_DOWNLOAD_STATE["message"] = "正在映射 MiniMax-H3 BF16 33B DiT 主干权重 (13 分片)..."
                dit_paths = [_resolve_and_stage_model_file(target_id, fn) for fn in dit_rel_names]

                MODEL_DOWNLOAD_STATE["progress"] = 0.58
                MODEL_DOWNLOAD_STATE["message"] = "正在映射 3D 视频 VAE 与 32kHz 立体声 VAE 权重..."
                vvae_path = _resolve_and_stage_model_file(
                    target_id, "FL2VA/video_vae/source/model.safetensors"
                )
                avae_path = _resolve_and_stage_model_file(
                    target_id, "FL2VA/audio_vae/model.safetensors"
                )

                # 基于分片映射完成后的真实可用系统内存自适应选择热备策略（严守 >=24GB CPU 内存安全余量，杜绝 /dev/shm + 匿名内存叠加触发 Cgroup OOM）：
                # - Tier 1 (如 G4 176GB RAM, 可用 >=148GB)：text_encoder(62.1GB) + dit(61.7GB) 双主干常驻系统内存，PCIe 极速互换 (~4-6s)！
                # - Tier 2 (可用 88~148GB RAM)：text_encoder(62.1GB) 常驻系统内存 (~4-6s 极速解析提示词)，dit 顺序缓存至本地高速 NVMe！
                # - Tier 3 (低内存机型 <88GB RAM)：保持磁盘按需映射，若本地 NVMe 空间充裕则脱离慢速云盘 FUSE。
                total_ram_gb, avail_ram_gb = _get_system_ram_gb()
                ram_residency_tag = ""
                if avail_ram_gb >= 148.0:
                    te_cfg = cpu_fast_vram_config
                    dit_cfg = cpu_fast_vram_config
                    vae_cfg = cpu_fast_vram_config
                    ram_residency_tag = "双主干内存热备"
                    print(
                        f"⚡ [G4 大内存热备] 检测到充裕系统内存 ({avail_ram_gb:.1f}/{total_ram_gb:.1f} GB)，"
                        "已启用 32B 文本编码器 + 33B DiT 双主干 DDR5 内存常驻热备（阶段切换仅需 ~5 秒）！"
                    )
                    MODEL_DOWNLOAD_STATE["progress"] = 0.68
                    MODEL_DOWNLOAD_STATE["message"] = (
                        f"正在将 MiniMax-H3 预装载至 {total_ram_gb:.0f}GB 高速系统内存热备池（首次载入后换提示词仅需 ~5 秒）..."
                    )
                elif avail_ram_gb >= 88.0:
                    te_cfg = cpu_fast_vram_config
                    dit_paths = _stage_model_files_to_local_nvme(
                        target_id, dit_paths, dit_rel_names, "MiniMax-H3 33B DiT 主干"
                    )
                    dit_cfg = vram_config
                    vae_cfg = cpu_fast_vram_config
                    ram_residency_tag = "编码器内存热备"
                    print(
                        f"⚡ [混合内存热备] 系统可用内存 {avail_ram_gb:.1f}/{total_ram_gb:.1f} GB，"
                        "已将 32B 文本编码器锁入系统内存热备（换提示词 ~5 秒极速换入），DiT 走本地高速存储！"
                    )
                    MODEL_DOWNLOAD_STATE["progress"] = 0.68
                    MODEL_DOWNLOAD_STATE["message"] = (
                        "正在将 32B 文本编码器装载至系统内存热备池（后续解析新提示词仅需 ~5 秒）..."
                    )
                else:
                    te_paths = _stage_model_files_to_local_nvme(
                        target_id, te_paths, te_rel_names, "MiniMax-H3 多模态编码器"
                    )
                    dit_paths = _stage_model_files_to_local_nvme(
                        target_id, dit_paths, dit_rel_names, "MiniMax-H3 33B DiT 主干"
                    )
                    te_cfg = vram_config
                    dit_cfg = vram_config
                    vae_cfg = vram_config
                    MODEL_DOWNLOAD_STATE["progress"] = 0.72
                    MODEL_DOWNLOAD_STATE["message"] = "正在构建 MiniMax-H3 BF16 原生 GPU 计算图与 Processor..."

                new_pipe = MiniMaxH3Pipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device="cuda",
                    model_configs=[
                        ModelConfig(path=te_paths, **te_cfg),
                        ModelConfig(path=dit_paths, **dit_cfg),
                        ModelConfig(path=vvae_path, **vae_cfg),
                        ModelConfig(path=avae_path, **vae_cfg),
                    ],
                    processor_config=ModelConfig(
                        model_id=repo_id or "MiniMaxAI/MiniMax-H3",
                        origin_file_pattern="FL2VA/processor/",
                        download_source="huggingface",
                    ),
                    vram_limit=vram_limit,
                )
                mode_label = f"{spec['name']} · BF16 原生" + (" · Ref2VA 参考主干" if h3_dit_variant == "Ref2VA" else "")
                residency_tag = _enable_high_vram_optimizations(
                    new_pipe,
                    total_vram_gb=total_vram_gb,
                    preload_vram=True,
                    skip_preload_attrs=("dit",) if free_before_gb >= 76.0 else ("text_encoder", "dit"),
                )
                tags = [t for t in (ram_residency_tag, residency_tag) if t]
                if tags:
                    mode_label = f"{mode_label} · {' / '.join(tags)}"

            elif pipeline_arch == "qwen_image_21":
                from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
                MODEL_DOWNLOAD_STATE["progress"] = 0.45
                MODEL_DOWNLOAD_STATE["message"] = f"正在映射 {spec['name']} 权重并构建 GPU 管线..."
                dit_paths = [
                    _resolve_and_stage_model_file(target_id, f"transformer/diffusion_pytorch_model-{i:05d}-of-00002.safetensors")
                    for i in range(1, 3)
                ]
                te_paths = [
                    _resolve_and_stage_model_file(target_id, f"text_encoder/model-{i:05d}-of-00004.safetensors")
                    for i in range(1, 5)
                ]
                vae_path = _resolve_and_stage_model_file(target_id, "vae/diffusion_pytorch_model.safetensors")
                new_pipe = QwenImage21Pipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device="cuda",
                    model_configs=[
                        ModelConfig(path=dit_paths, **vram_config),
                        ModelConfig(path=te_paths, **vram_config),
                        ModelConfig(path=vae_path, **vram_config),
                    ],
                    processor_config=ModelConfig(
                        model_id=repo_id or "Qwen/Qwen-Image-2.1",
                        origin_file_pattern="processor/",
                        download_source="huggingface",
                    ),
                    vram_limit=vram_limit,
                )
                _enable_high_vram_optimizations(new_pipe, total_vram_gb=total_vram_gb, preload_vram=True)
                mode_label = f"{spec['name']} · BF16"

            elif pipeline_arch == "z_image_turbo":
                from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
                MODEL_DOWNLOAD_STATE["progress"] = 0.45
                MODEL_DOWNLOAD_STATE["message"] = f"正在映射 {spec['name']} 权重并构建 8 步极速生图管线..."
                dit_paths = [
                    _resolve_and_stage_model_file(target_id, f"transformer/diffusion_pytorch_model-{i:05d}-of-00003.safetensors")
                    for i in range(1, 4)
                ]
                te_paths = [
                    _resolve_and_stage_model_file(target_id, f"text_encoder/model-{i:05d}-of-00003.safetensors")
                    for i in range(1, 4)
                ]
                vae_path = _resolve_and_stage_model_file(target_id, "vae/diffusion_pytorch_model.safetensors")
                new_pipe = ZImagePipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device="cuda",
                    model_configs=[
                        ModelConfig(path=dit_paths, **vram_config),
                        ModelConfig(path=te_paths, **vram_config),
                        ModelConfig(path=vae_path, **vram_config),
                    ],
                    tokenizer_config=ModelConfig(
                        model_id=repo_id or "Tongyi-MAI/Z-Image-Turbo",
                        origin_file_pattern="tokenizer/",
                        download_source="huggingface",
                    ),
                    vram_limit=vram_limit,
                )
                _enable_high_vram_optimizations(new_pipe, total_vram_gb=total_vram_gb, preload_vram=True)
                mode_label = f"{spec['name']} · 8步极速 BF16"

            elif pipeline_arch == "diffusers_image":
                MODEL_DOWNLOAD_STATE["progress"] = 0.45
                MODEL_DOWNLOAD_STATE["message"] = f"正在使用 Diffusers 装载自定义图像模型 {spec['name']}..."
                first_rel = spec["files"][0][0] if spec.get("files") else ""
                first_existing = _find_existing_model_file(target_id, first_rel) if first_rel else None
                model_dir = os.path.dirname(first_existing) if first_existing else _get_model_dirs(target_id)[2]
                if "/" in first_rel and first_existing:
                    # 若分片在子目录 (如 transformer/xxx.safetensors)，取顶层模型目录
                    sub_depth = len(first_rel.split("/")) - 1
                    for _ in range(sub_depth):
                        model_dir = os.path.dirname(model_dir)
                from diffusers import AutoPipelineForText2Image, DiffusionPipeline
                try:
                    new_pipe = AutoPipelineForText2Image.from_pretrained(
                        model_dir if os.path.exists(os.path.join(model_dir, "model_index.json")) else repo_id,
                        torch_dtype=torch.bfloat16,
                    ).to("cuda")
                except Exception:
                    new_pipe = DiffusionPipeline.from_pretrained(
                        model_dir if os.path.exists(os.path.join(model_dir, "model_index.json")) else repo_id,
                        torch_dtype=torch.bfloat16,
                    ).to("cuda")
                mode_label = f"{spec['name']} · Diffusers BF16"

            elif pipeline_arch in ("qwen38_27b", "transformers"):
                if target_id == "qwen38_27b" and free_before_gb < 49.0:
                    raise RuntimeError(
                        f"当前 GPU 可用显存 ({free_before_gb:.1f} GB / 总计 {total_vram_gb:.1f} GB) 不足以加载 "
                        f"Qwen3.8-27B BF16 原生权重（需约 52 GB 显存，推荐 A100 80GB）。"
                        f"若在 L4 (24GB) 或中低显存运行，请在模型中心选择独立的「Qwen3.8-27B-GGUF (UD-Q4_K_M)」量化版！"
                    )

                MODEL_DOWNLOAD_STATE["progress"] = 0.30
                MODEL_DOWNLOAD_STATE["message"] = (
                    f"正在装载 {spec['name']} Transformers 原生权重 (上下文上限 {ctx_len // 1024}K)..."
                )
                first_rel = spec["files"][0][0] if spec.get("files") else "model-00001-of-00018.safetensors"
                first_shard = _find_existing_model_file(target_id, first_rel)
                model_dir = os.path.dirname(first_shard) if first_shard else _get_model_dirs(target_id)[2]

                # 确保配置文件在同一目录中完备
                if not os.path.exists(os.path.join(model_dir, "config.json")):
                    from huggingface_hub import snapshot_download
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=model_dir,
                        ignore_patterns=["*.safetensors", "*.bin", "*.pth", "*.gguf", "*.pt"],
                    )

                if not os.path.exists(os.path.join(model_dir, "config.json")):
                    raise RuntimeError(
                        f"仓库 {repo_id} 未包含标准 Transformers 配置文件 (config.json / tokenizer.json)，"
                        f"仅包含独立权重分片（如 3D 重建 / 专用 SDK / ComfyUI 单体组件），无法作为对话或视觉大模型直接加载。"
                        f"请选用 GGUF 量化模型、含 config.json 的标准 Transformers 模型，或在 ComfyUI 工作台中使用单体权重。"
                    )

                def _load_transformers_llm():
                    _ensure_torchaudio_compatible()
                    _ensure_qwen_fast_kernels()
                    from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM

                    def _load_tok():
                        try:
                            return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                        except Exception:
                            return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)

                    try:
                        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
                        if not hasattr(processor, "tokenizer"):
                            tok = _load_tok()
                            processor = _CausalLMProcessorAdapter(tok)
                    except Exception:
                        tok = _load_tok()
                        processor = _CausalLMProcessorAdapter(tok)

                    try:
                        mdl = AutoModelForImageTextToText.from_pretrained(
                            model_dir,
                            dtype=torch.bfloat16,
                            attn_implementation="sdpa",
                            device_map="cuda",
                            trust_remote_code=True,
                        )
                    except Exception:
                        mdl = AutoModelForCausalLM.from_pretrained(
                            model_dir,
                            torch_dtype=torch.bfloat16,
                            attn_implementation="sdpa",
                            device_map="auto",
                            trust_remote_code=True,
                        )
                    mode_tag = f"{spec['name']} · BF16 原生 · {ctx_len // 1024}K"
                    mdl.eval()
                    try:
                        _optimize_qwen35_for_fast_inference(mdl, max_context_len=ctx_len)
                    except Exception:
                        pass
                    return mdl, processor, mode_tag

                try:
                    mdl, proc, mode_tag = _load_transformers_llm()
                except (KeyError, ValueError, ImportError) as up_e:
                    print(f"检测到分词器或 transformers 组件缺失 ({up_e})，正在自动升级 transformers 及分词器依赖...")
                    subprocess.check_call(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "-q",
                            "-U",
                            "transformers",
                            "accelerate",
                            "qwen-vl-utils",
                            "sentencepiece",
                            "tiktoken",
                            "protobuf",
                            "blobfile",
                        ]
                    )
                    import importlib
                    import transformers
                    importlib.reload(transformers)
                    mdl, proc, mode_tag = _load_transformers_llm()

                new_llm = {
                    "backend": "transformers",
                    "model": mdl,
                    "processor": proc,
                    "model_id": target_id,
                    "model_name": spec["name"],
                    "context_length": ctx_len,
                }
                mode_label = mode_tag

            elif pipeline_arch in ("qwen38_27b_gguf", "gguf"):
                MODEL_DOWNLOAD_STATE["progress"] = 0.25
                MODEL_DOWNLOAD_STATE["message"] = f"正在预热 {spec['name']} GGUF 权重并启动 llama.cpp CUDA 引擎..."
                if target_id == "qwen38_27b_gguf":
                    gguf_path = _resolve_and_stage_model_file("qwen38_27b_gguf", QWEN_GGUF_MAIN_FILE)
                    mmproj_path = _resolve_and_stage_model_file("qwen38_27b_gguf", QWEN_GGUF_MMPROJ_FILE)

                    mtp_rel = "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
                    mtp_path: Optional[str] = None
                    try:
                        if not _find_existing_model_file("qwen38_27b_gguf", mtp_rel):
                            MODEL_DOWNLOAD_STATE["progress"] = 0.38
                            MODEL_DOWNLOAD_STATE["message"] = "正在自动补齐 Unsloth MTP 投机解码加速头 (1.31 GB)..."
                            print("[GGUF 引擎] 检测到尚未缓存 MTP 投机解码权重，正在 8 路并发极速补齐...")
                            ram_d, nvme_d, drive_d = _get_model_dirs("qwen38_27b_gguf")
                            dl_dir = ram_d if ram_d else (drive_d if DRIVE_MOUNTED else nvme_d)
                            os.makedirs(dl_dir, exist_ok=True)
                            dl_mtp = fast_parallel_hf_download(
                                repo_id="unsloth/Qwen3.8-27B-GGUF",
                                filename=mtp_rel,
                                dest_dir=dl_dir,
                                hf_token=os.environ.get("HF_TOKEN", ""),
                                num_connections=8,
                            )
                            if DRIVE_MOUNTED and dl_dir != drive_d:
                                threading.Thread(
                                    target=_async_backup_to_drive,
                                    args=(dl_mtp, drive_d, mtp_rel),
                                    daemon=True,
                                ).start()
                        mtp_path = _resolve_and_stage_model_file("qwen38_27b_gguf", mtp_rel)
                    except Exception as mtp_e:
                        print(f"[GGUF 引擎] MTP 投机权重获取跳过，将以标准模式启动: {mtp_e}")
                        mtp_path = None
                else:
                    all_gguf_rels = [fn for fn, _ in spec.get("files", []) if fn.lower().endswith(".gguf")]
                    mmproj_rels = [fn for fn in all_gguf_rels if "mmproj" in fn.lower()]
                    mtp_rels = [fn for fn in all_gguf_rels if ("/mtp" in fn.lower() or os.path.basename(fn.lower()).startswith("mtp"))]
                    main_rels = sorted([fn for fn in all_gguf_rels if fn not in mmproj_rels and fn not in mtp_rels])
                    if not main_rels:
                        raise RuntimeError(f"未在 {spec['name']} 中找到主 GGUF 权重文件")
                    staged_mains = [_resolve_and_stage_model_file(target_id, fn) for fn in main_rels]
                    gguf_path = staged_mains[0]
                    mmproj_path = _resolve_and_stage_model_file(target_id, mmproj_rels[0]) if mmproj_rels else None
                    mtp_path = _resolve_and_stage_model_file(target_id, mtp_rels[0]) if mtp_rels else None

                new_llm = _start_qwen_gguf_server(
                    gguf_path=gguf_path,
                    mmproj_path=mmproj_path,
                    ctx_len=ctx_len,
                    cur_free_gb=free_before_gb,
                    mtp_path=mtp_path,
                    model_id=target_id,
                    model_name=spec["name"],
                    total_weights_gb=approx_gb,
                )
                mode_label = new_llm["mode_label"]

            free_after_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
            vram_delta_gb = round(max(0.5, free_before_gb - free_after_gb), 1)

            LOADED_RUNTIMES[target_id] = {
                "model_id": target_id,
                "name": spec["name"],
                "modality": spec["modality"],
                "mode_label": mode_label,
                "pipeline": new_pipe,
                "llm_bundle": new_llm,
                "server_proc": h3_image_proc,
                "vram_gb": vram_delta_gb,
                "loaded_at": time.time(),
                "h3_dit_variant": h3_dit_variant if pipeline_arch == "minimax_h3" else "",
            }
            _sync_legacy_globals()

            elapsed_s = round(time.time() - t_start, 1)
            MODEL_DOWNLOAD_STATE["progress"] = 1.0
            co_count = len(LOADED_RUNTIMES)
            evict_tip = f"（因显存上限已自动腾退 {', '.join(evicted_names)}）" if evicted_names else ""
            co_tip = f" | 当前共 {co_count} 个模型同时常驻显存，切换 0 等待" if co_count > 1 else ""
            msg = f"{spec['name']} 已装载至显存 ({mode_label}，约 {vram_delta_gb}GB，耗时 {elapsed_s}s){co_tip}！{evict_tip}{auto_switched_notice}"
            MODEL_DOWNLOAD_STATE["message"] = msg
            print(msg)
            return True, msg

        except Exception as e:
            _sync_legacy_globals()
            err = f"加载 {spec['name']} 至显存失败: {str(e)}"
            MODEL_DOWNLOAD_STATE["message"] = err
            print(err)
            return False, err
        finally:
            MODEL_DOWNLOAD_STATE["is_loading_vram"] = False
            touch_activity()


def load_minimax_h3_to_vram_core(
    precision_mode: str = "bf16",
    offload_mode: str = "fast_mmap",
    hf_token: str = "",
) -> Tuple[bool, str]:
    return load_model_to_vram_core("minimax_h3", "bf16", offload_mode, hf_token)


def _enable_high_vram_optimizations(
    pipe,
    total_vram_gb: float,
    preload_vram: bool = True,
    skip_preload_attrs: Tuple[str, ...] = (),
) -> str:
    """
    针对 G4 (96GB) / A100 (80GB) 等大显存实例 (>45GB VRAM) 的深度加速（完美兼容多模型显存共驻与 BF16 超大模型）：
    1. 拦截 DiffSynth 默认在每个阶段切换时强制触发的 module.offload()，基于实时剩余显存智能判断：
       - 对于 MiniMax-H3 BF16 (text_encoder 62.1GB + dit 61.7GB > 96GB)，在 text_encoder 与 dit 互斥切换时精准腾退，
         而在 dit -> video_vae (9.7GB) -> audio_vae (0.56GB) 阶段只要余量 >= 25GB 即全量常驻，连抽换种子 0 重载！
    2. 为 Qwen3-VL 32B PromptEmbedder 挂载零触发 LRU 缓存：命中缓存时连 load_models_to_device(["text_encoder"]) 亦直接跳过，
       保证 61.7GB DiT 主干在 G4 96GB 显存中全程锁死不卸载！
    """
    import types

    is_minimax_bf16 = pipe.__class__.__name__ == "MiniMaxH3Pipeline"

    # 1. 挂载 PromptEmbedder 编码缓存（连抽多条或同 Prompt 换 Seed 时 0ms 瞬时跳过 32B Qwen3-VL 推理及显存换入）
    try:
        for unit in getattr(pipe, "units", []):
            if unit.__class__.__name__ == "MiniMaxH3Unit_PromptEmbedder" and not hasattr(
                unit, "_orig_process"
            ):
                unit._orig_process = unit.process
                unit._orig_onload_names = getattr(unit, "onload_model_names", ("text_encoder",))
                # 置空外层自动 onload，仅在缓存未命中时按需加载 text_encoder，避免同提示词连抽时把常驻 DiT 挤出显存
                unit.onload_model_names = ()
                unit._embed_cache = {}

                def _cached_prompt_process(
                    self_unit,
                    pipe_obj,
                    prompt,
                    keyframes=None,
                    ref_blocks=None,
                    height=None,
                    width=None,
                    text_embedding=None,
                ):
                    if ref_blocks is None and text_embedding is None and prompt is not None:
                        kf_sig = None
                        if keyframes:
                            try:
                                kf_sig = tuple(
                                    (img.size, img.mode, img.tobytes()[:512]) for img in keyframes
                                )
                            except Exception:
                                kf_sig = id(keyframes)
                        cache_key = (prompt, height, width, kf_sig)
                        cached = self_unit._embed_cache.get(cache_key)
                        if cached is not None:
                            print("⚡ [PromptCache] 命中 Qwen3-VL 32B 多模态编码缓存，0ms 瞬时跳过文本编码阶段！")
                            return {
                                "prompt_embeds": cached["prompt_embeds"].to(
                                    pipe_obj.device, pipe_obj.torch_dtype
                                ),
                                "text_token_tags": cached["text_token_tags"].to(
                                    pipe_obj.device, torch.long
                                ),
                            }
                        t_pe = time.time()
                        pipe_obj.load_models_to_device(self_unit._orig_onload_names)
                        res = self_unit._orig_process(
                            pipe_obj,
                            prompt,
                            keyframes=keyframes,
                            ref_blocks=ref_blocks,
                            height=height,
                            width=width,
                            text_embedding=text_embedding,
                        )
                        self_unit._embed_cache.clear()
                        self_unit._embed_cache[cache_key] = {
                            "prompt_embeds": res["prompt_embeds"].detach().clone(),
                            "text_token_tags": res["text_token_tags"].detach().clone(),
                        }
                        print(
                            f"⚡ [PromptEmbedder] 多模态提示词解析完成，总耗时 {time.time() - t_pe:.1f}s"
                            "（已写入缓存，同提示词换种子后续 0ms）"
                        )
                        return res
                    t_pe = time.time()
                    pipe_obj.load_models_to_device(self_unit._orig_onload_names)
                    res = self_unit._orig_process(
                        pipe_obj,
                        prompt,
                        keyframes=keyframes,
                        ref_blocks=ref_blocks,
                        height=height,
                        width=width,
                        text_embedding=text_embedding,
                    )
                    print(f"⚡ [PromptEmbedder] 多模态提示词解析完成，总耗时 {time.time() - t_pe:.1f}s")
                    return res

                unit.process = types.MethodType(_cached_prompt_process, unit)
    except Exception as e:
        print(f"ℹ️ Prompt 缓存挂载跳过: {e}")

    if total_vram_gb < 45.0:
        return ""

    def _is_submodel_fully_prepared_on_cuda(mdl) -> bool:
        if mdl is None:
            return False
        found_stateful = False
        for mod in mdl.modules():
            if hasattr(mod, "state"):
                found_stateful = True
                if getattr(mod, "state", 0) != 2:
                    return False
        if found_stateful:
            return True
        p = next(mdl.parameters(), None)
        return bool(p is not None and p.device.type == "cuda")

    def _offload_submodel_to_host(mdl) -> bool:
        if mdl is None or not getattr(mdl, "vram_management_enabled", False):
            return False
        has_active = any(
            hasattr(mod, "state") and getattr(mod, "state", 0) != 0
            for mod in mdl.modules()
        )
        if not has_active:
            return False
        if hasattr(mdl, "offload"):
            mdl.offload()
        else:
            for mod in mdl.modules():
                if hasattr(mod, "offload"):
                    mod.offload()
        return True

    # 2. 替换 load_models_to_device：智能阶段显存驻留 + 整块 PCIe 极速换入
    def _smart_load_models_to_device(self_pipe, model_names):
        if not getattr(self_pipe, "vram_management_enabled", False):
            return
        target_set = set(model_names or ())
        if not target_set:
            return

        # 快速路径：若请求的所有目标子模块均已 100% 处于 GPU 显存 (state=2)，0ms 直接放行，绝不误卸其他常驻模块
        target_children = [
            (name, mdl)
            for name, mdl in self_pipe.named_children()
            if name in target_set and mdl is not None
        ]
        if target_children and all(_is_submodel_fully_prepared_on_cuda(mdl) for _, mdl in target_children):
            return

        def _get_free_vram_gb() -> float:
            try:
                return torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
            except Exception:
                return 0.0

        free_gb = _get_free_vram_gb()
        need_heavy_backbone = is_minimax_bf16 and (("dit" in target_set) or ("text_encoder" in target_set))
        req_free_gb = 66.5 if need_heavy_backbone else 24.0

        # 第一级腾退：切换 MiniMax-H3 两大 62GB 主干时，优先仅腾退对侧闲置大主干，让轻量的 video_vae / audio_vae 在 G4 96GB 中全程钉死显存
        if need_heavy_backbone and free_gb < req_free_gb:
            offloaded_heavy = False
            for name, model in self_pipe.named_children():
                if name not in target_set and name in ("dit", "text_encoder") and model is not None:
                    if _offload_submodel_to_host(model):
                        offloaded_heavy = True
            if offloaded_heavy:
                gc.collect()
                torch.cuda.empty_cache()
                free_gb = _get_free_vram_gb()

        # 第二级腾退：仅当腾退大主干后显存仍不足 (例如与 21GB Qwen-GGUF 共驻显存时)，再按需腾退其余非目标子模块
        if free_gb < req_free_gb:
            offloaded_rest = False
            for name, model in self_pipe.named_children():
                if name not in target_set and model is not None:
                    if _offload_submodel_to_host(model):
                        offloaded_rest = True
            if offloaded_rest:
                gc.collect()
                torch.cuda.empty_cache()
                free_gb = _get_free_vram_gb()

        # 目标模块整块换入：显存充裕时直接一次性执行 preparing() 锁入 GPU (state=2)，彻底消灭推理时 800+ 层逐层同步钩子开销
        for name, model in target_children:
            if _is_submodel_fully_prepared_on_cuda(model):
                continue
            if getattr(model, "vram_management_enabled", False):
                t_sw = time.time()
                if hasattr(model, "onload"):
                    model.onload()
                else:
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()

                cur_free_gb = _get_free_vram_gb()
                bulk_need_gb = 66.0 if (is_minimax_bf16 and name in ("dit", "text_encoder")) else 22.0
                if cur_free_gb >= bulk_need_gb:
                    for module in model.modules():
                        if hasattr(module, "preparing") and getattr(module, "state", 2) != 2:
                            try:
                                module.preparing()
                            except Exception:
                                pass
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    dt_sw = time.time() - t_sw
                    if dt_sw >= 0.3:
                        print(f"⚡ [极速显存换入] 子模块 {name} 已整块载入 GPU 显存，耗时 {dt_sw:.1f}s")

    pipe.load_models_to_device = types.MethodType(_smart_load_models_to_device, pipe)
    pipe._high_vram_fast_mode = True

    # 3. 预热指定子模块至 GPU 显存 (state=2)，保留 >=22GB 安全激活余量
    if preload_vram:
        children = [
            (name, mdl)
            for name, mdl in pipe.named_children()
            if mdl is not None and name not in skip_preload_attrs
        ]
        for idx, (attr_name, model) in enumerate(children):
            free_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
            if free_gb < 22.0:
                break
            prog = round(0.80 + 0.18 * ((idx + 1) / max(1, len(children))), 2)
            MODEL_DOWNLOAD_STATE["progress"] = prog
            MODEL_DOWNLOAD_STATE["message"] = f"[显存常驻加速] 正在将子模块 {attr_name} 预热锁入 GPU 显存..."
            for module in model.modules():
                if hasattr(module, "preparing") and getattr(module, "state", 2) != 2:
                    try:
                        module.preparing()
                    except Exception:
                        pass
        _trim_linux_cpu_heap()
        used_gb = (
            torch.cuda.mem_get_info("cuda")[1] - torch.cuda.mem_get_info("cuda")[0]
        ) / (1024 ** 3)
        print(
            f"[大显存常驻已激活] 管线子模块已预热至显存 (当前总驻留 {used_gb:.1f} GB / {total_vram_gb:.1f} GB)"
        )
        return "大显存常驻"

    return "大显存常驻加速"


def _reclaim_transient_gpu_memory():
    """在连续视频/图像生成前后主动回收 Python 引用与 CUDA 缓存碎片，杜绝第 2 条视频中途 OOM 崩溃"""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _run_pipeline_with_smart_tiling(pipe, call_kwargs: dict):
    """仅当实时空闲显存 >= 46GB 时启用整帧 VAE 解码 (tiled=False)；超长帧序列 (如 15s / 362帧) 触顶时自动开启梯度检查点兜底杜绝 OOM"""
    # 若生成高分辨率超长视频 (如 720P + >=260帧/11~15秒)，采样前若 video_vae 驻留显存且总空闲偏紧，先释放 VAE 留足 33GB+ 激活空间
    num_f = int(call_kwargs.get("num_frames") or 124)
    pix = int(call_kwargs.get("height") or 480) * int(call_kwargs.get("width") or 832)
    if pipe.__class__.__name__ == "MiniMaxH3Pipeline" and num_f >= 260 and pix >= 1280 * 700:
        try:
            vvae = getattr(pipe, "video_vae", None)
            if vvae is not None and hasattr(vvae, "offload"):
                vvae.offload()
                _reclaim_transient_gpu_memory()
            elif vvae is not None:
                for mod in vvae.modules():
                    if hasattr(mod, "offload"):
                        mod.offload()
                _reclaim_transient_gpu_memory()
        except Exception:
            pass

    if getattr(pipe, "_high_vram_fast_mode", False) and "tiled" not in call_kwargs:
        try:
            free_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
        except Exception:
            free_gb = 0.0
        if free_gb >= 46.0:
            fast_kwargs = dict(call_kwargs)
            fast_kwargs["tiled"] = False
            try:
                return pipe(**fast_kwargs)
            except torch.cuda.OutOfMemoryError:
                print("⚠️ 整帧 VAE 解码显存触顶，正在清理缓存并自动切换为 Tiled 分块安全解码...")
                _reclaim_transient_gpu_memory()
    safe_kwargs = dict(call_kwargs)
    safe_kwargs.setdefault("tiled", True)
    try:
        return pipe(**safe_kwargs)
    except torch.cuda.OutOfMemoryError:
        if pipe.__class__.__name__ == "MiniMaxH3Pipeline" and not safe_kwargs.get("use_gradient_checkpointing"):
            print("⚠️ 超长序列注意力显存触顶，正在自动释放闲置子模块并启用梯度检查点安全重试...")
            for sub_name in ("video_vae", "audio_vae", "text_encoder"):
                sub_m = getattr(pipe, sub_name, None)
                if sub_m is not None:
                    if hasattr(sub_m, "offload"):
                        sub_m.offload()
                    else:
                        for mod in sub_m.modules():
                            if hasattr(mod, "offload"):
                                mod.offload()
            _reclaim_transient_gpu_memory()
            gc_kwargs = dict(safe_kwargs)
            gc_kwargs["use_gradient_checkpointing"] = True
            return pipe(**gc_kwargs)
        raise


# ==============================================================================
# 视频生成核心函数
# ==============================================================================
RESOLUTION_MAP = {
    "832 × 480 (16:9 横屏标准 - 推荐)": (832, 480),
    "480 × 832 (9:16 竖屏短视频)": (480, 832),
    "1280 × 720 (16:9 720P 高清)": (1280, 720),
    "720 × 1280 (9:16 720P 竖屏高清)": (720, 1280),
    "640 × 640 (1:1 方形)": (640, 640),
}


def generate_video(
    task_mode: str,
    visual_prompt: str,
    soundscape_prompt: str,
    music_prompt: str,
    first_frame_img: Optional[Image.Image],
    last_frame_img: Optional[Image.Image],
    resolution_label: str,
    num_frames: int,
    num_inference_steps: int,
    fps: int,
    seed: int,
    randomize_seed: bool,
    save_dir: str,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], str, int, List[str]]:
    global PIPELINE
    touch_activity()

    if not visual_prompt or not visual_prompt.strip():
        return None, "⚠️ 请输入画面描述提示词 (Prompt)！", seed, get_history_videos(save_dir)

    if PIPELINE is None:
        return (
            None,
            "⚠️ 模型尚未加载到 GPU 显存！请先在左侧点击「1️⃣ 下载模型权重」与「2️⃣ 加载模型至显存」。",
            seed,
            get_history_videos(save_dir),
        )

    if (LOADED_RUNTIMES.get("minimax_h3") or {}).get("h3_dit_variant") == "Ref2VA":
        return (
            None,
            "⚠️ 显存里现在是 Ref2VA 参考主干（客户端「参考生音视频」切换的），文生与首尾帧要用 FL2VA 主干：请重新点「2️⃣ 加载模型至显存」，或在客户端用文生/首尾帧模式生成一次自动切回。",
            seed,
            get_history_videos(save_dir),
        )

    if randomize_seed or seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

    width, height = RESOLUTION_MAP.get(resolution_label, (832, 480))
    full_prompt = build_structured_prompt(visual_prompt, soundscape_prompt, music_prompt)

    keyframes = []
    keyframe_indices = []
    if task_mode == "fl2va":
        if first_frame_img is not None:
            keyframes.append(first_frame_img.convert("RGB"))
            keyframe_indices.append(0)
        if last_frame_img is not None:
            keyframes.append(last_frame_img.convert("RGB"))
            keyframe_indices.append(-1)
        if not keyframes:
            return (
                None,
                "⚠️ 在「首尾帧图生视频」模式下，请至少上传一张【首帧图片】或【尾帧图片】！",
                seed,
                get_history_videos(save_dir),
            )

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"minimax_h3_{task_mode}_{timestamp}_seed{seed}.mp4"
    output_path = os.path.join(save_dir, filename)

    start_time = time.time()
    with GPU_LOCK:
        try:
            from diffsynth.utils.data.audio_video import write_video_audio

            if progress is not None:
                progress(0.05, desc="正在准备潜空间与多模态编码...")

            call_kwargs = dict(
                prompt=full_prompt,
                height=height,
                width=width,
                num_frames=int(num_frames),
                num_inference_steps=int(num_inference_steps),
                seed=int(seed),
            )
            if keyframes:
                call_kwargs["keyframes"] = keyframes
                call_kwargs["keyframe_indices"] = keyframe_indices

            _reclaim_transient_gpu_memory()
            video, audio = _run_pipeline_with_smart_tiling(PIPELINE, call_kwargs)

            if progress is not None:
                progress(0.92, desc="正在合成 32kHz 立体声音轨与 MP4 视频...")

            write_video_audio(
                video=video,
                audio=audio,
                output_path=output_path,
                fps=int(fps),
                audio_sample_rate=32000,
            )
            video = None
            audio = None
            _reclaim_transient_gpu_memory()

            elapsed = time.time() - start_time
            status_msg = (
                f"✅ 生成完成！耗时: {elapsed:.1f} 秒 | 分辨率: {width}×{height} | "
                f"帧数: {num_frames} ({num_frames/fps:.1f}s) | Seed: {seed}\n"
                f"📁 已保存至: {output_path}"
            )
            return output_path, status_msg, seed, get_history_videos(save_dir)

        except Exception as e:
            _reclaim_transient_gpu_memory()
            return None, f"❌ 生成过程中出错: {str(e)}", seed, get_history_videos(save_dir)
        finally:
            touch_activity()


def get_history_videos(save_dir: str) -> List[str]:
    if not os.path.exists(save_dir):
        return []
    files = glob.glob(os.path.join(save_dir, "*.mp4"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:12]


# ==============================================================================
# ComfyUI 云端节点工作台 + Google Drive 自动软链与双向持久化引擎
# ==============================================================================
COMFY_STANDARD_MODEL_SUBDIRS = [
    "checkpoints",
    "diffusion_models",
    "unet",
    "vae",
    "loras",
    "text_encoders",
    "clip",
    "clip_vision",
    "controlnet",
    "upscale_models",
    "latent_upscale_models",
    "audio_encoders",
    "embeddings",
    "style_models",
    "gguf",
]

_COMFY_PROTECTED_PIP_PKGS = re.compile(
    r"^\s*(torch|torchvision|torchaudio|xformers)($|[<>=!~;\s\[])",
    re.IGNORECASE,
)
_COMFY_BG_SYNC_STARTED = False
_COMFY_ADOPT_CHECKED = False
_COMFY_STATE_FILE = "/tmp/.easycolab_comfy_state.json"
_COMFY_LOG_FILE = "/tmp/easycolab_comfyui.log"
_COMFY_TUNNEL_LOG_FILE = "/tmp/easycolab_cloudflared.log"


def _spawn_logged_detached(cmd: List[str], log_path: str, cwd: Optional[str] = None) -> subprocess.Popen:
    """子进程输出直接写日志文件、不走管道：app.py 热更新被杀后，ComfyUI / cloudflared 再写日志也不会因管道断开（SIGPIPE）而退出"""
    with open(log_path, "w", encoding="utf-8") as log_f:
        return subprocess.Popen(
            cmd, cwd=cwd, stdout=log_f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
        )


def _follow_log_file(path: str, proc: subprocess.Popen, on_line) -> None:
    """边写边读子进程的日志文件，直到子进程退出"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                ln = f.readline()
                if ln:
                    on_line(ln.rstrip())
                elif proc.poll() is not None:
                    return
                else:
                    time.sleep(0.3)
    except Exception:
        pass


def _start_comfy_tunnel(wait_s: float) -> Tuple[subprocess.Popen, str]:
    """起一条指向 ComfyUI 入口的 Cloudflare 隧道，在 wait_s 秒内从日志文件里读出公网地址（读不到返回空串）"""
    cf_bin = _ensure_cloudflared_binary()
    proc = _spawn_logged_detached(
        [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{COMFY_INGRESS}", "--no-autoupdate", "--protocol", "http2"],
        _COMFY_TUNNEL_LOG_FILE,
    )
    cf_re = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    t_end = time.time() + wait_s
    while time.time() < t_end and proc.poll() is None:
        try:
            m = cf_re.search(open(_COMFY_TUNNEL_LOG_FILE, "r", encoding="utf-8", errors="replace").read())
        except Exception:
            m = None
        if m:
            return proc, m.group(0)
        time.sleep(0.5)
    return proc, ""


def _detached_proc_state(pattern: str) -> str:
    """按命令行找实例上的进程：none＝没有；pipe＝旧版本用管道起的（读管道的 app.py 已死，再写日志就会出错或退出）；ok＝正常"""
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    except Exception:
        return "none"
    if not out:
        return "none"
    for pid in out:
        try:
            if os.readlink(f"/proc/{pid}/fd/1").startswith("pipe:"):
                return "pipe"
        except Exception:
            pass
    return "ok"
_COMFY_STATUS_CACHE: Dict[str, Any] = {"ts": 0.0, "counts": (0, 0, 0), "outputs": [], "packs": []}

COMFY_TEMPLATE_RAW_BASE = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/"

# MiniMax-H3 预置工作流默认选用的社区 Heretic 文本编码器；官方编码器仍会下载，可在工作流下拉框切回。
H3_HERETIC_TEXT_ENCODER = "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"

# 100% 免审核 (Gated=False) 官方原生节点预置图生视频 / 首尾帧运镜工作流包
# H3 Director 插件自带示例工作流 -> 画布里的中文文件名（随导演台附加包写入 workflows 目录）
H3_DIRECTOR_NODE_DIR = "ComfyUI_MiniMaxH3_Director"
H3_DIRECTOR_REPO = "https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director.git"
H3_DIRECTOR_EXAMPLE_WORKFLOWS: List[Tuple[str, str]] = [
    ("minimax_h3_director_fl2v.json", "10_H3导演台_首尾帧多段(fl2v).json"),
    ("minimax_h3_director_t2v.json", "11_H3导演台_文生多段(t2v).json"),
    ("minimax_h3_director_r2v.json", "12_H3导演台_参考主体素材组(r2v).json"),
    ("minimax_h3_director_v2v.json", "13_H3导演台_源视频改视频(v2v).json"),
    ("minimax_h3_director_rv2v.json", "14_H3导演台_源视频加参考(rv2v).json"),
    ("minimax_h3_director_external_groups_i2v.json", "15_H3导演台_外部多组接线(i2v).json"),
    ("minimax_h3_director_external_groups_r2v.json", "16_H3导演台_外部多组接线(r2v).json"),
    ("minimax_h3_director_二采_加速.json", "17_H3导演台_二采放大(Refine).json"),
    ("minimax_h3_director_加速版.json", "18_H3导演台_加速版(SageAttention).json"),
]
# 示例工作流里引用、但公开仓库里没有的权重 -> 本附加包实际下载的同类权重（目前都能找到原文件，留空）
H3_DIRECTOR_MODEL_RENAMES: Dict[str, str] = {}


# Comfyit「MiniMax H3 模型和工作流」资源包里的导演台 / 官方模板工作流（GitHub 仓库里没有），zlib+base64 内嵌，随导演台附加包写入画布
H3_DIRECTOR_BUNDLED_WORKFLOWS: Dict[str, Tuple[str, str]] = {   # 画布文件名 -> (所属附加包, 压缩内容)
    '19_H3导演台_全能工作流(混合主干).json': ('plus', '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'),
    '20_H3导演台_全能工作流(Pika六步加速).json': ('plus', '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'),
    '21_H3导演台_外接参数版(i2v).json': ('core', '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'),
    '22_H3导演台_外接参数版(r2v).json': ('core', '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'),
    '23_H3官方模板_图生有声视频(i2v).json': ('core', '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'),
    '24_H3官方模板_参考主体(r2v).json': ('core', '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'),
    '25_H3官方模板_文生有声视频(t2v).json': ('core', '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'),
}

# 导演台满血 bf16 参考主体包：4 套 r2v 工作流另存一份（不覆盖 int8 原版），主干 / 编码器下拉默认换成 bf16
H3_DIRECTOR_BF16_R2V_WORKFLOWS: Dict[str, str] = {   # 原工作流 -> 满血版文件名
    "12_H3导演台_参考主体素材组(r2v).json": "12_H3导演台_参考主体素材组(r2v)_满血bf16.json",
    "16_H3导演台_外部多组接线(r2v).json": "16_H3导演台_外部多组接线(r2v)_满血bf16.json",
    "22_H3导演台_外接参数版(r2v).json": "22_H3导演台_外接参数版(r2v)_满血bf16.json",
    "24_H3官方模板_参考主体(r2v).json": "24_H3官方模板_参考主体(r2v)_满血bf16.json",
}
H3_DIRECTOR_BF16_RENAMES: Dict[str, str] = {
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors": "minimax_h3_ref2va_bf16.safetensors",
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors": "minimax_h3_ref2va_bf16.safetensors",   # 16 / 22 示例默认挂的是 fl2va，r2v 应该用 ref2va
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": "qwen3vl_32b_minimax_h3_bf16.safetensors",
}



COMFY_PRESET_WORKFLOW_PACKS: List[Dict[str, Any]] = [
    {
        "id": "wan22_i2v_flf2v",
        "title": "万相 Wan2.2 14B · 图生视频 + 首尾帧双工作流",
        "badge": "新手首选 · 4步极速",
        "scenario": "含「单图动起来」与「首尾双图丝滑过渡」2 套中文画布，G4 约 25 秒出片，纯官方内置节点零红框报错",
        "approx_size_gb": 35.4,
        "workflows": [
            "01_Wan2.2_极速单图生视频(4步傻瓜版).json",
            "02_Wan2.2_首尾帧丝滑过渡(双图4步极速).json",
        ],
        "models": [
            ("text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6.27),
            ("vae", "wan_2.1_vae.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan_2.1_vae.safetensors", 0.24),
            ("diffusion_models", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", 13.31),
            ("diffusion_models", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", 13.31),
            ("loras", "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", 1.14),
            ("loras", "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", 1.14),
        ],
    },
    {
        "id": "minimax_h3_i2v_flf2v",
        "title": "MiniMax-H3 Turbo · 图生有声视频 + 首尾帧双工作流",
        "badge": "自带立体声音效 · 8步加速",
        "scenario": "含「单图生有声视频」与「首尾双图有声过渡」2 套极简画布，一次生成高清动态画面 + 同步环境音效/配乐/对白",
        "approx_size_gb": 53.7,
        "workflows": [
            "03_MiniMax-H3_极速图生有声视频(单图带立体声).json",
            "04_MiniMax-H3_首尾帧有声过渡(双图带立体声).json",
        ],
        "models": [
            ("vae", "minimax_h3_video_vae_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_int8_convrot.safetensors", 2.62),
            ("vae", "minimax_h3_audio_vae_fp32.safetensors", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors", 0.56),
            ("diffusion_models", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", 19.53),
            ("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 14.61),
            ("text_encoders", H3_HERETIC_TEXT_ENCODER, "Momoking/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4", H3_HERETIC_TEXT_ENCODER, 14.61),
            ("loras", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "lightx2v/Minimax-h3-Turbo", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 1.82),
        ],
    },
    # 下面两个是 H3 基础包的附加包：只多下高精度主干与同档编码器（int8 包配 Heretic 编码器，bf16 包配官方编码器），供 03/04 工作流的下拉框切换。
    # G4 实例开机后本地盘约 180 GB 可用，三者全装放不下，所以拆开按需装。
    {
        "id": "minimax_h3_hq_int8",
        "title": "MiniMax-H3 附加 · 不剪枝 int8 主干（首尾帧 + 参考主体）+ 官方 / 无审查 int8 编码器",
        "badge": "自动补齐基础包文件 · 可常驻显存",
        "scenario": "给 03/04 与导演台工作流的下拉框多四个选项：不剪枝 int8 的 fl2va / ref2va 主干，以及官方与 Heretic 两个 int8 编码器，本包约 113 GB；连同 03/04 工作流默认要用的基础包文件合计约 152 GB，不要和满血 bf16 包同时装",
        "approx_size_gb": 113.23,
        "workflows": [
            "03_MiniMax-H3_极速图生有声视频(单图带立体声).json",
            "04_MiniMax-H3_首尾帧有声过渡(双图带立体声).json",
        ],
        # 03/04 工作流默认加载、但放在 H3 基础包里的文件：点本包「一键配齐」时一起补齐，缺了不算配齐
        "deps": ["minimax_h3_audio_vae_fp32.safetensors", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "minimax_h3_video_vae_int8_convrot.safetensors", H3_HERETIC_TEXT_ENCODER],
        "models": [
            ("diffusion_models", "minimax_h3_fl2va_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors", 31.70),
            ("diffusion_models", "minimax_h3_ref2va_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors", 31.70),
            ("text_encoders", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", 25.28),
            ("text_encoders", "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors", "ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot", "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors", 24.55),
        ],
    },
    {
        "id": "minimax_h3_hq_bf16",
        "title": "MiniMax-H3 附加 · 满血 bf16 主干 + 官方 bf16 编码器 + fp16 视频 VAE",
        "badge": "自动补齐基础包文件 · 最高精度",
        "scenario": "给 03/04 工作流的下拉框多两个选项：满血 bf16 主干、官方 bf16 编码器与 fp16 视频 VAE，本包约 115 GB；连同 03/04 工作流默认要用的基础包文件合计约 154 GB，不要再同时装其它大包",
        "approx_size_gb": 114.55,
        "workflows": [
            "03_MiniMax-H3_极速图生有声视频(单图带立体声).json",
            "04_MiniMax-H3_首尾帧有声过渡(双图带立体声).json",
        ],
        # 03/04 工作流默认加载、但放在 H3 基础包里的文件：点本包「一键配齐」时一起补齐，缺了不算配齐
        "deps": ["minimax_h3_audio_vae_fp32.safetensors", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "minimax_h3_video_vae_int8_convrot.safetensors", H3_HERETIC_TEXT_ENCODER],
        "models": [
            ("diffusion_models", "minimax_h3_fl2va_bf16.safetensors", "Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_fl2va_bf16.safetensors", 61.73),
            ("vae", "minimax_h3_video_vae_fp16.safetensors", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_fp16.safetensors", 4.85),
            ("text_encoders", "qwen3vl_32b_minimax_h3_bf16.safetensors", "Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors", 47.97),
        ],
    },
    # H3 Director 导演台插件（开机自动装）的全部可选权重 + 9 套示例工作流；主干/编码器/8 步 LoRA 用 H3 基础包里的
    {
        "id": "minimax_h3_director",
        "title": "MiniMax-H3 导演台 · 多段 + 参考主体 + 改视频 + 语义桥 / 二采放大 / 修脸",
        "badge": "自动补齐基础包文件 · 导演台全功能",
        "scenario": "H3 Director 插件的 9 套示例工作流 + Comfyit 资源包里的外接参数版与官方 i2v/r2v/t2v 模板共 5 套，以及全部可选权重：参考主体 ref2va 主干（int8 与 w4a8 两种）、fp16 视频 VAE、4 步 Turbo LoRA、语义桥（原版 + BUNNY V1/V2）、3D 潜空间放大（bf16 / fp16 / fp32）、UltraSharp 与 RealESRGAN 像素放大、TAE 实时预览、修脸检测，本包约 41 GB；连同工作流要用的基础包编码器、fl2va 主干、音频 VAE 与 8 步 LoRA 合计约 77 GB",
        "approx_size_gb": 40.69,
        "workflows": [name for _, name in H3_DIRECTOR_EXAMPLE_WORKFLOWS] + [n for n, (pk, _) in H3_DIRECTOR_BUNDLED_WORKFLOWS.items() if pk == "core"],
        # 工作流加载、但放在 H3 基础包里的文件（编码器、fl2va 主干、音频 VAE、8 步 LoRA）：一起补齐，缺了不算配齐
        "deps": ["minimax_h3_audio_vae_fp32.safetensors", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
        "models": [
            ("diffusion_models", "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", 19.53),
            # 示例「二采放大」工作流用的 w4a8 混合量化 ref2va 主干（官方仓库没有；与资源包同一份的公开副本）
            ("diffusion_models", "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors", "starsfriday/MiniMax-H3-w4a8", "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors", 11.68),   # SHA256 与 Comfyit 资源包里的同名文件逐字节一致
            ("vae", "minimax_h3_video_vae_fp16.safetensors", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_fp16.safetensors", 4.85),
            ("loras", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", "lightx2v/Minimax-h3-Turbo", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", 1.82),
            ("latent_upscale_models", "minimax_h3_latent_upscaler_3d_bf16.safetensors", "LBH-123-AI/Minimax_h3_latent_Upscaler", "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors", 0.64),
            ("latent_upscale_models", "minimax_h3_latent_upscaler_3d_fp16.safetensors", "LBH-123-AI/Minimax_h3_latent_Upscaler", "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors", 0.64),
            ("latent_upscale_models", "minimax_h3_latent_upscaler_3d_fp32.pth", "LBH-123-AI/Minimax_h3_latent_Upscaler", "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp32.pth", 1.29),
            # Refine 选 lanczos 时可另接的像素放大模型
            ("upscale_models", "4x-UltraSharp.pth", "Kim2091/UltraSharp", "4x-UltraSharp.pth", 0.06),
            ("upscale_models", "RealESRGAN_x4plus.pth", "lllyasviel/Annotators", "RealESRGAN_x4plus.pth", 0.06),
            ("semantic_bridge", "MiniMaxH3_SemanticBridge_v1.safetensors", "speach1sdef178/MiniMax-H3-Semantic-Bridge", "MiniMaxH3_SemanticBridge_v1.safetensors", 0.01),
            ("semantic_bridge", "BUNNY_H3_ActionLogic_Bridge_V1.safetensors", "JOKER141/BUNNY_H3_Conditioning_Bridge", "BUNNY_H3_ActionLogic_Bridge_V1.safetensors", 0.02),
            ("semantic_bridge", "BUNNY_H3_ActionLogic_Bridge_V2.safetensors", "JOKER141/BUNNY_H3_Conditioning_Bridge", "BUNNY_H3_ActionLogic_Bridge_V2.safetensors", 0.02),
            ("vae_approx", "taeh3.safetensors", "t8star/Taeh3-Comfy", "models/vae_approx/taeh3.safetensors", 0.02),
            ("ultralytics/bbox", "face_yolov8m.pt", "Bingsu/adetailer", "face_yolov8m.pt", 0.05),
        ],
    },
    # 资源包里的两套「导演台全能工作流」另需第三方混合主干与 Pika LoRA，单独成包，避免和满血 bf16 包一起把磁盘挤满
    {
        "id": "minimax_h3_director_plus",
        "title": "MiniMax-H3 导演台扩展 · 全能工作流（混合主干 + Pika 六步加速）",
        "badge": "自动补齐基础包与导演台文件",
        "scenario": "Comfyit 资源包的两套导演台全能工作流（一采 + SelfLift + 二采放大 + 修脸一条龙）；另下 fl2va/ref2va 混合 int8 主干与 Pika H3 Flex 的 fl2va / ref2va 两个 LoRA，本包约 23 GB；连同工作流要用的基础包与导演台文件合计约 87 GB",
        "approx_size_gb": 23.17,
        "workflows": [n for n, (pk, _) in H3_DIRECTOR_BUNDLED_WORKFLOWS.items() if pk == "plus"],
        # 19/20 工作流加载、但放在 H3 基础包与导演台包里的文件：一起补齐，缺了不算配齐（修脸 YOLO 节点默认旁路，但很小，一并装上）
        "deps": [
            "minimax_h3_audio_vae_fp32.safetensors", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "minimax_h3_video_vae_fp16.safetensors",
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "minimax_h3_latent_upscaler_3d_bf16.safetensors",
            "face_yolov8m.pt",
        ],
        "models": [
            ("diffusion_models", "minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors", "smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models", "minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors", 19.53),
            ("loras", "pika_h3_flex_fl2va_v1_bf16.safetensors", "Pika-Labs/Pika-H3-Flex", "comfy/loras/pika_h3_flex_fl2va_comfy_bf16.safetensors", 1.82),
            ("loras", "pika_h3_flex_ref2va_v1_bf16.safetensors", "Pika-Labs/Pika-H3-Flex", "comfy/loras/pika_h3_flex_ref2va_comfy_bf16.safetensors", 1.82),
        ],
    },
    # 导演台满血版：只装 bf16 ref2va 主干 + 官方 bf16 编码器 + 小件，不带导演台包的 int8 / w4a8 主干，也不补 nvfp4 编码器和 8 步 LoRA
    {
        "id": "minimax_h3_director_bf16",
        "title": "MiniMax-H3 导演台 · 满血 bf16（参考主体）",
        "badge": "满血主干 · 参考主体 r2v",
        "scenario": "导演台 4 套参考主体（r2v）工作流的满血版：主干 / 编码器下拉默认是 bf16 ref2va 与官方 bf16 编码器，另含 fp16 视频 VAE、音频 VAE、TAE 实时预览、3D 潜空间放大与修脸检测，合计约 116 GB；盘只够单独装它，不要和满血 bf16 首尾帧包或导演台 int8 包同时装",
        "approx_size_gb": 109.7,
        "workflows": list(H3_DIRECTOR_BF16_R2V_WORKFLOWS.values()),
        # 这 4 套工作流要用、但放在别的包里的小文件；不补 int8 主干、nvfp4 编码器和 8 步 LoRA
        "deps": [
            "minimax_h3_audio_vae_fp32.safetensors",
            "minimax_h3_video_vae_fp16.safetensors",
            "minimax_h3_latent_upscaler_3d_bf16.safetensors",
            "taeh3.safetensors",
            "face_yolov8m.pt",
        ],
        "models": [
            ("diffusion_models", "minimax_h3_ref2va_bf16.safetensors", "Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_ref2va_bf16.safetensors", 61.73),
            ("text_encoders", "qwen3vl_32b_minimax_h3_bf16.safetensors", "Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors", 47.97),
        ],
    },
    {
        "id": "minimax_h3_image_t2i_edit",
        "title": "MiniMax-H3-Image · 文生图 + 参考图改图 + 多参考图合成",
        "badge": "6步出图 · 多张参考图",
        "scenario": "含「文生图」「按原图尺寸改图」「多张参考图合成新画面」3 套画布；主干是官方剪枝 bf16 版，要满血版请用左侧的 MiniMax-H3-Image 模型",
        "approx_size_gb": 70.7,
        "workflows": [
            "07_MiniMax-H3-Image_文生图(4MP高清).json",
            "08_MiniMax-H3-Image_参考图改图(按原图尺寸).json",
            "09_MiniMax-H3-Image_多参考图合成新画面(16比9).json",
        ],
        "models": [
            ("diffusion_models", "minimax_h3_image_pruned_bf16.safetensors", "Comfy-Org/MiniMax-H3-Image", "diffusion_models/minimax_h3_image_pruned_bf16.safetensors", 37.47),
            ("text_encoders", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3-Image", "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", 25.28),
            # 工作流里的 enable_PE (自动扩写提示词) 用的小模型；默认关闭，但节点在图里，文件必须存在
            ("text_encoders", "qwen3.5_9b_w4a8_gptq.safetensors", "Comfy-Org/MiniMax-H3-Image", "text_encoders/qwen3.5_9b_w4a8_gptq.safetensors", 6.84),
            ("vae", "minimax_h3_image_vae_fp16.safetensors", "Comfy-Org/MiniMax-H3-Image", "vae/minimax_h3_image_vae_fp16.safetensors", 1.09),
        ],
    },
    {
        "id": "ltx23_flf2v",
        "title": "LTX-2.3 (22B) · 首尾帧电影级运镜有声工作流",
        "badge": "电影推拉摇移 · 仅 2 个权重",
        "scenario": "专为首尾关键帧大幅度镜头运动优化（如高空俯拍平滑推镜至低角度特写 + 自动合成配乐音轨）",
        "approx_size_gb": 36.3,
        "workflows": [
            "05_LTX-2.3_首尾帧电影运镜(双图极速带音效).json",
        ],
        "models": [
            ("checkpoints", "ltx-2.3-22b-distilled-fp8.safetensors", "Lightricks/LTX-2.3-fp8", "ltx-2.3-22b-distilled-fp8.safetensors", 27.50),
            ("text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors", "Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", 8.80),
        ],
    },
    {
        "id": "hunyuan15_i2v_1080p",
        "title": "混元 HunyuanVideo 1.5 · 单图直出 1080P 超分工作流",
        "badge": "720P + 1080P 双路超清",
        "scenario": "已默认激活潜空间超分管线，传 1 张图一键自动渲染 720P 原片与 1920×1080 超高清成片",
        "approx_size_gb": 43.5,
        "workflows": [
            "06_Hunyuan1.5_单图直出1080P超清视频.json",
        ],
        "models": [
            ("clip_vision", "sigclip_vision_patch14_384.safetensors", "Comfy-Org/sigclip_vision_384", "sigclip_vision_patch14_384.safetensors", 0.80),
            ("text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", 8.74),
            ("text_encoders", "byt5_small_glyphxl_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors", 0.41),
            ("latent_upscale_models", "hunyuanvideo15_latent_upsampler_1080p.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/latent_upscale_models/hunyuanvideo15_latent_upsampler_1080p.safetensors", 0.45),
            ("diffusion_models", "hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/diffusion_models/hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors", 15.20),
            ("vae", "hunyuanvideo15_vae_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/vae/hunyuanvideo15_vae_fp16.safetensors", 2.35),
            ("diffusion_models", "hunyuanvideo1.5_720p_i2v_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/diffusion_models/hunyuanvideo1.5_720p_i2v_fp16.safetensors", 15.51),
        ],
    },
]


def _comfy_pack_all_models(pack: Dict[str, Any]) -> List[Tuple[str, str, str, str, float]]:
    """本包自己的权重 + deps：工作流实际加载、但放在别的包里的权重（按文件名到其它包里找下载来源）"""
    models = list(pack["models"])
    names = {m[1] for m in models}
    for dep in pack.get("deps", []):
        if dep in names:
            continue
        models.append(next(m for p in COMFY_PRESET_WORKFLOW_PACKS for m in p["models"] if m[1] == dep))
        names.add(dep)
    return models


def _comfy_missing_models(persist_root: str, models: List[Tuple[str, str, str, str, float]]) -> List[Tuple[str, str, str, str, float]]:
    return [m for m in models if not os.path.isfile(os.path.join(persist_root, "models", m[0], m[1]))]


def _disk_free_gb(path: str) -> float:
    while path and not os.path.isdir(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path or "/").free / (1024 ** 3)




def _ensure_comfyui_beginner_assets(comfy_root: str, persist_root: str):
    """
    为新手免折腾准备：
    1. 默认将 ComfyUI 界面语言设为简体中文 (Comfy.Locale = zh)；
    2. 生成 1280x720 预设示例首帧/尾帧图片至 input/ 目录并建立软链，即使未传图直接点「运行」也零报错。
    """
    try:
        settings_dir = os.path.join(persist_root, "user", "default")
        os.makedirs(settings_dir, exist_ok=True)
        settings_path = os.path.join(settings_dir, "comfy.settings.json")
        cfg_obj: Dict[str, Any] = {}
        if os.path.isfile(settings_path):
            try:
                cfg_obj = json.load(open(settings_path, "r", encoding="utf-8"))
            except Exception:
                cfg_obj = {}
        if not cfg_obj.get("Comfy.Locale"):
            cfg_obj["Comfy.Locale"] = "zh"
            with open(settings_path, "w", encoding="utf-8") as sf:
                json.dump(cfg_obj, sf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    try:
        from PIL import ImageDraw
        input_persist = os.path.join(persist_root, "input")
        os.makedirs(input_persist, exist_ok=True)
        if os.path.isdir(comfy_root):
            _safe_symlink_dir(input_persist, os.path.join(comfy_root, "input"))

        start_img_p = os.path.join(input_persist, "easycolab_sample_start.png")
        end_img_p = os.path.join(input_persist, "easycolab_sample_end.png")

        if not os.path.isfile(start_img_p):
            w, h = 1280, 720
            im = Image.new("RGB", (w, h), (16, 24, 48))
            dr = ImageDraw.Draw(im)
            for y in range(h):
                t = y / float(h - 1)
                r = int(18 * (1 - t) + 64 * t)
                g = int(28 * (1 - t) + 108 * t)
                b = int(68 * (1 - t) + 168 * t)
                dr.line([(0, y), (w, y)], fill=(r, g, b))
            # 太阳光晕与海平面剪影（首帧：朝阳初升左侧）
            dr.ellipse([260, 210, 440, 390], fill=(255, 206, 126))
            dr.ellipse([285, 235, 415, 365], fill=(255, 240, 196))
            dr.polygon([(0, 520), (240, 430), (520, 495), (860, 410), (1140, 485), (1280, 450), (1280, 720), (0, 720)], fill=(18, 28, 46))
            dr.rectangle([0, 565, w, h], fill=(12, 22, 38))
            im.save(start_img_p, format="PNG")

        if not os.path.isfile(end_img_p):
            w, h = 1280, 720
            im = Image.new("RGB", (w, h), (32, 18, 52))
            dr = ImageDraw.Draw(im)
            for y in range(h):
                t = y / float(h - 1)
                r = int(42 * (1 - t) + 232 * t)
                g = int(24 * (1 - t) + 118 * t)
                b = int(76 * (1 - t) + 88 * t)
                dr.line([(0, y), (w, y)], fill=(r, g, b))
            # 尾帧：晚霞金日移至右侧高空，测试首尾帧运镜与光影平滑过渡
            dr.ellipse([800, 130, 1000, 330], fill=(255, 168, 92))
            dr.ellipse([830, 160, 970, 300], fill=(255, 232, 178))
            dr.polygon([(0, 540), (210, 460), (490, 510), (820, 395), (1090, 470), (1280, 430), (1280, 720), (0, 720)], fill=(24, 16, 36))
            dr.rectangle([0, 575, w, h], fill=(16, 12, 28))
            im.save(end_img_p, format="PNG")
    except Exception as e:
        print(f"[ComfyUI] 生成默认占位参考图提示: {e}")


def _fetch_official_comfy_template(template_filename: str) -> Dict[str, Any]:
    url = COMFY_TEMPLATE_RAW_BASE + template_filename
    req = urllib.request.Request(url, headers={"User-Agent": "EasyColab/2.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _make_zh_guide_note(node_id: int, pos: List[float], title: str, steps_md: str) -> Dict[str, Any]:
    return {
        "id": node_id,
        "type": "MarkdownNote",
        "pos": pos,
        "size": [430, 340],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "title": title,
        "properties": {},
        "widgets_values": [steps_md],
        "widgets_values_named": {"text": steps_md},
        "color": "#1e3a5f",
        "bgcolor": "#13263e",
    }


def _ensure_easycolab_comfy_preset_helper(comfy_root: str, persist_root: str):
    """
    预置轻量前端扩展：每次通过全新 *.trycloudflare.com 域名首次打开 ComfyUI 时，
    由于新域名 localStorage 为空，ComfyUI 默认会加载内置的 Z-Image Turbo 文生图模板，
    并把缺失的模型自动替换成目录里字母序第一个文件（例如把 minimax_h3_audio_vae 塞进图像 VAEDecode 导致维度 24!=32 报错）。
    本扩展会在检测到默认 Z-Image Turbo 占位图时，自动切换为用户已安装的 EasyColab 中文预置工作流。
    """
    if not os.path.isdir(comfy_root):
        return
    try:
        helper_dir = os.path.join(comfy_root, "custom_nodes", "easycolab_preset_helper")
        js_dir = os.path.join(helper_dir, "js")
        os.makedirs(js_dir, exist_ok=True)

        init_py = '''import os
import json
from aiohttp import web
import folder_paths
from server import PromptServer

WEB_DIRECTORY = "./js"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    if getattr(PromptServer, "instance", None) is not None:
        @PromptServer.instance.routes.get("/easycolab/default_workflow")
        async def _easycolab_get_default_workflow(request):
            try:
                get_ud = getattr(folder_paths, "get_user_directory", None)
                user_dir = get_ud() if callable(get_ud) else os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "user",
                )
                default_dir = os.path.join(user_dir, "default")
                wf_dir = os.path.join(default_dir, "workflows")
                target_fn = ""
                ptr_path = os.path.join(default_dir, "easycolab_active_preset.json")
                if os.path.isfile(ptr_path):
                    try:
                        with open(ptr_path, "r", encoding="utf-8") as pf:
                            ptr = json.load(pf)
                        cand = str(ptr.get("workflow_filename") or "").strip()
                        if cand and os.path.isfile(os.path.join(wf_dir, cand)):
                            target_fn = cand
                    except Exception:
                        pass
                if not target_fn and os.path.isdir(wf_dir):
                    cands = sorted(
                        f for f in os.listdir(wf_dir)
                        if f.lower().endswith(".json") and not f.startswith(".")
                    )
                    if cands:
                        target_fn = cands[0]
                if target_fn:
                    wf_path = os.path.join(wf_dir, target_fn)
                    with open(wf_path, "r", encoding="utf-8") as wf:
                        wf_obj = json.load(wf)
                    return web.json_response({"ok": True, "filename": target_fn, "workflow": wf_obj})
            except Exception as exc:
                return web.json_response({"ok": False, "error": str(exc)})
            return web.json_response({"ok": False})
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
'''

        autoload_js = '''import { app } from "../../scripts/app.js";

function isBuiltInZImageStarterGraph(graph) {
    if (!graph || !Array.isArray(graph._nodes)) return false;
    if (graph._nodes.length === 0) return false;
    return graph._nodes.some((n) => {
        const models = n?.properties?.models;
        return Array.isArray(models) && models.some((m) => String(m?.url || "").includes("z_image_turbo"));
    });
}

app.registerExtension({
    name: "EasyColab.PresetWorkflowAutoLoader",
    async setup() {
        if (sessionStorage.getItem("easycolab_preset_autoloaded_v1") === "1") {
            return;
        }
        const tryAutoLoad = async (allowEmpty) => {
            if (sessionStorage.getItem("easycolab_preset_autoloaded_v1") === "1") return true;
            const g = app.graph;
            const isStarter = isBuiltInZImageStarterGraph(g);
            const isEmpty = allowEmpty && g && Array.isArray(g._nodes) && g._nodes.length === 0;
            if (!isStarter && !isEmpty) return false;

            sessionStorage.setItem("easycolab_preset_autoloaded_v1", "1");
            try {
                const resp = await fetch("/easycolab/default_workflow", { cache: "no-store" });
                if (!resp.ok) return true;
                const data = await resp.json();
                if (data && data.ok && data.workflow) {
                    const wfTitle = String(data.filename || "EasyColab_Preset").replace(/\\.json$/i, "");
                    await app.loadGraphData(data.workflow, true, true, wfTitle);
                    console.log("[EasyColab] 已自动载入预置中文工作流:", data.filename);
                }
            } catch (err) {
                console.warn("[EasyColab] 自动载入预置工作流跳过:", err);
            }
            return true;
        };

        setTimeout(() => { tryAutoLoad(false); }, 350);
        setTimeout(() => { tryAutoLoad(true); }, 1100);
    },
});
'''
        with open(os.path.join(helper_dir, "__init__.py"), "w", encoding="utf-8") as f_py:
            f_py.write(init_py)
        with open(os.path.join(js_dir, "autoload.js"), "w", encoding="utf-8") as f_js:
            f_js.write(autoload_js)
    except Exception as e:
        print(f"[ComfyUI] 写入 EasyColab 默认工作流引导扩展提示: {e}")


def _prune_workflow_dangling_links(wf_data: Dict[str, Any]):
    """确保每个节点的 inputs/outputs 链接表与顶层 links 完全一致，彻底杜绝 LiteGraph 断链警告"""
    node_ids = {n.get("id") for n in wf_data.get("nodes", [])}
    valid_links = [
        lk for lk in wf_data.get("links", [])
        if isinstance(lk, list) and len(lk) >= 5 and lk[1] in node_ids and lk[3] in node_ids
    ]
    wf_data["links"] = valid_links
    valid_lids = {lk[0] for lk in valid_links}
    for n in wf_data.get("nodes", []):
        for inp in n.get("inputs", []) or []:
            if inp.get("link") is not None and inp.get("link") not in valid_lids:
                inp["link"] = None
        for out in n.get("outputs", []) or []:
            if isinstance(out.get("links"), list):
                out["links"] = [lid for lid in out["links"] if lid in valid_lids]


def _expose_h3_base_steps(wf: Dict[str, Any], min_steps: int = 20) -> bool:
    """官方 H3 模板里关闭 Turbo LoRA 时的采样步数写死在子图内部 (20)。这里把它露到主节点面板，并用 max(min_steps, a) 保底。
    模板结构与预期不符时不做任何改动并返回 False，工作流照常生成。"""
    try:
        sg = wf["definitions"]["subgraphs"][0]
        nodes = {n["id"]: n for n in sg["nodes"]}
        links = {lk["id"]: lk for lk in sg["links"]}
        switch = next(
            n for n in sg["nodes"]
            if n.get("type") == "ComfySwitchNode" and "Steps" in (n.get("title") or "")
        )
        false_in = next(i for i in switch["inputs"] if i.get("name") == "on_false")
        base_link = links[false_in["link"]]
        base = nodes[base_link["origin_id"]]
        outer = next(n for n in wf["nodes"] if n.get("type") == sg["id"])
        widget_inputs = sum(1 for i in sg["inputs"] if i.get("type") != "IMAGE")
        if (
            base.get("type") != "PrimitiveInt"
            or base.get("inputs")
            or not isinstance(outer.get("widgets_values"), list)
            or len(outer["widgets_values"]) != widget_inputs
        ):
            return False
    except (KeyError, IndexError, StopIteration, TypeError):
        return False

    state = sg.setdefault("state", {})
    in_link = max(int(state.get("lastLinkId") or 0), max(links)) + 1
    out_link = in_link + 1
    guard_id = max(int(state.get("lastNodeId") or 0), max(nodes)) + 1
    label = f"关闭加速时的步数 (最少 {min_steps})"
    last_pos = sg["inputs"][-1].get("pos") or [0, 0]

    # 子图新增一个输入口，接到原本写死的步数节点上
    sg["links"].append({
        "id": in_link, "origin_id": -10, "origin_slot": len(sg["inputs"]),
        "target_id": base["id"], "target_slot": 0, "type": "INT",
    })
    sg["inputs"].append({
        "id": str(uuid.uuid4()), "name": "base_steps", "type": "INT",
        "linkIds": [in_link], "label": label, "pos": [last_pos[0], last_pos[1] + 20],
    })
    base["inputs"] = [{
        "localized_name": "value", "name": "value", "type": "INT",
        "widget": {"name": "value"}, "link": in_link,
    }]

    # 步数节点 -> max(min_steps, a) -> 步数开关的 on_false
    expr = f"max({int(min_steps)}, a)"
    base_pos = base.get("pos") or [0, 0]
    sg["nodes"].append({
        "id": guard_id,
        "type": "ComfyMathExpression",
        "pos": [base_pos[0], base_pos[1] - 230],
        "size": [270, 170],
        "flags": {},
        "order": switch.get("order", 0),
        "mode": 0,
        "inputs": [
            {"label": "a", "localized_name": "values.a", "name": "values.a", "type": "FLOAT,INT,BOOLEAN", "link": base_link["id"]},
            {"label": "b", "localized_name": "values.b", "name": "values.b", "shape": 7, "type": "FLOAT,INT,BOOLEAN", "link": None},
        ],
        "outputs": [
            {"localized_name": "FLOAT", "name": "FLOAT", "type": "FLOAT", "links": None},
            {"localized_name": "INT", "name": "INT", "type": "INT", "links": [out_link]},
            {"localized_name": "BOOL", "name": "BOOL", "type": "BOOLEAN", "links": None},
        ],
        "title": f"步数保底 {expr}",
        "properties": {**(base.get("properties") or {}), "Node name for S&R": "ComfyMathExpression"},
        "widgets_values": [expr],
        "widgets_values_named": {"expression": expr},
    })
    base_link["target_id"] = guard_id
    base_link["target_slot"] = 0
    sg["links"].append({
        "id": out_link, "origin_id": guard_id, "origin_slot": 1,
        "target_id": switch["id"], "target_slot": switch["inputs"].index(false_in), "type": "INT",
    })
    false_in["link"] = out_link
    state["lastLinkId"] = out_link
    state["lastNodeId"] = guard_id

    # 主节点面板上对应多出一个步数控件
    outer.setdefault("inputs", []).append({
        "label": label, "name": "base_steps", "type": "INT",
        "widget": {"name": "base_steps"}, "link": None,
    })
    outer["widgets_values"].append(int(min_steps))
    if isinstance(outer.get("widgets_values_named"), dict):
        outer["widgets_values_named"]["base_steps"] = int(min_steps)
    return True


def _build_and_install_comfy_preset_workflows(
    persist_root: str,
    pack_ids: Optional[List[str]] = None,
) -> List[str]:
    """拉取官方最新模板并转换为全中文标题、内置三步傻瓜注释、G4 高清参数与零缺图报错的开箱即用工作流"""
    wf_dir = os.path.join(persist_root, "user", "default", "workflows")
    os.makedirs(wf_dir, exist_ok=True)
    target_set = set(pack_ids) if pack_ids else {p["id"] for p in COMFY_PRESET_WORKFLOW_PACKS}
    installed_files: List[str] = []

    # 1. 万相 Wan2.2 14B (单图 I2V + 首尾帧 FLF2V 4步极速版)
    if "wan22_i2v_flf2v" in target_set:
        try:
            d_i2v = _fetch_official_comfy_template("video_wan2_2_14B_i2v.json")
            kept_nodes = []
            for n in d_i2v.get("nodes", []):
                ntype = n.get("type")
                if ntype == "MarkdownNote":
                    continue
                if n.get("id") == 97 and ntype == "LoadImage":
                    n["title"] = "① 点击此处选择起始图片 (Choose file to upload)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif n.get("id") == 129:
                    n["title"] = "② 填写画面动作与镜头描述 (G4 已预设 1280×720 · 4步极速)"
                    wv = list(n.get("widgets_values") or [])
                    if len(wv) >= 4:
                        wv[0] = "镜头缓慢向前推进，晨曦阳光洒在海面与远山上，云层轻柔流动，水面泛起细腻金光涟漪，电影级光影质感。"
                        wv[1] = 1280
                        wv[2] = 720
                    n["widgets_values"] = wv
                    n.pop("widgets_values_named", None)
                elif ntype == "SaveVideo":
                    n["title"] = "③ 生成结果预览与自动保存 (右键可下载视频)"
                    n["widgets_values"] = ["video/Wan2.2_单图生视频", "auto", "auto"]
                    n.pop("widgets_values_named", None)
                kept_nodes.append(n)
            kept_nodes.append(_make_zh_guide_note(
                501,
                [-520, -420],
                "【EasyColab 三步傻瓜出片指南 · 单图生视频】",
                "## 三步直接出片（无需连线！）\n\n"
                "1. **传图**：点击右侧 **「① 点击此处选择起始图片」** 节点底部的上传按钮，选一张你的图片。\n"
                "2. **写动作**：在中间 **「② 填写画面动作与镜头描述」** 第一个文本框里用中文或英文写希望画面怎么动（如：镜头推进、人物转头微笑、发丝随风飘动）。\n"
                "3. **点运行**：点击网页右上角蓝色的 **「运行 (Run)」** 按钮，G4 显卡约 **25 秒** 即可在右侧直接预览成片！\n\n"
                "> 提示：所有生成的视频都会自动同步至你的 Google Drive，也可回到 EasyColab 客户端一键拉取到 Mac 本地。",
            ))
            d_i2v["nodes"] = kept_nodes
            d_i2v["last_node_id"] = max(int(d_i2v.get("last_node_id") or 200), 505)
            d_i2v.setdefault("extra", {})["ds"] = {"scale": 0.72, "offset": [580, 560]}
            _prune_workflow_dangling_links(d_i2v)
            out_1 = os.path.join(wf_dir, "01_Wan2.2_极速单图生视频(4步傻瓜版).json")
            with open(out_1, "w", encoding="utf-8") as wf:
                json.dump(d_i2v, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_1))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] Wan2.2 I2V 模板处理异常: {e}")

        try:
            d_flf = _fetch_official_comfy_template("video_wan2_2_14B_flf2v.json")
            # 仅提取官方模板中 4-Step LightX2V LoRA 极速分支连通分量 (原 mode=4)，彻底剔除慢速重复节点并自动激活 (mode=0)
            active_ids = {n["id"] for n in d_flf.get("nodes", []) if n.get("mode") == 4}
            flf_nodes = []
            for n in d_flf.get("nodes", []):
                nid = n.get("id")
                if nid not in active_ids or n.get("type") == "Note":
                    continue
                n["mode"] = 0
                if isinstance(n.get("pos"), list) and len(n["pos"]) >= 2:
                    n["pos"] = [n["pos"][0], n["pos"][1] - 1600]
                if nid == 68 and n.get("type") == "LoadImage":
                    # link 157 -> node 67 slot 5 (start_image)
                    n["title"] = "① 点击上传起始首帧图片 (Start Frame)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 62 and n.get("type") == "LoadImage":
                    # link 158 -> node 67 slot 6 (end_image)
                    n["title"] = "② 点击上传结束尾帧图片 (End Frame)"
                    n["widgets_values"] = ["easycolab_sample_end.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 6 and n.get("type") == "CLIPTextEncode":
                    n["title"] = "③ 输入首尾帧过渡动作与运镜描述 (正向提示词)"
                    n["widgets_values"] = [
                        "清晨海面的朝阳缓缓升起并化作绚丽的傍晚漫天晚霞，镜头平稳向右推进，云层与海浪自然流动过渡，光影由冷蓝渐变为温暖金红，画面连贯丝滑。"
                    ]
                    n.pop("widgets_values_named", None)
                elif nid == 7 and n.get("type") == "CLIPTextEncode":
                    n["title"] = "反向提示词 (保持默认即可)"
                elif nid == 67 and n.get("type") == "WanFirstLastFrameToVideo":
                    n["title"] = "④ 视频尺寸与总帧数 (81帧≈5秒 · 1280×720)"
                    n["widgets_values"] = [1280, 720, 81, 1]
                    n.pop("widgets_values_named", None)
                elif nid == 61 and n.get("type") == "SaveVideo":
                    n["title"] = "⑤ 首尾帧过渡成片预览与自动保存"
                    n["widgets_values"] = ["video/Wan2.2_首尾帧过渡", "auto", "auto"]
                    n.pop("widgets_values_named", None)
                flf_nodes.append(n)

            flf_nodes.append(_make_zh_guide_note(
                502,
                [-430, 490],
                "【EasyColab 三步傻瓜出片指南 · 首尾帧过渡】",
                "## 首尾两张图丝滑变形/运镜（4步极速已开启）\n\n"
                "1. **传首尾两张图**：在右侧 **「① 起始首帧图片」** 和 **「② 结束尾帧图片」** 分别上传开头和结尾画面。\n"
                "2. **写过渡过程**：在 **「③ 输入首尾帧过渡动作与运镜描述」** 写清楚从图 1 怎么变化/运镜到图 2。\n"
                "3. **点右上角运行**：无需按任何快捷键，4 步加速 LoRA 已自动配好，约 **25~35 秒** 直接生成丝滑过渡视频！",
            ))
            d_flf["nodes"] = flf_nodes
            d_flf["links"] = [
                lk for lk in d_flf.get("links", [])
                if lk[1] in active_ids and lk[3] in active_ids
            ]
            zh_group_titles = {
                1: "自动模型加载区 (无需改动)",
                2: "第一步：上传起始首帧图片",
                5: "第二步：上传结束尾帧图片",
                3: "第三步：输入画面过渡描述",
                4: "第四步：视频分辨率与帧数 (默认 1280×720)",
            }
            kept_groups = []
            for g in d_flf.get("groups", []):
                gid = g.get("id")
                if gid in zh_group_titles:
                    b = list(g.get("bounding", [0, 0, 400, 400]))
                    if len(b) >= 2:
                        b[1] = b[1] - 1600
                    g["bounding"] = b
                    g["title"] = zh_group_titles[gid]
                    kept_groups.append(g)
            d_flf["groups"] = kept_groups
            d_flf["last_node_id"] = max(int(d_flf.get("last_node_id") or 200), 505)
            d_flf.setdefault("extra", {})["ds"] = {"scale": 0.62, "offset": [480, 120]}
            _prune_workflow_dangling_links(d_flf)
            out_2 = os.path.join(wf_dir, "02_Wan2.2_首尾帧丝滑过渡(双图4步极速).json")
            with open(out_2, "w", encoding="utf-8") as wf:
                json.dump(d_flf, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_2))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] Wan2.2 FLF2V 模板处理异常: {e}")

    # 2. MiniMax-H3 Turbo (单图有声视频 + 首尾帧有声视频 8步极速双工作流)
    if target_set & {"minimax_h3_i2v_flf2v", "minimax_h3_hq_int8", "minimax_h3_hq_bf16"}:
        try:
            import copy
            d_h3_raw = _fetch_official_comfy_template("video_minimax_h3_i2v.json")

            def _customize_h3(is_dual_frame: bool) -> Dict[str, Any]:
                d_h3 = copy.deepcopy(d_h3_raw)
                _expose_h3_base_steps(d_h3)
                nodes_out = []
                for n in d_h3.get("nodes", []):
                    nid = n.get("id")
                    ntype = n.get("type")
                    if ntype == "MarkdownNote" or nid in (119, 120):
                        continue
                    if nid == 114 and ntype == "LoadImage":
                        n["title"] = "① 点击上传起始首帧图片 (First Frame)"
                        n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                        n["widgets_values_named"] = {"image": "easycolab_sample_start.png", "upload": "image"}
                        n["pos"] = [-1320, 4820]
                    elif nid == 115 and ntype == "ResolutionSelector":
                        n["title"] = "画幅比例与清晰度 (0.6≈1056×608 · 0.98=1344×768)"
                        n["widgets_values"] = ["16:9 (Widescreen)", 0.6, 32]
                        n["widgets_values_named"] = {
                            "aspect_ratio": "16:9 (Widescreen)",
                            "megapixels": 0.6,
                            "multiple": 32,
                        }
                        n["pos"] = [-1320, 5660] if is_dual_frame else [-1320, 5260]
                    elif nid == 105:
                        n["title"] = (
                            "③ 填写过渡动作 + 音效配乐描述 (8步 Turbo 极速已开启)"
                            if is_dual_frame
                            else "② 填写画面动作 + 音效配乐描述 (8步 Turbo 极速已开启)"
                        )
                        zh_prompt = (
                            "电影级航拍镜头：画面从清晨宁静的蔚蓝海岸平稳推进并过渡至漫天金红晚霞的黄昏海湾，海浪轻拍礁石，光影自然流转。\n"
                            "Audio: 轻柔的海浪拍岸声、温暖的海风吹拂声，伴随舒缓治愈的钢琴与大提琴电影配乐。"
                            if is_dual_frame
                            else "电影级运镜缓慢向前推进，清晨阳光穿透薄雾洒在海面，金光波光粼粼，飞鸟掠过天际。\n"
                                 "Audio: 清脆的海浪声、微风声与远处海鸥鸣叫，搭配温柔空灵的钢琴环境配乐。"
                        )
                        wv = list(n.get("widgets_values") or [])
                        if len(wv) >= 10:
                            wv[0] = zh_prompt
                            wv[6] = H3_HERETIC_TEXT_ENCODER
                            wv[9] = True  # 默认启用 8-step Turbo LoRA 极速出片
                        n["widgets_values"] = wv
                        if isinstance(n.get("widgets_values_named"), dict):
                            n["widgets_values_named"]["prompt"] = zh_prompt
                            n["widgets_values_named"]["clip_name"] = H3_HERETIC_TEXT_ENCODER
                            n["widgets_values_named"]["value"] = True
                    elif nid == 92 and ntype == "SaveVideo":
                        n["title"] = "④ 有声视频预览与自动保存 (请打开播放器音量!)"
                        vid_prefix = "video/MiniMax_H3_首尾帧有声" if is_dual_frame else "video/MiniMax_H3_图生有声"
                        n["widgets_values"] = [vid_prefix, "auto", "auto"]
                        n["widgets_values_named"] = {
                            "filename_prefix": vid_prefix,
                            "format": "auto",
                            "codec": "auto",
                        }
                    nodes_out.append(n)

                links_out = [lk for lk in d_h3.get("links", []) if lk[0] != 228]
                if is_dual_frame:
                    end_node_id = 201
                    new_link_id = 250
                    nodes_out.append({
                        "id": end_node_id,
                        "type": "LoadImage",
                        "pos": [-1320, 5240],
                        "size": [315, 314],
                        "flags": {},
                        "order": 1,
                        "mode": 0,
                        "inputs": [],
                        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [new_link_id]}, {"name": "MASK", "type": "MASK", "links": None}],
                        "title": "② 点击上传结束尾帧图片 (Last Frame)",
                        "properties": {"Node name for S&R": "LoadImage"},
                        "widgets_values": ["easycolab_sample_end.png", "image"],
                        "widgets_values_named": {"image": "easycolab_sample_end.png", "upload": "image"},
                    })
                    for n in nodes_out:
                        if n.get("id") == 105:
                            for inp in n.get("inputs", []):
                                if inp.get("name") == "last_frame":
                                    inp["link"] = new_link_id
                    links_out.append([new_link_id, end_node_id, 0, 105, 1, "IMAGE"])
                    d_h3["last_link_id"] = max(int(d_h3.get("last_link_id") or 230), new_link_id + 1)

                nodes_out.append(_make_zh_guide_note(
                    503,
                    [-1800, 4820],
                    "【EasyColab 三步出片指南 · MiniMax-H3 原生有声视频】",
                    "## 画面 + 立体声音效/配乐一次生成！\n\n"
                    + (
                        "1. **传首尾两张图**：在 **「① 起始首帧」** 与 **「② 结束尾帧」** 上传起止图片。\n"
                        "2. **写画面 + 声音**：在中间 **「③ 过渡动作 + 音效配乐描述」** 写清画面变化，并在 `Audio:` 后写想要的环境音/台词/音乐。\n"
                        "3. **点运行**：已默认开启 **8 步 Turbo 加速**，生成后记得打开右侧播放器喇叭图标听立体声！"
                        if is_dual_frame
                        else "1. **传首帧图**：在 **「① 起始首帧」** 点击上传你的参考图。\n"
                             "2. **写画面 + 声音**：在中间 **「② 画面动作 + 音效配乐描述」** 描述动作，并在 `Audio:` 后写音效与配乐风格。\n"
                             "3. **点运行**：已默认开启 **8 步 Turbo 加速**，生成后打开右侧视频喇叭即可试听同步立体声！"
                    )
                    + "\n\n**想要更高质量**：先在客户端配齐「MiniMax-H3 附加」包，再在主节点把 `unet_name` 换成不剪枝 int8 或满血 bf16 主干，"
                      "`clip_name` 换成 int8 的 Heretic 编码器或 bf16 的官方编码器；"
                      "关掉 `turbo_mode` 后按「关闭加速时的步数」采样（最少 20 步）。",
                ))
                d_h3["nodes"] = nodes_out
                d_h3["links"] = links_out
                d_h3["last_node_id"] = max(int(d_h3.get("last_node_id") or 210), 505)
                d_h3.setdefault("extra", {})["ds"] = {"scale": 0.68, "offset": [1860, -4740]}
                _prune_workflow_dangling_links(d_h3)
                return d_h3

            out_3 = os.path.join(wf_dir, "03_MiniMax-H3_极速图生有声视频(单图带立体声).json")
            with open(out_3, "w", encoding="utf-8") as wf:
                json.dump(_customize_h3(is_dual_frame=False), wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_3))

            out_4 = os.path.join(wf_dir, "04_MiniMax-H3_首尾帧有声过渡(双图带立体声).json")
            with open(out_4, "w", encoding="utf-8") as wf:
                json.dump(_customize_h3(is_dual_frame=True), wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_4))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] MiniMax-H3 模板处理异常: {e}")

    # 2b. MiniMax-H3-Image (文生图 + 按原图尺寸改图 + 多参考图合成新画面)
    if "minimax_h3_image_t2i_edit" in target_set:
        try:
            import copy
            h3i_unet = "minimax_h3_image_pruned_bf16.safetensors"

            def _set_widget(node: Dict[str, Any], index: int, name: str, value: Any):
                node["widgets_values"][index] = value
                if isinstance(node.get("widgets_values_named"), dict):
                    node["widgets_values_named"][name] = value

            # 文生图：官方模板把整条链路收在一个子图节点里
            d_t2i = _fetch_official_comfy_template("image_minimax_h3_image_t2i.json")
            sg_ids = {sg.get("id") for sg in (d_t2i.get("definitions") or {}).get("subgraphs") or []}
            t2i_nodes = []
            for n in d_t2i.get("nodes", []):
                ntype = n.get("type")
                if ntype == "MarkdownNote":
                    continue
                if ntype in sg_ids:
                    n["title"] = "① 写提示词 (固定 6 步 · enable_PE 是自动扩写，自己写了长提示词就保持关闭)"
                    _set_widget(n, 0, "replace", "电影剧照质感的写实摄影：雨后的海边小镇清晨，一位撑着透明雨伞的女孩走在湿漉漉的石板路上，远处是薄雾里的灯塔，冷灰蓝色调，浅景深，35mm 胶片颗粒。")
                    _set_widget(n, 1, "switch", False)
                    _set_widget(n, 7, "unet_name", h3i_unet)
                elif ntype == "ResolutionSelector":
                    n["title"] = "画幅与像素量 (4 ≈ 400 万像素；这个模型低分辨率效果差，别低于 2)"
                    n["widgets_values"] = ["16:9 (Widescreen)", 4, 32]
                    n["widgets_values_named"] = {"aspect_ratio": "16:9 (Widescreen)", "megapixels": 4, "multiple": 32}
                elif ntype == "SaveImage":
                    n["title"] = "② 出图预览与自动保存"
                    _set_widget(n, 0, "filename_prefix", "image/MiniMax_H3_Image_文生图")
                t2i_nodes.append(n)
            t2i_nodes.append(_make_zh_guide_note(
                601,
                [-120, 380],
                "【EasyColab 两步出图指南 · MiniMax-H3-Image 文生图】",
                "## 写一段话，6 步出一张 400 万像素的图\n\n"
                "1. **写提示词**：在 **「① 写提示词」** 里描述画面，中英文都可以，越具体越好。\n"
                "2. **点运行**：右侧 **「② 出图预览」** 出图并自动保存。\n\n"
                "**画幅**：左下角的节点选比例和像素量。\n\n"
                "**enable_PE**：打开后会先用一个小模型把短提示词扩写成长描述再出图；自己已经写了详细提示词时保持关闭，否则会被改写。\n\n"
                "**换编码器**：`clip_name` 下拉框可以换成 H3 附加包里的 Heretic 编码器。",
            ))
            d_t2i["nodes"] = t2i_nodes
            d_t2i["last_node_id"] = max(int(d_t2i.get("last_node_id") or 0), 601)
            _prune_workflow_dangling_links(d_t2i)

            # 改图 / 多参考图：官方模板是平铺节点，参考图 1 决定画布尺寸
            d_edit_raw = _fetch_official_comfy_template("image_minimax_h3_image_edit.json")

            def _customize_h3_image_edit(free_canvas: bool) -> Dict[str, Any]:
                d = copy.deepcopy(d_edit_raw)
                nodes = [n for n in d.get("nodes", []) if n.get("type") != "MarkdownNote"]
                if free_canvas:
                    # 画布不跟参考图走时，“改前改后对比”没有意义
                    nodes = [n for n in nodes if n.get("type") != "ImageCompare"]
                links = d.get("links", [])
                cond = next(n for n in nodes if n.get("type") == "MiniMaxH3ImageConditioning")
                next_node = max(int(d.get("last_node_id") or 0), max(n["id"] for n in nodes)) + 1
                next_link = max(int(d.get("last_link_id") or 0), max(lk[0] for lk in links)) + 1

                for n in nodes:
                    ntype = n.get("type")
                    if ntype == "UNETLoader":
                        _set_widget(n, 0, "unet_name", h3i_unet)
                    elif ntype == "PrimitiveBoolean":
                        n["title"] = "enable_PE 自动扩写指令 (自己写清楚了就保持关闭)"
                        _set_widget(n, 0, "value", False)
                    elif ntype == "PrimitiveStringMultiline" and "user_prompt" in (n.get("title") or ""):
                        n["title"] = (
                            "② 描述要生成的新画面 (用 <Picture 1>、<Picture 2> 指代参考图里的人和物)"
                            if free_canvas
                            else "② 写修改指令 (用 <Picture 1> 指代要改的图，<Picture 2> 起是参考图)"
                        )
                        _set_widget(n, 0, "value", (
                            "电影剧照质感的写实摄影：<Picture 1> 里的人物站在雨后的海边栈桥上回头看向镜头，远处是薄雾里的灯塔，冷灰蓝色调，浅景深。人物的长相、服装和随身物品与 <Picture 1> 完全一致。"
                            if free_canvas
                            else "把 <Picture 1> 的天空换成漫天金红色晚霞，其余内容、构图和人物保持不变。"
                        ))
                    elif ntype == "LoadImage":
                        n["title"] = "① 参考图 1 (角色或物品)" if free_canvas else "① 要修改的图 (输出尺寸跟它走)"
                        n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                        n["widgets_values_named"] = {"image": "easycolab_sample_start.png", "upload": "image"}
                    elif ntype == "SaveImage":
                        n["title"] = "③ 出图预览与自动保存"
                        _set_widget(n, 0, "filename_prefix", "image/MiniMax_H3_Image_" + ("多参考图合成" if free_canvas else "参考图改图"))

                # 预留参考图 2~4：已接好线但默认旁路 (选中按 Ctrl+B 启用)，不启用就不参与出图
                for k in (2, 3, 4):
                    slot = next(i for i, inp in enumerate(cond["inputs"]) if inp.get("name") == f"images.image_{k}")
                    cond["inputs"][slot]["link"] = next_link
                    nodes.append({
                        "id": next_node,
                        "type": "LoadImage",
                        "pos": [50 + (k - 2) * 440, 1100],
                        "size": [420, 360],
                        "flags": {},
                        "order": 0,
                        "mode": 4,
                        "inputs": [],
                        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [next_link]}, {"name": "MASK", "type": "MASK", "links": None}],
                        "title": f"参考图 {k} (选中后按 Ctrl+B 启用)",
                        "properties": {"Node name for S&R": "LoadImage"},
                        "widgets_values": ["easycolab_sample_end.png", "image"],
                        "widgets_values_named": {"image": "easycolab_sample_end.png", "upload": "image"},
                    })
                    links.append([next_link, next_node, 0, cond["id"], slot, "IMAGE"])
                    next_node += 1
                    next_link += 1

                if free_canvas:
                    # 画布改由比例选择器决定，不再取参考图 1 的尺寸
                    size_links = {}
                    for key in ("width", "height"):
                        slot = next(i for i, inp in enumerate(cond["inputs"]) if inp.get("name") == key)
                        size_links[key] = next(lk for lk in links if lk[3] == cond["id"] and lk[4] == slot)
                    old_src = next(n for n in nodes if n["id"] == size_links["width"][1])
                    for out in old_src.get("outputs", []):
                        if isinstance(out.get("links"), list):
                            out["links"] = [lid for lid in out["links"] if lid not in (size_links["width"][0], size_links["height"][0])]
                    nodes.append({
                        "id": next_node,
                        "type": "ResolutionSelector",
                        "pos": [870, 800],
                        "size": [270, 150],
                        "flags": {},
                        "order": 0,
                        "mode": 0,
                        "showAdvanced": True,
                        "inputs": [],
                        "outputs": [
                            {"name": "width", "type": "INT", "links": [size_links["width"][0]]},
                            {"name": "height", "type": "INT", "links": [size_links["height"][0]]},
                        ],
                        "title": "画幅与像素量 (4 ≈ 400 万像素)",
                        "properties": {"Node name for S&R": "ResolutionSelector"},
                        "widgets_values": ["16:9 (Widescreen)", 4, 32],
                        "widgets_values_named": {"aspect_ratio": "16:9 (Widescreen)", "megapixels": 4, "multiple": 32},
                    })
                    size_links["width"][1], size_links["width"][2] = next_node, 0
                    size_links["height"][1], size_links["height"][2] = next_node, 1
                    next_node += 1

                nodes.append(_make_zh_guide_note(
                    next_node,
                    [-1050, -60],
                    "【EasyColab 三步出图指南 · MiniMax-H3-Image "
                    + ("多参考图合成新画面】" if free_canvas else "参考图改图】"),
                    (
                        "## 给它几张参考图，生成一张全新的画面\n\n"
                        "1. **传参考图**：在 **「① 参考图 1」** 上传角色或物品的图；还有的话，选中下面的 **「参考图 2/3/4」** 按 `Ctrl+B` 启用再上传。\n"
                        "2. **写画面**：在 **「② 描述要生成的新画面」** 里写场景，用 `<Picture 1>`、`<Picture 2>` 指代各张参考图里的人和物。\n"
                        "3. **点运行**：画布大小由 **「画幅与像素量」** 决定，默认 16:9。\n\n"
                        "适合用同一套角色定妆图出多张不同场景的图。\n\n"
                        if free_canvas else
                        "## 改一张图，其余保持不变\n\n"
                        "1. **传要改的图**：在 **「① 要修改的图」** 上传，输出尺寸跟它一样。\n"
                        "2. **写指令**：在 **「② 写修改指令」** 里说要改什么，用 `<Picture 1>` 指代这张图。\n"
                        "3. **点运行**：最右边可以拖动对比改前改后。\n\n"
                        "要参考别的图（比如把某件衣服换上去），选中下面的 **「参考图 2/3/4」** 按 `Ctrl+B` 启用再上传，指令里用 `<Picture 2>` 指代。\n\n"
                    )
                    + "**enable_PE**：打开后会先把你的指令自动扩写再出图；自己写清楚了就保持关闭。\n\n"
                    "**更多参考图**：把新的「加载图像」节点连到 `image_5` 之后的口上，模型最多认 16 张。",
                ))
                d["nodes"] = nodes
                d["last_node_id"] = next_node
                d["last_link_id"] = next_link - 1
                _prune_workflow_dangling_links(d)
                return d

            for wf_name, wf_data in (
                ("07_MiniMax-H3-Image_文生图(4MP高清).json", d_t2i),
                ("08_MiniMax-H3-Image_参考图改图(按原图尺寸).json", _customize_h3_image_edit(free_canvas=False)),
                ("09_MiniMax-H3-Image_多参考图合成新画面(16比9).json", _customize_h3_image_edit(free_canvas=True)),
            ):
                out_h3i = os.path.join(wf_dir, wf_name)
                with open(out_h3i, "w", encoding="utf-8") as wf:
                    json.dump(wf_data, wf, ensure_ascii=False, indent=2)
                installed_files.append(wf_name)
        except Exception as e:
            print(f"[ComfyUI 工作流生成] MiniMax-H3-Image 模板处理异常: {e}")

    # 3. LTX-2.3 22B Distilled (首尾帧电影运镜 + 音轨)
    if "ltx23_flf2v" in target_set:
        try:
            d_ltx = _fetch_official_comfy_template("video_ltx2_3_flf2v.json")
            ltx_nodes = []
            for n in d_ltx.get("nodes", []):
                nid = n.get("id")
                ntype = n.get("type")
                if ntype == "MarkdownNote":
                    continue
                if nid == 31 and ntype == "LoadImage":
                    n["title"] = "① 点击上传起始首帧图片 (First Frame)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 39 and ntype == "LoadImage":
                    n["title"] = "② 点击上传结束尾帧图片 (Last Frame)"
                    n["widgets_values"] = ["easycolab_sample_end.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 129:
                    n["title"] = "③ 首尾帧电影运镜与音轨参数 (LTX-2.3 22B Distilled)"
                elif nid == 68 and ntype == "SaveVideo":
                    n["title"] = "④ 电影运镜成片预览与自动保存"
                    n["widgets_values"] = ["video/LTX2.3_首尾帧运镜", "auto", "auto"]
                    n.pop("widgets_values_named", None)
                ltx_nodes.append(n)

            ltx_nodes.append(_make_zh_guide_note(
                504,
                [-490, 2250],
                "【EasyColab 三步傻瓜出片指南 · LTX-2.3 首尾帧运镜】",
                "## 电影级大幅度推拉摇移运镜（自带音效配乐）\n\n"
                "1. **传首尾帧**：在右侧 **「① 起始首帧」** 与 **「② 结束尾帧」** 上传两张不同机位或状态的画面。\n"
                "2. **写运镜轨迹**：在中间 **「③ 首尾帧电影运镜与音轨参数」** 文本框描述镜头怎么移动（建议英文描述运镜 + `Music:` 描述音乐）。\n"
                "3. **点运行**：采用 Distilled 极速蒸馏权重，一键直出高帧率带音轨视频！",
            ))
            d_ltx["nodes"] = ltx_nodes
            d_ltx["last_node_id"] = max(int(d_ltx.get("last_node_id") or 200), 506)
            d_ltx.setdefault("extra", {})["ds"] = {"scale": 0.66, "offset": [560, -2160]}
            _prune_workflow_dangling_links(d_ltx)
            out_5 = os.path.join(wf_dir, "05_LTX-2.3_首尾帧电影运镜(双图极速带音效).json")
            with open(out_5, "w", encoding="utf-8") as wf:
                json.dump(d_ltx, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_5))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] LTX-2.3 FLF2V 模板处理异常: {e}")

    # 4. 混元 HunyuanVideo 1.5 (单图 720P -> 1080P 自动潜空间超分)
    if "hunyuan15_i2v_1080p" in target_set:
        try:
            d_hy = _fetch_official_comfy_template("video_hunyuan_video_1.5_720p_i2v.json")
            sr_enable_ids = {108, 111, 109, 113, 114, 117, 118, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141}
            drop_ids = {104, 120, 112, 121, 122, 123, 124, 136}
            hy_nodes = []
            for n in d_hy.get("nodes", []):
                nid = n.get("id")
                if nid in drop_ids:
                    continue
                if nid in sr_enable_ids:
                    n["mode"] = 0
                if nid == 80 and n.get("type") == "LoadImage":
                    n["title"] = "① 点击上传起始首帧图片 (720P → 1080P 超分)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 44 and n.get("type") == "CLIPTextEncode":
                    n["title"] = "② 输入画面动作与光影描述 (中英文皆可)"
                    n["widgets_values"] = [
                        "清晨阳光洒在壮阔的海岸山脉上，镜头平稳向前推进，海面波光粼粼，云雾在山谷间缓慢流动，极致写实电影摄影质感。"
                    ]
                    n.pop("widgets_values_named", None)
                elif nid in (127, 131) and n.get("type") == "RandomNoise":
                    wv = list(n.get("widgets_values") or [42, "randomize"])
                    if len(wv) >= 2:
                        wv[1] = "randomize"
                    n["widgets_values"] = wv
                    n.pop("widgets_values_named", None)
                elif nid == 102 and n.get("type") == "SaveVideo":
                    n["title"] = "③ 720P 基础成片预览"
                elif nid == 118 and n.get("type") == "SaveVideo":
                    n["title"] = "④ 1920×1080 超高清成片预览与保存"
                hy_nodes.append(n)

            hy_nodes.append(_make_zh_guide_note(
                506,
                [-1120, 480],
                "【EasyColab 三步傻瓜出片指南 · 混元 1.5 直出 1080P】",
                "## 单图直出 1920×1080 超高清视频（超分管线已激活）\n\n"
                "1. **传图片**：在右侧 **「① 点击上传起始首帧图片」** 选择你的原图。\n"
                "2. **改提示词**：在上方 **「② 输入画面动作与光影描述」** 输入动态描述。\n"
                "3. **点运行**：无需按 `Ctrl+B`，工作流会先生成 720P 动态原片，再自动接力潜空间超分输出 **1920×1080** 超清成片！",
            ))
            d_hy["nodes"] = hy_nodes
            d_hy["links"] = [
                lk for lk in d_hy.get("links", [])
                if lk[1] not in drop_ids and lk[3] not in drop_ids
            ]
            for g in d_hy.get("groups", []):
                if g.get("id") == 3:
                    g["title"] = "第一步：上传起始图片"
                elif g.get("id") == 2:
                    g["title"] = "第二步：输入画面动作描述"
                elif g.get("id") == 8:
                    g["title"] = "1080P 超高清潜空间放大管线 (已默认激活)"
            d_hy["last_node_id"] = max(int(d_hy.get("last_node_id") or 200), 508)
            d_hy.setdefault("extra", {})["ds"] = {"scale": 0.56, "offset": [1180, 120]}
            _prune_workflow_dangling_links(d_hy)
            out_6 = os.path.join(wf_dir, "06_Hunyuan1.5_单图直出1080P超清视频.json")
            with open(out_6, "w", encoding="utf-8") as wf:
                json.dump(d_hy, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_6))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] Hunyuan 1.5 模板处理异常: {e}")

    # 10–25. H3 Director 导演台：插件自带的 9 套示例工作流（插件不在时先拉取）+ 内嵌的资源包工作流
    for pk_id, pk_tag in (("minimax_h3_director", "core"), ("minimax_h3_director_plus", "plus")):
        if pk_id not in target_set:
            continue
        for dst_name, (pk, blob) in H3_DIRECTOR_BUNDLED_WORKFLOWS.items():
            if pk != pk_tag:
                continue
            try:
                import zlib as _zlib
                with open(os.path.join(wf_dir, dst_name), "wb") as wf:
                    wf.write(_zlib.decompress(base64.b64decode(blob)))
                installed_files.append(dst_name)
            except Exception as e:
                print(f"[ComfyUI 工作流生成] 导演台内嵌工作流 {dst_name} 写入异常: {e}")
    def _h3_director_example_dir() -> str:
        comfy_root_d = _resolve_comfyui_persistence_paths()[0]
        node_dir = os.path.join(comfy_root_d, "custom_nodes", H3_DIRECTOR_NODE_DIR)
        if not os.path.isdir(os.path.join(node_dir, "example_workflows")):
            # ComfyUI 本体可能还没装：只在临时目录里拉一份读示例，绝不提前创建 /content/ComfyUI（否则之后克隆 ComfyUI 会失败）
            node_dir = "/tmp/easycolab_h3_director_src"
            if not os.path.isdir(os.path.join(node_dir, "example_workflows")):
                shutil.rmtree(node_dir, ignore_errors=True)
                subprocess.run(["git", "clone", "--depth", "1", H3_DIRECTOR_REPO, node_dir], check=False)
        return node_dir

    if "minimax_h3_director" in target_set:
        try:
            node_dir = _h3_director_example_dir()
            for src_name, dst_name in H3_DIRECTOR_EXAMPLE_WORKFLOWS:
                src_p = os.path.join(node_dir, "example_workflows", src_name)
                if not os.path.isfile(src_p):
                    print(f"[ComfyUI 工作流生成] H3 导演台示例缺失: {src_name}")
                    continue
                txt = open(src_p, "r", encoding="utf-8").read()
                for old_name, new_name in H3_DIRECTOR_MODEL_RENAMES.items():
                    txt = txt.replace(old_name, new_name)
                with open(os.path.join(wf_dir, dst_name), "w", encoding="utf-8") as wf:
                    wf.write(txt)
                installed_files.append(dst_name)
        except Exception as e:
            print(f"[ComfyUI 工作流生成] H3 导演台示例处理异常: {e}")

    # 导演台满血版：4 套 r2v 工作流另存一份，主干 / 编码器默认换成 bf16
    if "minimax_h3_director_bf16" in target_set:
        import zlib as _zlib
        ex_src = {dst: src for src, dst in H3_DIRECTOR_EXAMPLE_WORKFLOWS}
        for base_name, bf16_name in H3_DIRECTOR_BF16_R2V_WORKFLOWS.items():
            try:
                if base_name in H3_DIRECTOR_BUNDLED_WORKFLOWS:
                    txt = _zlib.decompress(base64.b64decode(H3_DIRECTOR_BUNDLED_WORKFLOWS[base_name][1])).decode("utf-8")
                else:
                    src_p = os.path.join(_h3_director_example_dir(), "example_workflows", ex_src[base_name])
                    txt = open(src_p, "r", encoding="utf-8").read()
                for old_name, new_name in H3_DIRECTOR_BF16_RENAMES.items():
                    txt = txt.replace(old_name, new_name)
                with open(os.path.join(wf_dir, bf16_name), "w", encoding="utf-8") as wf:
                    wf.write(txt)
                installed_files.append(bf16_name)
            except Exception as e:
                print(f"[ComfyUI 工作流生成] 导演台满血版工作流 {bf16_name} 写入异常: {e}")

    if installed_files:
        try:
            with open(os.path.join(wf_dir, ".easycolab_wf_schema_v3"), "w", encoding="utf-8") as sf:
                sf.write("v2")
            ptr_path = os.path.join(persist_root, "user", "default", "easycolab_active_preset.json")
            if (pack_ids and len(pack_ids) == 1) or not os.path.isfile(ptr_path):
                with open(ptr_path, "w", encoding="utf-8") as pf:
                    json.dump({"workflow_filename": installed_files[0]}, pf, ensure_ascii=False, indent=2)
        except Exception:
            pass

    _COMFY_STATUS_CACHE["ts"] = 0.0
    return installed_files


def _get_comfy_preset_packs_status(persist_root: str) -> List[Dict[str, Any]]:
    wf_dir = os.path.join(persist_root, "user", "default", "workflows")
    results: List[Dict[str, Any]] = []
    free_gb = round(_disk_free_gb(persist_root), 1)
    for pack in COMFY_PRESET_WORKFLOW_PACKS:
        models = _comfy_pack_all_models(pack)
        installed_m = 0
        for sub, fname, _, _, _ in models:
            p = os.path.join(persist_root, "models", sub, fname)
            if os.path.isfile(p) and os.path.getsize(p) > 1024 * 1024:
                installed_m += 1
        wfs = pack["workflows"]
        installed_wf = sum(1 for w in wfs if os.path.isfile(os.path.join(wf_dir, w)))
        ready = (installed_m == len(models)) and (installed_wf == len(wfs))
        results.append({
            "id": pack["id"],
            "title": pack["title"],
            "badge": pack["badge"],
            "scenario": pack["scenario"],
            "approx_size_gb": round(pack["approx_size_gb"] + sum(m[4] for m in models[len(pack["models"]):]), 2),
            "workflows": wfs,
            "missing_gb": round(sum(m[4] for m in _comfy_missing_models(persist_root, models)), 1),
            "disk_free_gb": free_gb,
            "installed_models": installed_m,
            "total_models": len(models),
            "workflows_installed": installed_wf == len(wfs),
            "ready": ready,
        })
    return results


def _resolve_comfyui_persistence_paths() -> Tuple[str, str, str, str, bool]:
    comfy_root = "/content/ComfyUI" if os.path.exists("/content") else os.path.join(os.getcwd(), "ComfyUI")
    if _native_drive_mounted():
        persist_root = "/content/drive/MyDrive/EasyColab_ComfyUI"
        return comfy_root, persist_root, "drive_symlink", "Google Drive 实时软链直连", True
    if _has_rclone_drive_remote():
        persist_root = "/content/drive_mirror/EasyColab_ComfyUI"
        return comfy_root, persist_root, "rclone_drive_sync", "Google Drive 自动软链持久化", True
    persist_root = (
        "/content/EasyColab_ComfyUI_Data"
        if os.path.exists("/content")
        else os.path.join(os.getcwd(), "EasyColab_ComfyUI_Data")
    )
    return comfy_root, persist_root, "local_only", "实例本地高速存储", False


def _repair_dangling_dir_link(path: str):
    # 之前的版本会把 models/ultralytics/bbox 这类子目录链成指向自己的死循环链接，导致 makedirs 报 File exists
    if os.path.islink(path) and not os.path.isdir(path):
        os.unlink(path)


def _safe_symlink_dir(target_dir: str, link_path: str):
    _repair_dangling_dir_link(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    # 上级目录已经是指向持久化目录的链接时，link_path 本身就是 target_dir，不能再删掉重链（会删光里面的文件并链成自环）
    if os.path.realpath(link_path) == os.path.realpath(target_dir):
        return
    if os.path.islink(link_path):
        cur_target = os.readlink(link_path)
        if os.path.abspath(cur_target) == os.path.abspath(target_dir):
            return
        os.unlink(link_path)
    elif os.path.isdir(link_path):
        for entry in os.listdir(link_path):
            src_item = os.path.join(link_path, entry)
            dst_item = os.path.join(target_dir, entry)
            if not os.path.exists(dst_item):
                try:
                    shutil.move(src_item, dst_item)
                except Exception:
                    pass
        shutil.rmtree(link_path, ignore_errors=True)
    elif os.path.exists(link_path):
        os.remove(link_path)
    os.symlink(target_dir, link_path)


def _scan_comfyui_counts_and_outputs() -> Tuple[Tuple[int, int, int], List[Dict[str, Any]]]:
    now = time.time()
    if (now - _COMFY_STATUS_CACHE["ts"]) < 6.0:
        return _COMFY_STATUS_CACHE["counts"], list(_COMFY_STATUS_CACHE["outputs"])

    comfy_root, persist_root, _, _, _ = _resolve_comfyui_persistence_paths()
    cn_count = 0
    cn_base = os.path.join(comfy_root, "custom_nodes")
    if not os.path.isdir(cn_base):
        cn_base = os.path.join(persist_root, "custom_nodes")
    if os.path.isdir(cn_base):
        try:
            for d in os.listdir(cn_base):
                if d.startswith(".") or d == "__pycache__":
                    continue
                if os.path.isdir(os.path.join(cn_base, d)):
                    cn_count += 1
        except Exception:
            pass

    model_exts = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}
    m_count = 0
    m_base = os.path.join(persist_root, "models")
    if os.path.isdir(m_base):
        try:
            for root_d, _, fnames in os.walk(m_base):
                for fn in fnames:
                    if os.path.splitext(fn)[1].lower() in model_exts:
                        m_count += 1
        except Exception:
            pass

    wf_count = 0
    wf_dir = os.path.join(persist_root, "user", "default", "workflows")
    if os.path.isdir(wf_dir):
        try:
            wf_count = sum(1 for fn in os.listdir(wf_dir) if fn.lower().endswith(".json"))
        except Exception:
            pass

    recent_outputs: List[Dict[str, Any]] = []
    out_dir = os.path.join(comfy_root, "output")
    if not os.path.isdir(out_dir):
        out_dir = os.path.join(persist_root, "output")
    valid_media = {".mp4", ".webm", ".mov", ".webp", ".gif", ".png", ".jpg", ".jpeg"}
    if os.path.isdir(out_dir):
        collected = []
        try:
            for root_d, _, fnames in os.walk(out_dir):
                for fn in fnames:
                    if fn.startswith("."):
                        continue
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in valid_media:
                        continue
                    full_p = os.path.join(root_d, fn)
                    try:
                        st = os.stat(full_p)
                        if st.st_size < 512:
                            continue
                        rel_p = os.path.relpath(full_p, out_dir).replace("\\", "/")
                        mtype = "video" if ext in {".mp4", ".webm", ".mov"} else "image"
                        collected.append((st.st_mtime, {
                            "rel_path": rel_p,
                            "filename": fn,
                            "media_type": mtype,
                            "size_mb": round(st.st_size / (1024 ** 2), 2),
                            "mtime": int(st.st_mtime),
                        }))
                    except Exception:
                        pass
            collected.sort(key=lambda x: x[0], reverse=True)
            recent_outputs = [item for _, item in collected[:16]]
        except Exception:
            pass

    _COMFY_STATUS_CACHE["ts"] = now
    _COMFY_STATUS_CACHE["counts"] = (cn_count, m_count, wf_count)
    _COMFY_STATUS_CACHE["outputs"] = recent_outputs
    return (cn_count, m_count, wf_count), recent_outputs


def _try_adopt_running_comfyui():
    """若 app.py 热更新重启前实例上已有在线 ComfyUI 与 cloudflared 隧道，自动无缝接管，不断开用户已打开的画布"""
    global _COMFY_ADOPT_CHECKED, _COMFY_BG_SYNC_STARTED
    if _COMFY_ADOPT_CHECKED or COMFYUI_STATE.get("is_running") or COMFYUI_STATE.get("is_starting"):
        return
    _COMFY_ADOPT_CHECKED = True

    port = int(COMFYUI_STATE.get("local_port") or COMFY_BACKEND)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/system_stats", timeout=1.0) as resp:
            if resp.status != 200:
                return
    except Exception:
        return

    pub_url = ""
    tunnel_state = _detached_proc_state(f"cloudflared.*{COMFY_INGRESS}")
    if tunnel_state == "pipe":
        # 旧版本起的隧道：输出接在已经死掉的 app.py 管道上，迟早会退出，现在就换一条新的
        subprocess.run(["pkill", "-f", f"cloudflared.*{COMFY_INGRESS}"], check=False)
        tunnel_state = "none"
    if tunnel_state == "ok" and os.path.isfile(_COMFY_STATE_FILE):
        try:
            st_obj = json.load(open(_COMFY_STATE_FILE, "r", encoding="utf-8"))
            pub_url = str(st_obj.get("public_url") or "").strip()
        except Exception:
            pub_url = ""

    if not pub_url and tunnel_state == "ok":
        cf_re = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
        for m_port in range(20241, 20247):
            for ep in ("/quicktunnel", "/metrics"):
                try:
                    with opener.open(f"http://127.0.0.1:{m_port}{ep}", timeout=0.6) as mr:
                        body = mr.read().decode("utf-8", errors="ignore")
                    m = cf_re.search(body)
                    if m:
                        pub_url = m.group(0)
                        break
                    m_host = re.search(r'"hostname"\s*:\s*"([a-zA-Z0-9-]+\.trycloudflare\.com)"', body)
                    if m_host:
                        pub_url = "https://" + m_host.group(1)
                        break
                except Exception:
                    pass
            if pub_url:
                break

    if not pub_url:
        try:
            subprocess.run(["pkill", "-f", f"cloudflared.*{COMFY_INGRESS}"], check=False)
            tunnel_proc, pub_url = _start_comfy_tunnel(20.0)
            COMFYUI_STATE["tunnel_proc"] = tunnel_proc
        except Exception:
            pass

    comfy_root, persist_root, _, label, _ = _resolve_comfyui_persistence_paths()
    _ensure_comfyui_beginner_assets(comfy_root, persist_root)
    COMFYUI_STATE["is_running"] = True
    COMFYUI_STATE["public_url"] = pub_url
    COMFYUI_STATE["progress"] = 1.0
    COMFYUI_STATE["status_message"] = f"已无缝接管在线 ComfyUI 画布（{label}）"
    if _detached_proc_state("ComfyUI/main.py") == "pipe":
        # 旧版本起的 ComfyUI 输出接在已死的管道上，跑任务时打印进度可能报 Broken pipe；没有任务时点一次「重启」换成新的启动方式
        COMFYUI_STATE["status_message"] += "；这个 ComfyUI 是旧版本服务端启动的，跑任务可能报 Broken pipe，请在没有任务时点一次「重启」"
    if pub_url:
        try:
            with open(_COMFY_STATE_FILE, "w", encoding="utf-8") as sf:
                json.dump({"public_url": pub_url, "port": port}, sf)
        except Exception:
            pass
    if not _COMFY_BG_SYNC_STARTED:
        _COMFY_BG_SYNC_STARTED = True
        threading.Thread(target=_comfyui_periodic_drive_sync_loop, daemon=True).start()
    print(f"[ComfyUI] 已自动接管实例上正在运行的 ComfyUI 服务: {pub_url or f'127.0.0.1:{port}'}")


def _poll_local_comfyui_runtime_quiet():
    proc = COMFYUI_STATE.get("comfy_proc")
    if proc is not None and proc.poll() is not None:
        COMFYUI_STATE["is_running"] = False
        COMFYUI_STATE["queue_running"] = 0
        COMFYUI_STATE["queue_pending"] = 0
        if not COMFYUI_STATE["is_starting"]:
            COMFYUI_STATE["status_message"] = f"ComfyUI 进程已退出 (code={proc.returncode})，可点击重新启动"
        COMFYUI_STATE["comfy_proc"] = None
        return

    if not COMFYUI_STATE.get("is_running"):
        _try_adopt_running_comfyui()
        if not COMFYUI_STATE.get("is_running"):
            return

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    port = int(COMFYUI_STATE.get("local_port") or COMFY_BACKEND)
    try:
        with opener.open(f"http://127.0.0.1:{port}/queue", timeout=1.2) as r:
            q_data = json.loads(r.read().decode("utf-8", errors="replace"))
        q_run = len(q_data.get("queue_running") or [])
        q_pend = len(q_data.get("queue_pending") or [])
        COMFYUI_STATE["queue_running"] = q_run
        COMFYUI_STATE["queue_pending"] = q_pend
        if q_run > 0 or q_pend > 0:
            touch_activity()
    except Exception:
        pass

    try:
        with opener.open(f"http://127.0.0.1:{port}/history?max_items=5", timeout=1.2) as r:
            h_data = json.loads(r.read().decode("utf-8", errors="replace"))
        if isinstance(h_data, dict):
            h_sig = hash(tuple(sorted(h_data.keys())))
            if h_sig != COMFYUI_STATE.get("last_history_count", 0) and COMFYUI_STATE.get("last_history_count", 0) != 0:
                touch_activity()
                _COMFY_STATUS_CACHE["ts"] = 0.0
            COMFYUI_STATE["last_history_count"] = h_sig
    except Exception:
        pass


def get_comfyui_status_dict() -> Dict[str, Any]:
    _, persist_root, mode, label, is_drive = _resolve_comfyui_persistence_paths()
    (cn_count, m_count, wf_count), recent_outputs = _scan_comfyui_counts_and_outputs()
    preset_packs = _get_comfy_preset_packs_status(persist_root)
    COMFYUI_STATE["persistence_mode"] = mode
    COMFYUI_STATE["persistence_label"] = label
    COMFYUI_STATE["drive_persisted"] = is_drive
    COMFYUI_STATE["custom_nodes_count"] = cn_count
    COMFYUI_STATE["models_count"] = m_count
    COMFYUI_STATE["workflows_count"] = wf_count
    return {
        "is_running": bool(COMFYUI_STATE["is_running"]),
        "is_starting": bool(COMFYUI_STATE["is_starting"]),
        "is_syncing_drive": bool(COMFYUI_STATE["is_syncing_drive"]),
        "is_downloading_weight": bool(COMFYUI_STATE["is_downloading_weight"]),
        "installing_pack_id": str(COMFYUI_STATE.get("installing_pack_id") or ""),
        "progress": float(COMFYUI_STATE["progress"]),
        "status_message": str(COMFYUI_STATE["status_message"]),
        "public_url": str(COMFYUI_STATE["public_url"] or ""),
        "local_port": int(COMFYUI_STATE["local_port"]),
        "queue_running": int(COMFYUI_STATE["queue_running"]),
        "queue_pending": int(COMFYUI_STATE["queue_pending"]),
        "persistence_mode": mode,
        "persistence_label": label,
        "drive_persisted": is_drive,
        "persist_root": persist_root,
        "custom_nodes_count": cn_count,
        "models_count": m_count,
        "workflows_count": wf_count,
        "last_sync_ts": int(COMFYUI_STATE["last_sync_ts"]),
        "recent_outputs": recent_outputs,
        "preset_packs": preset_packs,
    }


def free_comfyui_vram_core() -> Tuple[bool, str]:
    if not COMFYUI_STATE.get("is_running"):
        return False, "ComfyUI 当前未运行"
    port = int(COMFYUI_STATE.get("local_port") or COMFY_BACKEND)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/free",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(req, timeout=8.0) as _:
            pass
        _reclaim_transient_gpu_memory()
        msg = "已释放 ComfyUI 驻留模型与显存缓存（画布保持在线）"
        COMFYUI_STATE["status_message"] = msg
        print(f"[ComfyUI] {msg}")
        return True, msg
    except Exception as e:
        return False, f"释放 ComfyUI 显存失败: {e}"


def sync_comfyui_to_drive_core(quiet: bool = False) -> Tuple[bool, str]:
    if COMFYUI_STATE["is_syncing_drive"]:
        return True, "正在后台同步 ComfyUI 数据到 Google Drive..."
    comfy_root, persist_root, mode, label, is_drive = _resolve_comfyui_persistence_paths()
    if not os.path.isdir(comfy_root) and not os.path.isdir(persist_root):
        return False, "ComfyUI 尚未初始化"

    COMFYUI_STATE["is_syncing_drive"] = True
    try:
        os.makedirs(os.path.join(persist_root, "models"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "custom_nodes"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "user"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "output"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "input"), exist_ok=True)

        # 1. 自动接管自定义节点在 models/ 下新建的非软链目录，迁移至持久化目录并建立软链
        models_dir = os.path.join(comfy_root, "models")
        if os.path.isdir(models_dir):
            for entry in os.listdir(models_dir):
                if entry in ("configs", "__pycache__") or entry.startswith("."):
                    continue
                local_sub = os.path.join(models_dir, entry)
                if os.path.isdir(local_sub) and not os.path.islink(local_sub):
                    persist_sub = os.path.join(persist_root, "models", entry)
                    _safe_symlink_dir(persist_sub, local_sub)

        # 2. 将本地高速 NVMe 上的 custom_nodes 增量镜像同步到持久化目录，并记录 Git 源清单
        cn_local = os.path.join(comfy_root, "custom_nodes")
        cn_persist = os.path.join(persist_root, "custom_nodes")
        manifest: Dict[str, Any] = {}
        manifest_path = os.path.join(persist_root, "custom_nodes_manifest.json")
        if os.path.exists(manifest_path):
            try:
                manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
            except Exception:
                manifest = {}

        if os.path.isdir(cn_local):
            for entry in sorted(os.listdir(cn_local)):
                if entry.startswith(".") or entry == "__pycache__":
                    continue
                src_node = os.path.join(cn_local, entry)
                if not os.path.isdir(src_node):
                    continue
                dst_node = os.path.join(cn_persist, entry)
                if shutil.which("rsync"):
                    subprocess.run(
                        ["rsync", "-a", "--delete", "--exclude=__pycache__", f"{src_node}/", f"{dst_node}/"],
                        check=False,
                    )
                else:
                    shutil.copytree(
                        src_node,
                        dst_node,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    )
                git_cfg = os.path.join(src_node, ".git", "config")
                if os.path.exists(git_cfg):
                    try:
                        cfg_txt = open(git_cfg, "r", encoding="utf-8", errors="ignore").read()
                        m_url = re.search(r"url\s*=\s*(\S+)", cfg_txt)
                        if m_url:
                            manifest[entry] = {"git_url": m_url.group(1).strip()}
                    except Exception:
                        pass

            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, ensure_ascii=False, indent=2)

        # 3. 若处于 rclone 云盘镜像模式，通过 Google Cloud 内网 8 并发同步至 Google Drive
        if mode == "rclone_drive_sync" and _has_rclone_drive_remote():
            if not quiet:
                print("[ComfyUI 云盘同步] 正在通过高速通道将自定义节点、模型与工作流同步至 Google Drive...")
            subprocess.run(
                [
                    "rclone", "copy",
                    persist_root,
                    "easycolab_gdrive:EasyColab_Drive/EasyColab_ComfyUI",
                    "--transfers", "8",
                    "--checkers", "16",
                    "--drive-chunk-size", "64M",
                    "--exclude", "**/__pycache__/**",
                    "--exclude", "**/*.part",
                    "--exclude", "**/*.tmp",
                ],
                check=False,
            )

        COMFYUI_STATE["last_sync_ts"] = time.time()
        _COMFY_STATUS_CACHE["ts"] = 0.0
        (cn_count, m_count, wf_count), _ = _scan_comfyui_counts_and_outputs()
        msg = f"已持久化至 {label}（节点 {cn_count} 个 · 权重 {m_count} 个 · 工作流 {wf_count} 个）"
        if not quiet:
            COMFYUI_STATE["status_message"] = msg
            print(f"[ComfyUI 云盘持久化] {msg}")
        return True, msg
    except Exception as e:
        err = f"ComfyUI 云盘同步异常: {e}"
        if not quiet:
            print(f"[ComfyUI] {err}")
        return False, err
    finally:
        COMFYUI_STATE["is_syncing_drive"] = False


def _comfyui_periodic_drive_sync_loop():
    while True:
        time.sleep(60.0)
        if COMFYUI_STATE.get("is_running") and not COMFYUI_STATE.get("is_starting"):
            if COMFYUI_STATE.get("queue_running", 0) == 0:
                sync_comfyui_to_drive_core(quiet=True)


def _ensure_comfyui_manager_tunnel_config(comfy_root: str, persist_root: str):
    """允许通过 trycloudflare 公网隧道在 ComfyUI-Manager 内直接安装 Git 节点与模型，不受默认 normal 安全锁拦截"""
    cfg_targets = [
        os.path.join(persist_root, "user", "default", "ComfyUI-Manager", "config.ini"),
        os.path.join(persist_root, "user", "__manager", "config.ini"),
        os.path.join(comfy_root, "custom_nodes", "ComfyUI-Manager", "config.ini"),
    ]
    for cfg_p in cfg_targets:
        try:
            os.makedirs(os.path.dirname(cfg_p), exist_ok=True)
            if os.path.exists(cfg_p):
                txt = open(cfg_p, "r", encoding="utf-8", errors="ignore").read()
                if "security_level" in txt:
                    new_txt = re.sub(r"security_level\s*=\s*\S+", "security_level = weak", txt)
                    if new_txt != txt:
                        with open(cfg_p, "w", encoding="utf-8") as wf:
                            wf.write(new_txt)
                    continue
            with open(cfg_p, "w", encoding="utf-8") as wf:
                wf.write("[default]\nsecurity_level = weak\npreview_method = auto\n")
        except Exception:
            pass


def _install_persisted_custom_node_requirements(comfy_root: str):
    """扫描已从 Google Drive 恢复的所有自定义节点 requirements.txt，过滤掉 torch/xformers 后批量补齐依赖"""
    import hashlib
    cn_dir = os.path.join(comfy_root, "custom_nodes")
    if not os.path.isdir(cn_dir):
        return
    collected_reqs: List[str] = []
    for entry in sorted(os.listdir(cn_dir)):
        req_file = os.path.join(cn_dir, entry, "requirements.txt")
        if not os.path.isfile(req_file):
            continue
        try:
            for raw_line in open(req_file, "r", encoding="utf-8", errors="ignore"):
                line = raw_line.split("#", 1)[0].strip()
                if not line or line.startswith("-"):
                    continue
                if _COMFY_PROTECTED_PIP_PKGS.match(line):
                    continue
                if line not in collected_reqs:
                    collected_reqs.append(line)
        except Exception:
            pass

    if not collected_reqs:
        return
    digest = hashlib.sha256("\n".join(collected_reqs).encode("utf-8")).hexdigest()
    stamp_file = "/tmp/.easycolab_comfy_node_reqs.sha256"
    if os.path.exists(stamp_file):
        try:
            if open(stamp_file, "r").read().strip() == digest:
                return
        except Exception:
            pass

    print(f"[ComfyUI] 正在为已持久化的自定义节点自动恢复 {len(collected_reqs)} 项 Python 依赖...")
    tmp_req = "/tmp/easycolab_comfy_merged_reqs.txt"
    with open(tmp_req, "w", encoding="utf-8") as wf:
        wf.write("\n".join(collected_reqs) + "\n")
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q", "-r", tmp_req])
    if rc == 0:
        with open(stamp_file, "w") as sf:
            sf.write(digest)


def _install_h3_director_optional_extras(comfy_root: str):
    """H3 导演台的可选加速依赖：SageAttention（加速版工作流）与 NVIDIA RTX VSR（Refine 放大）。装不上不影响其它功能"""
    if not os.path.isdir(os.path.join(comfy_root, "custom_nodes", H3_DIRECTOR_NODE_DIR)):
        return
    stamp_file = "/tmp/.easycolab_h3_director_extras"
    if os.path.exists(stamp_file):
        return
    print("[ComfyUI] 正在为 H3 导演台安装可选加速依赖 (SageAttention / RTX VSR)...")
    for args in (
        ["sageattention"],
        ["nvidia-vfx", "--extra-index-url", "https://pypi.nvidia.com"],
    ):
        rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q"] + args)
        if rc != 0:
            print(f"[ComfyUI] 可选依赖 {args[0]} 安装失败，导演台对应的加速选项不可用，其余功能不受影响")
    with open(stamp_file, "w") as sf:
        sf.write("1")


def _ensure_cloudflared_binary() -> str:
    cf_bin = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
    if os.path.isfile(cf_bin) and os.access(cf_bin, os.X_OK):
        return cf_bin
    dl_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    os.makedirs(os.path.dirname(cf_bin), exist_ok=True)
    urllib.request.urlretrieve(dl_url, cf_bin)
    os.chmod(cf_bin, 0o755)
    return cf_bin


def stop_comfyui_core(sync_before_stop: bool = True) -> Tuple[bool, str]:
    with COMFYUI_LOCK:
        if sync_before_stop and (COMFYUI_STATE.get("is_running") or os.path.isdir("/content/ComfyUI")):
            sync_comfyui_to_drive_core(quiet=True)
        for key in ("tunnel_proc", "comfy_proc"):
            proc = COMFYUI_STATE.get(key)
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                COMFYUI_STATE[key] = None
        subprocess.run(["pkill", "-f", "ComfyUI/main.py"], check=False)
        subprocess.run(["pkill", "-f", "cloudflared.*8188"], check=False)
        if os.path.exists(_COMFY_STATE_FILE):
            try:
                os.remove(_COMFY_STATE_FILE)
            except Exception:
                pass
        _reclaim_transient_gpu_memory()
        COMFYUI_STATE["is_running"] = False
        COMFYUI_STATE["is_starting"] = False
        COMFYUI_STATE["public_url"] = ""
        COMFYUI_STATE["queue_running"] = 0
        COMFYUI_STATE["queue_pending"] = 0
        COMFYUI_STATE["progress"] = 0.0
        COMFYUI_STATE["status_message"] = "ComfyUI 已停止（节点、模型与工作流已持久化保存至 Google Drive）"
        print("[ComfyUI] 服务与公网隧道已停止，GPU 显存已回收。")
        return True, COMFYUI_STATE["status_message"]


def start_comfyui_core(unload_existing: bool = False, force_restart: bool = False) -> Tuple[bool, str]:
    global _COMFY_BG_SYNC_STARTED
    touch_activity()
    _try_adopt_running_comfyui()
    if COMFYUI_STATE["is_starting"]:
        return True, "ComfyUI 正在部署启动中，请稍候..."
    if COMFYUI_STATE["is_running"] and COMFYUI_STATE.get("public_url") and not force_restart:
        return True, f"ComfyUI 已在运行: {COMFYUI_STATE['public_url']}"

    if force_restart and COMFYUI_STATE["is_running"]:
        stop_comfyui_core(sync_before_stop=True)

    COMFYUI_STATE["is_starting"] = True
    COMFYUI_STATE["progress"] = 0.06
    COMFYUI_STATE["status_message"] = "正在初始化 ComfyUI 环境与 Google Drive 持久化目录..."

    try:
        if unload_existing:
            COMFYUI_STATE["status_message"] = "正在释放现有常驻模型显存，为 ComfyUI 腾出完整 96GB 空间..."
            unload_model_from_vram_core("all")

        comfy_root, persist_root, mode, label, is_drive = _resolve_comfyui_persistence_paths()

        # 1. 若为 rclone 云盘同步模式且本实例首次拉起，先从 Google Drive 拉回已持久化的节点与工作流；
        #    权重不在开机时整盘拉回，等用户点某个预置包的「一键配齐」时只拉那个包的权重（install_comfyui_preset_pack_core）
        pull_flag = "/tmp/.easycolab_comfy_drive_pulled"
        if mode == "rclone_drive_sync" and _has_rclone_drive_remote() and not os.path.exists(pull_flag):
            COMFYUI_STATE["progress"] = 0.14
            COMFYUI_STATE["status_message"] = "正在从 Google Drive 恢复历史持久化的 ComfyUI 节点与工作流（权重在点「一键配齐」时再拉）..."
            print("[ComfyUI] 正在从 Google Drive 同步已保存的节点与工作流（不含权重）...")
            _rclone_pull_comfy_from_drive(persist_root, None, 0.14, 0.22, "历史节点与工作流", exclude_models=True)
            with open(pull_flag, "w") as pf:
                pf.write("1")

        # 2. 克隆或检查 ComfyUI 主程序 (置于本地高速 NVMe，保证秒级导入)
        if not os.path.exists(os.path.join(comfy_root, "main.py")):
            COMFYUI_STATE["progress"] = 0.24
            COMFYUI_STATE["status_message"] = "正在克隆官方最新 ComfyUI 核心引擎..."
            print("[ComfyUI] 正在克隆官方最新 ComfyUI 仓库...")
            # 目录已被提前建出（例如预置包先建了 models 软链或 custom_nodes）时，先挪开再克隆，克隆后把其中的自定义节点搬回
            stale_dir = ""
            if os.path.isdir(comfy_root) and os.listdir(comfy_root):
                stale_dir = f"{comfy_root}.pre_{int(time.time())}"
                os.rename(comfy_root, stale_dir)
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", comfy_root],
                check=True,
            )
            if stale_dir:
                stale_nodes = os.path.join(stale_dir, "custom_nodes")
                if os.path.isdir(stale_nodes):
                    for entry in os.listdir(stale_nodes):
                        dst_p = os.path.join(comfy_root, "custom_nodes", entry)
                        if not os.path.exists(dst_p):
                            shutil.move(os.path.join(stale_nodes, entry), dst_p)
                shutil.rmtree(stale_dir, ignore_errors=True)
            COMFYUI_STATE["progress"] = 0.36
            COMFYUI_STATE["status_message"] = "正在安装 ComfyUI 核心依赖（保留当前 G4 Blackwell CUDA PyTorch）..."
            raw_reqs = open(os.path.join(comfy_root, "requirements.txt"), "r", encoding="utf-8").readlines()
            safe_reqs = [
                ln.strip() for ln in raw_reqs
                if ln.strip() and not ln.strip().startswith("#") and not _COMFY_PROTECTED_PIP_PKGS.match(ln)
            ]
            tmp_core_req = "/tmp/comfy_core_safe_reqs.txt"
            with open(tmp_core_req, "w", encoding="utf-8") as wf:
                wf.write("\n".join(safe_reqs) + "\n")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", tmp_core_req], check=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "gguf", "opencv-python-headless", "imageio-ffmpeg", "gitpython", "toml"],
                check=False,
            )

        # 3. 建立 Google Drive 软链体系：models/* 各子目录、user 工作流配置、output 成片输出全部自动软链至持久化根目录
        COMFYUI_STATE["progress"] = 0.48
        COMFYUI_STATE["status_message"] = f"正在建立 {label} 目录软链 (models / workflows / output)..."
        os.makedirs(os.path.join(persist_root, "models"), exist_ok=True)
        all_model_subdirs = set(COMFY_STANDARD_MODEL_SUBDIRS)
        for existing_sub in os.listdir(os.path.join(persist_root, "models")):
            if not existing_sub.startswith(".") and os.path.isdir(os.path.join(persist_root, "models", existing_sub)):
                all_model_subdirs.add(existing_sub)
        for sub in sorted(all_model_subdirs):
            _safe_symlink_dir(
                os.path.join(persist_root, "models", sub),
                os.path.join(comfy_root, "models", sub),
            )
        _safe_symlink_dir(os.path.join(persist_root, "user"), os.path.join(comfy_root, "user"))
        _safe_symlink_dir(os.path.join(persist_root, "output"), os.path.join(comfy_root, "output"))
        _ensure_comfyui_beginner_assets(comfy_root, persist_root)

        # 若云盘中已有任何预置包的全部权重，但缺少中文工作流文件或工作流版本尚未升级至 v2 (修复 widgets_values_named 兼容)，自动重新生成
        wf_dir_chk = os.path.join(persist_root, "user", "default", "workflows")
        schema_v2_ok = os.path.isfile(os.path.join(wf_dir_chk, ".easycolab_wf_schema_v3"))
        for pk in COMFY_PRESET_WORKFLOW_PACKS:
            if all(
                os.path.isfile(os.path.join(persist_root, "models", sub, fn))
                for sub, fn, _, _, _ in pk["models"]
            ):
                if (not schema_v2_ok) or (not all(os.path.isfile(os.path.join(wf_dir_chk, w)) for w in pk["workflows"])):
                    _build_and_install_comfy_preset_workflows(persist_root, [pk["id"]])

        # 4. 恢复已持久化的自定义节点至本地高速 NVMe，并预装 ComfyUI-Manager、VideoHelperSuite 视频核心节点与 MiniMax H3 导演台
        COMFYUI_STATE["progress"] = 0.58
        COMFYUI_STATE["status_message"] = "正在恢复已持久化的自定义节点并配置 ComfyUI-Manager..."
        cn_local = os.path.join(comfy_root, "custom_nodes")
        cn_persist = os.path.join(persist_root, "custom_nodes")
        os.makedirs(cn_local, exist_ok=True)
        os.makedirs(cn_persist, exist_ok=True)

        for entry in sorted(os.listdir(cn_persist)):
            if entry.startswith(".") or entry in ("__pycache__", "easycolab_preset_helper"):
                continue
            src_p = os.path.join(cn_persist, entry)
            dst_p = os.path.join(cn_local, entry)
            if os.path.isdir(src_p) and not os.path.exists(dst_p):
                if shutil.which("rsync"):
                    subprocess.run(["rsync", "-a", f"{src_p}/", f"{dst_p}/"], check=False)
                else:
                    shutil.copytree(src_p, dst_p, dirs_exist_ok=True)

        # 若 manifest 中记录了节点但目录缺失，自动按 Git URL 补齐
        manifest_path = os.path.join(persist_root, "custom_nodes_manifest.json")
        if os.path.exists(manifest_path):
            try:
                manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
                for node_name, meta in manifest.items():
                    git_url = (meta or {}).get("git_url", "")
                    dst_p = os.path.join(cn_local, node_name)
                    if git_url and not os.path.exists(dst_p):
                        subprocess.run(["git", "clone", "--depth", "1", git_url, dst_p], check=False)
            except Exception:
                pass

        default_nodes = [
            ("ComfyUI-Manager", "https://github.com/ltdrdata/ComfyUI-Manager.git"),
            ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"),
            # MiniMax H3 多段导演台（首尾帧 / 参考图 / 段间引导 / 语义桥 / 二采）；依赖由下面的 requirements 扫描自动补齐
            (H3_DIRECTOR_NODE_DIR, H3_DIRECTOR_REPO),
            # 导演台「加速版 / 外部多组」示例工作流里的 SageAttention 补丁节点来自 KJNodes
            ("ComfyUI-KJNodes", "https://github.com/kijai/ComfyUI-KJNodes.git"),
        ]
        for node_dir_name, repo_url in default_nodes:
            target_node_dir = os.path.join(cn_local, node_dir_name)
            if not os.path.exists(target_node_dir):
                print(f"[ComfyUI] 正在预装核心插件 {node_dir_name}...")
                subprocess.run(["git", "clone", "--depth", "1", repo_url, target_node_dir], check=False)

        _ensure_easycolab_comfy_preset_helper(comfy_root, persist_root)
        _ensure_comfyui_manager_tunnel_config(comfy_root, persist_root)

        COMFYUI_STATE["progress"] = 0.70
        COMFYUI_STATE["status_message"] = "正在检查并安装已保存自定义节点的 Python 依赖包..."
        _install_persisted_custom_node_requirements(comfy_root)
        _install_h3_director_optional_extras(comfy_root)
        sync_comfyui_to_drive_core(quiet=True)

        # 5. 启动 ComfyUI 服务进程 (loopback) 并实时镜像输出至 EasyColab 日志面板
        COMFYUI_STATE["progress"] = 0.80
        COMFYUI_STATE["status_message"] = "正在拉起 ComfyUI 节点推理引擎 (本地后端)..."
        subprocess.run(["pkill", "-f", "ComfyUI/main.py"], check=False)
        subprocess.run(["pkill", "-f", "cloudflared.*8188"], check=False)
        time.sleep(0.6)

        port = int(COMFYUI_STATE.get("local_port") or COMFY_BACKEND)
        comfy_cmd = [
            sys.executable, "-u", os.path.join(comfy_root, "main.py"),
            "--listen", "127.0.0.1",
            "--port", str(port),
            "--preview-method", "auto",
        ]
        comfy_proc = _spawn_logged_detached(comfy_cmd, _COMFY_LOG_FILE, cwd=comfy_root)
        COMFYUI_STATE["comfy_proc"] = comfy_proc
        threading.Thread(
            target=_follow_log_file,
            args=(_COMFY_LOG_FILE, comfy_proc, lambda ln: ln and print(f"[ComfyUI] {ln}")),
            daemon=True,
        ).start()

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        ready = False
        for _ in range(90):
            if comfy_proc.poll() is not None:
                raise RuntimeError(f"ComfyUI 启动失败提前退出 (code={comfy_proc.returncode})，请查看右侧运行日志。")
            try:
                with opener.open(f"http://127.0.0.1:{port}/system_stats", timeout=1.5) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(1.0)

        if not ready:
            raise RuntimeError("等待 ComfyUI 本地后端端口 就绪超时，请查看右侧运行日志。")

        # 6. 启动 Cloudflare WebSocket 高速公网隧道
        COMFYUI_STATE["progress"] = 0.92
        COMFYUI_STATE["status_message"] = "ComfyUI 内核已就绪，正在建立 Cloudflare 高速全屏画布隧道..."
        tunnel_proc, cf_url = _start_comfy_tunnel(22.5)
        COMFYUI_STATE["tunnel_proc"] = tunnel_proc
        cf_url_box: List[str] = [cf_url]

        if not cf_url_box[0]:
            raise RuntimeError("未能获取 Cloudflare 公网隧道地址，请重试。")

        COMFYUI_STATE["public_url"] = cf_url_box[0]
        COMFYUI_STATE["is_running"] = True
        COMFYUI_STATE["progress"] = 1.0
        try:
            with open(_COMFY_STATE_FILE, "w", encoding="utf-8") as sf:
                json.dump({"public_url": cf_url_box[0], "port": port}, sf)
        except Exception:
            pass
        _COMFY_STATUS_CACHE["ts"] = 0.0
        (cn_count, m_count, _), _ = _scan_comfyui_counts_and_outputs()
        ok_msg = f"ComfyUI 已就绪（{label} · 节点 {cn_count} 个 · 权重 {m_count} 个）"
        COMFYUI_STATE["status_message"] = ok_msg
        print(f"[ComfyUI] 画布公网隧道已开通: {cf_url_box[0]}")

        if not _COMFY_BG_SYNC_STARTED:
            _COMFY_BG_SYNC_STARTED = True
            threading.Thread(target=_comfyui_periodic_drive_sync_loop, daemon=True).start()

        return True, ok_msg
    except Exception as e:
        err_msg = f"ComfyUI 启动出错: {e}"
        COMFYUI_STATE["status_message"] = err_msg
        COMFYUI_STATE["progress"] = 0.0
        print(f"[ComfyUI] {err_msg}")
        return False, err_msg
    finally:
        COMFYUI_STATE["is_starting"] = False
        touch_activity()


def download_comfyui_model_core(
    url_or_repo: str,
    filename: str = "",
    subfolder: str = "diffusion_models",
    hf_token: str = "",
) -> Tuple[bool, str]:
    """8 线程并发下载模型权重直达持久化的 ComfyUI/models/<subfolder>，下次重开 G4 免重复下载"""
    touch_activity()
    raw_src = (url_or_repo or "").strip()
    if not raw_src:
        return False, "请输入 HuggingFace 仓库/直链或模型下载 URL"
    if COMFYUI_STATE["is_downloading_weight"]:
        return False, "当前已有 ComfyUI 权重正在下载中，请稍候"

    clean_sub = re.sub(r"[^a-zA-Z0-9_.-]", "", (subfolder or "diffusion_models").strip()) or "diffusion_models"
    comfy_root, persist_root, mode, label, _ = _resolve_comfyui_persistence_paths()
    dest_dir = os.path.join(persist_root, "models", clean_sub)
    _repair_dangling_dir_link(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.isdir(comfy_root):
        _safe_symlink_dir(dest_dir, os.path.join(comfy_root, "models", clean_sub))

    COMFYUI_STATE["is_downloading_weight"] = True
    COMFYUI_STATE["progress"] = 0.05
    try:
        token = (hf_token or "").strip() or None
        hf_m = re.match(r"^https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/[^/]+/(.+?)(\?.*)?$", raw_src)
        if hf_m:
            repo_id = hf_m.group(1)
            rel_file = hf_m.group(2)
            base_name = os.path.basename(filename.strip() or rel_file)
            COMFYUI_STATE["status_message"] = f"正在 8 线程极速下载 {base_name} 至 models/{clean_sub}..."
            print(f"[ComfyUI 极速下载] {repo_id} -> {rel_file} -> models/{clean_sub}/{base_name}")
            dl_path = fast_parallel_hf_download(repo_id=repo_id, filename=rel_file, dest_dir=dest_dir, hf_token=token)
            final_path = os.path.join(dest_dir, base_name)
            if os.path.abspath(dl_path) != os.path.abspath(final_path):
                shutil.move(dl_path, final_path)
        elif raw_src.startswith("http://") or raw_src.startswith("https://"):
            url_path = urllib.parse.urlparse(raw_src).path
            base_name = os.path.basename(filename.strip() or url_path) or f"model_{int(time.time())}.safetensors"
            final_path = os.path.join(dest_dir, base_name)
            COMFYUI_STATE["status_message"] = f"正在下载 {base_name} 至 models/{clean_sub}..."
            print(f"[ComfyUI 直链下载] {raw_src} -> models/{clean_sub}/{base_name}")
            headers = {"User-Agent": "EasyColab/2.0"}
            if token and urllib.parse.urlsplit(raw_src).scheme == "https" and urllib.parse.urlsplit(raw_src).hostname == "huggingface.co":
                headers["Authorization"] = f"Bearer {token}"
            req = urllib.request.Request(raw_src, headers=headers)
            tmp_p = final_path + ".part"
            with urllib.request.build_opener(_StripAuthRedirectHandler()).open(req, timeout=120) as resp, open(tmp_p, "wb") as wf:
                total_sz = int(resp.headers.get("Content-Length") or 0)
                done_sz = 0
                t0 = time.time()
                last_log = 0.0
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    wf.write(chunk)
                    done_sz += len(chunk)
                    now_t = time.time()
                    if total_sz > 0 and (now_t - last_log) >= 0.6:
                        last_log = now_t
                        pct = done_sz / float(total_sz)
                        COMFYUI_STATE["progress"] = round(min(0.98, pct), 3)
                        spd = (done_sz / (1024 ** 2)) / max(0.01, now_t - t0)
                        print(f"{base_name}: {int(pct * 100)}%| {done_sz / (1024**3):.2f}G/{total_sz / (1024**3):.2f}G [{spd:.1f}MB/s]")
            os.replace(tmp_p, final_path)
        else:
            if not filename.strip():
                return False, "填写 HuggingFace 仓库 ID (如 Comfy-Org/Wan_2.1_ComfyUI_repackaged) 时请同时填写文件路径"
            rel_file = filename.strip()
            base_name = os.path.basename(rel_file)
            COMFYUI_STATE["status_message"] = f"正在 8 线程极速下载 {base_name} 至 models/{clean_sub}..."
            dl_path = fast_parallel_hf_download(repo_id=raw_src, filename=rel_file, dest_dir=dest_dir, hf_token=token)
            final_path = os.path.join(dest_dir, base_name)
            if os.path.abspath(dl_path) != os.path.abspath(final_path):
                shutil.move(dl_path, final_path)

        if mode == "rclone_drive_sync":
            sync_comfyui_to_drive_core(quiet=True)
        _COMFY_STATUS_CACHE["ts"] = 0.0
        sz_gb = os.path.getsize(final_path) / (1024 ** 3)
        msg = f"权重 {base_name} ({sz_gb:.2f} GB) 已下载至 models/{clean_sub} 并持久化到 {label}（在画布按 R 键即可刷新选用）"
        COMFYUI_STATE["progress"] = 1.0
        COMFYUI_STATE["status_message"] = msg
        print(f"[ComfyUI] {msg}")
        return True, msg
    except Exception as e:
        err = f"下载 ComfyUI 权重失败: {e}"
        COMFYUI_STATE["status_message"] = err
        print(f"[ComfyUI] {err}")
        return False, err
    finally:
        COMFYUI_STATE["is_downloading_weight"] = False
        touch_activity()


_RCLONE_STATS_RE = re.compile(
    r"([\d.]+\s*\S?i?B)\s*/\s*([\d.]+\s*\S?i?B),\s*(\d+)%(?:,\s*([\d.]+\s*\S?i?B/s))?(?:,\s*ETA\s*(\S+))?"
)


def _rclone_pull_comfy_from_drive(
    persist_root: str,
    rel_files: Optional[List[str]] = None,
    prog_lo: float = 0.03,
    prog_hi: float = 0.05,
    what: str = "已持久化的历史文件",
    exclude_models: bool = False,
) -> None:
    """从 Google Drive 拉回 ComfyUI 持久化目录，把 rclone 的实时进度写进 COMFYUI_STATE；rel_files 不为空时只拉这些相对路径；exclude_models 时不拉 models/ 下的权重"""
    os.makedirs(persist_root, exist_ok=True)
    cmd = [
        "rclone", "copy",
        "easycolab_gdrive:EasyColab_Drive/EasyColab_ComfyUI",
        persist_root,
        "--transfers", "8",
        "--checkers", "16",
        "--drive-chunk-size", "64M",
        "--stats", "2s",
        "--stats-one-line",
        "--stats-log-level", "NOTICE",
    ]
    if rel_files is not None:
        list_path = "/tmp/.easycolab_comfy_pull_list.txt"
        with open(list_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(rel_files) + "\n")
        cmd += ["--files-from-raw", list_path]
    elif exclude_models:
        cmd += ["--exclude", "models/**"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        m = _RCLONE_STATS_RE.search(line)
        if not m:
            print(f"[ComfyUI 云盘拉取] {line.rstrip()}")
            continue
        pct = int(m.group(3))
        extra = "".join(
            s for s in (
                f"，{m.group(4)}" if m.group(4) else "",
                f"，剩余 {m.group(5)}" if m.group(5) and m.group(5) != "-" else "",
            )
        )
        COMFYUI_STATE["progress"] = round(prog_lo + (prog_hi - prog_lo) * pct / 100.0, 3)
        COMFYUI_STATE["status_message"] = f"正在从 Google Drive 拉回{what}：{m.group(1)} / {m.group(2)}（{pct}%{extra}）"
        touch_activity()
    proc.wait()


def install_comfyui_preset_pack_core(
    pack_id: str,
    auto_start: bool = True,
    hf_token: str = "",
    weight_source: str = "hf",
) -> Tuple[bool, str]:
    """一键配齐预置中文工作流包：8 线程下载缺失模型权重 + 写入中文傻瓜注释工作流 + 生成默认参考图 + 自动拉起/刷新画布
    weight_source：hf（默认）＝不从 Drive 拉，直接从 HF 下；auto＝先从 Drive 拉、缺的再去 HF；drive＝只从 Drive，缺的报出来不去 HF"""
    touch_activity()
    if weight_source not in ("auto", "drive", "hf"):
        return False, f"未知的权重来源: {weight_source}（可选 auto / drive / hf）"
    target_pack = next((p for p in COMFY_PRESET_WORKFLOW_PACKS if p["id"] == pack_id), None)
    if not target_pack:
        return False, f"未找到预置工作流包: {pack_id}"
    if COMFYUI_STATE["is_downloading_weight"] or COMFYUI_STATE.get("installing_pack_id"):
        return False, "当前已有模型或工作流包正在高速下载配齐中，请稍候..."

    comfy_root, persist_root, mode, label, _ = _resolve_comfyui_persistence_paths()
    models_list = _comfy_pack_all_models(target_pack)   # 含 deps：工作流要用、但放在别的包里的权重
    total_m = len(models_list)

    COMFYUI_STATE["is_downloading_weight"] = True
    COMFYUI_STATE["installing_pack_id"] = pack_id
    COMFYUI_STATE["progress"] = 0.03
    COMFYUI_STATE["status_message"] = f"正在准备配齐「{target_pack['title']}」（共 {total_m} 个官方权重）..."

    try:
        token = (hf_token or "").strip() or ""
        # 先算空间再动手：从云盘拉回和从 HF 下载都写本地盘，放不下就一个都不下，免得装到一半把盘写满
        missing_now = _comfy_missing_models(persist_root, models_list)
        missing_gb = sum(m[4] for m in missing_now)
        free_gb = _disk_free_gb(persist_root)
        if missing_now and missing_gb + 5.0 > free_gb:
            raise RuntimeError(
                f"磁盘空间不足，没有开始下载：本包还缺 {len(missing_now)} 个权重共约 {missing_gb:.0f} GB（另留 5 GB 余量），"
                f"盘上只剩 {free_gb:.0f} GB，还差约 {missing_gb + 5.0 - free_gb:.0f} GB。请先在 ComfyUI 模型目录删掉不用的权重"
            )
        if weight_source == "drive" and not (
            mode == "drive_symlink" or (mode == "rclone_drive_sync" and _has_rclone_drive_remote())
        ):
            raise RuntimeError("当前实例没有连上 Google Drive，无法「只从 Drive」配齐；请改选「自动」或「只从 Hugging Face」")
        # 若为 rclone 模式，先从云盘拉回本包缺的那几个权重（开机时不拉权重），云盘没有的再去 HF 下载（「只从 HF」时跳过）
        # 旧版本可能把 models/ultralytics/bbox 这类子目录链成了自环，先修好再拉，免得 rclone 也卡在上面
        for sub in {m[0] for m in models_list}:
            _repair_dangling_dir_link(os.path.join(persist_root, "models", sub))
        if weight_source != "hf" and mode == "rclone_drive_sync" and _has_rclone_drive_remote():
            pack_missing = [f"models/{sub}/{fname}" for sub, fname, _, _, _ in missing_now]
            if pack_missing:
                COMFYUI_STATE["status_message"] = "正在先从 Google Drive 检查已持久化的历史模型权重..."
                _rclone_pull_comfy_from_drive(persist_root, pack_missing, 0.03, 0.05, "本包已持久化的权重")

        _ensure_comfyui_beginner_assets(comfy_root, persist_root)

        if weight_source == "drive":
            still_missing = [
                f"models/{sub}/{fname}" for sub, fname, _, _, _ in models_list
                if not os.path.isfile(os.path.join(persist_root, "models", sub, fname))
            ]
            if still_missing:
                raise RuntimeError(
                    f"Google Drive 里缺 {len(still_missing)} 个权重（已选「只从 Drive」，不会去 Hugging Face 下载）：" + "、".join(still_missing)
                )

        os.makedirs(persist_root, exist_ok=True)

        for idx, (sub, fname, repo_id, repo_rel_path, approx_gb) in enumerate(models_list, start=1):
            touch_activity()
            dest_dir = os.path.join(persist_root, "models", sub)
            _repair_dangling_dir_link(dest_dir)
            os.makedirs(dest_dir, exist_ok=True)
            if os.path.isdir(comfy_root):
                _safe_symlink_dir(dest_dir, os.path.join(comfy_root, "models", sub))

            final_path = os.path.join(dest_dir, fname)
            base_prog = 0.05 + 0.80 * ((idx - 1) / float(max(1, total_m)))
            COMFYUI_STATE["progress"] = round(base_prog, 3)

            if os.path.isfile(final_path) and os.path.getsize(final_path) > 1024 * 1024:
                print(f"[ComfyUI 预置包] ({idx}/{total_m}) 已存在跳过: models/{sub}/{fname}")
                continue

            COMFYUI_STATE["status_message"] = (
                f"[{idx}/{total_m}] 正在 8 线程极速下载 {fname} (约 {approx_gb:.1f} GB)..."
            )
            print(f"[ComfyUI 预置包] ({idx}/{total_m}) 开始 8 线程下载 {repo_id}/{repo_rel_path} -> models/{sub}/{fname}")
            dl_path = fast_parallel_hf_download(
                repo_id=repo_id,
                filename=repo_rel_path,
                dest_dir=dest_dir,
                hf_token=token,
                num_connections=8,
            )
            if os.path.abspath(dl_path) != os.path.abspath(final_path):
                shutil.move(dl_path, final_path)
                # 清理 fast_parallel_hf_download 因嵌套路径创建的空子目录 (如 split_files/*)
                top_nested = repo_rel_path.split("/", 1)[0]
                nested_dir = os.path.join(dest_dir, top_nested)
                if top_nested and os.path.isdir(nested_dir):
                    try:
                        shutil.rmtree(nested_dir, ignore_errors=True)
                    except Exception:
                        pass

        COMFYUI_STATE["progress"] = 0.88
        COMFYUI_STATE["status_message"] = f"正在生成「{target_pack['title']}」全中文三步傻瓜注释工作流..."
        installed_wfs = _build_and_install_comfy_preset_workflows(persist_root, [pack_id])
        if os.path.isdir(comfy_root):
            _safe_symlink_dir(os.path.join(persist_root, "user"), os.path.join(comfy_root, "user"))
            _safe_symlink_dir(os.path.join(persist_root, "output"), os.path.join(comfy_root, "output"))
            _ensure_comfyui_beginner_assets(comfy_root, persist_root)

        if mode == "rclone_drive_sync":
            COMFYUI_STATE["progress"] = 0.93
            COMFYUI_STATE["status_message"] = "正在将配齐的模型与中文工作流持久化同步至 Google Drive..."
            sync_comfyui_to_drive_core(quiet=True)

        _COMFY_STATUS_CACHE["ts"] = 0.0
        wf_names_str = "、".join(installed_wfs) if installed_wfs else "、".join(target_pack["workflows"])

        # 若 ComfyUI 正在运行，主动轻触 /object_info 刷新模型下拉缓存；若尚未启动且开启了 auto_start，直接自动拉起画布
        if COMFYUI_STATE.get("is_running"):
            try:
                port = int(COMFYUI_STATE.get("local_port") or COMFY_BACKEND)
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(f"http://127.0.0.1:{port}/object_info/UNETLoader", timeout=3.0) as _:
                    pass
            except Exception:
                pass
            COMFYUI_STATE["progress"] = 1.0
            ok_msg = f"「{target_pack['title']}」已全部配齐！打开画布左侧「工作流」点击 {wf_names_str} 即可直接传图运行"
            COMFYUI_STATE["status_message"] = ok_msg
            print(f"[ComfyUI 预置包] {ok_msg}")
            return True, ok_msg
    except Exception as e:
        err = f"配齐预置工作流包失败: {e}"
        COMFYUI_STATE["progress"] = 0.0
        COMFYUI_STATE["status_message"] = err
        print(f"[ComfyUI 预置包] {err}")
        return False, err
    finally:
        COMFYUI_STATE["is_downloading_weight"] = False
        COMFYUI_STATE["installing_pack_id"] = ""
        touch_activity()

    # 若尚未拉起 ComfyUI，在释放下载锁后自动拉起云端画布隧道
    if auto_start and not COMFYUI_STATE.get("is_running") and not COMFYUI_STATE.get("is_starting"):
        print("[ComfyUI 预置包] 权重与工作流已就绪，正在自动拉起 ComfyUI 全屏画布...")
        ok_s, msg_s = start_comfyui_core(unload_existing=False, force_restart=False)
        if ok_s:
            final_msg = f"「{target_pack['title']}」已配齐且画布已启动！在左侧「工作流」打开中文模板即可直接出片"
            COMFYUI_STATE["status_message"] = final_msg
            return True, final_msg
        return False, msg_s

    return True, f"「{target_pack['title']}」模型与中文工作流已配齐！"


def disconnect_colab_runtime() -> str:
    try:
        if os.path.isdir("/content/ComfyUI") or COMFYUI_STATE.get("is_running"):
            sync_comfyui_to_drive_core(quiet=True)
    except Exception:
        pass
    try:
        from google.colab import runtime
        runtime.unassign()
        return "🛑 已向 Colab 发送释放运行时指令，实例即将关闭并停止扣除计算单元。"
    except ImportError:
        return "ℹ️ 当前不在 Google Colab 环境中，无需释放云端运行时。"
    except Exception as e:
        return f"⚠️ 释放运行时失败: {e}"


# ==============================================================================
# 供 macOS 原生客户端调用的异步 REST API 接口 (/minimax_api/* · 多模型通用)
# ==============================================================================
class SelectModelRequest(BaseModel):
    model_id: str = "minimax_h3"
    repo_id: str = ""
    include_pattern: str = ""
    custom_name: str = ""
    modality: str = "auto"
    hf_token: str = ""


class DownloadModelRequest(BaseModel):
    model_id: Optional[str] = None
    hf_token: str = ""
    repo_id: str = ""
    include_pattern: str = ""
    custom_name: str = ""
    modality: str = "auto"


class LoadModelRequest(BaseModel):
    model_id: Optional[str] = None
    precision_mode: str = "bf16"
    offload_mode: str = "disk"
    hf_token: str = ""
    context_length: Optional[int] = None
    repo_id: str = ""
    include_pattern: str = ""
    custom_name: str = ""
    modality: str = "auto"


class ContextConfigRequest(BaseModel):
    context_length: int = 32768


class IdleConfigRequest(BaseModel):
    idle_timeout_minutes: int = 30


class ComfyStartRequest(BaseModel):
    unload_existing: bool = False
    force_restart: bool = False


class ComfyWeightDownloadRequest(BaseModel):
    url_or_repo: str
    filename: str = ""
    subfolder: str = "diffusion_models"
    hf_token: str = ""


class ComfyInstallPackRequest(BaseModel):
    pack_id: str
    auto_start: bool = True
    hf_token: str = ""
    weight_source: str = "hf"  # hf（默认）/ auto / drive


class GenerateVideoRequest(BaseModel):
    model_id: Optional[str] = None
    task_mode: str = "t2va"
    visual_prompt: str
    negative_prompt: str = ""
    soundscape_prompt: str = ""
    music_prompt: str = ""
    first_frame_base64: Optional[str] = None
    last_frame_base64: Optional[str] = None
    # MiniMax-H3-Image 参考图编辑：前两张之外追加的参考图，按顺序接在 <Picture 2> 后面
    extra_reference_images_base64: Optional[List[str]] = None
    # MiniMax-H3 参考生音视频 (ref2va)：参考图、参考视频、参考音频各自按上传顺序编号，对应提示词里的 <Picture N> / <Video N> / <Audio N>
    reference_images_base64: Optional[List[str]] = None
    reference_videos_base64: Optional[List[str]] = None
    reference_audios_base64: Optional[List[str]] = None
    # 参考视频带声音时，把原声也作为一段 <Audio N> 参考（编号排在独立参考音频前面）
    ref2va_use_video_audio: bool = True
    width: int = 832
    height: int = 480
    num_frames: int = 124
    num_inference_steps: int = 35
    fps: int = 24
    seed: int = 42
    randomize_seed: bool = True
    batch_count: int = 1
    denoising_strength: float = 0.70
    enable_thinking: bool = True
    max_new_tokens: int = 16384
    context_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: Optional[float] = None
    repeat_penalty: float = 1.0
    reasoning_effort: str = "medium"
    chat_messages: Optional[List[Dict[str, str]]] = None


def decode_base64_image(b64_str: Optional[str]) -> Optional[Image.Image]:
    if not b64_str:
        return None
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _probe_media_file(path: str) -> Tuple[float, bool, int, int]:
    """返回 (时长秒, 是否有音轨, 视频宽, 视频高)；纯音频文件的宽高为 0"""
    import av
    with av.open(path) as container:
        dur = float(container.duration or 0) / 1_000_000.0
        streams = list(container.streams.video) + list(container.streams.audio)
        if dur <= 0:
            for st in streams:
                if st.duration and st.time_base:
                    dur = max(dur, float(st.duration * st.time_base))
        has_audio = len(container.streams.audio) > 0
        vw = vh = 0
        if container.streams.video:
            vw = int(container.streams.video[0].codec_context.width or 0)
            vh = int(container.streams.video[0].codec_context.height or 0)
    return dur, has_audio, vw, vh


def _build_h3_ref2va_references(req: "GenerateVideoRequest") -> Tuple[List[Dict[str, Any]], str]:
    """把客户端传来的参考图/视频/音频解码成 DiffSynth MiniMaxH3Pipeline 的 references 列表，并返回各素材对应的提示词标签说明。
    参考顺序固定为：全部图片 → 全部视频 → 全部音频；管线按类型分别编号，带原声的视频会先占用一个 <Audio N>。"""
    import tempfile
    from diffsynth.utils.data.audio_video import read_video_audio
    from diffsynth.utils.data.audio import read_audio

    lim = H3_REF2VA_LIMITS
    img_b64 = [b for b in (req.reference_images_base64 or []) if b]
    vid_b64 = [b for b in (req.reference_videos_base64 or []) if b]
    aud_b64 = [b for b in (req.reference_audios_base64 or []) if b]
    if not (img_b64 or vid_b64 or aud_b64):
        raise ValueError("参考生音视频模式下请至少上传一张参考图、一段参考视频或一段参考音频")
    if len(img_b64) > lim["image"] or len(vid_b64) > lim["video"] or len(aud_b64) > lim["audio"]:
        raise ValueError(f"参考素材超出官方上限：图片最多 {lim['image']} 张、视频最多 {lim['video']} 段、音频最多 {lim['audio']} 段")
    if len(img_b64) + len(vid_b64) + len(aud_b64) > lim["total"]:
        raise ValueError(f"参考素材合计最多 {lim['total']} 个")

    def _check_dur(kind: str, idx: int, dur: float):
        if dur < lim["min_s"] - 0.05 or dur > lim["max_s"] + 0.05:
            raise ValueError(f"参考{kind} {idx} 时长 {dur:.1f} 秒，官方要求每段 {lim['min_s']:.0f}~{lim['max_s']:.0f} 秒")

    refs: List[Dict[str, Any]] = []
    labels: List[str] = []
    for i, b in enumerate(img_b64, start=1):
        refs.append({"type": "image", "image": decode_base64_image(b)})
        labels.append(f"<Picture {i}>=参考图{i}")

    tmp_dir = tempfile.mkdtemp(prefix="h3_ref2va_")
    try:
        def _to_file(b64: str, name: str) -> str:
            if "," in b64[:200]:
                b64 = b64.split(",", 1)[1]
            p = os.path.join(tmp_dir, name)
            with open(p, "wb") as wf:
                wf.write(base64.b64decode(b64))
            return p

        audio_no = 0
        total_v = 0.0
        for i, b in enumerate(vid_b64, start=1):
            p = _to_file(b, f"video_{i}.mp4")  # imageio 按扩展名选读取插件，没有扩展名会退化成读不出帧数的 LegacyReader；ffmpeg 实际按内容识别 mov/mp4
            dur, has_audio, vw, vh = _probe_media_file(p)
            _check_dur("视频", i, dur)
            total_v += dur
            if total_v > lim["max_s"] + 0.05:
                raise ValueError(f"参考视频合计 {total_v:.1f} 秒，官方要求合计不超过 {lim['max_s']:.0f} 秒")
            # 读帧时就按短边 768 缩小（保持原比例、16 的倍数），避免 4K 素材整段解码占满内存；管线内部还会再按官方规则缩放
            scale = min(1.0, 768.0 / max(1, min(vw, vh)), (768 * 1344 / max(1, vw * vh)) ** 0.5) if vw and vh else 1.0
            tw = max(16, int(round(vw * scale / 16.0)) * 16) if vw else None
            th = max(16, int(round(vh * scale / 16.0)) * 16) if vh else None
            frames, wav, sr = read_video_audio(p, height=th, width=tw, duration=dur, fps=24, audio_sample_rate=32000)
            if not frames:
                raise ValueError(f"参考视频 {i} 解不出画面，请换成 mp4/mov 文件")
            if req.ref2va_use_video_audio and has_audio and wav is not None:
                audio_no += 1
                refs.append({"type": "video_audio", "video": frames, "audio": wav, "sample_rate": sr})
                labels.append(f"<Video {i}>=参考视频{i}（原声为 <Audio {audio_no}>）")
            else:
                refs.append({"type": "video", "video": frames})
                labels.append(f"<Video {i}>=参考视频{i}")

        total_a = 0.0
        for i, b in enumerate(aud_b64, start=1):
            p = _to_file(b, f"audio_{i}.m4a")
            dur, _, _, _ = _probe_media_file(p)
            _check_dur("音频", i, dur)
            total_a += dur
            if total_a > lim["max_s"] + 0.05:
                raise ValueError(f"参考音频合计 {total_a:.1f} 秒，官方要求合计不超过 {lim['max_s']:.0f} 秒")
            wav, sr = read_audio(p, duration=dur, resample=True, resample_rate=32000)
            audio_no += 1
            refs.append({"type": "audio", "audio": wav, "sample_rate": sr})
            labels.append(f"<Audio {audio_no}>=参考音频{i}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return refs, "；".join(labels)


def _ensure_h3_ref2va_dit_downloaded(job_id: str) -> None:
    """按需下载 Ref2VA DiT 主干（13 分片约 66 GB）到 MiniMax-H3 的同一个模型目录；文本编码器与 VAE 复用 FL2VA 那份"""
    missing = [fn for fn in MINIMAX_H3_DIT_FILES["Ref2VA"] if not _find_existing_model_file("minimax_h3", fn)]
    if not missing:
        return
    if MODEL_DOWNLOAD_STATE["is_downloading"]:
        raise RuntimeError("另一个模型正在下载，等它下完再用参考生音视频")
    _, nvme_dir, drive_dir = _get_model_dirs("minimax_h3")
    need_gb = 5.2 * len(missing)
    os.makedirs(nvme_dir, exist_ok=True)
    nvme_free_gb = shutil.disk_usage(nvme_dir).free / (1024 ** 3)
    drive_free_gb = 0.0
    if DRIVE_MOUNTED and os.path.exists("/content/drive/MyDrive"):
        try:
            drive_free_gb = shutil.disk_usage("/content/drive/MyDrive").free / (1024 ** 3)
        except Exception:
            drive_free_gb = 0.0
    if nvme_free_gb >= need_gb + 22.0:
        dest_dir = nvme_dir
    elif drive_free_gb >= need_gb + 2.0:
        dest_dir = drive_dir
    else:
        raise RuntimeError(
            f"Ref2VA 主干还需约 {need_gb:.0f} GB，本地盘只剩 {nvme_free_gb:.0f} GB、云盘 {drive_free_gb:.0f} GB，空间不够"
        )
    backup_to_drive = dest_dir == nvme_dir and drive_free_gb >= need_gb + 2.0

    MODEL_DOWNLOAD_STATE["is_downloading"] = True
    try:
        for idx, fname in enumerate(missing, start=1):
            touch_activity()
            JOBS[job_id]["message"] = (
                f"首次使用参考生音视频：正在下载 Ref2VA 主干 {idx}/{len(missing)}（共约 {need_gb:.0f} GB，只需下载一次）..."
            )
            JOBS[job_id]["progress"] = round(0.02 + 0.10 * (idx - 1) / len(missing), 3)
            MODEL_DOWNLOAD_STATE["message"] = JOBS[job_id]["message"]
            dl_path = fast_parallel_hf_download(
                repo_id=MODEL_CATALOG["minimax_h3"]["repo_id"],
                filename=fname,
                dest_dir=dest_dir,
                hf_token=os.environ.get("HF_TOKEN", ""),
                num_connections=8,
            )
            if backup_to_drive:
                threading.Thread(target=_async_backup_to_drive, args=(dl_path, drive_dir, fname), daemon=True).start()
    finally:
        MODEL_DOWNLOAD_STATE["is_downloading"] = False
        with _TELEMETRY_LOCK:
            _CACHED_FILE_STATUS_MAP.clear()


def _split_thinking_and_answer(raw_text: str) -> Tuple[str, str]:
    """解析 Qwen3.8 <think>...</think> 推理链与最终回复"""
    text = raw_text or ""
    if "</think>" in text:
        think_part, ans_part = text.split("</think>", 1)
        think_part = think_part.replace("<think>", "").strip()
        return think_part, ans_part.strip()
    if text.lstrip().startswith("<think>"):
        return text.replace("<think>", "", 1).strip(), ""
    return "", text.strip()


def _is_bundle_alive(bundle: Optional[Dict[str, Any]]) -> bool:
    if not bundle:
        return False
    if bundle.get("backend") == "gguf":
        proc = bundle.get("server_proc")
        return proc is not None and proc.poll() is None
    return bundle.get("model") is not None


def _get_qwen_llm_bundle(preferred_model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    candidates: List[str] = []
    if preferred_model_id:
        low = preferred_model_id.lower()
        if preferred_model_id in MODEL_CATALOG and _is_llm_model_id(preferred_model_id):
            candidates.append(preferred_model_id)
        for mid, mspec in MODEL_CATALOG.items():
            if mspec.get("modality") != "llm_vlm" or mid in candidates:
                continue
            if low == str(mspec.get("repo_id") or "").lower() or low == str(mspec.get("name") or "").lower():
                candidates.append(mid)
        if preferred_model_id in ("qwen38_27b", "qwen38_27b_gguf") and preferred_model_id not in candidates:
            candidates.append(preferred_model_id)
        elif ("gguf" in low or "q4_k" in low or "unsloth" in low) and "qwen38_27b_gguf" not in candidates:
            candidates.append("qwen38_27b_gguf")
        elif ("qwen3.8" in low or "qwen38" in low):
            for qid in ("qwen38_27b_gguf", "qwen38_27b"):
                if qid not in candidates:
                    candidates.append(qid)

    if _is_llm_model_id(SELECTED_MODEL_ID) and SELECTED_MODEL_ID not in candidates:
        candidates.append(SELECTED_MODEL_ID)

    for fallback_id in list(LOADED_RUNTIMES.keys()) + ["qwen38_27b_gguf", "qwen38_27b"]:
        if _is_llm_model_id(fallback_id) and fallback_id not in candidates:
            candidates.append(fallback_id)

    for cid in candidates:
        rt = LOADED_RUNTIMES.get(cid)
        if rt:
            b = rt.get("llm_bundle")
            if _is_bundle_alive(b):
                return b
            # GGUF 子进程意外退出时清理陈旧表项
            if b and b.get("backend") == "gguf":
                LOADED_RUNTIMES.pop(cid, None)

    if _is_bundle_alive(LLM_BUNDLE):
        return LLM_BUNDLE
    return None


def _count_bundle_tokens(bundle: Optional[Dict[str, Any]], text: str) -> int:
    if not text:
        return 1
    if bundle is not None:
        if bundle.get("backend") == "gguf":
            try:
                opener = _make_noproxy_opener()
                tok_req = urllib.request.Request(
                    f"{bundle['base_url']}/tokenize",
                    data=json.dumps({"content": text}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with opener.open(tok_req, timeout=4.0) as resp:
                    obj = json.loads(resp.read().decode("utf-8"))
                    toks = obj.get("tokens")
                    if isinstance(toks, list):
                        return max(1, len(toks))
            except Exception:
                pass
        elif bundle.get("processor") is not None:
            try:
                return max(1, len(bundle["processor"].tokenizer.encode(text)))
            except Exception:
                pass
    return max(1, len(text) // 3)


def _run_qwen_llm_job(job_id: str, req: GenerateVideoRequest, target_llm_id: Optional[str] = None):
    """运行 Qwen3.8-27B (BF16 原生 或 Unsloth UD-Q4_K_M GGUF) 深度推理对话与多模态视觉理解"""
    global QWEN_CONTEXT_LENGTH
    start_time = time.time()
    if req.context_length is not None:
        QWEN_CONTEXT_LENGTH = _clamp_context_length(req.context_length)

    bundle = _get_qwen_llm_bundle(preferred_model_id=target_llm_id or req.model_id)
    if bundle is None:
        req_name = MODEL_CATALOG.get(target_llm_id or req.model_id or SELECTED_MODEL_ID, {}).get("name", "大语言模型")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["message"] = f"{req_name} 尚未加载至 GPU 显存！请先在左侧点击「加载到显存」。"
        return

    bundle_mid = str(bundle.get("model_id") or target_llm_id or SELECTED_MODEL_ID)
    model_label = str(
        bundle.get("model_name")
        or MODEL_CATALOG.get(bundle_mid, {}).get("name")
        or ("Qwen3.8-27B-GGUF" if bundle.get("backend") == "gguf" else "Qwen3.8-27B")
    )
    first_img = decode_base64_image(req.first_frame_base64)
    last_img = decode_base64_image(req.last_frame_base64)
    attached_images = [im for im in (first_img, last_img) if im is not None]

    messages: List[Dict[str, Any]] = []
    if req.chat_messages:
        for m in req.chat_messages[:-1]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content:
                messages.append({"role": role, "content": [{"type": "text", "text": content}]})

    user_content: List[Dict[str, Any]] = []
    for im in attached_images:
        user_content.append({"type": "image", "image": im})
    user_content.append({"type": "text", "text": req.visual_prompt})
    messages.append({"role": "user", "content": user_content})

    JOBS[job_id]["media_type"] = "text"
    JOBS[job_id]["progress"] = 0.12
    JOBS[job_id]["message"] = f"{model_label} 正在编码多模态上下文并启动推理..."

    job_cancel_event = _JOB_CANCEL_EVENTS.setdefault(job_id, threading.Event())
    if job_cancel_event.is_set():
        raise GenerationAbortedError("已手动停止当前推理任务")
    ABORT_GENERATION_EVENT.clear()
    bundle_ctx = int(bundle.get("context_length") or QWEN_CONTEXT_LENGTH)
    max_toks = max(64, min(65536, int(req.max_new_tokens or 16384), max(256, bundle_ctx - 256)))
    think_on = bool(req.enable_thinking)
    # 遵循 Unsloth Qwen3.8 官方部署采样表：Thinking (temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence=0.0)；Non-Thinking (temp=0.7, top_p=0.80, top_k=20, min_p=0.0, presence=1.5)
    default_temp = 1.0 if think_on else 0.7
    default_top_p = 0.95 if think_on else 0.80
    default_presence = 0.0 if think_on else 1.5
    temp = max(0.0, min(2.0, float(req.temperature if req.temperature is not None else default_temp)))
    eff_top_p = max(0.05, min(1.0, float(req.top_p if req.top_p is not None else default_top_p)))
    eff_presence = max(0.0, min(2.0, float(req.presence_penalty if req.presence_penalty is not None else default_presence)))
    eff_top_k = max(0, int(req.top_k if req.top_k is not None else 20))
    eff_min_p = max(0.0, min(1.0, float(req.min_p if req.min_p is not None else 0.0)))
    eff_repeat = max(0.5, min(2.0, float(req.repeat_penalty if req.repeat_penalty is not None else 1.0)))
    eff_effort = str(req.reasoning_effort or "medium").strip().lower()

    streamer, _, start_in_thinking, gen_thread, err_box, cancel_event = _prepare_qwen_streamer_generation(
        hf_messages=messages,
        pil_images=attached_images,
        hf_tools=None,
        enable_thinking=think_on,
        max_new_tokens=max_toks,
        temperature=temp,
        top_p=eff_top_p,
        preferred_model_id=bundle.get("model_id"),
        top_k=eff_top_k,
        min_p=eff_min_p,
        presence_penalty=eff_presence,
        repeat_penalty=eff_repeat,
        reasoning_effort=eff_effort,
    )

    parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
    think_parts: List[str] = []
    ans_parts: List[str] = []
    token_chunks = 0
    last_ui_sync = 0.0

    try:
        for piece in streamer:
            if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                cancel_event.set()
                break
            token_chunks += 1
            for kind, val in parser.feed(piece):
                if kind == "thinking":
                    think_parts.append(val)
                elif kind == "text":
                    ans_parts.append(val)
            now_ts = time.time()
            # 按 ~10Hz 节流拼接流式字符串快照，避免 100+ tok/s 高速解码下逐 token O(N^2) 字符串拷贝争抢 GIL
            if token_chunks <= 2 or (now_ts - last_ui_sync) >= 0.10:
                last_ui_sync = now_ts
                think_txt = "".join(think_parts).strip()
                ans_txt = "".join(ans_parts).strip()
                JOBS[job_id]["thinking_output"] = think_txt
                JOBS[job_id]["text_output"] = ans_txt
                JOBS[job_id]["progress"] = min(0.95, round(0.15 + 0.80 * min(1.0, token_chunks / float(max(128.0, max_toks * 0.5))), 3))
                elapsed_now = round(now_ts - start_time, 1)
                JOBS[job_id]["elapsed"] = elapsed_now
                tps = streamer.tok_per_sec
                speed_tag = f" · {tps} tok/s" if tps > 0 else ""
                JOBS[job_id]["message"] = (
                    f"{model_label} 正在深度思考 ({elapsed_now}s{speed_tag})..."
                    if (think_txt and not ans_txt)
                    else f"{model_label} 正在流式输出回复 ({elapsed_now}s{speed_tag})..."
                )
                touch_activity()
    except Exception:
        cancel_event.set()
    finally:
        cancel_event.set()

    tail_events, _ = parser.finish()
    for kind, val in tail_events:
        if kind == "thinking":
            think_parts.append(val)
        elif kind == "text":
            ans_parts.append(val)

    gen_thread.join(timeout=10.0)
    if err_box[0] is not None:
        raise err_box[0]

    think_final = "".join(think_parts).strip()
    ans_final = "".join(ans_parts).strip()
    if req.enable_thinking and not think_final and "</think>" in ans_final:
        think_final, ans_final = _split_thinking_and_answer("<think>\n" + ans_final)

    if job_cancel_event.is_set() or (ABORT_GENERATION_EVENT.is_set() and not (think_final or ans_final)):
        raise GenerationAbortedError("已手动停止当前推理任务")

    elapsed = round(time.time() - start_time, 1)
    final_tps = streamer.tok_per_sec
    gen_toks = streamer.generated_tokens
    tps_summary = f" · 共 {gen_toks} tokens · {final_tps} tok/s" if gen_toks > 0 and final_tps > 0 else ""
    JOBS[job_id]["thinking_output"] = think_final
    JOBS[job_id]["text_output"] = ans_final
    JOBS[job_id]["status"] = "completed"
    JOBS[job_id]["progress"] = 1.0
    JOBS[job_id]["elapsed"] = elapsed
    JOBS[job_id]["message"] = f"{model_label} 推理完成！耗时 {elapsed}s{tps_summary}"
    print(f"[{model_label}] 推理完成 | {gen_toks} tokens | {final_tps} tok/s | 耗时 {elapsed}s")


def _fmt_mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"


from contextlib import contextmanager


@contextmanager
def _pipeline_job_execution_guard(
    pipe,
    job_id: str,
    job_cancel_event: threading.Event,
    start_time: float,
    base_prog: float,
    span_prog: float,
    prefix: str,
    model_name: str,
    media_type: str,
):
    """
    为所有 DiffSynth 视频/生图管线注入全链路真实进度与毫秒级 GPU 中断守卫：
    1. 挂载层块级 forward_pre_hook：点击停止推理后在当前 Transformer Block 结束时 (<0.1s) 立即抛出 GenerationAbortedError，
       瞬时释放 GPU_LOCK，彻底杜绝僵尸任务阻塞新任务。
    2. 接管 progress_bar_cmd 与阶段切换 load_models_to_device：
       - 0.04 ~ 0.08: 多模态提示词编码与潜空间初始化
       - 0.08 ~ 0.82 (图像 0.90): DiT 逐步去噪采样（每步实时刷新百分比与剩余时间，单行刷新不堆叠）
       - 0.84 ~ 0.93: 3D 视频 VAE 与 32kHz 立体声 VAE 解码（明确区分采样完成与后处理解码，绝不提前报 100%）
       - 0.94 ~ 0.99: MP4 音轨合成与落盘
    """
    global CURRENT_STAGE_BAR_LABEL
    import types

    dit_end = 0.90 if media_type == "image" else 0.82
    hook_handles = []
    orig_load_fn = getattr(pipe, "load_models_to_device", None)

    def _check_abort():
        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            raise GenerationAbortedError("已手动停止当前推理任务")

    def _abort_pre_hook(_mod, _inp):
        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            raise GenerationAbortedError("已手动停止当前推理任务")

    # 为管线主干及内部 Transformer Blocks 挂载轻量级中断钩子，使长耗时单步也能在 ~100ms 内立刻中止
    try:
        seen_mods = set()
        for _, child in pipe.named_children():
            if child is None or not isinstance(child, torch.nn.Module):
                continue
            if id(child) not in seen_mods:
                seen_mods.add(id(child))
                hook_handles.append(child.register_forward_pre_hook(_abort_pre_hook))
            for sub_name, sub_mod in child.named_modules():
                if isinstance(sub_mod, torch.nn.ModuleList) and any(
                    k in sub_name for k in ("blocks", "layers")
                ):
                    for blk in sub_mod:
                        if isinstance(blk, torch.nn.Module) and id(blk) not in seen_mods:
                            seen_mods.add(id(blk))
                            hook_handles.append(blk.register_forward_pre_hook(_abort_pre_hook))
    except Exception:
        pass

    if orig_load_fn is not None:
        def _stage_aware_load_models(self_pipe, model_names):
            global CURRENT_STAGE_BAR_LABEL
            _check_abort()
            names_set = set(model_names or ())
            if "text_encoder" in names_set:
                CURRENT_STAGE_BAR_LABEL = "多模态提示词编码"
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + 0.05 * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = f"{prefix}{model_name} 正在使用多模态编码器解析提示词 (5%)..."
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = "提示词编码:   5%|#---------| 正在解析画面、运镜与声音提示词..."
            elif "video_vae" in names_set or "vae" in names_set:
                v_frac = 0.92 if media_type == "image" else 0.85
                lbl = "高清图像 VAE 解码" if media_type == "image" else "3D 视频 VAE 解码"
                CURRENT_STAGE_BAR_LABEL = lbl
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + v_frac * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = f"{prefix}DiT 采样已完成，正在执行{lbl} ({int(v_frac * 100)}%)..."
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = f"{lbl}: {int(v_frac * 100):3d}%|########--| 正在将潜空间张量解码为高清像素..."
            elif "audio_vae" in names_set:
                CURRENT_STAGE_BAR_LABEL = "32kHz 立体声解码"
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + 0.92 * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = f"{prefix}视频帧解码完成，正在执行 32kHz 立体声音轨解码 (92%)..."
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = "32kHz 立体声解码:  92%|#########-| 正在解码双声道环境音与配乐波形..."
            res = orig_load_fn(model_names)
            _check_abort()
            touch_activity()
            return res

        pipe.load_models_to_device = types.MethodType(_stage_aware_load_models, pipe)

    def _job_progress_bar(iterable, *_args, **_kwargs):
        global CURRENT_STAGE_BAR_LABEL
        _check_abort()
        items = list(iterable)
        total = max(1, len(items))
        loop_t0 = time.time()
        CURRENT_STAGE_BAR_LABEL = None

        if job_id in JOBS:
            JOBS[job_id]["progress"] = round(base_prog + 0.08 * span_prog, 3)
            JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
            JOBS[job_id]["message"] = f"{prefix}{model_name} 正在执行去噪采样第 1/{total} 步 (8%)..."
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS["sampling"] = f"DiT 去噪采样:   0%|----------| 0/{total} [正在执行第 1/{total} 步...]"

        for idx, item in enumerate(items):
            _check_abort()
            yield item
            _check_abort()
            done = idx + 1
            step_frac = done / float(total)
            dt = max(0.01, time.time() - loop_t0)
            avg_s = dt / float(done)
            eta_s = max(0.0, avg_s * (total - done))
            overall_frac = 0.08 + (dit_end - 0.08) * step_frac
            fill = max(0, min(10, int(round(10 * step_frac))))
            touch_activity()

            if done < total:
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + overall_frac * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = (
                        f"{prefix}{model_name} 正在采样 {done}/{total} 步 "
                        f"(总进度 {int(overall_frac * 100)}% · {avg_s:.1f}s/步 · 采样剩余约 {int(round(eta_s))}s)..."
                    )
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = (
                        f"DiT 去噪采样: {int(step_frac * 100):3d}%|{'#' * fill}{'-' * (10 - fill)}| "
                        f"{done}/{total} [{_fmt_mmss(dt)}<{_fmt_mmss(eta_s)}, {avg_s:.2f}s/step] · 总进度 {int(overall_frac * 100)}%"
                    )
            else:
                post_frac = dit_end + 0.02
                next_lbl = "高清图像 VAE 解码" if media_type == "image" else "3D 视频与音频 VAE 解码"
                CURRENT_STAGE_BAR_LABEL = next_lbl
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + post_frac * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = (
                        f"{prefix}DiT 采样已完成 ({total}/{total})，正在进入{next_lbl} ({int(post_frac * 100)}%)..."
                    )
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = (
                        f"{next_lbl}: {int(post_frac * 100):3d}%|########--| "
                        f"DiT 采样已完成 ({total}/{total}，耗时 {_fmt_mmss(dt)}) -> 正在解码输出..."
                    )
                print(f"[{model_name}] DiT 采样 {total}/{total} 步完成 ({dt:.1f}s)，开始执行{next_lbl}...")

    try:
        yield _job_progress_bar
    finally:
        CURRENT_STAGE_BAR_LABEL = None
        if orig_load_fn is not None:
            pipe.load_models_to_device = orig_load_fn
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass


def _run_h3_image_job(
    job_id: str,
    req: GenerateVideoRequest,
    model_id: str,
    batch_count: int,
    job_cancel_event: threading.Event,
):
    """MiniMax-H3-Image：把请求转发给本机 SGLang 服务，结果落盘为 PNG。模型固定 6 步、无 CFG、不支持负面提示词。"""
    spec = MODEL_CATALOG[model_id]
    proc = (LOADED_RUNTIMES.get(model_id) or {}).get("server_proc")
    if proc is None or proc.poll() is not None:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["message"] = f"{spec['name']} 尚未加载到 GPU 显存！请先在左侧点击「加载到显存」。"
        return

    refs = []
    if req.task_mode == "i2i_edit":
        ref_sources = [req.first_frame_base64, req.last_frame_base64] + list(req.extra_reference_images_base64 or [])
        refs = [im for im in (decode_base64_image(b64) for b64 in ref_sources) if im is not None][:H3_IMAGE_MAX_REFERENCES]
        if not refs:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["message"] = "图像编辑模式下请至少拖入一张参考图"
            return

    JOBS[job_id]["media_type"] = "image"
    base_seed = req.seed
    if req.randomize_seed or base_seed < 0:
        base_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    base_url = f"http://127.0.0.1:{H3_IMAGE_SGLANG_PORT}"
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    start_time = time.time()
    filename = None
    current_seed = base_seed

    for batch_idx in range(batch_count):
        if job_cancel_event.is_set():
            raise GenerationAbortedError("已手动停止当前推理任务")
        if batch_idx > 0:
            current_seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if req.randomize_seed else base_seed + batch_idx
        JOBS[job_id]["seed"] = current_seed
        prefix = f"[{batch_idx + 1}/{batch_count}] " if batch_count > 1 else ""

        with GPU_LOCK:
            sub_start = time.time()
            touch_activity()
            JOBS[job_id]["progress"] = round((batch_idx + 0.3) / float(batch_count), 3)
            JOBS[job_id]["message"] = f"{prefix}{spec['name']} 正在 6 步采样并解码 {req.width}×{req.height} 图像 (Seed: {current_seed})..."
            # 参考图在提示词里按上传顺序称作 <Picture 1>、<Picture 2>
            fields = {
                "prompt": req.visual_prompt,
                "size": f"{int(req.width)}x{int(req.height)}",
                "seed": int(current_seed),
                "generator_device": "cpu",
                "response_format": "b64_json",
            }
            if refs:
                boundary = uuid.uuid4().hex
                body = io.BytesIO()
                for k, v in fields.items():
                    body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode("utf-8"))
                for i, im in enumerate(refs):
                    png = io.BytesIO()
                    im.save(png, format="PNG")
                    body.write(
                        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="ref{i + 1}.png"\r\n'
                        f"Content-Type: image/png\r\n\r\n".encode("utf-8")
                    )
                    body.write(png.getvalue())
                    body.write(b"\r\n")
                body.write(f"--{boundary}--\r\n".encode("utf-8"))
                http_req = urllib.request.Request(
                    base_url + "/v1/images/edits",
                    data=body.getvalue(),
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                )
            else:
                http_req = urllib.request.Request(
                    base_url + "/v1/images/generations",
                    data=json.dumps(fields).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
            try:
                with urllib.request.urlopen(http_req, timeout=900) as resp:
                    result = json.loads(resp.read())
            except urllib.error.HTTPError as http_err:
                raise RuntimeError(f"SGLang 返回 HTTP {http_err.code}: {http_err.read()[:400].decode('utf-8', 'ignore')}")

            mode_tag = "edit" if refs else "t2i"
            filename = f"minimax_h3_image_{mode_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{current_seed}.png"
            with open(os.path.join(DEFAULT_OUTPUT_DIR, filename), "wb") as f:
                f.write(base64.b64decode(result["data"][0]["b64_json"]))

        sub_elapsed = round(time.time() - sub_start, 1)
        print(f"[{spec['name']}] 图像生成完成 | 耗时 {sub_elapsed}s | Seed: {current_seed} | 文件: {filename}")
        JOBS[job_id]["completed_videos"] = list(JOBS[job_id]["completed_videos"]) + [{
            "index": batch_idx + 1,
            "seed": current_seed,
            "filename": filename,
            "video_url": f"/minimax_api/video/{filename}",
            "media_type": "image",
            "elapsed": sub_elapsed,
        }]
        JOBS[job_id]["completed_count"] = batch_idx + 1
        JOBS[job_id]["filename"] = filename
        JOBS[job_id]["video_url"] = f"/minimax_api/video/{filename}"
        JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
        touch_activity()

    elapsed = round(time.time() - start_time, 1)
    JOBS[job_id]["status"] = "completed"
    JOBS[job_id]["progress"] = 1.0
    JOBS[job_id]["elapsed"] = elapsed
    JOBS[job_id]["message"] = (
        f"批量生成 {batch_count} 项全部完成！总耗时 {elapsed}s"
        if batch_count > 1
        else f"生成完成！耗时 {elapsed}s (Seed: {current_seed})"
    )


def _run_async_generation_job(job_id: str, req: GenerateVideoRequest):
    touch_activity()
    job_cancel_event = _JOB_CANCEL_EVENTS.setdefault(job_id, threading.Event())
    active_model_id = (
        req.model_id
        if (req.model_id and req.model_id in MODEL_CATALOG)
        else (SELECTED_MODEL_ID if SELECTED_MODEL_ID in LOADED_RUNTIMES else (LOADED_MODEL_ID or SELECTED_MODEL_ID))
    )
    batch_count = max(1, min(8, int(getattr(req, "batch_count", 1) or 1)))

    JOBS[job_id]["model_id"] = active_model_id
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["progress"] = 0.01
    JOBS[job_id]["batch_count"] = batch_count
    JOBS[job_id]["completed_count"] = 0
    JOBS[job_id]["completed_videos"] = []
    start_time = time.time()
    JOBS[job_id]["started_at"] = start_time

    try:
        if job_cancel_event.is_set():
            raise GenerationAbortedError("已手动停止当前推理任务")

        # 分支 A：Qwen3.8-27B / Qwen3.8-27B-GGUF 视觉理解与深度推理对话
        if _is_llm_model_id(active_model_id):
            _run_qwen_llm_job(job_id, req, target_llm_id=active_model_id)
            return

        if MODEL_CATALOG.get(active_model_id, {}).get("pipeline_arch") == "minimax_h3_image":
            _run_h3_image_job(job_id, req, active_model_id, batch_count, job_cancel_event)
            return

        rt = LOADED_RUNTIMES.get(active_model_id)
        active_pipe = rt.get("pipeline") if rt else (PIPELINE if LOADED_MODEL_ID == active_model_id else None)
        if active_pipe is None:
            req_name = MODEL_CATALOG.get(active_model_id, {}).get("name", active_model_id)
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["message"] = f"{req_name} 尚未加载到 GPU 显存！请先在左侧点击「加载到显存」（显存充裕时可与现有模型同时常驻）。"
            return

        spec = MODEL_CATALOG.get(active_model_id, MODEL_CATALOG["minimax_h3"])
        pipeline_arch = spec.get("pipeline_arch") or active_model_id
        modality = spec["modality"]
        media_type = "image" if modality == "image" else "video"
        JOBS[job_id]["media_type"] = media_type
        JOBS[job_id]["progress"] = 0.02
        JOBS[job_id]["message"] = (
            f"正在准备 {spec['name']} 推理管线 (共 {batch_count} 项)..."
            if batch_count > 1
            else f"正在准备 {spec['name']} 推理管线..."
        )

        base_seed = req.seed
        if req.randomize_seed or base_seed < 0:
            base_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        JOBS[job_id]["seed"] = base_seed

        first_img = decode_base64_image(req.first_frame_base64)
        last_img = decode_base64_image(req.last_frame_base64)

        keyframes = []
        keyframe_indices = []
        h3_references = None
        # MiniMax-H3 FL2VA 主干的三种关键帧子模式：i2va 只用首帧、l2va 只用尾帧、fl2va 首尾帧（只给一张也能跑）
        if pipeline_arch == "minimax_h3" and req.task_mode in ("fl2va", "i2va", "l2va"):
            if first_img is not None and req.task_mode != "l2va":
                keyframes.append(first_img)
                keyframe_indices.append(0)
            if last_img is not None and req.task_mode != "i2va":
                keyframes.append(last_img)
                keyframe_indices.append(-1)
            need_msg = {
                "i2va": "首帧生视频模式下请提供首帧图片",
                "l2va": "尾帧生视频模式下请提供尾帧图片",
                "fl2va": "首尾帧模式下请至少提供首帧或尾帧图片",
            }[req.task_mode]
            if not keyframes:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["message"] = need_msg
                return
        elif pipeline_arch == "minimax_h3" and req.task_mode == "ref2va":
            JOBS[job_id]["message"] = "正在解码参考图、参考视频与参考音频..."
            try:
                h3_references, ref_labels = _build_h3_ref2va_references(req)
            except ValueError as ve:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["message"] = str(ve)
                return
            print(f"[MiniMax-H3 Ref2VA] 参考素材标签：{ref_labels}")
        elif pipeline_arch == "minimax_h3" and req.task_mode != "t2va":
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["message"] = f"MiniMax-H3 不支持任务模式 {req.task_mode}"
            return
        elif modality == "image" and req.task_mode == "i2i_edit" and first_img is None and last_img is None:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["message"] = "图像编辑/重绘模式下请至少拖入一张输入图片"
            return

        # Ref2VA 与 FL2VA 只差 DiT 主干：显存里的主干和本次任务不一致时，先（按需下载并）重新装载对应主干
        if pipeline_arch == "minimax_h3":
            want_variant = "Ref2VA" if req.task_mode == "ref2va" else "FL2VA"
            have_variant = (rt or {}).get("h3_dit_variant") or "FL2VA"
            if want_variant != have_variant:
                if want_variant == "Ref2VA":
                    _ensure_h3_ref2va_dit_downloaded(job_id)
                if job_cancel_event.is_set():
                    raise GenerationAbortedError("已手动停止当前推理任务")
                JOBS[job_id]["progress"] = 0.12
                JOBS[job_id]["message"] = (
                    f"正在把显存里的 {have_variant} 主干换成 {want_variant} 主干并重新装载（几分钟，左侧状态栏可看装载进度）..."
                )
                ok_sw, msg_sw = load_model_to_vram_core("minimax_h3", h3_dit_variant=want_variant)
                rt = LOADED_RUNTIMES.get("minimax_h3")
                active_pipe = rt.get("pipeline") if (ok_sw and rt) else None
                if active_pipe is None:
                    JOBS[job_id]["status"] = "failed"
                    JOBS[job_id]["message"] = f"切换到 {want_variant} 主干失败：{msg_sw}"
                    return

        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        last_filename = None
        last_seed = base_seed

        for batch_idx in range(batch_count):
            if job_cancel_event.is_set():
                raise GenerationAbortedError("已手动停止当前推理任务")

            if batch_idx == 0:
                current_seed = base_seed
            elif req.randomize_seed:
                current_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            else:
                current_seed = base_seed + batch_idx

            last_seed = current_seed
            JOBS[job_id]["seed"] = current_seed
            prefix = f"[{batch_idx + 1}/{batch_count}] " if batch_count > 1 else ""
            base_prog = batch_idx / float(batch_count)
            span_prog = 1.0 / float(batch_count)

            if GPU_LOCK.locked():
                JOBS[job_id]["progress"] = round(base_prog + 0.01 * span_prog, 3)
                JOBS[job_id]["message"] = f"{prefix}正在等待上一任务释放 GPU 显存锁..."

            with GPU_LOCK:
                if job_cancel_event.is_set():
                    raise GenerationAbortedError("已手动停止当前推理任务")
                ABORT_GENERATION_EVENT.clear()

                # 获取 GPU 锁后重置本轮起始计时并清空所有旧进度条，确保新任务严格从 0% 与 0.0s 起步
                sub_start = time.time()
                if batch_idx == 0:
                    start_time = sub_start
                    JOBS[job_id]["started_at"] = start_time
                    JOBS[job_id]["elapsed"] = 0.0
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS.clear()

                JOBS[job_id]["progress"] = round(base_prog + 0.04 * span_prog, 3)
                JOBS[job_id]["message"] = (
                    f"{prefix}{spec['name']} 正在初始化潜空间并准备 {req.num_inference_steps} 步推理 (Seed: {current_seed})..."
                )
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                touch_activity()
                _reclaim_transient_gpu_memory()

                with _pipeline_job_execution_guard(
                    pipe=active_pipe,
                    job_id=job_id,
                    job_cancel_event=job_cancel_event,
                    start_time=start_time,
                    base_prog=base_prog,
                    span_prog=span_prog,
                    prefix=prefix,
                    model_name=spec["name"],
                    media_type=media_type,
                ) as step_progress_bar:
                    if pipeline_arch == "minimax_h3":
                        from diffsynth.utils.data.audio_video import write_video_audio
                        full_prompt = build_structured_prompt(
                            req.visual_prompt, req.soundscape_prompt, req.music_prompt
                        )
                        filename = f"minimax_h3_{req.task_mode}_{timestamp}_seed{current_seed}.mp4"
                        output_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
                        call_kwargs = dict(
                            prompt=full_prompt,
                            height=int(req.height),
                            width=int(req.width),
                            num_frames=int(req.num_frames),
                            num_inference_steps=int(req.num_inference_steps),
                            seed=int(current_seed),
                            progress_bar_cmd=step_progress_bar,
                        )
                        if keyframes:
                            call_kwargs["keyframes"] = keyframes
                            call_kwargs["keyframe_indices"] = keyframe_indices
                        if h3_references:
                            call_kwargs["references"] = h3_references
                        video, audio = _run_pipeline_with_smart_tiling(active_pipe, call_kwargs)
                        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            raise GenerationAbortedError("已手动停止当前推理任务")
                        JOBS[job_id]["progress"] = round(base_prog + 0.95 * span_prog, 3)
                        JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                        JOBS[job_id]["message"] = f"{prefix}VAE 解码完成，正在合成 32kHz 立体声音轨并封装 MP4 (95%)..."
                        with CONSOLE_LOCK:
                            ACTIVE_TQDM_BARS["sampling"] = "音视频封装:  95%|#########-| 正在编码 H.264 视频与 32kHz 立体声 MP4..."
                        write_video_audio(
                            video=video,
                            audio=audio,
                            output_path=output_path,
                            fps=int(req.fps),
                            audio_sample_rate=32000,
                        )
                        video = None
                        audio = None

                    elif pipeline_arch == "qwen_image_21":
                        filename = f"qwen_image21_{req.task_mode}_{timestamp}_seed{current_seed}.png"
                        output_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
                        edit_imgs = [im for im in (first_img, last_img) if im is not None] if req.task_mode == "i2i_edit" else None
                        img_out = active_pipe(
                            prompt=req.visual_prompt,
                            negative_prompt=req.negative_prompt.strip() or " ",
                            edit_image=edit_imgs if edit_imgs else None,
                            height=int(req.height),
                            width=int(req.width),
                            num_inference_steps=int(req.num_inference_steps),
                            seed=int(current_seed),
                            progress_bar_cmd=step_progress_bar,
                        )
                        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            raise GenerationAbortedError("已手动停止当前推理任务")
                        JOBS[job_id]["progress"] = round(base_prog + 0.96 * span_prog, 3)
                        JOBS[job_id]["message"] = f"{prefix}正在写入无损 PNG 图像文件 (96%)..."
                        img_out.save(output_path)
                        img_out = None
                        edit_imgs = None

                    elif pipeline_arch == "z_image_turbo":
                        filename = f"z_image_turbo_{req.task_mode}_{timestamp}_seed{current_seed}.png"
                        output_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
                        w, h = int(req.width), int(req.height)
                        call_kw = dict(
                            prompt=req.visual_prompt,
                            negative_prompt=req.negative_prompt or "",
                            height=h,
                            width=w,
                            num_inference_steps=int(req.num_inference_steps),
                            seed=int(current_seed),
                            rand_device="cuda",
                            progress_bar_cmd=step_progress_bar,
                        )
                        if req.task_mode == "i2i_edit" and first_img is not None:
                            call_kw["input_image"] = first_img.resize((w, h))
                            call_kw["denoising_strength"] = max(0.15, min(1.0, float(req.denoising_strength or 0.70)))
                        img_out = active_pipe(**call_kw)
                        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            raise GenerationAbortedError("已手动停止当前推理任务")
                        JOBS[job_id]["progress"] = round(base_prog + 0.96 * span_prog, 3)
                        JOBS[job_id]["message"] = f"{prefix}正在写入无损 PNG 图像文件 (96%)..."
                        img_out.save(output_path)
                        img_out = None
                        call_kw.clear()

                    elif pipeline_arch == "diffusers_image":
                        filename = f"{active_model_id}_{req.task_mode}_{timestamp}_seed{current_seed}.png"
                        output_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
                        w, h = int(req.width), int(req.height)
                        gen = torch.Generator(device="cuda").manual_seed(int(current_seed))
                        out_res = active_pipe(
                            prompt=req.visual_prompt,
                            height=h,
                            width=w,
                            num_inference_steps=int(req.num_inference_steps),
                            generator=gen,
                        )
                        img_out = out_res.images[0] if hasattr(out_res, "images") else out_res
                        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            raise GenerationAbortedError("已手动停止当前推理任务")
                        JOBS[job_id]["progress"] = round(base_prog + 0.96 * span_prog, 3)
                        JOBS[job_id]["message"] = f"{prefix}正在写入无损 PNG 图像文件 (96%)..."
                        img_out.save(output_path)
                        img_out = None

                    else:
                        raise ValueError(f"未知模型 ID: {active_model_id}")

                if not os.path.exists(output_path) or os.path.getsize(output_path) < 256:
                    raise RuntimeError("输出媒体文件未成功生成或写入为空")

                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS.clear()
                _reclaim_transient_gpu_memory()

            sub_elapsed = round(time.time() - sub_start, 1)
            last_filename = filename
            kind_cn = "图像" if media_type == "image" else "视频"
            print(f"[{spec['name']}] {kind_cn}生成完成 | 耗时 {sub_elapsed}s | Seed: {current_seed} | 文件: {filename}")
            media_item = {
                "index": batch_idx + 1,
                "seed": current_seed,
                "filename": filename,
                "video_url": f"/minimax_api/video/{filename}",
                "media_type": media_type,
                "elapsed": sub_elapsed,
            }
            JOBS[job_id]["completed_videos"] = list(JOBS[job_id]["completed_videos"]) + [media_item]
            JOBS[job_id]["completed_count"] = batch_idx + 1
            JOBS[job_id]["filename"] = filename
            JOBS[job_id]["video_url"] = f"/minimax_api/video/{filename}"
            JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
            if batch_count > 1 and batch_idx + 1 < batch_count:
                JOBS[job_id]["progress"] = round((batch_idx + 1) / float(batch_count), 3)
                JOBS[job_id]["message"] = (
                    f"第 {batch_idx + 1}/{batch_count} 项已完成 ({sub_elapsed}s)，已同步回传 Mac！继续生成第 {batch_idx + 2} 项..."
                )
            touch_activity()

        elapsed = round(time.time() - start_time, 1)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 1.0
        JOBS[job_id]["elapsed"] = elapsed
        JOBS[job_id]["filename"] = last_filename
        JOBS[job_id]["video_url"] = f"/minimax_api/video/{last_filename}" if last_filename else None
        if batch_count > 1:
            JOBS[job_id]["message"] = (
                f"批量生成 {batch_count} 项全部完成！总耗时 {elapsed}s (平均 {round(elapsed / batch_count, 1)}s/项)"
            )
        else:
            JOBS[job_id]["message"] = f"生成完成！耗时 {elapsed}s (Seed: {last_seed})"

    except GenerationAbortedError as abort_err:
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()
        _reclaim_transient_gpu_memory()
        if job_id in JOBS:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["progress"] = 0.0
            JOBS[job_id]["message"] = str(abort_err) or "已手动停止当前推理任务"
        print(f"[任务控制] 任务 {job_id} 已中断并释放 GPU 显存锁。")
    except Exception as e:
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()
        _reclaim_transient_gpu_memory()
        if job_id in JOBS:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["message"] = f"生成出错: {str(e)}"
    finally:
        _JOB_CANCEL_EVENTS.pop(job_id, None)
        touch_activity()


class ApiKeyConfigRequest(BaseModel):
    api_key: str = ""
    initialize_only: bool = False


class UnloadModelRequest(BaseModel):
    model_id: str = ""


class RegisterCustomModelRequest(BaseModel):
    repo_id: str
    include_pattern: str = ""
    custom_name: str = ""
    modality: str = "auto"
    modality_hint: str = "auto"
    hf_token: str = ""
    auto_download: bool = False


class RemoveCustomModelRequest(BaseModel):
    model_id: str
    delete_files: bool = False


# ==============================================================================
# 标准对外网关：OpenAI (/v1/chat/completions, /v1/models, /v1/images/generations)
#            与 Anthropic (/v1/messages, /v1/messages/count_tokens) 双协议兼容引擎
# ==============================================================================
def _verify_external_api_auth(request: Request, protocol: str = "openai") -> Optional[JSONResponse]:
    """同时支持 Authorization: Bearer <key> 与 x-api-key: <key> 鉴权"""
    expected = (API_AUTH_KEY or "").strip()
    auth_hdr = (request.headers.get("authorization") or "").strip()
    bearer_tok = auth_hdr[7:].strip() if auth_hdr.lower().startswith("bearer ") else ""
    x_api_tok = (request.headers.get("x-api-key") or "").strip()
    provided = bearer_tok or x_api_tok
    if expected and secrets.compare_digest(provided.encode(), expected.encode()):
        return None
    if protocol == "anthropic":
        return JSONResponse(
            status_code=401,
            content={"type": "error", "error": {"type": "authentication_error", "message": "Invalid API Key"}},
        )
    return JSONResponse(
        status_code=401,
        content={"error": {"message": "Invalid API Key", "type": "invalid_request_error", "code": "invalid_api_key"}},
    )


def _load_pil_from_url_or_b64(src: str) -> Optional[Image.Image]:
    if not src:
        return None
    src = src.strip()
    if src.startswith("http://") or src.startswith("https://"):
        req = urllib.request.Request(src, headers={"User-Agent": "Qwen3.8-API-Gateway/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    return decode_base64_image(src)


def _extract_qwen_tool_calls(raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """解析 Qwen3 / Qwen3.8 输出中的 <tool_call>...</tool_call> (兼容 JSON 与 XML 参数格式)"""
    if "<tool_call>" not in raw_text:
        return raw_text, []

    tool_calls: List[Dict[str, Any]] = []
    tc_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    for match in tc_pattern.finditer(raw_text):
        inner = match.group(1).strip()
        call_id = f"toolu_{uuid.uuid4().hex[:16]}"
        parsed_name = None
        parsed_args: Dict[str, Any] = {}

        # 格式 1: JSON {"name": "...", "arguments": {...}}
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict) and "name" in obj:
                parsed_name = str(obj["name"])
                raw_args = obj.get("arguments", obj.get("parameters", {}))
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except Exception:
                        parsed_args = {"raw": raw_args}
                elif isinstance(raw_args, dict):
                    parsed_args = raw_args
        except Exception:
            pass

        # 格式 2: Qwen3.5/3.8 XML <function=fn_name><parameter=k>v</parameter></function>
        if not parsed_name:
            fn_m = re.search(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", inner, re.DOTALL)
            if fn_m:
                parsed_name = fn_m.group(1).strip()
                body = fn_m.group(2)
                for pm in re.finditer(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", body, re.DOTALL):
                    p_key = pm.group(1).strip()
                    p_val_raw = pm.group(2).strip()
                    try:
                        parsed_args[p_key] = json.loads(p_val_raw)
                    except Exception:
                        parsed_args[p_key] = p_val_raw

        if parsed_name:
            tool_calls.append({"id": call_id, "name": parsed_name, "arguments": parsed_args})

    cleaned_text = tc_pattern.sub("", raw_text).strip()
    return cleaned_text, tool_calls


class _IncrementalQwenStreamParser:
    """流式增量拆分 <think>...</think>、正文与 <tool_call>...</tool_call>，杜绝半截标签外泄"""

    def __init__(self, start_in_thinking: bool):
        self.in_thinking = start_in_thinking
        self.in_tool_call = False
        self.buf = ""
        self.tool_raw = ""

    def feed(self, piece: str) -> List[Tuple[str, str]]:
        events: List[Tuple[str, str]] = []
        self.buf += piece

        while self.buf:
            if self.in_tool_call:
                self.tool_raw += self.buf
                self.buf = ""
                break

            if self.in_thinking:
                # 忽略开头的显式 <think>
                if self.buf.lstrip().startswith("<think>"):
                    idx = self.buf.find("<think>")
                    self.buf = self.buf[idx + len("<think>"):]
                    continue
                end_idx = self.buf.find("</think>")
                if end_idx != -1:
                    before = self.buf[:end_idx]
                    if before:
                        events.append(("thinking", before))
                    self.buf = self.buf[end_idx + len("</think>"):].lstrip("\n")
                    self.in_thinking = False
                    continue
                # 保留末尾可能未闭合的 '<' 前缀
                lt = self.buf.rfind("<")
                if lt != -1 and len(self.buf) - lt < 12 and "</think>".startswith(self.buf[lt:]):
                    safe = self.buf[:lt]
                    if safe:
                        events.append(("thinking", safe))
                    self.buf = self.buf[lt:]
                    break
                events.append(("thinking", self.buf))
                self.buf = ""
            else:
                if self.buf.lstrip().startswith("<think>"):
                    idx = self.buf.find("<think>")
                    before = self.buf[:idx]
                    if before:
                        events.append(("text", before))
                    self.buf = self.buf[idx + len("<think>"):]
                    self.in_thinking = True
                    continue

                tc_idx = self.buf.find("<tool_call>")
                if tc_idx != -1:
                    before = self.buf[:tc_idx]
                    if before:
                        events.append(("text", before))
                    self.tool_raw += self.buf[tc_idx:]
                    self.buf = ""
                    self.in_tool_call = True
                    break

                lt = self.buf.rfind("<")
                if lt != -1 and len(self.buf) - lt < 14 and (
                    "<tool_call>".startswith(self.buf[lt:]) or "<think>".startswith(self.buf[lt:])
                ):
                    safe = self.buf[:lt]
                    if safe:
                        events.append(("text", safe))
                    self.buf = self.buf[lt:]
                    break

                events.append(("text", self.buf))
                self.buf = ""

        return events

    def finish(self) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
        events: List[Tuple[str, str]] = []
        if self.buf:
            if self.in_tool_call:
                self.tool_raw += self.buf
            elif self.in_thinking:
                events.append(("thinking", self.buf))
            else:
                events.append(("text", self.buf))
            self.buf = ""

        _, tool_calls = _extract_qwen_tool_calls(self.tool_raw) if self.tool_raw else ("", [])
        return events, tool_calls


def _normalize_to_hf_messages_and_images(
    raw_messages: List[Dict[str, Any]],
    system_prompt: Any = None,
    protocol: str = "openai",
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """将 OpenAI 或 Anthropic 消息数组统一转换为 Qwen3.8 Processor 标准多模态结构"""
    hf_messages: List[Dict[str, Any]] = []
    pil_images: List[Image.Image] = []

    if system_prompt:
        if isinstance(system_prompt, str) and system_prompt.strip():
            hf_messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]})
        elif isinstance(system_prompt, list):
            sys_txt = "\n".join(
                b.get("text", "") for b in system_prompt if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            if sys_txt:
                hf_messages.append({"role": "system", "content": [{"type": "text", "text": sys_txt}]})

    for msg in raw_messages or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "tool":
            txt = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            hf_messages.append({"role": "tool", "content": txt})
            continue

        blocks: List[Dict[str, Any]] = []
        if isinstance(content, str):
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype == "text":
                    blocks.append({"type": "text", "text": str(part.get("text", ""))})
                elif ptype == "image_url":
                    img_obj = part.get("image_url", {})
                    url_str = img_obj.get("url", "") if isinstance(img_obj, dict) else str(img_obj)
                    im = _load_pil_from_url_or_b64(url_str)
                    if im is not None:
                        pil_images.append(im)
                        blocks.append({"type": "image", "image": im})
                elif ptype == "image":
                    src = part.get("source", {})
                    im = None
                    if isinstance(src, dict):
                        if src.get("type") == "base64":
                            im = decode_base64_image(src.get("data", ""))
                        elif src.get("type") == "url":
                            im = _load_pil_from_url_or_b64(src.get("url", ""))
                    if im is not None:
                        pil_images.append(im)
                        blocks.append({"type": "image", "image": im})
                elif ptype == "tool_use":
                    tc_str = json.dumps({"name": part.get("name", ""), "arguments": part.get("input", {})}, ensure_ascii=False)
                    blocks.append({"type": "text", "text": f"<tool_call>\n{tc_str}\n</tool_call>"})
                elif ptype == "tool_result":
                    tr_content = part.get("content", "")
                    if isinstance(tr_content, list):
                        tr_txt = "\n".join(x.get("text", "") for x in tr_content if isinstance(x, dict))
                    else:
                        tr_txt = str(tr_content)
                    hf_messages.append({"role": "tool", "content": tr_txt})

        # OpenAI assistant historical tool_calls
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                raw_a = fn.get("arguments", "{}")
                try:
                    a_obj = json.loads(raw_a) if isinstance(raw_a, str) else raw_a
                except Exception:
                    a_obj = {}
                tc_json = json.dumps({"name": fn.get("name", ""), "arguments": a_obj}, ensure_ascii=False)
                blocks.append({"type": "text", "text": f"<tool_call>\n{tc_json}\n</tool_call>"})

        if blocks:
            hf_messages.append({"role": role, "content": blocks})

    return hf_messages, pil_images


def _normalize_hf_tools(raw_tools: Optional[List[Dict[str, Any]]], protocol: str) -> Optional[List[Dict[str, Any]]]:
    if not raw_tools:
        return None
    hf_tools: List[Dict[str, Any]] = []
    for t in raw_tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            hf_tools.append(t)
        elif "name" in t:
            # Anthropic tool schema -> HF OpenAI function tool schema
            hf_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
    return hf_tools if hf_tools else None


class _GGUFStreamIterator:
    """与 _TimedStreamer 接口一致的 GGUF 流式迭代器，统一供原生客户端与 OpenAI/Anthropic 网关消费"""

    def __init__(self, est_prompt_tokens: int = 1):
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self.prompt_tokens: int = max(1, int(est_prompt_tokens))
        self.generated_tokens: int = 0
        self.first_token_ts: Optional[float] = None
        self._server_tps: float = 0.0

    def put_text(self, text: str, token_inc: int = 1):
        if not text:
            return
        now = time.time()
        if self.first_token_ts is None:
            self.first_token_ts = now
        if token_inc > 0:
            self.generated_tokens += token_inc
        self._q.put(text)

    def end(self):
        self._q.put(None)

    @property
    def tok_per_sec(self) -> float:
        if self._server_tps > 0:
            return round(self._server_tps, 1)
        if self.first_token_ts is None or self.generated_tokens <= 1:
            return 0.0
        dt = max(0.05, time.time() - self.first_token_ts)
        return round((self.generated_tokens - 1) / dt, 1)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item


def _pil_to_jpeg_data_url(im: Image.Image) -> str:
    rgb = im.convert("RGB")
    max_side = max(rgb.size)
    if max_side > 2048:
        scale = 2048.0 / float(max_side)
        rgb = rgb.resize(
            (max(64, int(rgb.size[0] * scale)), max(64, int(rgb.size[1] * scale))),
            Image.Resampling.LANCZOS,
        )
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _hf_messages_to_gguf_openai_messages(hf_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    oai_msgs: List[Dict[str, Any]] = []
    est_chars = 0
    est_imgs = 0

    for msg in hf_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "tool":
            s_txt = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            est_chars += len(s_txt)
            oai_msgs.append({"role": "tool", "content": s_txt, "tool_call_id": msg.get("tool_call_id", "call_0")})
            continue

        if isinstance(content, str):
            est_chars += len(content)
            oai_msgs.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            has_img = any(isinstance(b, dict) and b.get("type") == "image" and b.get("image") is not None for b in content)
            if not has_img:
                txt = "".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
                est_chars += len(txt)
                oai_msgs.append({"role": role, "content": txt})
            else:
                parts: List[Dict[str, Any]] = []
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "image" and b.get("image") is not None:
                        est_imgs += 1
                        parts.append({"type": "image_url", "image_url": {"url": _pil_to_jpeg_data_url(b["image"])}})
                    elif b.get("type") == "text":
                        t_str = str(b.get("text", ""))
                        est_chars += len(t_str)
                        parts.append({"type": "text", "text": t_str})
                oai_msgs.append({"role": role, "content": parts})

    est_tokens = max(1, est_chars // 3 + est_imgs * 512)
    return oai_msgs, est_tokens


def _prepare_gguf_streamer_generation(
    bundle: Dict[str, Any],
    hf_messages: List[Dict[str, Any]],
    hf_tools: Optional[List[Dict[str, Any]]],
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: Optional[float] = None,
    repeat_penalty: float = 1.0,
    reasoning_effort: Optional[str] = None,
):
    """通过驻留 GPU 的 llama-server 流式执行 Qwen3.8-27B-UD-Q4_K_M 推理 (遵循 Unsloth 官方 GGUF 部署参数规范，支持最高 256K 上下文与视觉输入)"""
    oai_msgs, est_prompt_tokens = _hf_messages_to_gguf_openai_messages(hf_messages)
    bundle_ctx = int(bundle.get("context_length") or QWEN_CONTEXT_LENGTH)
    max_toks = max(16, min(65536, int(max_new_tokens or 16384), max(256, bundle_ctx - min(est_prompt_tokens, bundle_ctx - 256))))

    streamer = _GGUFStreamIterator(est_prompt_tokens=est_prompt_tokens)
    cancel_event = threading.Event()
    err_box: List[Optional[Exception]] = [None]

    eff_presence = (
        float(presence_penalty)
        if presence_penalty is not None
        else (0.0 if enable_thinking else 1.5)
    )
    tpl_kwargs: Dict[str, Any] = {"enable_thinking": bool(enable_thinking)}
    if enable_thinking and reasoning_effort and reasoning_effort not in ("default", "none"):
        tpl_kwargs["reasoning_effort"] = str(reasoning_effort)

    eff_temp = max(0.0, min(2.0, float(temperature)))
    eff_top_p_val = max(0.05, min(1.0, float(top_p)))
    eff_top_k_val = max(0, int(top_k))
    eff_min_p_val = max(0.0, min(1.0, float(min_p)))
    eff_pres_val = max(0.0, min(2.0, eff_presence))
    eff_rep_val = max(0.5, min(2.0, float(repeat_penalty)))

    # 针对 Qwen3.8 超大词表 (248,320) + MTP 多词元投机解码定制采样管线：
    # - 无惩罚项时 (Thinking 默认) 仅保留 GPU 原生采样器，100% 在 GPU 端完成采样，零 PCIe Logits 拷贝；
    # - 启用惩罚项时 (Non-Thinking presence_penalty=1.5) 先在 GPU 执行 top_k 将 248K 词表筛至 20 项，再计算 penalties。
    has_penalties = (eff_pres_val > 1e-5) or (abs(eff_rep_val - 1.0) > 1e-4)
    req_samplers: List[str] = []
    if eff_top_k_val > 0:
        req_samplers.append("top_k")
    if has_penalties:
        req_samplers.append("penalties")
    if eff_top_p_val < 0.9999:
        req_samplers.append("top_p")
    if eff_min_p_val > 1e-6:
        req_samplers.append("min_p")
    req_samplers.append("temperature")

    payload: Dict[str, Any] = {
        "model": "Qwen3.8-27B-UD-Q4_K_M",
        "messages": oai_msgs,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_toks,
        "temperature": eff_temp,
        "top_p": eff_top_p_val,
        "top_k": eff_top_k_val,
        "min_p": eff_min_p_val,
        "presence_penalty": eff_pres_val,
        "repeat_penalty": eff_rep_val,
        "samplers": req_samplers,
        "chat_template_kwargs": tpl_kwargs,
    }
    if hf_tools:
        payload["tools"] = hf_tools

    def _locked_gguf_generate():
        if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            streamer.end()
            return

        with GPU_LOCK:
            if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                streamer.end()
                return

            srv_proc = bundle.get("server_proc")
            if srv_proc is not None and srv_proc.poll() is not None:
                err_box[0] = RuntimeError(
                    f"llama-server 子进程已意外退出 (code={srv_proc.returncode})，请在左侧重新点击「加载到显存」。"
                )
                streamer.end()
                return

            opener = _make_noproxy_opener()
            url = f"{bundle['base_url']}/v1/chat/completions"
            req_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            http_req = urllib.request.Request(
                url,
                data=req_bytes,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                method="POST",
            )

            in_reasoning = False
            tc_acc: Dict[int, Dict[str, str]] = {}
            resp = None
            raw_sock = None
            stop_watch = threading.Event()

            try:
                # 严禁将 raw_sock.settimeout 设为短秒数：Python 3.13 SocketIO 一旦触发单次超时会将 _timeout_occurred 永久置 True，
                # 导致后续 resp.readline() 立即抛出 OSError("cannot read from timed out object")。
                # 此处保持长超时 900s 以覆盖 256K 超长上下文预填充，并由独立守护线程监听中断信号关闭套接字。
                resp = opener.open(http_req, timeout=900.0)
                raw_sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)

                def _abort_watcher():
                    while not stop_watch.wait(0.25):
                        if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            if raw_sock is not None:
                                try:
                                    raw_sock.shutdown(socket.SHUT_RDWR)
                                except Exception:
                                    pass
                            if resp is not None:
                                try:
                                    resp.close()
                                except Exception:
                                    pass
                            break

                threading.Thread(target=_abort_watcher, daemon=True).start()

                while True:
                    if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                        break
                    raw_line = resp.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue

                    if isinstance(chunk.get("error"), dict):
                        raise RuntimeError(str(chunk["error"].get("message") or chunk["error"]))

                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        if usage.get("prompt_tokens"):
                            streamer.prompt_tokens = int(usage["prompt_tokens"])
                        if usage.get("completion_tokens"):
                            streamer.generated_tokens = max(streamer.generated_tokens, int(usage["completion_tokens"]))

                    timings = chunk.get("timings")
                    if isinstance(timings, dict) and timings.get("predicted_per_second"):
                        streamer._server_tps = float(timings["predicted_per_second"])

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}

                    r_txt = delta.get("reasoning_content")
                    if r_txt:
                        if not in_reasoning:
                            streamer.put_text("<think>\n" + str(r_txt), token_inc=1)
                            in_reasoning = True
                        else:
                            streamer.put_text(str(r_txt), token_inc=1)

                    c_txt = delta.get("content")
                    if c_txt:
                        if in_reasoning:
                            streamer.put_text("</think>\n" + str(c_txt), token_inc=1)
                            in_reasoning = False
                        else:
                            streamer.put_text(str(c_txt), token_inc=1)

                    d_tcs = delta.get("tool_calls")
                    if isinstance(d_tcs, list):
                        for tc_item in d_tcs:
                            if not isinstance(tc_item, dict):
                                continue
                            idx = int(tc_item.get("index", 0))
                            slot = tc_acc.setdefault(idx, {"name": "", "arguments": ""})
                            fn_obj = tc_item.get("function") or {}
                            if fn_obj.get("name"):
                                slot["name"] += str(fn_obj["name"])
                            if fn_obj.get("arguments"):
                                slot["arguments"] += str(fn_obj["arguments"])

                if in_reasoning:
                    streamer.put_text("</think>\n", token_inc=0)
                    in_reasoning = False

                if tc_acc:
                    for idx in sorted(tc_acc.keys()):
                        slot = tc_acc[idx]
                        if not slot["name"]:
                            continue
                        try:
                            arg_val = json.loads(slot["arguments"]) if slot["arguments"] else {}
                        except Exception:
                            arg_val = {"raw": slot["arguments"]}
                        tc_str = json.dumps({"name": slot["name"], "arguments": arg_val}, ensure_ascii=False)
                        streamer.put_text(f"<tool_call>\n{tc_str}\n</tool_call>", token_inc=0)

            except urllib.error.HTTPError as he:
                err_body = ""
                try:
                    err_body = he.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                if he.code == 400 and ("context" in err_body.lower() or "exceed" in err_body.lower()):
                    err_box[0] = RuntimeError(
                        f"输入长度超出当前 GGUF 上下文窗口 ({bundle_ctx // 1024}K = {bundle_ctx} tokens)！"
                        f"请在左侧调大「上下文长度」（最高 256K）后点击重新加载模型。({err_body[:180]})"
                    )
                else:
                    err_box[0] = RuntimeError(f"GGUF 推理服务返回 HTTP {he.code}: {err_body[:240]}")
            except Exception as ex:
                if not (cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set()):
                    err_box[0] = ex
            finally:
                stop_watch.set()
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
                streamer.end()
                touch_activity()

    t = threading.Thread(target=_locked_gguf_generate, daemon=True)
    t.start()
    return streamer, streamer.prompt_tokens, False, t, err_box, cancel_event


def _prepare_qwen_streamer_generation(
    hf_messages: List[Dict[str, Any]],
    pil_images: List[Image.Image],
    hf_tools: Optional[List[Dict[str, Any]]],
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    preferred_model_id: Optional[str] = None,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: Optional[float] = None,
    repeat_penalty: float = 1.0,
    reasoning_effort: Optional[str] = None,
):
    """构建 Qwen3.8 (GGUF 或 BF16 Transformers) 后台流式推理任务并返回 (streamer, prompt_tokens, start_in_thinking, gen_thread, err_box, cancel_event)"""
    bundle = _get_qwen_llm_bundle(preferred_model_id=preferred_model_id)
    if bundle is not None and bundle.get("backend") == "gguf":
        return _prepare_gguf_streamer_generation(
            bundle=bundle,
            hf_messages=hf_messages,
            hf_tools=hf_tools,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            reasoning_effort=reasoning_effort,
        )

    model = bundle["model"]
    processor = bundle["processor"]
    bundle_ctx = int(bundle.get("context_length") or QWEN_CONTEXT_LENGTH)

    tpl_kwargs: Dict[str, Any] = dict(
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if enable_thinking and reasoning_effort and reasoning_effort not in ("default", "none"):
        tpl_kwargs["reasoning_effort"] = str(reasoning_effort)
    if hf_tools:
        tpl_kwargs["tools"] = hf_tools

    try:
        prompt_text = processor.apply_chat_template(hf_messages, **tpl_kwargs)
    except TypeError:
        tpl_kwargs.pop("reasoning_effort", None)
        try:
            prompt_text = processor.apply_chat_template(hf_messages, **tpl_kwargs)
        except TypeError:
            tpl_kwargs.pop("enable_thinking", None)
            prompt_text = processor.apply_chat_template(hf_messages, **tpl_kwargs)

    start_in_thinking = enable_thinking and (
        prompt_text.rstrip().endswith("<think>") or "<think>" in prompt_text[-24:]
    ) and not prompt_text.rstrip().endswith("</think>")

    prompt_tokens = max(1, len(processor.tokenizer.encode(prompt_text)))
    streamer = _make_timed_token_streamer(processor.tokenizer, timeout=180.0)

    max_toks = max(16, min(65536, int(max_new_tokens or 16384), max(128, bundle_ctx - prompt_tokens)))
    do_sample = temperature > 0.01
    cancel_event = threading.Event()
    err_box: List[Optional[Exception]] = [None]

    def _locked_generate():
        if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            try:
                streamer.end()
            except Exception:
                pass
            return

        with GPU_LOCK:
            if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                try:
                    streamer.end()
                except Exception:
                    pass
                return

            inputs = None
            try:
                if pil_images:
                    inputs = processor(text=[prompt_text], images=pil_images, padding=True, return_tensors="pt").to(model.device)
                else:
                    inputs = processor(text=[prompt_text], padding=True, return_tensors="pt").to(model.device)

                seq_in_len = int(inputs["input_ids"].shape[-1])
                if seq_in_len >= bundle_ctx:
                    raise RuntimeError(
                        f"输入长度 ({seq_in_len} tokens) 超出当前设定的上下文上限 ({bundle_ctx // 1024}K = {bundle_ctx} tokens)，"
                        f"请在左侧调大「上下文长度」（最高支持 256K）。"
                    )
                eff_max_toks = max(16, min(max_toks, bundle_ctx - seq_in_len))

                gen_kwargs: Dict[str, Any] = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=eff_max_toks,
                    do_sample=do_sample,
                    stopping_criteria=_make_abort_stopping_criteria(cancel_event),
                )
                if do_sample:
                    gen_kwargs["temperature"] = max(0.05, min(2.0, float(temperature)))
                    gen_kwargs["top_p"] = max(0.05, min(1.0, float(top_p)))
                    if int(top_k) > 0:
                        gen_kwargs["top_k"] = int(top_k)

                with torch.inference_mode():
                    model.generate(**gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                err_box[0] = RuntimeError(
                    "GPU 显存不足 (OOM)：BF16 原生精度长上下文峰值显存较高，建议切换使用独立的「Qwen3.8-27B-GGUF (UD-Q4_K_M)」量化模型，或适当调低上下文长度。"
                )
                try:
                    streamer.end()
                except Exception:
                    pass
            except Exception as ex:
                err_box[0] = ex
                try:
                    streamer.end()
                except Exception:
                    pass
            finally:
                if inputs is not None:
                    del inputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                touch_activity()

    t = threading.Thread(target=_locked_generate, daemon=True)
    t.start()
    return streamer, prompt_tokens, start_in_thinking, t, err_box, cancel_event


def attach_macos_api_routes(app: FastAPI):
    """在 Gradio 底层的 FastAPI 实例上挂载原生 macOS 客户端接口 + OpenAI / Anthropic 双协议 API 网关"""
    global API_AUTH_KEY, API_REQUEST_COUNT

    @app.get("/minimax_api/status")
    def api_status(model_id: Optional[str] = None, touch: Optional[int] = None):
        if touch:
            touch_activity()
        return get_gpu_status_dict(model_id=model_id)

    @app.post("/minimax_api/select_model")
    def api_select_model(req: SelectModelRequest):
        touch_activity()
        if req.model_id and req.model_id not in MODEL_CATALOG and req.repo_id:
            register_custom_hf_model_core(
                repo_id=req.repo_id,
                custom_name=req.custom_name,
                include_pattern=req.include_pattern,
                modality=req.modality or "auto",
                hf_token=req.hf_token,
                model_id_override=req.model_id,
            )
        return get_gpu_status_dict(model_id=req.model_id)

    @app.post("/minimax_api/config_idle")
    def api_config_idle(req: IdleConfigRequest):
        global IDLE_TIMEOUT_MINUTES
        IDLE_TIMEOUT_MINUTES = req.idle_timeout_minutes
        touch_activity()
        return get_gpu_status_dict()

    @app.post("/minimax_api/config_context")
    def api_config_context(req: ContextConfigRequest):
        global QWEN_CONTEXT_LENGTH
        QWEN_CONTEXT_LENGTH = _clamp_context_length(req.context_length)
        # BF16 Transformers 运行时支持直接热更新动态上下文上限；GGUF 需重载以重分配 KV 槽位
        for mid, rt in LOADED_RUNTIMES.items():
            b = rt.get("llm_bundle")
            if b and b.get("backend") == "transformers":
                b["context_length"] = QWEN_CONTEXT_LENGTH
                mname = MODEL_CATALOG.get(mid, {}).get("name", mid)
                rt["mode_label"] = f"{mname} · BF16 原生 · {QWEN_CONTEXT_LENGTH // 1024}K"
        _sync_legacy_globals()
        touch_activity()
        print(f"[上下文配置] 目标最大上下文长度已设为 {QWEN_CONTEXT_LENGTH // 1024}K ({QWEN_CONTEXT_LENGTH} tokens)")
        return get_gpu_status_dict(model_id=SELECTED_MODEL_ID)

    @app.get("/minimax_api/hf_trending")
    def api_hf_trending(
        category: str = "all",
        limit: int = 12,
        hf_token: str = "",
        force_refresh: bool = False,
        search_query: str = "",
    ):
        items = fetch_hf_trending_models_core(
            category=category,
            limit=limit,
            hf_token=hf_token,
            force_refresh=force_refresh,
            search_query=search_query,
        )
        return {"ok": True, "category": category, "search_query": search_query, "models": items}

    @app.get("/minimax_api/hf_model_detail")
    def api_hf_model_detail(repo_id: str = "", hf_token: str = ""):
        return fetch_hf_model_detail_core(repo_id=repo_id, hf_token=hf_token)

    @app.post("/minimax_api/custom_model/register")
    def api_register_custom_model(req: RegisterCustomModelRequest):
        touch_activity()
        eff_mod = req.modality if (req.modality and req.modality != "auto") else req.modality_hint
        ok, msg, spec = register_custom_hf_model_core(
            repo_id=req.repo_id,
            custom_name=req.custom_name,
            include_pattern=req.include_pattern,
            modality=eff_mod,
            hf_token=req.hf_token,
        )
        if ok and spec and req.auto_download and not MODEL_DOWNLOAD_STATE["is_downloading"]:
            threading.Thread(
                target=download_model_weights_core,
                args=(spec["id"], req.hf_token),
                daemon=True,
            ).start()
        res = get_gpu_status_dict(model_id=spec["id"] if (ok and spec) else SELECTED_MODEL_ID)
        res["action_ok"] = ok
        res["action_message"] = msg
        if spec:
            res["registered_model"] = {
                "id": spec["id"],
                "name": spec["name"],
                "repo_id": spec["repo_id"],
                "modality": spec["modality"],
                "category": spec["category"],
                "approx_size_gb": spec["approx_size_gb"],
                "include_pattern": spec.get("include_pattern", ""),
                "pipeline_arch": spec.get("pipeline_arch", "gguf"),
                "is_custom": True,
            }
        return res

    @app.post("/minimax_api/custom_model/remove")
    def api_remove_custom_model(req: RemoveCustomModelRequest):
        touch_activity()
        ok, msg = remove_custom_hf_model_core(req.model_id, delete_files=req.delete_files)
        res = get_gpu_status_dict(model_id=SELECTED_MODEL_ID)
        res["action_ok"] = ok
        res["action_message"] = msg
        return res

    @app.post("/minimax_api/config_api_key")
    def api_config_api_key(req: ApiKeyConfigRequest):
        global API_AUTH_KEY
        key = req.api_key.strip()
        if not key:
            raise HTTPException(status_code=422, detail="An inference API key is required")
        with API_KEY_LOCK:
            if not req.initialize_only or API_AUTH_KEY == INITIAL_API_KEY:
                API_AUTH_KEY = key
        touch_activity()
        return get_gpu_status_dict()

    # --------------------------------------------------------------------------
    # 1. OpenAI 兼容协议: GET /v1/models, POST /v1/chat/completions, POST /v1/images/generations
    # --------------------------------------------------------------------------
    @app.get("/v1/models")
    @app.get("/models")
    def openai_list_models(request: Request):
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        touch_activity()
        now_ts = int(time.time())
        any_qwen_loaded = ("qwen38_27b" in LOADED_RUNTIMES) or ("qwen38_27b_gguf" in LOADED_RUNTIMES)
        models_data = [
            {"id": "unsloth/Qwen3.8-27B-GGUF", "object": "model", "created": now_ts, "owned_by": "unsloth", "loaded": "qwen38_27b_gguf" in LOADED_RUNTIMES},
            {"id": "Qwen3.8-27B-UD-Q4_K_M", "object": "model", "created": now_ts, "owned_by": "unsloth", "loaded": "qwen38_27b_gguf" in LOADED_RUNTIMES},
            {"id": "Qwen/Qwen3.8-27B", "object": "model", "created": now_ts, "owned_by": "qwen", "loaded": any_qwen_loaded},
            {"id": "qwen3.8-27b", "object": "model", "created": now_ts, "owned_by": "qwen", "loaded": any_qwen_loaded},
            {"id": "Qwen/Qwen-Image-2.1", "object": "model", "created": now_ts, "owned_by": "qwen", "loaded": "qwen_image_21" in LOADED_RUNTIMES},
            {"id": "Tongyi-MAI/Z-Image-Turbo", "object": "model", "created": now_ts, "owned_by": "tongyi", "loaded": "z_image_turbo" in LOADED_RUNTIMES},
            {"id": "MiniMaxAI/MiniMax-H3", "object": "model", "created": now_ts, "owned_by": "minimax", "loaded": "minimax_h3" in LOADED_RUNTIMES},
        ]
        for cm in _get_custom_models_public_list():
            models_data.append({
                "id": cm["repo_id"],
                "object": "model",
                "created": now_ts,
                "owned_by": "custom-hf",
                "loaded": cm["id"] in LOADED_RUNTIMES,
            })
        return {"object": "list", "data": models_data}

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def openai_chat_completions(request: Request):
        global API_REQUEST_COUNT
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        touch_activity()

        body = await request.json()
        req_model = body.get("model") or "Qwen/Qwen3.8-27B"
        bundle = _get_qwen_llm_bundle(preferred_model_id=str(req_model))
        if bundle is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Qwen3.8-27B / Qwen3.8-27B-GGUF 尚未加载至 Colab GPU 显存，请先在 Mac 客户端点击「加载到显存」。",
                        "type": "server_error",
                        "code": "model_not_loaded",
                    }
                },
            )

        API_REQUEST_COUNT += 1
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 16384)

        # 支持通过 enable_thinking / chat_template_kwargs / reasoning_effort 控制思考模式与思考深度
        enable_thinking = True
        effort_val: Optional[str] = None
        ctk = body.get("chat_template_kwargs") if isinstance(body.get("chat_template_kwargs"), dict) else {}
        if "enable_thinking" in body:
            enable_thinking = bool(body["enable_thinking"])
        elif "enable_thinking" in ctk:
            enable_thinking = bool(ctk["enable_thinking"])
        raw_eff = str(body.get("reasoning_effort") or ctk.get("reasoning_effort") or "").strip().lower()
        if raw_eff in ("none", "disabled", "off"):
            enable_thinking = False
        elif raw_eff:
            effort_val = raw_eff

        temperature = float(body["temperature"]) if body.get("temperature") is not None else (1.0 if enable_thinking else 0.7)
        top_p = float(body["top_p"]) if body.get("top_p") is not None else (0.95 if enable_thinking else 0.80)
        top_k = int(body["top_k"]) if body.get("top_k") is not None else 20
        min_p = float(body["min_p"]) if body.get("min_p") is not None else 0.0
        presence_penalty = float(body["presence_penalty"]) if body.get("presence_penalty") is not None else (0.0 if enable_thinking else 1.5)
        repeat_penalty = float(body["repeat_penalty"]) if body.get("repeat_penalty") is not None else 1.0

        print(f"[OpenAI API] POST /v1/chat/completions | model={req_model} | backend={bundle.get('backend')} | stream={stream} | thinking={enable_thinking}")

        hf_messages, pil_images = _normalize_to_hf_messages_and_images(body.get("messages", []), protocol="openai")
        hf_tools = _normalize_hf_tools(body.get("tools"), protocol="openai")

        streamer, prompt_tokens, start_in_thinking, gen_thread, err_box, cancel_event = _prepare_qwen_streamer_generation(
            hf_messages=hf_messages,
            pil_images=pil_images,
            hf_tools=hf_tools,
            enable_thinking=enable_thinking,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            preferred_model_id=str(req_model),
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            reasoning_effort=effort_val,
        )

        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"
        created_ts = int(time.time())

        if not stream:
            parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
            think_parts: List[str] = []
            text_parts: List[str] = []
            out_chunks = 0
            with _ActiveApiStreamGuard():
                try:
                    for piece in streamer:
                        out_chunks += 1
                        if out_chunks % 64 == 0:
                            touch_activity()
                        for kind, val in parser.feed(piece):
                            if kind == "thinking":
                                think_parts.append(val)
                            elif kind == "text":
                                text_parts.append(val)
                finally:
                    cancel_event.set()
            tail_events, tool_calls = parser.finish()
            for kind, val in tail_events:
                if kind == "thinking":
                    think_parts.append(val)
                elif kind == "text":
                    text_parts.append(val)
            gen_thread.join(timeout=5.0)
            if err_box[0] is not None:
                return JSONResponse(status_code=500, content={"error": {"message": str(err_box[0]), "type": "server_error"}})

            full_thinking = "".join(think_parts).strip()
            full_text, extra_tools = _extract_qwen_tool_calls("".join(text_parts).strip())
            all_tools = tool_calls + extra_tools

            msg_obj: Dict[str, Any] = {"role": "assistant", "content": full_text if (full_text or not all_tools) else None}
            if full_thinking:
                msg_obj["reasoning_content"] = full_thinking
            if all_tools:
                msg_obj["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)},
                    }
                    for tc in all_tools
                ]

            final_prompt_tokens = int(getattr(streamer, "prompt_tokens", prompt_tokens) or prompt_tokens)
            comp_tokens = int(getattr(streamer, "generated_tokens", 0)) or (
                _count_bundle_tokens(bundle, full_thinking + full_text) if (full_thinking or full_text) else max(1, out_chunks)
            )
            finish_reason = "tool_calls" if all_tools else "stop"

            return {
                "id": cmpl_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": req_model,
                "choices": [{"index": 0, "message": msg_obj, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": final_prompt_tokens,
                    "completion_tokens": comp_tokens,
                    "total_tokens": final_prompt_tokens + comp_tokens,
                },
            }

        def _openai_sse_stream():
            parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
            comp_chars = 0
            chunk_idx = 0
            with _ActiveApiStreamGuard():
                try:
                    first_chunk = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req_model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

                    for piece in streamer:
                        comp_chars += len(piece)
                        chunk_idx += 1
                        if chunk_idx % 64 == 0:
                            touch_activity()
                        for kind, val in parser.feed(piece):
                            if not val:
                                continue
                            delta_payload = {"reasoning_content": val} if kind == "thinking" else {"content": val}
                            chunk_obj = {
                                "id": cmpl_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": req_model,
                                "choices": [{"index": 0, "delta": delta_payload, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n"

                    tail_events, tool_calls = parser.finish()
                    for kind, val in tail_events:
                        if not val:
                            continue
                        delta_payload = {"reasoning_content": val} if kind == "thinking" else {"content": val}
                        chunk_obj = {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": req_model,
                            "choices": [{"index": 0, "delta": delta_payload, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n"

                    gen_thread.join(timeout=5.0)
                    if getattr(streamer, "generated_tokens", 0) > 0:
                        print(f"[OpenAI API] 流式完成 | {streamer.generated_tokens} tokens | {streamer.tok_per_sec} tok/s")
                finally:
                    cancel_event.set()

            if tool_calls:
                tc_deltas = [
                    {
                        "index": idx,
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)},
                    }
                    for idx, tc in enumerate(tool_calls)
                ]
                tc_chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {"tool_calls": tc_deltas}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n"

            final_prompt_toks = int(getattr(streamer, "prompt_tokens", prompt_tokens) or prompt_tokens)
            est_out_toks = int(getattr(streamer, "generated_tokens", 0)) or max(1, comp_chars // 3)
            final_chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": req_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}],
                "usage": {
                    "prompt_tokens": final_prompt_toks,
                    "completion_tokens": est_out_toks,
                    "total_tokens": final_prompt_toks + est_out_toks,
                },
            }
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _openai_sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/images/generations")
    @app.post("/images/generations")
    async def openai_images_generations(request: Request):
        """当显存中驻留了 Qwen-Image-2.1 或 Z-Image-Turbo（可与 Qwen3.8-27B 同时共驻）时，提供标准 OpenAI 生图接口"""
        global API_REQUEST_COUNT
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        touch_activity()

        body = await request.json()
        req_model_str = str(body.get("model") or "").lower()
        img_target_id: Optional[str] = None
        if ("z-image" in req_model_str or "z_image" in req_model_str) and "z_image_turbo" in LOADED_RUNTIMES:
            img_target_id = "z_image_turbo"
        elif "qwen" in req_model_str and "qwen_image_21" in LOADED_RUNTIMES:
            img_target_id = "qwen_image_21"
        else:
            for cand in ("z_image_turbo", "qwen_image_21"):
                if cand in LOADED_RUNTIMES:
                    img_target_id = cand
                    break

        if not img_target_id:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "请先在客户端加载 Qwen-Image-2.1 或 Z-Image-Turbo 至显存（可与 Qwen3.8-27B 同时常驻）。", "type": "server_error"}},
            )

        img_pipe = LOADED_RUNTIMES[img_target_id]["pipeline"]
        API_REQUEST_COUNT += 1
        prompt = body.get("prompt", "")
        size_str = str(body.get("size", "1024x1024")).lower()
        try:
            w, h = [int(x) for x in size_str.split("x", 1)]
        except Exception:
            w, h = 1024, 1024
        n = max(1, min(4, int(body.get("n", 1))))
        steps = int(body.get("steps") or (8 if img_target_id == "z_image_turbo" else 40))

        results = []
        with GPU_LOCK:
            for _ in range(n):
                cur_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
                if img_target_id == "z_image_turbo":
                    img = img_pipe(prompt=prompt, width=w, height=h, num_inference_steps=steps, seed=cur_seed, rand_device="cuda")
                else:
                    img = img_pipe(prompt=prompt, negative_prompt=" ", width=w, height=h, num_inference_steps=steps, seed=cur_seed)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                results.append({"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")})
        return {"created": int(time.time()), "data": results}

    # --------------------------------------------------------------------------
    # 2. Anthropic Messages 协议: POST /v1/messages & POST /v1/messages/count_tokens
    #    完美兼容 Anthropic Python/TS SDK、Claude Code (ANTHROPIC_BASE_URL) 等工具
    # --------------------------------------------------------------------------
    @app.post("/v1/messages/count_tokens")
    @app.post("/messages/count_tokens")
    @app.post("/v1/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request):
        auth_err = _verify_external_api_auth(request, protocol="anthropic")
        if auth_err:
            return auth_err
        touch_activity()
        body = await request.json()
        raw_txt = json.dumps(body.get("messages", []), ensure_ascii=False)
        bundle = _get_qwen_llm_bundle(preferred_model_id=str(body.get("model") or ""))
        toks = _count_bundle_tokens(bundle, raw_txt)
        return {"input_tokens": toks}

    @app.post("/v1/messages")
    @app.post("/messages")
    @app.post("/v1/v1/messages")
    async def anthropic_messages(request: Request):
        global API_REQUEST_COUNT
        auth_err = _verify_external_api_auth(request, protocol="anthropic")
        if auth_err:
            return auth_err
        touch_activity()

        body = await request.json()
        req_model = body.get("model") or "Qwen/Qwen3.8-27B"
        bundle = _get_qwen_llm_bundle(preferred_model_id=str(req_model))
        if bundle is None:
            return JSONResponse(
                status_code=503,
                content={
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "Qwen3.8-27B / Qwen3.8-27B-GGUF 尚未加载至 Colab GPU 显存，请先在 Mac 客户端点击「加载到显存」。",
                    },
                },
            )

        API_REQUEST_COUNT += 1
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_tokens") or 16384)

        thinking_cfg = body.get("thinking")
        if isinstance(thinking_cfg, dict):
            enable_thinking = thinking_cfg.get("type") != "disabled"
        else:
            enable_thinking = True

        temperature = float(body["temperature"]) if body.get("temperature") is not None else (1.0 if enable_thinking else 0.7)
        top_p = float(body["top_p"]) if body.get("top_p") is not None else (0.95 if enable_thinking else 0.80)
        top_k = int(body["top_k"]) if body.get("top_k") is not None else 20

        print(f"[Anthropic API] POST /v1/messages | model={req_model} | backend={bundle.get('backend')} | stream={stream} | thinking={enable_thinking}")

        hf_messages, pil_images = _normalize_to_hf_messages_and_images(
            body.get("messages", []),
            system_prompt=body.get("system"),
            protocol="anthropic",
        )
        hf_tools = _normalize_hf_tools(body.get("tools"), protocol="anthropic")

        streamer, prompt_tokens, start_in_thinking, gen_thread, err_box, cancel_event = _prepare_qwen_streamer_generation(
            hf_messages=hf_messages,
            pil_images=pil_images,
            hf_tools=hf_tools,
            enable_thinking=enable_thinking,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            preferred_model_id=str(req_model),
            top_k=top_k,
            min_p=0.0,
            presence_penalty=0.0 if enable_thinking else 1.5,
            repeat_penalty=1.0,
        )

        msg_id = f"msg_{uuid.uuid4().hex[:22]}"

        if not stream:
            parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
            think_parts: List[str] = []
            text_parts: List[str] = []
            out_chunks = 0
            with _ActiveApiStreamGuard():
                try:
                    for piece in streamer:
                        out_chunks += 1
                        if out_chunks % 64 == 0:
                            touch_activity()
                        for kind, val in parser.feed(piece):
                            if kind == "thinking":
                                think_parts.append(val)
                            elif kind == "text":
                                text_parts.append(val)
                finally:
                    cancel_event.set()
            tail_events, tool_calls = parser.finish()
            for kind, val in tail_events:
                if kind == "thinking":
                    think_parts.append(val)
                elif kind == "text":
                    text_parts.append(val)
            gen_thread.join(timeout=5.0)
            if err_box[0] is not None:
                return JSONResponse(
                    status_code=500,
                    content={"type": "error", "error": {"type": "api_error", "message": str(err_box[0])}},
                )

            full_thinking = "".join(think_parts).strip()
            full_text, extra_tools = _extract_qwen_tool_calls("".join(text_parts).strip())
            all_tools = tool_calls + extra_tools

            content_blocks: List[Dict[str, Any]] = []
            if full_thinking:
                content_blocks.append({"type": "thinking", "thinking": full_thinking, "signature": "qwen38-sig"})
            if full_text or not all_tools:
                content_blocks.append({"type": "text", "text": full_text})
            for tc in all_tools:
                content_blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})

            final_prompt_tokens = int(getattr(streamer, "prompt_tokens", prompt_tokens) or prompt_tokens)
            out_tokens = int(getattr(streamer, "generated_tokens", 0)) or (
                _count_bundle_tokens(bundle, full_thinking + full_text) if (full_thinking or full_text) else 1
            )
            stop_reason = "tool_use" if all_tools else "end_turn"

            return {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": req_model,
                "content": content_blocks,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": final_prompt_tokens, "output_tokens": out_tokens},
            }

        def _anthropic_sse_event(ev_name: str, payload: Dict[str, Any]) -> str:
            return f"event: {ev_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        def _anthropic_sse_stream():
            parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
            yield _anthropic_sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": req_model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": prompt_tokens, "output_tokens": 1},
                    },
                },
            )

            next_block_idx = 0
            active_block_type: Optional[str] = None  # "thinking" | "text" | None
            active_block_idx = 0
            comp_chars = 0

            def _ensure_block(target_type: str):
                nonlocal next_block_idx, active_block_type, active_block_idx
                out_strs: List[str] = []
                if active_block_type == target_type:
                    return out_strs
                if active_block_type is not None:
                    if active_block_type == "thinking":
                        out_strs.append(
                            _anthropic_sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": active_block_idx,
                                    "delta": {"type": "signature_delta", "signature": "qwen38-sig"},
                                },
                            )
                        )
                    out_strs.append(
                        _anthropic_sse_event(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": active_block_idx},
                        )
                    )
                active_block_idx = next_block_idx
                next_block_idx += 1
                active_block_type = target_type
                block_init = (
                    {"type": "thinking", "thinking": ""}
                    if target_type == "thinking"
                    else {"type": "text", "text": ""}
                )
                out_strs.append(
                    _anthropic_sse_event(
                        "content_block_start",
                        {"type": "content_block_start", "index": active_block_idx, "content_block": block_init},
                    )
                )
                return out_strs

            chunk_idx = 0
            with _ActiveApiStreamGuard():
                try:
                    for piece in streamer:
                        comp_chars += len(piece)
                        chunk_idx += 1
                        if chunk_idx % 64 == 0:
                            touch_activity()
                        for kind, val in parser.feed(piece):
                            if not val:
                                continue
                            for s in _ensure_block(kind):
                                yield s
                            delta_obj = (
                                {"type": "thinking_delta", "thinking": val}
                                if kind == "thinking"
                                else {"type": "text_delta", "text": val}
                            )
                            yield _anthropic_sse_event(
                                "content_block_delta",
                                {"type": "content_block_delta", "index": active_block_idx, "delta": delta_obj},
                            )

                    tail_events, tool_calls = parser.finish()
                    for kind, val in tail_events:
                        if not val:
                            continue
                        for s in _ensure_block(kind):
                            yield s
                        delta_obj = (
                            {"type": "thinking_delta", "thinking": val}
                            if kind == "thinking"
                            else {"type": "text_delta", "text": val}
                        )
                        yield _anthropic_sse_event(
                            "content_block_delta",
                            {"type": "content_block_delta", "index": active_block_idx, "delta": delta_obj},
                        )

                    gen_thread.join(timeout=5.0)
                    if getattr(streamer, "generated_tokens", 0) > 0:
                        print(f"[Anthropic API] 流式完成 | {streamer.generated_tokens} tokens | {streamer.tok_per_sec} tok/s")
                finally:
                    cancel_event.set()

            if active_block_type is not None:
                if active_block_type == "thinking":
                    yield _anthropic_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": active_block_idx,
                            "delta": {"type": "signature_delta", "signature": "qwen38-sig"},
                        },
                    )
                yield _anthropic_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": active_block_idx},
                )
                active_block_type = None
            elif not tool_calls:
                # 若全程无输出块，保底发送一个空文本块符合 Anthropic 协议规范
                yield _anthropic_sse_event(
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                )
                yield _anthropic_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
                next_block_idx = 1

            for tc in tool_calls:
                t_idx = next_block_idx
                next_block_idx += 1
                yield _anthropic_sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": t_idx,
                        "content_block": {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": {}},
                    },
                )
                yield _anthropic_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": t_idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(tc["arguments"], ensure_ascii=False),
                        },
                    },
                )
                yield _anthropic_sse_event("content_block_stop", {"type": "content_block_stop", "index": t_idx})

            est_out_toks = int(getattr(streamer, "generated_tokens", 0)) or max(1, comp_chars // 3)
            yield _anthropic_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use" if tool_calls else "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": est_out_toks},
                },
            )
            yield _anthropic_sse_event("message_stop", {"type": "message_stop"})

        return StreamingResponse(
            _anthropic_sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    @app.post("/minimax_api/download_model")
    def api_download_model(req: DownloadModelRequest):
        touch_activity()
        if req.model_id and req.model_id not in MODEL_CATALOG and req.repo_id:
            register_custom_hf_model_core(
                repo_id=req.repo_id,
                custom_name=req.custom_name,
                include_pattern=req.include_pattern,
                modality=req.modality or "auto",
                hf_token=req.hf_token,
                model_id_override=req.model_id,
            )
        target_id = req.model_id if (req.model_id and req.model_id in MODEL_CATALOG) else SELECTED_MODEL_ID
        if not MODEL_DOWNLOAD_STATE["is_downloading"]:
            t = threading.Thread(
                target=download_model_weights_core,
                args=(target_id, req.hf_token),
                daemon=True,
            )
            t.start()
        return get_gpu_status_dict(model_id=target_id)

    @app.post("/minimax_api/load_model")
    def api_load_model(req: LoadModelRequest):
        global QWEN_CONTEXT_LENGTH
        touch_activity()
        if req.context_length is not None:
            QWEN_CONTEXT_LENGTH = _clamp_context_length(req.context_length)
        if req.model_id and req.model_id not in MODEL_CATALOG and req.repo_id:
            register_custom_hf_model_core(
                repo_id=req.repo_id,
                custom_name=req.custom_name,
                include_pattern=req.include_pattern,
                modality=req.modality or "auto",
                hf_token=req.hf_token,
                model_id_override=req.model_id,
            )
        target_id = req.model_id if (req.model_id and req.model_id in MODEL_CATALOG) else SELECTED_MODEL_ID
        if not MODEL_DOWNLOAD_STATE["is_loading_vram"]:
            t = threading.Thread(
                target=load_model_to_vram_core,
                args=(target_id, req.precision_mode, req.offload_mode, req.hf_token, req.context_length),
                daemon=True,
            )
            t.start()
        return get_gpu_status_dict(model_id=target_id)

    @app.post("/minimax_api/unload_model")
    def api_unload_model(req: UnloadModelRequest):
        touch_activity()
        unload_model_from_vram_core(req.model_id or None)
        return get_gpu_status_dict(model_id=SELECTED_MODEL_ID)

    @app.post("/minimax_api/cancel")
    def api_cancel_generation():
        global CURRENT_STAGE_BAR_LABEL
        touch_activity()
        ABORT_GENERATION_EVENT.set()
        for ev in list(_JOB_CANCEL_EVENTS.values()):
            ev.set()
        for jid, j in list(JOBS.items()):
            if j.get("status") in ("queued", "running"):
                j["status"] = "failed"
                j["progress"] = 0.0
                j["message"] = "已手动停止当前推理任务"
        CURRENT_STAGE_BAR_LABEL = None
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()
        print("[任务控制] 已收到停止指令，正在中断当前 GPU 推理并清空进度缓存...")

        def _clear_abort_after_release():
            with GPU_LOCK:
                ABORT_GENERATION_EVENT.clear()
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS.clear()

        threading.Thread(target=_clear_abort_after_release, daemon=True).start()
        return get_gpu_status_dict(model_id=SELECTED_MODEL_ID)

    @app.post("/minimax_api/generate")
    def api_generate(req: GenerateVideoRequest):
        touch_activity()
        target_model_id = (
            req.model_id
            if (req.model_id and req.model_id in MODEL_CATALOG)
            else SELECTED_MODEL_ID
        )
        if not LOADED_RUNTIMES and PIPELINE is None and LLM_BUNDLE is None:
            return {
                "job_id": "none",
                "status": "failed",
                "progress": 0.0,
                "message": "尚未有任何模型加载至 GPU 显存！请先在左侧点击「下载权重」与「加载到显存」。",
                "seed": req.seed,
                "elapsed": 0.0,
                "filename": None,
                "video_url": None,
            }

        # 若请求的是 LLM/VLM 但当前显存中驻留的是另一变体（如选中了 BF16 但已加载 GGUF，或反之），自动无缝路由至已就绪变体
        if _is_llm_model_id(target_model_id) and target_model_id not in LOADED_RUNTIMES:
            alt_bundle = _get_qwen_llm_bundle(preferred_model_id=target_model_id)
            if alt_bundle and alt_bundle.get("model_id") in LOADED_RUNTIMES:
                target_model_id = str(alt_bundle["model_id"])
                req.model_id = target_model_id

        if target_model_id not in LOADED_RUNTIMES and LOADED_MODEL_ID != target_model_id:
            resident_names = "、".join(rt["name"] for rt in LOADED_RUNTIMES.values()) or "无"
            req_name = MODEL_CATALOG.get(target_model_id, {}).get("name", target_model_id)
            return {
                "job_id": "none",
                "status": "failed",
                "progress": 0.0,
                "message": f"{req_name} 尚未加载到显存（当前已常驻：{resident_names}）。请点击左侧「加载到显存」，显存充裕时可同时常驻无需卸载现有模型。",
                "seed": req.seed,
                "elapsed": 0.0,
                "filename": None,
                "video_url": None,
            }

        # 若此前有残留未结束的任务，先置位其专属中断信号并清空进度条，防止旧任务继续占用 GPU 或污染新任务进度
        for old_jid, old_j in list(JOBS.items()):
            if old_j.get("status") in ("queued", "running"):
                old_ev = _JOB_CANCEL_EVENTS.get(old_jid)
                if old_ev is not None:
                    old_ev.set()
                old_j["status"] = "failed"
                old_j["progress"] = 0.0
                old_j["message"] = "已被新提交任务接替中断"
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()

        is_llm_req = _is_llm_model_id(target_model_id)
        batch_count = 1 if is_llm_req else max(1, min(8, int(getattr(req, "batch_count", 1) or 1)))
        job_id = uuid.uuid4().hex[:12]
        _JOB_CANCEL_EVENTS[job_id] = threading.Event()
        JOBS[job_id] = {
            "job_id": job_id,
            "model_id": target_model_id,
            "status": "queued",
            "progress": 0.0,
            "message": f"任务已提交至 Colab 队列 (共 {batch_count} 项)..." if batch_count > 1 else "任务已提交至 Colab 队列...",
            "seed": req.seed,
            "started_at": time.time(),
            "elapsed": 0.0,
            "filename": None,
            "video_url": None,
            "media_type": "text" if is_llm_req else (
                "image" if MODEL_CATALOG.get(target_model_id, {}).get("modality") == "image" else "video"
            ),
            "text_output": "",
            "thinking_output": "",
            "batch_count": batch_count,
            "completed_count": 0,
            "completed_videos": [],
        }
        t = threading.Thread(target=_run_async_generation_job, args=(job_id, req), daemon=True)
        t.start()
        return JOBS[job_id]

    @app.get("/minimax_api/job/{job_id}")
    def api_job_status(job_id: str):
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Job not found")
        res = dict(JOBS[job_id])
        if res.get("status") in ("queued", "running"):
            touch_activity()
            if res.get("started_at"):
                res["elapsed"] = round(max(0.0, time.time() - float(res["started_at"])), 1)
        res["console_lines"] = get_live_console_Lines()
        res["telemetry"] = get_live_telemetry_snapshot()
        return res

    @app.get("/minimax_api/video/{filename}")
    @app.get("/minimax_api/media/{filename}")
    def api_download_media(filename: str):
        touch_activity()
        safe_name = os.path.basename(filename)
        file_path = os.path.join(DEFAULT_OUTPUT_DIR, safe_name)
        if not os.path.exists(file_path) and DRIVE_MOUNTED:
            for leg_out in LEGACY_DRIVE_OUTPUT_DIRS:
                alt_path = os.path.join(leg_out, safe_name)
                if os.path.exists(alt_path):
                    file_path = alt_path
                    break
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Media file not found")
        ext = os.path.splitext(safe_name)[1].lower()
        mime = (
            "image/png" if ext == ".png"
            else "image/jpeg" if ext in (".jpg", ".jpeg")
            else "image/webp" if ext == ".webp"
            else "video/mp4"
        )
        return FileResponse(file_path, media_type=mime, filename=safe_name)

    @app.post("/minimax_api/disconnect")
    def api_disconnect():
        msg = disconnect_colab_runtime()
        return {"ok": True, "message": msg}

    # --------------------------------------------------------------------------
    # ComfyUI 云端节点工作台与 Google Drive 持久化专用接口 (/minimax_api/comfyui/*)
    # --------------------------------------------------------------------------
    @app.get("/minimax_api/comfyui/status")
    def api_comfyui_status():
        return {"ok": True, "comfyui": get_comfyui_status_dict()}

    @app.post("/minimax_api/comfyui/start")
    def api_comfyui_start(req: ComfyStartRequest):
        touch_activity()
        if not COMFYUI_STATE["is_starting"]:
            threading.Thread(
                target=start_comfyui_core,
                args=(req.unload_existing, req.force_restart),
                daemon=True,
            ).start()
        return get_gpu_status_dict(model_id=SELECTED_MODEL_ID)

    @app.post("/minimax_api/comfyui/stop")
    def api_comfyui_stop():
        touch_activity()
        stop_comfyui_core(sync_before_stop=True)
        return get_gpu_status_dict(model_id=SELECTED_MODEL_ID)

    @app.post("/minimax_api/comfyui/free_vram")
    def api_comfyui_free_vram():
        touch_activity()
        ok, msg = free_comfyui_vram_core()
        res = get_gpu_status_dict(model_id=SELECTED_MODEL_ID)
        res["action_ok"] = ok
        res["action_message"] = msg
        return res

    @app.post("/minimax_api/comfyui/sync_drive")
    def api_comfyui_sync_drive():
        touch_activity()
        ok, msg = sync_comfyui_to_drive_core(quiet=False)
        res = get_gpu_status_dict(model_id=SELECTED_MODEL_ID)
        res["action_ok"] = ok
        res["action_message"] = msg
        return res

    @app.post("/minimax_api/comfyui/download_weight")
    def api_comfyui_download_weight(req: ComfyWeightDownloadRequest):
        touch_activity()
        if not COMFYUI_STATE["is_downloading_weight"]:
            threading.Thread(
                target=download_comfyui_model_core,
                args=(req.url_or_repo, req.filename, req.subfolder, req.hf_token),
                daemon=True,
            ).start()
        return get_gpu_status_dict(model_id=SELECTED_MODEL_ID)

    @app.post("/minimax_api/comfyui/install_pack")
    def api_comfyui_install_pack(req: ComfyInstallPackRequest):
        touch_activity()
        if not COMFYUI_STATE["is_downloading_weight"] and not COMFYUI_STATE.get("installing_pack_id"):
            threading.Thread(
                target=install_comfyui_preset_pack_core,
                args=(req.pack_id, req.auto_start, req.hf_token, req.weight_source),
                daemon=True,
            ).start()
        return get_gpu_status_dict(model_id=SELECTED_MODEL_ID)

    @app.get("/minimax_api/comfyui/output_media/{rel_path:path}")
    def api_comfyui_output_media(rel_path: str):
        touch_activity()
        comfy_root, persist_root, _, _, _ = _resolve_comfyui_persistence_paths()
        for base_out in (os.path.join(comfy_root, "output"), os.path.join(persist_root, "output")):
            if not os.path.isdir(base_out):
                continue
            cand = os.path.normpath(os.path.join(base_out, rel_path))
            if not os.path.abspath(cand).startswith(os.path.abspath(base_out)):
                raise HTTPException(status_code=400, detail="Invalid output path")
            if os.path.isfile(cand):
                safe_name = os.path.basename(cand)
                ext = os.path.splitext(safe_name)[1].lower()
                mime = (
                    "image/png" if ext == ".png"
                    else "image/jpeg" if ext in (".jpg", ".jpeg")
                    else "image/webp" if ext == ".webp"
                    else "image/gif" if ext == ".gif"
                    else "video/webm" if ext == ".webm"
                    else "video/mp4"
                )
                return FileResponse(cand, media_type=mime, filename=safe_name)
        raise HTTPException(status_code=404, detail="ComfyUI output file not found")


# ==============================================================================
# 预设提示词模板与 Gradio 界面
# ==============================================================================
PRESET_EXAMPLES = [
    [
        "Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, vibrant cinematic lighting, smooth 24fps keyframe animation.",
        "Metallic clacking of train wheels on railway tracks, rhythmic dinging of the railroad crossing warning bell, warm spring wind rustling through leaves, a soft gasp from the girl.",
        "Emotional and nostalgic solo piano melody joined by warm acoustic guitar and soaring cinematic strings.",
        "832 × 480 (16:9 横屏标准 - 推荐)",
        124,
        35,
    ],
    [
        "High-budget ufotable anime action sakuga, dynamic fast-paced camera tracking. A silver-haired anime swordsman in a dark haori coat dashes across a shattered stone temple under a blood-red full moon, swinging a katana wreathed in crackling blue lightning and swirling water dragon effects. Sparks and glowing embers fly toward the camera in slow-motion impact frames, intense anime shading, dramatic motion blur.",
        "Sharp metallic sword unsheathing and clashing clang, roaring thunder and crackling electric arcs, heavy stone debris crumbling, fierce battle cry.",
        "Epic dark orchestral battle music with fast taiko drums, intense choir, and distorted electric guitar riffs.",
        "832 × 480 (16:9 横屏标准 - 推荐)",
        124,
        40,
    ],
]

CUSTOM_CSS = """
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
    color: #ffffff;
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 16px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}
.header-banner h1 {
    margin: 0 0 6px 0;
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
}
.header-banner p {
    margin: 0;
    font-size: 0.95rem;
    color: #cbd5e1;
}
.status-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
}
"""


WEBUI_HEADER_ZH = """
<div class="header-banner">
    <h1>⚡ EasyColab · Google Colab 多模态 AI 推理与生成服务端</h1>
    <p>支持 <b>ComfyUI 节点工作台 (云盘软链持久化) · MiniMax-H3 · Qwen3.8-27B (GGUF+MTP / BF16) · Qwen-Image-2.1 · Z-Image-Turbo</b> · 自动关机看门狗</p>
</div>
"""

WEBUI_HEADER_EN = """
<div class="header-banner">
    <h1>⚡ EasyColab · Google Colab Multimodal AI Inference & Generation Server</h1>
    <p>Supports <b>ComfyUI Node Studio (Google Drive Persistence) · MiniMax-H3 · Qwen3.8-27B (GGUF+MTP / BF16) · Qwen-Image-2.1 · Z-Image-Turbo</b> · Idle Auto-Shutdown Watchdog</p>
</div>
"""

RESOLUTION_CHOICES_ZH = [
    ("832 × 480 (16:9 横屏标准 - 推荐)", "832 × 480 (16:9 横屏标准 - 推荐)"),
    ("480 × 832 (9:16 竖屏短视频)", "480 × 832 (9:16 竖屏短视频)"),
    ("1280 × 720 (16:9 720P 高清)", "1280 × 720 (16:9 720P 高清)"),
    ("720 × 1280 (9:16 720P 竖屏高清)", "720 × 1280 (9:16 720P 竖屏高清)"),
    ("640 × 640 (1:1 方形)", "640 × 640 (1:1 方形)"),
]

RESOLUTION_CHOICES_EN = [
    ("832 × 480 (16:9 Landscape · Recommended)", "832 × 480 (16:9 横屏标准 - 推荐)"),
    ("480 × 832 (9:16 Portrait Short)", "480 × 832 (9:16 竖屏短视频)"),
    ("1280 × 720 (16:9 720P HD)", "1280 × 720 (16:9 720P 高清)"),
    ("720 × 1280 (9:16 720P Portrait HD)", "720 × 1280 (9:16 720P 竖屏高清)"),
    ("640 × 640 (1:1 Square)", "640 × 640 (1:1 方形)"),
]


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="EasyColab · Colab 多模型 AI 服务端") as demo:
        ui_lang_state = gr.State(value="zh")

        with gr.Row():
            with gr.Column(scale=9):
                header_html = gr.HTML(WEBUI_HEADER_ZH)
            with gr.Column(scale=1, min_width=130):
                lang_toggle_btn = gr.Button("🌐 English", size="sm", variant="secondary")

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Group():
                    sec1_md = gr.Markdown("### 📦 1. 模型权重管理中心 (先下载检查，再加载显存)")
                    gpu_status_text = gr.Textbox(
                        label="硬件、算力消耗与模型检查报告",
                        value=get_gpu_and_model_summary_text(),
                        lines=9,
                        interactive=False,
                        elem_classes=["status-box"],
                    )
                    refresh_gpu_btn = gr.Button("🔍 检查 / 刷新模型与算力状态", size="sm")

                    precision_Choice = gr.Radio(
                        choices=[
                            ("BF16 原生精度版 (官方满血权重 · G4 96GB 极速推理)", "bf16"),
                        ],
                        value="bf16",
                        label="模型权重精度",
                    )

                    offload_choice = gr.Radio(
                        choices=[
                            ("⚡ 零拷贝极速映射 + GPU 显存常驻 (推荐 · 5秒秒加载)", "disk"),
                            ("CPU 内存全量预载 (耗时较长 · 约3~5分钟)", "cpu"),
                        ],
                        value="disk",
                        label="显存加载与卸载策略 (VRAM Strategy)",
                    )

                    with gr.Accordion("🔑 Hugging Face Token (可选 / Optional)", open=False):
                        hf_token_input = gr.Textbox(
                            label="HF Access Token",
                            placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx",
                            type="password",
                        )

                    with gr.Accordion("➕ 添加自定义 HuggingFace 模型仓库 (Custom HF Repo Import)", open=False):
                        with gr.Row():
                            trending_hf_dropdown = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="🔥 HuggingFace 实时热门趋势模型 (点击右侧刷新拉取最新热榜)",
                                allow_custom_value=True,
                                scale=4,
                            )
                            refresh_trending_hf_btn = gr.Button("🔄 拉取实时热榜", size="sm", scale=1)
                        custom_repo_input = gr.Textbox(
                            label="HuggingFace Repo ID 或完整链接",
                            placeholder="从上方实时热榜选择，或输入任意 Owner/Repo",
                        )
                        custom_pattern_input = gr.Textbox(
                            label="文件名 / 通配符过滤 (Include Pattern，可选)",
                            placeholder="例如: *Q4_K_M*.gguf 或 *.safetensors（留空自动优选）",
                        )
                        custom_name_input = gr.Textbox(
                            label="自定义显示名称 (可选)",
                            placeholder="留空则默认使用仓库名",
                        )
                        add_custom_model_btn = gr.Button("➕ 解析并导入自定义 HF 模型", variant="secondary", size="sm")

                    with gr.Row():
                        download_model_btn = gr.Button(
                            "📥 1. 极速下载模型权重",
                            variant="secondary",
                        )
                        load_model_btn = gr.Button(
                            "🚀 2. 加载模型至显存",
                            variant="primary",
                        )

                with gr.Group():
                    sec2_md = gr.Markdown("### ⚙️ 2. 视频与采样参数")
                    resolution_dropdown = gr.Dropdown(
                        choices=RESOLUTION_CHOICES_ZH,
                        value="832 × 480 (16:9 横屏标准 - 推荐)",
                        label="输出分辨率 (Resolution)",
                    )
                    with gr.Row():
                        num_frames_slider = gr.Slider(
                            minimum=90,
                            maximum=362,
                            step=17,
                            value=124,
                            label="生成总帧数 (90~362帧 ≈ 4~15秒, 17n+5 对齐)",
                        )
                        fps_slider = gr.Slider(
                            minimum=16,
                            maximum=30,
                            step=1,
                            value=24,
                            label="帧率 (FPS)",
                        )
                    steps_slider = gr.Slider(
                        minimum=10,
                        maximum=60,
                        step=5,
                        value=35,
                        label="采样步数 (Inference Steps，推荐 30~50)",
                    )
                    with gr.Row():
                        seed_input = gr.Number(
                            value=42,
                            precision=0,
                            label="随机种子 (Seed)",
                        )
                        randomize_seed_chk = gr.Checkbox(
                            value=True,
                            label="🎲 每次随机种子",
                        )

                    save_dir_input = gr.Textbox(
                        value=DEFAULT_OUTPUT_DIR,
                        label="📂 视频自动保存路径",
                    )

                with gr.Accordion("💡 Google Colab 算力省额度工具 / Compute Quota Saver", open=True):
                    disconnect_btn = gr.Button("🛑 停止任务并释放 Colab 运行时 (省计算单元)", variant="stop")

            with gr.Column(scale=6):
                task_mode_state = gr.State(value="t2va")

                with gr.Tabs() as mode_tabs:
                    with gr.TabItem("✨ 文生音视频 / T2VA", id="t2va") as tab_t2va:
                        t2va_desc_md = gr.Markdown("输入画面描述与声音细节，同步生成高清视频与 **32kHz 立体声音轨**。")

                    with gr.TabItem("🖼️ 首尾帧生音视频 / FL2VA", id="fl2va") as tab_fl2va:
                        with gr.Row():
                            first_frame_input = gr.Image(
                                label="📌 首帧图片 (First Frame - index 0)",
                                type="pil",
                                height=220,
                            )
                            last_frame_input = gr.Image(
                                label="🏁 尾帧图片 (Last Frame - index -1，可选)",
                                type="pil",
                                height=220,
                            )

                with gr.Group():
                    visual_prompt_input = gr.Textbox(
                        label="🎥 画面主体与镜头描述 (Visual & Camera Prompt)",
                        placeholder="描述画面主体、动作、运镜方式、光影风格...",
                        lines=4,
                    )
                    with gr.Row():
                        soundscape_prompt_input = gr.Textbox(
                            label="🔊 环境音效与人声描述 (Soundscape & Foley - 可选)",
                            placeholder="例如：雨滴敲击声、脚步踩水声、温柔的女声独白...",
                            lines=2,
                        )
                        music_prompt_input = gr.Textbox(
                            label="🎵 背景配乐风格 (Background Music - 可选)",
                            placeholder="例如：低沉的赛博朋克合成器电子乐、恢弘管弦乐...",
                            lines=2,
                        )

                generate_btn = gr.Button(
                    "🎬 开始生成音视频 (Generate Video + Stereo Audio)",
                    variant="primary",
                    size="lg",
                )

                with gr.Group():
                    output_video = gr.Video(
                        label="🎞️ 生成结果预览 (请开启播放器声音体验原生立体声)",
                        height=420,
                        autoplay=True,
                    )
                    run_status_box = gr.Textbox(
                        label="运行日志与状态",
                        value="请先在左侧点击「1. 极速下载模型权重」并「2. 加载模型至显存」...",
                        interactive=False,
                        lines=2,
                        elem_classes=["status-box"],
                    )

                with gr.Accordion("📚 提示词一键体验模板 / Preset Prompt Templates", open=True):
                    gr.Examples(
                        examples=PRESET_EXAMPLES,
                        inputs=[
                            visual_prompt_input,
                            soundscape_prompt_input,
                            music_prompt_input,
                            resolution_dropdown,
                            num_frames_slider,
                            steps_slider,
                        ],
                    )

                with gr.Accordion("🗂️ 本次会话历史生成记录 / Session History", open=False):
                    history_gallery = gr.Gallery(
                        label="已生成的视频文件",
                        value=get_history_videos(DEFAULT_OUTPUT_DIR),
                        columns=3,
                        height=240,
                    )
                    refresh_history_btn = gr.Button("🔄 刷新历史记录", size="sm")

        def _toggle_webui_lang(cur_lang: str, cur_prec: str, cur_offload: str, cur_res: str, cur_status: str):
            new_lang = "en" if cur_lang == "zh" else "zh"
            is_en = new_lang == "en"
            default_zh_status = "请先在左侧点击「1. 极速下载模型权重」并「2. 加载模型至显存」..."
            default_en_status = "Click '1. Download Weights' then '2. Load to VRAM' on the left to get started..."
            if is_en and cur_status == default_zh_status:
                next_status = default_en_status
            elif not is_en and cur_status == default_en_status:
                next_status = default_zh_status
            else:
                next_status = cur_status

            return (
                new_lang,
                WEBUI_HEADER_EN if is_en else WEBUI_HEADER_ZH,
                "🌐 简体中文" if is_en else "🌐 English",
                "### 📦 1. Model Weights Center (Download First, Then Load to VRAM)" if is_en else "### 📦 1. 模型权重管理中心 (先下载检查，再加载显存)",
                gr.update(label="Hardware, CU Usage & Model Inspection Report" if is_en else "硬件、算力消耗与模型检查报告"),
                "🔍 Check / Refresh GPU & Model Status" if is_en else "🔍 检查 / 刷新模型与算力状态",
                gr.update(
                    choices=[
                        ("Native BF16 Full Precision (Recommended for G4 96GB)" if is_en else "BF16 原生精度版 (官方满血权重 · G4 96GB 极速推理)", "bf16"),
                    ],
                    value=cur_prec or "bf16",
                    label="Model Precision" if is_en else "模型权重精度",
                ),
                gr.update(
                    choices=[
                        ("⚡ Zero-Copy Fast Mapping + GPU VRAM Resident (Recommended · ~5s)" if is_en else "⚡ 零拷贝极速映射 + GPU 显存常驻 (推荐 · 5秒秒加载)", "disk"),
                        ("Full CPU RAM Preload (Slower · ~3-5 min)" if is_en else "CPU 内存全量预载 (耗时较长 · 约3~5分钟)", "cpu"),
                    ],
                    value=cur_offload or "disk",
                    label="VRAM Loading Strategy" if is_en else "显存加载与卸载策略 (VRAM Strategy)",
                ),
                "📥 1. Download Model Weights" if is_en else "📥 1. 极速下载模型权重",
                "🚀 2. Load Model to VRAM" if is_en else "🚀 2. 加载模型至显存",
                "### ⚙️ 2. Video & Sampling Parameters" if is_en else "### ⚙️ 2. 视频与采样参数",
                gr.update(
                    choices=RESOLUTION_CHOICES_EN if is_en else RESOLUTION_CHOICES_ZH,
                    value=cur_res or "832 × 480 (16:9 横屏标准 - 推荐)",
                    label="Output Resolution" if is_en else "输出分辨率 (Resolution)",
                ),
                gr.update(label="Total Frames (90~362 frames ≈ 4~15s, 17n+5 aligned)" if is_en else "生成总帧数 (90~362帧 ≈ 4~15秒, 17n+5 对齐)"),
                gr.update(label="Frame Rate (FPS)" if is_en else "帧率 (FPS)"),
                gr.update(label="Inference Steps (30~50 recommended)" if is_en else "采样步数 (Inference Steps，推荐 30~50)"),
                gr.update(label="Seed" if is_en else "随机种子 (Seed)"),
                gr.update(label="🎲 Randomize Seed Every Run" if is_en else "🎲 每次随机种子"),
                gr.update(label="📂 Output Save Directory" if is_en else "📂 视频自动保存路径"),
                "🛑 Stop Jobs & Release Colab Runtime (Save CU)" if is_en else "🛑 停止任务并释放 Colab 运行时 (省计算单元)",
                "Describe visual scenes and soundscapes to generate HD video with **32kHz stereo audio**." if is_en else "输入画面描述与声音细节，同步生成高清视频与 **32kHz 立体声音轨**。",
                gr.update(label="📌 First Frame Image (index 0)" if is_en else "📌 首帧图片 (First Frame - index 0)"),
                gr.update(label="🏁 Last Frame Image (index -1, Optional)" if is_en else "🏁 尾帧图片 (Last Frame - index -1，可选)"),
                gr.update(
                    label="🎥 Visual & Camera Prompt" if is_en else "🎥 画面主体与镜头描述 (Visual & Camera Prompt)",
                    placeholder="Describe subject, motion, camera movement, lighting, and visual style..." if is_en else "描述画面主体、动作、运镜方式、光影风格...",
                ),
                gr.update(
                    label="🔊 Soundscape & Dialogue (Optional)" if is_en else "🔊 环境音效与人声描述 (Soundscape & Foley - 可选)",
                    placeholder="e.g., Raindrops hitting pavement, footsteps in puddles, soft spoken dialogue..." if is_en else "例如：雨滴敲击声、脚步踩水声、温柔的女声独白...",
                ),
                gr.update(
                    label="🎵 Background Music Style (Optional)" if is_en else "🎵 背景配乐风格 (Background Music - 可选)",
                    placeholder="e.g., Moody cyberpunk synthwave, soaring orchestral strings..." if is_en else "例如：低沉的赛博朋克合成器电子乐、恢弘管弦乐...",
                ),
                "🎬 Generate Video + Stereo Audio" if is_en else "🎬 开始生成音视频 (Generate Video + Stereo Audio)",
                gr.update(label="🎞️ Output Preview (Unmute player for native stereo audio)" if is_en else "🎞️ 生成结果预览 (请开启播放器声音体验原生立体声)"),
                gr.update(label="Run Status & Logs" if is_en else "运行日志与状态", value=next_status),
                gr.update(label="Generated Videos" if is_en else "已生成的视频文件"),
                "🔄 Refresh History" if is_en else "🔄 刷新历史记录",
            )

        lang_toggle_btn.click(
            fn=_toggle_webui_lang,
            inputs=[ui_lang_state, precision_Choice, offload_choice, resolution_dropdown, run_status_box],
            outputs=[
                ui_lang_state,
                header_html,
                lang_toggle_btn,
                sec1_md,
                gpu_status_text,
                refresh_gpu_btn,
                precision_Choice,
                offload_choice,
                download_model_btn,
                load_model_btn,
                sec2_md,
                resolution_dropdown,
                num_frames_slider,
                fps_slider,
                steps_slider,
                seed_input,
                randomize_seed_chk,
                save_dir_input,
                disconnect_btn,
                t2va_desc_md,
                first_frame_input,
                last_frame_input,
                visual_prompt_input,
                soundscape_prompt_input,
                music_prompt_input,
                generate_btn,
                output_video,
                run_status_box,
                history_gallery,
                refresh_history_btn,
            ],
        )

        tab_t2va.select(fn=lambda: "t2va", inputs=None, outputs=task_mode_state)
        tab_fl2va.select(fn=lambda: "fl2va", inputs=None, outputs=task_mode_state)

        refresh_gpu_btn.click(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        def _refresh_gradio_trending_choices(tok: str):
            items = fetch_hf_trending_models_core("all", 14, tok, force_refresh=True)
            choices = [
                (
                    f"🔥 {m['repo_id']} ({(m.get('param_label') + ' · ') if m.get('param_label') else ''}{m.get('deploy_label_zh', '')} · {'🔒需Token · ' if m.get('is_gated') else ''}{m.get('pipeline_tag', '')} · ❤️{m.get('likes', 0)})",
                    f"{m['repo_id']}|{m.get('suggested_pattern', '')}",
                )
                for m in items
            ]
            return gr.update(choices=choices, value=choices[0][1] if choices else None)

        def _on_select_trending_repo(val: str):
            if not val:
                return gr.update(), gr.update()
            parts = str(val).split("|", 1)
            repo = parts[0].strip()
            pat = parts[1].strip() if len(parts) > 1 else ""
            return repo, pat

        refresh_trending_hf_btn.click(
            fn=_refresh_gradio_trending_choices,
            inputs=[hf_token_input],
            outputs=[trending_hf_dropdown],
        )
        trending_hf_dropdown.change(
            fn=_on_select_trending_repo,
            inputs=[trending_hf_dropdown],
            outputs=[custom_repo_input, custom_pattern_input],
        )

        add_custom_model_btn.click(
            fn=lambda repo, pat, cname, tok: register_custom_hf_model_core(repo, cname, pat, "auto", tok)[1],
            inputs=[custom_repo_input, custom_pattern_input, custom_name_input, hf_token_input],
            outputs=run_status_box,
        ).then(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        download_model_btn.click(
            fn=lambda tok: download_model_weights_core(SELECTED_MODEL_ID, tok)[1],
            inputs=[hf_token_input],
            outputs=run_status_box,
        ).then(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        load_model_btn.click(
            fn=lambda prec, off, tok: load_model_to_vram_core(SELECTED_MODEL_ID, prec, off, tok)[1],
            inputs=[precision_Choice, offload_choice, hf_token_input],
            outputs=run_status_box,
        ).then(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        generate_btn.click(
            fn=generate_video,
            inputs=[
                task_mode_state,
                visual_prompt_input,
                soundscape_prompt_input,
                music_prompt_input,
                first_frame_input,
                last_frame_input,
                resolution_dropdown,
                num_frames_slider,
                steps_slider,
                fps_slider,
                seed_input,
                randomize_seed_chk,
                save_dir_input,
            ],
            outputs=[output_video, run_status_box, seed_input, history_gallery],
        ).then(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        refresh_history_btn.click(
            fn=get_history_videos,
            inputs=save_dir_input,
            outputs=history_gallery,
        )

        disconnect_btn.click(
            fn=disconnect_colab_runtime,
            inputs=None,
            outputs=run_status_box,
        )

    return demo


if __name__ == "__main__":
    import getpass
    from gradio import networking

    if authenticated_ingress is not None and (_DEPLOY_PASSWORD or sys.stdin.isatty()):
        password = _DEPLOY_PASSWORD or getpass.getpass("Management password (24+ URL-safe characters): ")
        ingress_ctx = authenticated_ingress(password)
    else:
        ingress_ctx = contextlib.nullcontext()

    # Unwind the ingress context on a headless hot redeploy (pkill sends TERM).
    def stop_server(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop_server)
    with ingress_ctx:
        ui = create_ui()
        ui.queue()
        app, _, _ = ui.launch(
            server_name="127.0.0.1", server_port=GRADIO_BACKEND, share=False,
            inline=False, prevent_thread_lock=True, css=CUSTOM_CSS,
        )
        attach_macos_api_routes(app)
        tunnel = networking.setup_tunnel(
            local_host="127.0.0.1", local_port=GRADIO_INGRESS, share_token=ui.share_token,
            share_server_address=None, share_server_tls_certificate=None,
        )
        share_url = urllib.parse.urlsplit(tunnel)._replace(scheme="https").geturl()
        print(f"EasyColab URL: {share_url}", flush=True)
        try:
            ui.block_thread()
        finally:
            ui.close()
