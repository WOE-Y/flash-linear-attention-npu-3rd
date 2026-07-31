# RecurrentKda

`RecurrentKda` 是 KDA 的 fused recurrent 前向算子。在一个 AIV kernel 内完成 Q/K L2 normalize、raw gate 转换、beta sigmoid、state decay、delta 更新、输出计算与 state 写回，功能语义对齐 `fused_recurrent_kda_fwd`。

完整 aclnn 4.0 接口见 [API 文档](docs/api.md)，实现方案见 [设计文档](docs/design.md)。

## Python 接口

```python
from fla_npu.ops.ascendc import recurrent_kda

out, final_state = recurrent_kda(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    *,
    cu_seqlens=None,
    ssm_state_indices=None,
    A_log=None,
    dt_bias=None,
    num_accepted_tokens=None,
    layout="BSND",
    scale=None,
    output_final_state=False,
    inplace_final_state=True,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=None,
    state_v_first=False,
)
```

## 主要语义

- `layout="BSND"`：`q/k=[B,T,H,K]`，`v=[B,T,HV,V]`，`g=[B,T,HV,K]`，`beta=[B,T,HV]`。
- `layout="TND"`：`q/k=[T,H,K]`，`v=[T,HV,V]`，`g=[T,HV,K]`，`beta=[T,HV]`。
- `state_v_first=True` 时 state 为 `[state_capacity,HV,V,K]`；默认 `False` 时为 `[state_capacity,HV,K,V]`。
- `inplace_final_state=True` 时必须传 `initial_state`，递推结果原地写回该 tensor；为 `False` 时允许 `initial_state=None`，wrapper 使用 FP32 全零状态。
- `output_final_state=True` 时返回 final state；否则第二个返回值为 `None`。
- `cu_seqlens` 为可选 host 累积 offset，可传 Python 整数序列或 INT32/INT64 tensor；为空时 BSND 使用每个 batch 的定长边界，TND 视为一条序列。
- `scale=None` 时使用 `K ** -0.5`。
- `use_qk_l2norm_in_kernel=True` 时使用 `x / sqrt(sum(x*x) + 1e-6)` 对每个 token 的 q/k 归一化。
- `use_gate_in_kernel=False` 时，`g` 是预计算的 step log gate。
- `use_gate_in_kernel=True` 时必须传 `A_log`，可选 `dt_bias`：
  - `safe_gate=False`：`gate = -exp(A_log) * softplus(g + dt_bias)`；
  - `safe_gate=True`：`gate = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`。
- `use_beta_sigmoid_in_kernel=True` 时使用 `sigmoid(beta)`；`allow_neg_eigval=True` 时再乘 2。
- `ssm_state_indices` 支持 packed `[T]` 和 speculative `[seq_num,max_step]` state slot 索引；`num_accepted_tokens` 用于 speculative decode。

每个 token 的递推为：

```text
S = exp(gate_t) * S
delta = beta_t * (v_t - S @ k_t)
S = S + outer(delta, k_t)
o_t = S @ (q_t * scale)
```

## 当前限制

- 芯片：本次验收为 `ascend910b`。
- `q/k/v/out`：BF16；`gate/beta` 公开入口支持 FP16/BF16/FP32，aclnn 内部转换为 FP32。
- state：BF16 或 FP32。
- `K=128`，`V=128` 或 `V=256`，且 `HV % H == 0`。
- 仅支持 `BSND` 和 `TND`；每条 recurrent 序列长度不超过 8。
- 不提供 `ssm_state_indices` 时，`state_capacity` 必须等于逻辑序列数。

## 构建与验证

按仓库 README 的方式 B 构建 `recurrent_kda` run 包与 Python wheel。验收入口：

```bash
python3 fla/ops/ascendc/kda/recurrent_kda/tests/pta/test_accuracy.py
```

开发过程中的编译和运行时问题见 [开发问题记录](docs/开发问题记录.md)。
