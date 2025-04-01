import mlx.core as mx


def affine_quantize(w, group_size=64, bits=8):

    n_bins = (1 << bits) - 1
    eps = 1e-7

    orig_shape = w.shape
    reshaped = w.reshape(*w.shape[:-1], w.shape[-1] // group_size, group_size)

    w_max = mx.max(reshaped, axis=-1, keepdims=True)
    w_min = mx.min(reshaped, axis=-1, keepdims=True)

    group_max = mx.distributed.all_max(w_max, stream=mx.cpu)
    group_min = mx.distributed.all_min(w_min, stream=mx.cpu)

    mask = mx.abs(group_min) > mx.abs(group_max)
    scale = (group_max - group_min) / n_bins
    scale = mx.maximum(scale, eps)
    scale = mx.where(mask, scale, -scale)

    edge = mx.where(mask, group_min, group_max)
    q0 = mx.round(edge / scale)

    scale = mx.where(q0 != 0, edge / q0, scale)
    bias = mx.where(q0 == 0, 0.0, edge)

    scale_reshaped = mx.broadcast_to(scale, reshaped.shape).reshape(orig_shape)
    bias_reshaped = mx.broadcast_to(bias, reshaped.shape).reshape(orig_shape)

    n_bins = (1 << bits) - 1
    zero = mx.array(0, dtype=mx.float32)
    q = mx.round((w - bias_reshaped) / scale_reshaped)
    q = mx.clip(q, zero, n_bins).astype(getattr(mx, f"int{bits}"))

    return q, scale, bias


def affine_dequantize(q, scale, bias, group_size=64, bits=8):

    orig_shape = q.shape
    num_groups = scale.shape[1]
    q_group_shape = (q.shape[0], num_groups, group_size)
    q_grouped = q.reshape(q_group_shape)
    w_grouped_float = q_grouped * scale + bias

    w_grouped_float = w_grouped_float.reshape(orig_shape)

    return w_grouped_float
