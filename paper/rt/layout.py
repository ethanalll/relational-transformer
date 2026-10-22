"""Only the tensor permutations used by the archived model; no learned state."""

try:
    from einops import rearrange
    from einops._torch_specific import allow_ops_in_compiled_graph
except ImportError:

    def allow_ops_in_compiled_graph():
        pass

    def rearrange(x, pattern, **axes):
        if pattern == "b s (h d) -> b h s d":
            b, s, hd = x.shape
            h = axes["h"]
            return x.reshape(b, s, h, hd // h).transpose(1, 2)
        if pattern == "b h s d -> b s (h d)":
            b, h, s, d = x.shape
            return x.transpose(1, 2).reshape(b, s, h * d)
        raise ValueError(f"Unsupported archived tensor layout: {pattern}")
