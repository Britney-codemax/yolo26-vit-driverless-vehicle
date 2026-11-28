"""Read-only structure and transfer checks used by the round-four notebook."""
import torch

from models.phase4_layers import M_END, M_SKIP, V_HEAD, V_SKIP, V_OLD, ROUTES


def compare_transfer(target, source):
    incoming, rebuilt = source.state_dict(), target.state_dict()
    for dst, src in target.transfer["loaded"].items():
        assert torch.equal(rebuilt[dst], incoming[src]), dst
    return target.transfer["matched_tensors"], target.transfer["target_tensors"]


@torch.no_grad()
def print_graph(model, image):
    outputs = {}
    def capture(layer, args, output):
        outputs[layer.i] = tuple(output.shape) if isinstance(output, torch.Tensor) else tuple(output[0].shape)
    handles = [layer.register_forward_hook(capture) for layer in model.model]
    try:
        model.eval()(image)
    finally:
        for handle in handles:
            handle.remove()
    for layer in model.model:
        print(f"{layer.i:2}  {str(layer.f):>12}  {type(layer).__name__:12}  {outputs[layer.i]}")


@torch.no_grad()
def check_group(key, full, model, native, image, coco):
    assert not full.transfer["unused_source"], "Some original COCO tensors were lost"
    original_count, _ = compare_transfer(full, coco)
    matched, total = compare_transfer(model, full)
    assert model.model[-1].nc == 2 and full.model[-1].nc == 80
    assert set(model.transfer["unmatched"]) == set(native.transfer["unmatched"])
    gates = [p for n, p in model.named_parameters() if n.endswith(".alpha")]
    routes = {edge for edge in ROUTES[key] if edge}
    assert len(gates) == len(routes) and all(g.item() == 0 for g in gates)
    expected = native(image)
    torch.testing.assert_close(model(image), expected, rtol=0, atol=0)
    for gate in gates:
        gate.fill_(0.25)

    producers, consumers, tokens = {}, {}, {}
    def save_output(i):
        def record(module, args, output):
            producers[i] = output
        return record
    def save_inputs(i):
        def record(module, args):
            x = args[0]
            consumers[i] = tuple(x) if isinstance(x, (tuple, list)) else (x,)
        return record
    def save_tokens(module, args):
        tokens["vit"] = tuple(args[0].shape)

    handles = [model.model[i].register_forward_hook(save_output(i)) for i in (10, 13, 16, 19, 22)]
    handles += [model.model[i].register_forward_pre_hook(save_inputs(i)) for i in (11, 14, 18, 20, 21, 23)]
    if V_OLD in routes:
        handles.append(model.model[10].base.register_forward_hook(save_output("before_old_vit")))
    if ROUTES[key][1]:
        vit_edge = ROUTES[key][1]
        src, dst = vit_edge
        consumer = model.model[dst]
        vit = consumer if vit_edge == V_OLD else consumer.input_branches[str(consumer.f.index(src))]
        handles.append(vit.encoder.register_forward_pre_hook(save_tokens))
    try:
        actual = model(image)
        assert not torch.equal(actual[1]["one2one"]["scores"], expected[1]["one2one"]["scores"])
    finally:
        for handle in handles:
            handle.remove()
        for gate in gates:
            gate.zero_()
    assert [tuple(x.shape) for x in consumers[23]] == [(1, 64, 80, 80), (1, 128, 40, 40), (1, 256, 20, 20)]
    edges = {(10, 11): 0, (13, 14): 0, (19, 20): 0, (16, 23): 0,
             M_SKIP: 1, M_END: 2, V_SKIP: 1, V_HEAD: 1}
    print(f"两类参数 {sum(p.numel() for p in model.parameters()):,}；COCO 原始张量已核对 {original_count} 项")
    print(f"80 → 2 类迁移 {matched}/{total} 项；零门控输出与基线逐元素完全一致")
    for edge, pos in edges.items():
        src, dst = edge
        x, y = producers[src], consumers[dst][pos]
        assert x.shape == y.shape and torch.isfinite(y).all()
        changed = not torch.equal(x, y)
        assert changed == (edge in routes), f"{key}: incorrect route {edge}"
        if changed:
            print(f"L{src} → L{dst}   {tuple(x.shape)} → {tuple(y.shape)}")
    if ROUTES[key][0]:
        src, dst = ROUTES[key][0]
        layer = model.model[dst]
        scan = layer.input_branches[str(layer.f.index(src))]
        assert (scan.select.shape[0], scan.in_proj.out_channels, scan.state_size) == (4, 32, 8)
        print(f"{type(scan).__name__}  四方向 × 400 个位置，宽度 {scan.in_proj.out_channels}，状态维度 {scan.state_size}")
    if V_OLD in routes:
        assert not torch.equal(producers["before_old_vit"], producers[10])
        print("L10 原版 P5 ViT   (1, 256, 20, 20) → (1, 256, 20, 20)，两条下游路径均读取其输出")
    if tokens:
        assert tokens["vit"] == ((1, 400, 128) if V_OLD in routes else (1, 100, 64))
        heads, feedforward = vit.encoder.self_attention.num_heads, vit.encoder.mlp[0].out_features
        assert (heads, feedforward) == (4, 512 if V_OLD in routes else 128)
        print(f"{type(vit).__name__}  注意力输入 {tokens['vit']}，{heads} 头，前馈宽度 {feedforward}")
    print("Detect 实收 P3 / P4 / P5", [tuple(x.shape[1:]) for x in consumers[23]])
