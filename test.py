import torch


model = 1

def hook_fn(module, input, output):
    """记录中间层输出的范围"""
    if isinstance(output, tuple):
        output = output[0]  # 处理多输出层（如 LSTM）
    output_abs_max = output.abs().max().item()
    print(f"Layer: {module.__class__.__name__}, Output Max: {output_abs_max:.6f}")
    if output_abs_max > 1e4:
        print(f"⚠️ 异常输出：{module.__class__.__name__}，输出最大值 = {output_abs_max}")

# 为关键层注册钩子（如全连接层、卷积层）
hooks = []
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LSTM)):
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)

# 训练结束后移除钩子
for hook in hooks:
    hook.remove()