from models.dit import MFDiT
import torch
import torchvision
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from meanflow import MeanFlow
from accelerate import Accelerator
import time
import math
import os


def track_gradients(model, writer=None, step=0):
    """跟踪各层参数的梯度 norm，可写入 TensorBoard"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            print(f"Layer: {name}, Grad Norm: {grad_norm:.6f}")
            # 若梯度 norm 超过 1e4，可能存在异常
            if grad_norm > 1e4:
                print(f"⚠️ 异常梯度：{name}，梯度 norm = {grad_norm}")
            # 写入 TensorBoard（可选）
            if writer:
                writer.add_scalar(f"grad_norm/{name}", grad_norm, step)
    return


def adjust_learning_rate_1(optimizer, epoch, warm_epoch, epochs, fast_decay_epochs):
    """调整学习率：前三轮线性增长，中间快速衰减，后期缓慢衰减"""
    # 阶段1：热身（epoch 0 到 warm_epoch-1），从0线性增长到6e-5（确保warm_epoch结束时达目标）
    if epoch < warm_epoch:
        # (epoch+1)/warm_epoch 确保 epoch=warm_epoch-1 时 lr=6e-5，与下一阶段无缝衔接
        lr = 6e-5 * (epoch + 1) / warm_epoch

        # 阶段2：快速衰减（epoch warm_epoch 到 warm_epoch+fast_decay_epochs-1）
    elif warm_epoch <= epoch < warm_epoch + fast_decay_epochs:
        decay_ratio = (epoch - warm_epoch) / fast_decay_epochs  # 0→1
        # 余弦衰减：从6e-5平滑过渡到2e-5（衔接点无跳变）
        lr = 2e-5 + (6e-5 - 2e-5) * 0.5 * (1. + math.cos(math.pi * decay_ratio))

    # 阶段3：缓慢衰减（剩余轮次）
    else:
        start_epoch = warm_epoch + fast_decay_epochs  # 缓慢衰减起始epoch
        remaining_epochs = epochs - start_epoch  # 剩余总轮次
        remaining_ratio = (epoch - start_epoch) / remaining_epochs  # 0→1
        # 余弦衰减：从2e-5平滑过渡到1e-6（衔接点无跳变）
        lr = 1e-6 + (2e-5 - 1e-6) * 0.5 * (1. + math.cos(math.pi * remaining_ratio))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    print('lr', lr)
    return lr


def adjust_learning_rate(optimizer, epoch, warm_epoch, decay_gamma=0.95, ):
    """
    调整学习率：支持两种策略（二选一）
    1. 指数衰减：热身阶段后按固定gamma指数衰减
    2. 阶梯衰减：在指定里程碑处按gamma衰减
    """
    # 基础学习率（热身结束时达到的值）
    base_lr = 6e-5
    epoch = 2 * epoch
    print('epoch: ', epoch)
    print('decay_gamma ** (epoch - warm_epoch): ',decay_gamma ** (epoch - warm_epoch))

    # 阶段1：热身（epoch 0 到 warm_epoch-1），从0线性增长到base_lr
    if epoch < warm_epoch:
        lr = base_lr * (epoch) / warm_epoch

        # 阶段2：学习率衰减（选择以下一种策略）
    else:
        lr = base_lr * (decay_gamma ** (epoch - warm_epoch))

        # 策略B：阶梯衰减（在指定里程碑处衰减）
        # 可注释掉策略A，启用策略B
        """
        lr = base_lr
        for milestone in milestones:
            if epoch >= milestone:
                lr *= decay_gamma
            else:
                break
        """

        # 防止学习率过低
        lr = max(lr, 1e-7)
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    print('lr', lr)
    return lr


if __name__ == '__main__':
    epochs = 100
    warm_epoch = 2
    fast_decay_epochs = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 100
    os.makedirs('images', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    accelerator = Accelerator(mixed_precision='bf16')

    dataset = torchvision.datasets.CIFAR10(
        root="MeanFlow-main/cifar",
        train=True,
        download=False,
        transform=T.Compose([T.ToTensor(), T.RandomHorizontalFlip()]),
    )


    # dataset = torchvision.datasets.MNIST(
    #     root="mnist",
    #     train=True,
    #     download=True,
    #     transform=T.Compose([T.Resize((32, 32)), T.ToTensor(),]),
    # )

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i


    sampler = RandomSampler(dataset)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=8
    )
    # loader = cycle(train_dataloader)
    loader = train_dataloader

    model = MFDiT(
        input_size=32,
        patch_size=2,
        in_channels=3,
        dim=512,
        num_classes=10,
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    meanflow = MeanFlow(channels=3,
                        image_size=32,
                        num_classes=10,
                        flow_ratio=0.50,
                        time_dist=['lognorm', -0.4, 1.0],
                        cfg_ratio=0.10,
                        cfg_scale=2.0,
                        # experimental
                        cfg_uncond='u')

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    global_step = 0.0
    losses = 0.0
    mse_losses = 0.0

    log_step = 500
    sample_step = 500

    for epoch in range(epochs):
        model.train()
        with tqdm(enumerate(loader), total=len(loader)) as tq:
            for data_iter_step, samples in tq:
                # we use a per iteration (instead of per epoch) lr scheduler
                if data_iter_step % 1 == 0:
                    adjust_learning_rate(optimizer, data_iter_step / len(loader) + epoch, warm_epoch,
                                         # epochs,
                                         # fast_decay_epochs
                                         )

                x = samples[0].to(accelerator.device)
                c = samples[1].to(accelerator.device)

                loss, mse_val = meanflow.loss(model, x, c)

                optimizer.zero_grad()

                accelerator.backward(loss)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                track_gradients(model, step=epoch)  # 打印当前轮次各层梯度

                optimizer.step()

                global_step += 1
                losses += loss.item()
                mse_losses += mse_val.item()

                if accelerator.is_main_process:
                    if global_step % log_step == 0:
                        current_time = time.asctime(time.localtime(time.time()))
                        batch_info = f'Global Step: {global_step}'
                        loss_info = f'Loss: {losses / log_step:.6f}    MSE_Loss: {mse_losses / log_step:.6f}'

                        # Extract the learning rate from the optimizer
                        lr = optimizer.param_groups[0]['lr']
                        lr_info = f'Learning Rate: {lr:.6f}'

                        log_message = f'{current_time}\n{batch_info}    {loss_info}    {lr_info}\n'

                        with open('log.txt', mode='a') as n:
                            n.write(log_message)

                        losses = 0.0
                        mse_losses = 0.0

                if global_step % sample_step == 0:
                    if accelerator.is_main_process:
                        model_module = model.module if hasattr(model, 'module') else model
                        z = meanflow.sample_each_class(model_module, 1)
                        log_img = make_grid(z, nrow=10)
                        img_save_path = f"MeanFlow-main/images/step_{global_step}.png"
                        save_image(log_img, img_save_path)
                    accelerator.wait_for_everyone()
                    model.train()

                loss_value = loss.item()
                print('loss_value', loss_value)
                if not math.isfinite(loss_value):
                    print('loss_value', loss_value)
                else:
                    assert 'loss inf'

    if accelerator.is_main_process:
        ckpt_path = f"checkpoints/step_{global_step}.pt"
        accelerator.save(model_module.state_dict(), ckpt_path)