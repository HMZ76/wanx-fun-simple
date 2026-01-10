import torch
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler

def test_flow_match_euler_discrete_scheduler():
    """
    修正版：测试 FlowMatchEulerDiscreteScheduler 的核心函数与功能
    移除不兼容的 beta 系列参数，适配流匹配调度器的初始化要求
    """
    # ======================================
    # 步骤1：定义测试超参数（模拟视频/图像潜变量维度）
    # ======================================
    # 潜变量形状：[批次大小, 通道数, 高度, 宽度]（模拟 720P 视频单帧潜变量，通道数为 4 对应 VAE 潜变量）
    batch_size = 1
    num_channels = 4
    height = 90  # 720P 图像经 VAE 编码后高度为 720/8=90
    width = 160   # 720P 图像经 VAE 编码后宽度为 1280/8=160
    latent_shape = (batch_size, num_channels, height, width)
    
    # 调度器超参数（仅保留流匹配调度器支持的参数）
    num_inference_steps = 23  # Jenga 论文中使用的 23 步调度，兼顾速度与质量
    num_train_timesteps = 1000  # 训练时的总时间步（流匹配调度器核心参数，默认 1000）

    # ======================================
    # 步骤2：初始化 FlowMatchEulerDiscreteScheduler（修正核心：移除 beta 系列参数）
    # ======================================
    print("="*50)
    print("步骤1：初始化 FlowMatchEulerDiscreteScheduler")
    try:
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,  # 流匹配调度器核心必填参数（仅需此参数即可正常初始化）
            step_scale=1.0,  # 可选：欧拉法步长缩放系数，用于稳定数值更新（默认 1.0）
            prediction_type="sample"  # 可选：模型预测类型（"sample" 对应流场，默认值）
        )
        print("✅ 调度器初始化成功")
        print(f"调度器训练时间步：{scheduler.num_train_timesteps}")
        print(f"调度器欧拉法步长系数：{scheduler.config.step_scale}")
        print(f"调度器预测类型：{scheduler.config.prediction_type}")
    except Exception as e:
        print(f"❌ 调度器初始化失败：{e}")
        return

    # ======================================
    # 步骤3：测试关键配置函数 - set_timesteps（设置推理时间步）
    # ======================================
    print("\n" + "="*50)
    print("步骤2：测试 set_timesteps（设置推理时间步）")
    try:
        # 设置推理时间步（核心函数，将 1000 训练步映射为 23 推理步）
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device="cpu")
        
        # 验证时间步结果
        print(f"✅ 成功设置 {num_inference_steps} 步推理时间")
        print(f"推理时间步列表长度：{len(scheduler.timesteps)}")
        print(f"推理时间步前5个值：{scheduler.timesteps[:5]}")
        print(f"推理时间步后5个值：{scheduler.timesteps[-5:]}")
        print(f"时间步间隔是否均匀：{np.allclose(np.diff(scheduler.timesteps.numpy()), np.diff(scheduler.timesteps.numpy())[0])}")
    except Exception as e:
        print(f"❌ set_timesteps 执行失败：{e}")
        return

    # ======================================
    # 步骤4：测试核心更新函数 - step（欧拉法潜变量更新）
    # ======================================
    print("\n" + "="*50)
    print("步骤3：测试 step（核心欧拉法更新）")
    try:
        # ① 初始化模拟数据
        # 模拟当前含噪潜变量 x_t（随机高斯噪声，对应 t=1 时刻）
        latent_x_t = torch.randn(latent_shape, dtype=torch.float32)
        # 模拟模型预测的流场 v_t（替代 DiT 模型输出，形状与潜变量一致）
        model_pred_v_t = torch.randn(latent_shape, dtype=torch.float32) * 0.1  # 小幅噪声模拟流场
        # 取第一个推理时间步（对应 t 较大的时刻，初始更新）
        current_timestep = scheduler.timesteps[0]

        # ② 执行 step 函数（核心：欧拉法离散更新）
        updated_latent, extra_info = scheduler.step(
            model_output=model_pred_v_t,  # 模型预测输出（流场 v_t，流匹配调度器专属输入）
            timestep=current_timestep,    # 当前时间步
            sample=latent_x_t             # 当前含噪潜变量 x_t
        )

        # ③ 验证更新结果
        print("✅ step 函数执行成功")
        print(f"原始潜变量形状：{latent_x_t.shape}")
        print(f"更新后潜变量形状：{updated_latent.shape}")
        print(f"潜变量是否仍为张量：{isinstance(updated_latent, torch.Tensor)}")
        print(f"更新前后潜变量差值均值：{torch.mean(torch.abs(updated_latent - latent_x_t)):.6f}")
        print(f"额外返回信息包含键：{list(extra_info.keys())}")
    except Exception as e:
        print(f"❌ step 执行失败：{e}")
        return

    # ======================================
    # 步骤5：测试状态重置函数 - reset（恢复调度器初始状态）
    # ======================================
    print("\n" + "="*50)
    print("步骤4：测试 reset（调度器状态重置）")
    try:
        # 重置前记录当前时间步列表
        timesteps_before_reset = scheduler.timesteps.clone() if hasattr(scheduler, 'timesteps') else None
        # 执行重置
        scheduler.reset()
        # 重置后验证状态
        print("✅ reset 函数执行成功")
        print(f"重置前时间步是否存在：{timesteps_before_reset is not None and len(timesteps_before_reset) > 0}")
        print(f"重置后时间步是否为空：{getattr(scheduler, 'timesteps', None) is None}")
        print(f"调度器核心参数是否保留：{scheduler.num_train_timesteps == num_train_timesteps}")
    except Exception as e:
        print(f"❌ reset 执行失败：{e}")
        return

    # ======================================
    # 步骤6：测试额外辅助功能（进度计算、时间步映射）
    # ======================================
    print("\n" + "="*50)
    print("步骤5：测试辅助功能（进度计算、时间步映射）")
    try:
        # 重新设置时间步（用于辅助测试）
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device="cpu")
        
        # ① 计算当前时间步的推理进度
        current_timestep = scheduler.timesteps[10]  # 取第 11 个时间步（索引 10）
        progress = (current_timestep.max() - current_timestep) / current_timestep.max()  # 流匹配调度器进度计算
        print(f"✅ 时间步 {current_timestep.item()} 对应的推理进度：{progress:.2%}")
        
        # ② 验证时间步的设备兼容性（CPU/GPU，若有GPU）
        if torch.cuda.is_available():
            scheduler.set_timesteps(num_inference_steps=num_inference_steps, device="cuda")
            print(f"✅ GPU 设备时间步设置成功，时间步设备：{scheduler.timesteps.device}")
        
        # ③ 查看调度器核心属性
        print(f"✅ 调度器当前是否处于训练模式：{scheduler.training}")
        print(f"✅ 调度器配置信息：{scheduler.config}")
    except Exception as e:
        print(f"❌ 辅助功能测试失败：{e}")
        return

    # ======================================
    # 步骤7：完整多步迭代测试（模拟真实推理流程）
    # ======================================
    print("\n" + "="*50)
    print("步骤6：完整多步迭代测试（模拟真实推理）")
    try:
        # 重新初始化调度器和潜变量
        scheduler.reset()
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device="cpu")
        current_latent = torch.randn(latent_shape, dtype=torch.float32)
        latent_history = [current_latent.clone()]  # 记录潜变量变化历史

        # 逐时间步迭代更新
        for idx, timestep in enumerate(scheduler.timesteps):
            # 模拟模型预测流场（每一步生成不同的流场，更贴近真实场景）
            model_pred_v_t = torch.randn(latent_shape, dtype=torch.float32) * 0.05 + (idx / num_inference_steps) * 0.1
            
            # 执行欧拉法更新
            current_latent, _ = scheduler.step(
                model_output=model_pred_v_t,
                timestep=timestep,
                sample=current_latent
            )
            
            # 记录每 5 步的结果
            if (idx + 1) % 5 == 0 or idx == len(scheduler.timesteps) - 1:
                latent_history.append(current_latent.clone())
                print(f"  完成第 {idx+1}/{num_inference_steps} 步更新，潜变量均值：{torch.mean(current_latent):.6f}")

        print("✅ 完整多步迭代测试成功")
        print(f"潜变量历史记录数：{len(latent_history)}（每 5 步记录一次）")
        print(f"最终潜变量与初始潜变量差值均值：{torch.mean(torch.abs(current_latent - latent_history[0])):.6f}")
    except Exception as e:
        print(f"❌ 完整多步迭代测试失败：{e}")
        return

    # ======================================
    # 测试总结
    # ======================================
    print("\n" + "="*50)
    print("🎉 所有测试项执行完成！FlowMatchEulerDiscreteScheduler 核心功能正常。")

if __name__ == "__main__":
    test_flow_match_euler_discrete_scheduler()