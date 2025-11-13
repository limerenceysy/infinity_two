#1.获取模型结构与权重
import torch
from infinity.models.infinity import Infinity
from tools.run_infinity import gen_one_img
model = Infinity(
    vae_local=None,
    embed_dim=2048,
    depth=32,
    num_heads=16,
    mlp_ratio=4,
)

state_dict = torch.load("weights/infinity_2b_reg.pth", map_location="cpu")
model.load_state_dict(state_dict)

#2.获取所有CrossAttention的mat_q权重
for name, module in model.named_modules():
    if module.__class__.__name__ == "CrossAttention":
        print(f'CrossAttention: {name} | shape: {module.mat_q.weight.shape}')
        print(module.mat_q.weight)
        print('-'*60)

#3.手动调整权重
#这里采取按比例缩放的方式，将权重缩放为0.8倍
scale_factor =0.8
print("将权重缩放为0.8倍：\n")
for name, module in model.named_modules():
    if module.__class__.__name__ == "CrossAttention":
        module.mat_q.weight.mul_(scale_factor)
        print(f'CrossAttention: {name} | shape: {module.mat_q.weight.shape}')
        print(module.mat_q.weight)
        print('-'*60)

#4.保存权重
torch.save(model.state_dict(), "weights/infinity_2b_reg_scale0.8.pth")

#5.生成图像以肉眼观察效果
prompt = "a beautiful landscape with mountains"
image = gen_one_img(model, prompt)
image.save("EraseInfinity/results/infinity_2b_reg_scale0.8.png")

#6.绘制权重分布图，观察扰动前后权重分布的变化
import matplotlib.pyplot as plt
import numpy as np

# 获取所有CrossAttention的mat_q权重
weights = []
for name, module in model.named_modules():
    if module.__class__.__name__ == "CrossAttention":
        weights.append(module.mat_q.weight.detach().numpy())

# 绘制权重分布图
plt.hist(weights, bins=100, density=True)
plt.show()