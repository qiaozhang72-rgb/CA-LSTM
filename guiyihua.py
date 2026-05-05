# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:51:47 2026

@author: User
"""

"""
数据归一化脚本
功能：将原始数据归一化到 [-1, 1] 范围，并保存归一化后的数据和参数
"""

import numpy as np
import torch
import os
import json

# ==================== 配置部分 ====================

Mach='0.76'
alpha='6'

# 输入文件路径
input_f='modecoeff_M' + Mach + '_AOA' + alpha + '.dat'
output_f='normalized_' + input_f
file_path = './dataSet/POD/'
input_file_path = file_path + input_f

# 输出文件路径
output_dir = file_path
os.makedirs(output_dir, exist_ok=True)

output_data_file = os.path.join(output_dir, output_f)
output_params_file = os.path.join(output_dir, output_f + '.json')

# ==================== 读取原始数据 ====================
print("="*60)
print("开始数据归一化处理...")
print("="*60)

print(f"📂 读取文件：{input_file_path}")
data = np.loadtxt(input_file_path).T  # 转置后形状：[时间步，通道数]

print(f"✅ 原始数据形状：{data.shape}")
print(f"✅ 原始数据范围：[{data.min():.2e}, {data.max():.2e}]")
print(f"✅ 原始数据均值：{data.mean():.2e}")
print(f"✅ 原始数据标准差：{data.std():.2e}")

# ==================== 归一化方法选择 ====================
# 方法 1：Min-Max 归一化到 [-1, 1]（推荐）
# 公式：normalized = 2 * (x - min) / (max - min) - 1

data_min = data.min(axis=0)  # 每个通道的最小值 [通道数]
data_max = data.max(axis=0)  # 每个通道的最大值 [通道数]
data_range = data_max - data_min

print(f"\n📊 归一化参数（每个通道）：")
print(f"   Min 范围：[{data_min.min():.2e}, {data_min.max():.2e}]")
print(f"   Max 范围：[{data_max.min():.2e}, {data_max.max():.2e}]")
print(f"   Range 范围：[{data_range.min():.2e}, {data_range.max():.2e}]")

# 归一化
# 使用广播机制：data - data_min → [时间步，通道数] - [通道数] → [时间步，通道数]
normalized_data = 2 * (data - data_min) / (data_range + 1e-8) - 1

print(f"\n✅ 归一化后数据形状：{normalized_data.shape}")
print(f"✅ 归一化后数据范围：[{normalized_data.min():.6f}, {normalized_data.max():.6f}]")
print(f"✅ 归一化后数据均值：{normalized_data.mean():.6f}")
print(f"✅ 归一化后数据标准差：{normalized_data.std():.6f}")

# 验证归一化效果
print(f"\n🔍 验证归一化：")
print(f"   最小值是否接近 -1：{np.isclose(normalized_data.min(), -1, atol=1e-6)}")
print(f"   最大值是否接近 1：{np.isclose(normalized_data.max(), 1, atol=1e-6)}")

# ==================== 保存归一化后的数据 ====================
print(f"\n💾 保存归一化数据到：{output_data_file}")
np.savetxt(output_data_file, normalized_data.T, fmt='%.15e')  # 转置回原始格式
print(f"✅ 归一化数据保存成功！")

# ==================== 保存归一化参数 ====================
# 保存为 JSON 格式，方便后续读取
params = {
    'data_min': data_min.tolist(),
    'data_max': data_max.tolist(),
    'data_range': data_range.tolist(),
    'original_shape': list(data.shape),
    'normalized_min': float(normalized_data.min()),
    'normalized_max': float(normalized_data.max()),
    'normalization_method': 'Min-Max to [-1, 1]',
    'formula': '2 * (x - min) / (max - min) - 1'
}

print(f"\n💾 保存归一化参数到：{output_params_file}")
with open(output_params_file, 'w') as f:
    json.dump(params, f, indent=2)
print(f"✅ 归一化参数保存成功！")

# ==================== 打印参数摘要 ====================
print("\n" + "="*60)
print("归一化参数摘要：")
print("="*60)
print(f"通道数：{data.shape[1]}")
print(f"时间步数：{data.shape[0]}")
print(f"归一化方法：Min-Max 到 [-1, 1]")
print(f"\n前 3 个通道的归一化参数：")
for i in range(min(3, data.shape[1])):
    print(f"   通道 {i+1}: min={data_min[i]:.2e}, max={data_max[i]:.2e}")

# ==================== 反归一化示例代码 ====================
print("\n" + "="*60)
print("反归一化示例代码（用于预测结果还原）：")
print("="*60)
print("""
# 加载归一化参数
import json
import numpy as np

with open('normalization_params.json', 'r') as f:
    params = json.load(f)

data_min = np.array(params['data_min'])
data_max = np.array(params['data_max'])

# 假设 normalized_pred 是你的模型预测结果（归一化空间）
# 反归一化公式：x = (normalized + 1) / 2 * (max - min) + min
original_pred = (normalized_pred + 1) / 2 * (data_max - data_min) + data_min
""")

print("\n" + "="*60)
print("✅✅✅ 数据归一化完成！✅✅✅")
print("="*60)
print(f"\n📁 输出文件：")
print(f"   1. 归一化数据：{output_data_file}")
print(f"   2. 归一化参数：{output_params_file}")
print(f"\n💡 下一步：")
print(f"   在你的训练代码中，读取归一化后的数据文件即可")
print(f"   文件路径：{output_data_file}")