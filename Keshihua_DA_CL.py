#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整时间序列数据增强效果可视化（数据保存版）
功能：将滑动窗口数据拼接成完整时间序列，展示原始数据与增强数据的对比
      并保存完整数据以便重新绘图
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json


# ==================== 【🔧 配置区域】 ====================
CONFIG = {
    # 增强数据文件路径
    'augmented_data_path': r'./dataSet/POD/augmented_data_M0.71-AOA6-SZ-4-128-AR8-C10-TN4K_AR8_N10-50.npz',
    
    # 输出目录
    'output_dir': r'./results_train/visualization/augmentation/',
    
    # 可视化参数
    'mode_idx': 0,              # 可视化第几个模态（从 0 开始）
    'alpha_original': 1.0,      # 原始数据透明度
    'alpha_augmented': 0.3,     # 增强数据透明度
    'line_width_original': 2.0, # 原始数据线宽
    'line_width_augmented': 1.0,# 增强数据线宽
    'n_augmented_to_show': 5,   # 显示几个增强样本
    'save_dpi': 300,            # 保存图片分辨率
    
    # 数据保存参数
    'save_full_data': True,     # 是否保存完整时间序列数据
    'save_format': 'npz',       # 保存格式：'npz' 或 'csv'
}


# ==================== 【数据加载函数】 ====================
def load_augmented_data(file_path):
    """加载增强数据文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"增强数据文件未找到：{file_path}")
    
    data = np.load(file_path, allow_pickle=True)
    
    result = {
        'X_train': data['X_train'],              # [n_augmented, window, modes]
        'X_train_original': data['X_train_original'],  # [n_original, window, modes]
        'Y_train': data['Y_train'],
        'Y_train_original': data['Y_train_original'],
        'augment_ratio': data['augment_ratio'],
        'noise_range': data['noise_range'],
    }
    
    print(f"✅ 加载增强数据成功！")
    print(f"   原始数据形状：{result['X_train_original'].shape}")
    print(f"   增强数据形状：{result['X_train'].shape}")
    print(f"   增强倍数：{result['augment_ratio']}x")
    print(f"   窗口长度：{result['X_train_original'].shape[1]}")
    
    return result


# ==================== 【窗口数据拼接函数】 ====================
def reconstruct_full_sequence_from_windows(window_data, mode_idx=0):
    """
    从滑动窗口数据重建完整时间序列
    
    Args:
        window_data: [n_samples, window_size, n_modes] 窗口数据
        mode_idx: 要提取的模态索引
    
    Returns:
        full_sequence: 完整的时间序列 [total_length]
        time_indices: 对应的时间索引 [total_length]
    """
    n_samples, window_size, n_modes = window_data.shape
    
    # 总时间步数 = n_samples + window_size - 1
    total_length = n_samples + window_size - 1
    
    # 初始化完整序列
    full_sequence = np.zeros(total_length)
    count = np.zeros(total_length)  # 用于平均重叠部分
    
    # 提取指定模态的数据
    mode_data = window_data[:, :, mode_idx]  # [n_samples, window_size]
    
    # 填充完整序列（滑动窗口拼接）
    for i in range(n_samples):
        start_idx = i
        end_idx = i + window_size
        full_sequence[start_idx:end_idx] += mode_data[i, :]
        count[start_idx:end_idx] += 1
    
    # 对重叠部分取平均
    full_sequence = full_sequence / count
    
    time_indices = np.arange(total_length)
    
    return full_sequence, time_indices


# ==================== 【完整时间序列可视化 + 数据保存】 ====================
def plot_full_sequence_comparison_with_data_save(original_data, augmented_data, 
                                                   mode_idx=0, n_aug_samples=5,
                                                   alpha_orig=1.0, alpha_aug=0.3,
                                                   lw_orig=2.0, lw_aug=1.0,
                                                   output_dir='./',
                                                   save_full_data=True,
                                                   save_format='npz'):
    """
    绘制完整时间序列的原始数据与增强数据对比，并保存完整数据
    
    Args:
        original_data: [n_original, window, modes]
        augmented_data: [n_augmented, window, modes]
        mode_idx: 可视化的模态索引
        n_aug_samples: 显示的增强样本数量
        save_full_data: 是否保存完整数据
        save_format: 保存格式 ('npz' 或 'csv')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 绘制完整时间序列对比图（模态{mode_idx+1}）...")
    
    # 获取数据维度
    n_original, window_size, n_modes = original_data.shape
    n_augmented = augmented_data.shape[0]
    
    print(f"   窗口数量：原始={n_original}, 增强={n_augmented}")
    print(f"   窗口长度：{window_size}")
    
    # ========== 【重建完整时间序列】 ==========
    print("   重建完整时间序列...")
    
    # 1. 重建原始数据的完整序列
    full_original, time_indices = reconstruct_full_sequence_from_windows(
        original_data, mode_idx
    )
    
    print(f"   完整序列长度：{len(full_original)}")
    
    # 2. 随机选择几个增强样本的索引
    np.random.seed(42)
    aug_indices = np.random.choice(n_augmented, min(n_aug_samples, n_augmented), replace=False)
    
    # 3. 为每个选中的增强样本重建完整序列
    full_augmented_list = []
    for aug_idx in aug_indices:
        # 从增强数据集中选择连续的窗口来重建序列
        start_aug_idx = (aug_idx // n_original) * n_original
        end_aug_idx = min(start_aug_idx + n_original, n_augmented)
        
        # 选择一组连续的增强窗口
        aug_windows = augmented_data[start_aug_idx:end_aug_idx]
        
        if len(aug_windows) > 0:
            full_aug, _ = reconstruct_full_sequence_from_windows(aug_windows, mode_idx)
            full_augmented_list.append(full_aug)
    
    # 如果没有成功重建增强序列，使用简化方法
    if len(full_augmented_list) == 0:
        print("  ⚠️  无法重建增强序列，使用简化方法...")
        for i in range(n_aug_samples):
            noise = np.random.randn(len(full_original)) * 0.1
            full_augmented_list.append(full_original + noise)
    
    # ========== 【计算噪声统计】 ==========
    noise_list = [full_aug - full_original for full_aug in full_augmented_list]
    noise_array = np.array(noise_list)
    
    noise_mean = np.mean(noise_array, axis=0)
    noise_std = np.std(noise_array, axis=0)
    
    # ========== 【保存完整数据】 ==========
    if save_full_data:
        print(f"\n💾 保存完整时间序列数据...")
        
        # 准备保存的数据字典
        save_data = {
            # 时间索引
            'time_indices': time_indices,
            
            # 原始数据
            'original_full_sequence': full_original,
            'original_stats': {
                'mean': float(np.mean(full_original)),
                'std': float(np.std(full_original)),
                'min': float(np.min(full_original)),
                'max': float(np.max(full_original)),
                'length': len(full_original)
            },
            
            # 增强数据（多个样本）
            'augmented_full_sequences': np.array(full_augmented_list),
            'augmented_stats': {
                'n_samples': len(full_augmented_list),
                'mean_sequence': noise_mean + full_original,  # 增强的均值
                'std_sequence': noise_std,
                'mean': float(np.mean(np.array(full_augmented_list))),
                'std': float(np.std(np.array(full_augmented_list))),
            },
            
            # 噪声数据
            'noise_sequences': noise_array,
            'noise_stats': {
                'mean': float(np.mean(noise_array)),
                'std': float(np.std(noise_array)),
                'min': float(np.min(noise_array)),
                'max': float(np.max(noise_array)),
                'mean_over_time': noise_mean,
                'std_over_time': noise_std
            },
            
            # 配置信息
            'config': {
                'mode_idx': mode_idx,
                'mode_name': f'Mode {mode_idx+1}',
                'n_augmented_displayed': len(full_augmented_list),
                'alpha_original': alpha_orig,
                'alpha_augmented': alpha_aug,
                'line_width_original': lw_orig,
                'line_width_augmented': lw_aug
            }
        }
        
        # 保存为指定格式
        if save_format == 'npz':
            save_path = os.path.join(output_dir, f'full_sequence_data_mode{mode_idx+1}.npz')
            np.savez(save_path, **save_data)
            print(f"  ✅ NPZ 数据已保存：{os.path.basename(save_path)}")
            print(f"     文件大小：{os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
        
        elif save_format == 'csv':
            # 保存为 CSV（适合用 Excel 或其他工具查看）
            csv_dir = os.path.join(output_dir, f'csv_data_mode{mode_idx+1}')
            os.makedirs(csv_dir, exist_ok=True)
            
            # 时间索引
            np.savetxt(os.path.join(csv_dir, 'time_indices.csv'), 
                      time_indices, fmt='%d', delimiter=',', header='Time_Step')
            
            # 原始数据
            np.savetxt(os.path.join(csv_dir, 'original_sequence.csv'), 
                      full_original, fmt='%.10e', delimiter=',', header='Original_Value')
            
            # 增强数据（每个样本一个文件）
            for i, full_aug in enumerate(full_augmented_list):
                np.savetxt(os.path.join(csv_dir, f'augmented_sequence_{i+1}.csv'), 
                          full_aug, fmt='%.10e', delimiter=',', header='Augmented_Value')
            
            # 噪声数据
            for i, noise in enumerate(noise_list):
                np.savetxt(os.path.join(csv_dir, f'noise_sequence_{i+1}.csv'), 
                          noise, fmt='%.10e', delimiter=',', header='Noise_Value')
            
            # 噪声统计
            np.savetxt(os.path.join(csv_dir, 'noise_mean_over_time.csv'), 
                      noise_mean, fmt='%.10e', delimiter=',', header='Noise_Mean')
            np.savetxt(os.path.join(csv_dir, 'noise_std_over_time.csv'), 
                      noise_std, fmt='%.10e', delimiter=',', header='Noise_Std')
            
            # 统计信息（JSON 格式）
            stats_path = os.path.join(csv_dir, 'statistics.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(save_data['original_stats'], f, indent=2)
                json.dump(save_data['augmented_stats'], f, indent=2)
                json.dump(save_data['noise_stats'], f, indent=2)
            
            print(f"  ✅ CSV 数据已保存：{csv_dir}/")
            print(f"     包含 {len(full_augmented_list)} 个增强样本文件")
        
        # 保存配置信息
        config_path = os.path.join(output_dir, f'visualization_config_mode{mode_idx+1}.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(save_data['config'], f, indent=2, ensure_ascii=False)
        print(f"  ✅ 配置文件已保存：{os.path.basename(config_path)}")
    
    # ========== 【绘图】 ==========
    # 创建图像（大尺寸，便于观察长序列）
    fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True)
    fig.suptitle(f'Full Time Series Comparison - Mode {mode_idx+1}\n'
                f'Total Length: {len(full_original)} time steps', 
                fontsize=16, fontweight='bold')
    
    # ==================== 【上图：原始 + 增强数据叠加】 ====================
    ax1 = axes[0]
    
    # 绘制原始数据
    ax1.plot(time_indices, full_original, 'b-', linewidth=lw_orig, 
            label=f'Original (Full Sequence)', 
            alpha=alpha_orig, zorder=3)
    
    # 绘制增强样本（半透明，显示分布）
    for i, full_aug in enumerate(full_augmented_list):
        if len(full_aug) != len(time_indices):
            print(f"  ⚠️  警告：增强样本{i+1}长度{len(full_aug)}与时间轴{len(time_indices)}不匹配，跳过")
            continue
        
        ax1.plot(time_indices, full_aug, 'r-', linewidth=lw_aug, 
                label=f'Augmented #{i+1}' if i == 0 else None, 
                alpha=alpha_aug, zorder=2)
    
    ax1.set_ylabel('Coefficient Value', fontsize=12)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_title('Original vs Augmented Full Sequences (Overlay)', fontsize=14, fontweight='bold')
    
    # 添加统计信息
    stats_text = f'Original Range: [{full_original.min():.4f}, {full_original.max():.4f}]\n'
    stats_text += f'Original Mean: {full_original.mean():.4f}\n'
    stats_text += f'Original Std: {full_original.std():.4f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # ==================== 【中图：增强样本的分布带】 ====================
    ax2 = axes[1]
    
    if len(full_augmented_list) > 0:
        # 计算增强样本的均值和标准差
        full_augmented_array = np.array(full_augmented_list)
        aug_mean = np.mean(full_augmented_array, axis=0)
        aug_std = np.std(full_augmented_array, axis=0)
        
        # 绘制均值线
        ax2.plot(time_indices, aug_mean, 'r-', linewidth=1.5, 
                label='Augmented Mean', alpha=0.8, zorder=3)
        
        # 填充±1 标准差区域
        ax2.fill_between(time_indices, aug_mean - aug_std, aug_mean + aug_std, 
                        color='red', alpha=0.2, label='±1 Std Dev')
        
        # 填充±2 标准差区域
        ax2.fill_between(time_indices, aug_mean - 2*aug_std, aug_mean + 2*aug_std, 
                        color='red', alpha=0.1, label='±2 Std Dev')
    
    ax2.set_ylabel('Coefficient Value', fontsize=12)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_title('Augmented Data Distribution (Mean ± Std Dev)', fontsize=14, fontweight='bold')
    
    # ==================== 【下图：噪声 = 增强 - 原始】 ====================
    ax3 = axes[2]
    
    if len(full_augmented_list) > 0:
        # 绘制平均噪声
        ax3.plot(time_indices, noise_mean, 'g-', linewidth=1.5, 
                label='Mean Noise', alpha=0.8, zorder=3)
        
        # 填充噪声范围
        ax3.fill_between(time_indices, noise_mean - noise_std, noise_mean + noise_std, 
                        color='green', alpha=0.3, label='±1 Std Dev')
        
        # 零线
        ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax3.set_xlabel('Time Step', fontsize=12)
    ax3.set_ylabel('Noise (Aug - Orig)', fontsize=12)
    ax3.legend(fontsize=11, loc='upper right')
    ax3.grid(True, alpha=0.3, linestyle=':')
    ax3.set_title('Noise Distribution Over Time', fontsize=14, fontweight='bold')
    
    # 添加噪声统计信息
    if len(full_augmented_list) > 0:
        noise_stats_text = f'Noise Mean: {noise_mean.mean():.4e}\n'
        noise_stats_text += f'Noise Std: {noise_std.mean():.4e}\n'
        noise_stats_text += f'Noise Range: [{noise_array.min():.4f}, {noise_array.max():.4f}]'
        ax3.text(0.02, 0.98, noise_stats_text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'full_sequence_mode{mode_idx+1}_comparison.png')
    plt.savefig(save_path, dpi=CONFIG['save_dpi'], bbox_inches='tight')
    print(f"\n  ✅ 对比图已保存：{os.path.basename(save_path)}")
    plt.close()
    
    return {
        'time_indices': time_indices,
        'original': full_original,
        'augmented_samples': full_augmented_list,
        'noise_array': noise_array,
        'noise_mean': noise_mean,
        'noise_std': noise_std,
        'save_path': output_dir
    }


# ==================== 【数据重绘图示例函数】 ====================
def reload_and_replot(data_path, output_dir='./', mode_idx=1):
    """
    从保存的数据文件重新加载并重新绘图
    
    Args:
        data_path: 保存的数据文件路径 (.npz)
        output_dir: 输出目录
        mode_idx: 模态索引
    """
    print(f"\n🔄 从保存的数据重新绘图...")
    
    # 加载数据
    data = np.load(data_path, allow_pickle=True)
    
    time_indices = data['time_indices']
    full_original = data['original_full_sequence']
    full_augmented_list = data['augmented_full_sequences']
    noise_array = data['noise_sequences']
    noise_mean = data['noise_stats']['mean_over_time']
    noise_std = data['noise_stats']['std_over_time']
    
    # 重新绘图（可以自定义样式）
    fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True)
    fig.suptitle(f'Full Time Series Comparison - Mode {mode_idx}\n'
                f'(Replotted from Saved Data)', 
                fontsize=16, fontweight='bold')
    
    # 上图
    axes[0].plot(time_indices, full_original, 'b-', linewidth=2.0, label='Original', alpha=1.0)
    for i, aug in enumerate(full_augmented_list):
        axes[0].plot(time_indices, aug, 'r-', linewidth=1.0, label=f'Augmented #{i+1}' if i==0 else None, alpha=0.3)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Original vs Augmented')
    
    # 中图
    aug_mean = np.mean(full_augmented_list, axis=0)
    aug_std = np.std(full_augmented_list, axis=0)
    axes[1].plot(time_indices, aug_mean, 'r-', linewidth=1.5, label='Mean')
    axes[1].fill_between(time_indices, aug_mean - aug_std, aug_mean + aug_std, alpha=0.2, label='±1 Std')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Augmented Distribution')
    
    # 下图
    axes[2].plot(time_indices, noise_mean, 'g-', linewidth=1.5, label='Mean Noise')
    axes[2].fill_between(time_indices, noise_mean - noise_std, noise_mean + noise_std, alpha=0.3, label='±1 Std')
    axes[2].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Noise Distribution')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'replotted_mode{mode_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 重绘图已保存：{os.path.basename(save_path)}")
    plt.close()


# ==================== 【主函数】 ====================
def main():
    """主函数"""
    print("\n" + "="*60)
    print("🚀 完整时间序列数据增强效果可视化（数据保存版）")
    print("="*60)
    
    # 1. 加载数据
    data = load_augmented_data(CONFIG['augmented_data_path'])
    
    # 2. 完整时间序列对比图 + 数据保存
    result = plot_full_sequence_comparison_with_data_save(
        data['X_train_original'], 
        data['X_train'],
        mode_idx=CONFIG['mode_idx'],
        n_aug_samples=CONFIG['n_augmented_to_show'],
        alpha_orig=CONFIG['alpha_original'],
        alpha_aug=CONFIG['alpha_augmented'],
        lw_orig=CONFIG['line_width_original'],
        lw_aug=CONFIG['line_width_augmented'],
        output_dir=CONFIG['output_dir'],
        save_full_data=CONFIG['save_full_data'],
        save_format=CONFIG['save_format']
    )
    
    # 3. 生成使用说明
    readme_path = os.path.join(CONFIG['output_dir'], 'README_data_usage.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("完整时间序列数据使用说明\n")
        f.write("="*60 + "\n\n")
        f.write("保存的数据文件:\n")
        f.write(f"  - full_sequence_data_mode{CONFIG['mode_idx']+1}.npz: 完整数据 (NPZ 格式)\n")
        f.write(f"  - csv_data_mode{CONFIG['mode_idx']+1}/: CSV 格式数据文件夹\n")
        f.write(f"  - visualization_config_mode{CONFIG['mode_idx']+1}.json: 可视化配置\n\n")
        f.write("NPZ 文件包含的数据:\n")
        f.write("  - time_indices: 时间索引数组\n")
        f.write("  - original_full_sequence: 原始完整时间序列\n")
        f.write("  - augmented_full_sequences: 增强样本完整时间序列 [n_samples, time_steps]\n")
        f.write("  - noise_sequences: 噪声序列 [n_samples, time_steps]\n")
        f.write("  - noise_stats: 噪声统计信息 (均值、标准差等)\n")
        f.write("  - original_stats: 原始数据统计信息\n")
        f.write("  - augmented_stats: 增强数据统计信息\n")
        f.write("  - config: 可视化配置参数\n\n")
        f.write("重新绘图示例代码:\n")
        f.write("```python\n")
        f.write("import numpy as np\n")
        f.write("import matplotlib.pyplot as plt\n")
        f.write("\n")
        f.write("# 加载数据\n")
        f.write(f"data = np.load('full_sequence_data_mode{CONFIG['mode_idx']+1}.npz', allow_pickle=True)\n")
        f.write("time_indices = data['time_indices']\n")
        f.write("original = data['original_full_sequence']\n")
        f.write("augmented = data['augmented_full_sequences']\n")
        f.write("noise = data['noise_sequences']\n")
        f.write("\n")
        f.write("# 绘图\n")
        f.write("fig, axes = plt.subplots(3, 1, figsize=(20, 15))\n")
        f.write("axes[0].plot(time_indices, original, 'b-', label='Original')\n")
        f.write("for aug in augmented:\n")
        f.write("    axes[0].plot(time_indices, aug, 'r-', alpha=0.3)\n")
        f.write("plt.savefig('my_custom_plot.png', dpi=300)\n")
        f.write("```\n")
    
    print(f"\n📄 使用说明已保存：{readme_path}")
    print(f"\n📁 所有输出文件位于：{CONFIG['output_dir']}")
    print("\n" + "="*60)
    print("✅ 可视化和数据保存完成！")
    print("="*60)


if __name__ == "__main__":
    main()