# lstm 模型主体
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import random
from utils import data_generator as data_generator
import time
from predict import normalize_path

# ==================== 【数据增强模块】 ====================
class PODNoiseAugmenter:
    """POD 系数噪声增强器（课程学习）"""
    
    def __init__(self, 
                 noise_range=(0.01, 0.02),
                 temporal_correlation=True,
                 mode_dependent=True,
                 curriculum_learning=True,
                 random_seed=40):
        self.noise_min, self.noise_max = noise_range
        self.temporal_correlation = temporal_correlation
        self.mode_dependent = mode_dependent
        self.curriculum_learning = curriculum_learning
        self.random_seed = random_seed
        self.curriculum_progress = 0.0
        
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    def set_curriculum_progress(self, progress):
        self.curriculum_progress = max(0.0, min(1.0, progress))
    
    def get_current_noise_range(self):
        if not self.curriculum_learning:
            return self.noise_min, self.noise_max
        current_min = self.noise_min * (0.5 + 0.5 * self.curriculum_progress)
        current_max = self.noise_max * (0.5 + 1.0 * self.curriculum_progress)
        return current_min, current_max
    
    def get_mode_noise_scale(self, mode_idx, n_modes):
        if not self.mode_dependent:
            return 1.0
        mode_weight = 1.0 + (mode_idx / n_modes)
        return mode_weight
    
    def generate_temporal_noise(self, length, noise_scale):
        if self.temporal_correlation:
            noise = np.zeros(length)
            ar_coef = 0.7
            for t in range(length):
                if t == 0:
                    noise[t] = np.random.randn() * noise_scale
                else:
                    noise[t] = ar_coef * noise[t-1] + (1-ar_coef) * np.random.randn() * noise_scale
            return noise
        else:
            return np.random.randn(length) * noise_scale
    
    def augment_sample(self, X_sample, y_sample, epoch=0, total_epochs=100):
        if self.curriculum_learning:
            self.set_curriculum_progress(epoch / total_epochs)
        
        noise_min, noise_max = self.get_current_noise_range()
        X_aug = X_sample.copy()
        window_size, n_modes = X_aug.shape
        
        for mode_idx in range(n_modes):
            mode_scale = self.get_mode_noise_scale(mode_idx, n_modes)
            base_noise = self.generate_temporal_noise(window_size, 1.0)
            
            for t in range(window_size):
                temporal_weight = 0.5 + 0.5 * (t / window_size)
                current_noise = np.random.uniform(noise_min, noise_max)
                current_noise *= mode_scale * temporal_weight
                noise_type = np.random.choice(['amplitude', 'offset', 'mixed'])
                
                if noise_type == 'amplitude':
                    epsilon = base_noise[t] * current_noise
                    X_aug[t, mode_idx] *= (1 + epsilon)
                elif noise_type == 'offset':
                    epsilon = base_noise[t] * current_noise
                    X_aug[t, mode_idx] += epsilon
                elif noise_type == 'mixed':
                    epsilon1 = base_noise[t] * current_noise * 0.5
                    epsilon2 = base_noise[t] * current_noise * 0.5
                    X_aug[t, mode_idx] = X_aug[t, mode_idx] * (1 + epsilon1) + epsilon2
        
        X_aug = np.clip(X_aug, -2.0, 2.0)
        return X_aug, y_sample
    
    def augment_dataset(self, X_original, Y_original, augment_ratio=4, 
                   epoch=0, total_epochs=100):
        n_original = len(X_original)
        n_augmented = n_original * augment_ratio
    
        print(f" 开始数据增强：原始 {n_original} 个样本 → 增强 {n_augmented} 个样本")
    
        X_aug_list = [X_original.numpy()]
        Y_aug_list = [Y_original.numpy()]
    
        for i in range(n_augmented):
            idx = np.random.randint(0, n_original)
            X_sample = X_original[idx].numpy()
            y_sample = Y_original[idx].numpy()
        
            X_aug, y_aug = self.augment_sample(X_sample, y_sample, epoch, total_epochs)
        
            X_aug_list.append(X_aug.reshape(1, X_aug.shape[0], X_aug.shape[1]))
            Y_aug_list.append(y_aug.reshape(1, y_aug.shape[0], y_aug.shape[1]))
        
            if (i + 1) % 1000 == 0:
                print(f"  已增强 {i+1}/{n_augmented} 个样本")
    
        X_final = np.concatenate(X_aug_list, axis=0)
        Y_final = np.concatenate(Y_aug_list, axis=0)
    
        print(f"数据增强完成！最终形状：{X_final.shape}")
    
        return torch.FloatTensor(X_final), torch.FloatTensor(Y_final)


#定义损失函数
class myMseLoss(torch.nn.Module):
    def __init__(self):
        super(myMseLoss,self).__init__()
    def forward(self, y_t, y_prime_t):
        ran = torch.arange(0, y_t.shape[1]).type(torch.FloatTensor)
        coeff =60*torch.exp(-0.2*ran).view(1,-1,1)+1
        ey_t = (y_t - y_prime_t)*coeff
        return torch.mean(ey_t**2)
    
    
class myMseLoss_2(torch.nn.Module):
    def __init__(self):
        super(myMseLoss_2,self).__init__()
    def forward(self, y_t, y_prime_t):
        ran = torch.arange(0, y_t.shape[1]).type(torch.FloatTensor)
        coeff =100*torch.exp(-0.2*ran).view(1,-1,1)+1
        ey_t = (y_t - y_prime_t)*coeff
        return 5*torch.mean(ey_t**2)
    
#定义 LSTM 模型框架 
class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=10, num_layers=1):
        super(LSTM,self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        r_out, _ = self.rnn(x)
        out = self.out(r_out)
        return out

#define para init，定义模型参数初始化
def init_para(model):
    for name, param in model.named_parameters():
        if 'weight_ih' in name:
            torch.nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)
        else :
            torch.nn.init.xavier_uniform_(param.data)

#%% ==================== 神经网络参数设置 ====================

Mach='0.71'
alpha='5.2'
yanchi = 20
pre_c_num = 256
longtimepre=0
modelFilepath = "./model_trained/SZ-3-64-AR4-C20-TN4K-MC5cond-T1"
modelFileName = "40_3_64_AdamW_20_SZ-3-64-AR4-C20-TN4K-MC5cond-T1.pt01"

#modelFile=modelFilepath + '\' + modelFileName

modelPara, input_channels_load2, output_channels_load2, \
hidden_num_load2, num_layers_load2, bolaverage_load2, diffdatanum_load2 = torch.load(normalize_path(modelFilepath)+'/'+modelFileName)

modelpre2 = LSTM(input_size=input_channels_load2, output_size=output_channels_load2, 
           hidden_size=hidden_num_load2, num_layers=num_layers_load2)
modelpre2.load_state_dict(modelPara)
modelpre2.eval()


randomSeed = 40
#EPOCH = 1000
#LR = 1e-3
input_channels = input_channels_load2
output_channels = output_channels_load2
hidden_num = hidden_num_load2
num_layers = num_layers_load2
#batch_size = 64

#fileName = 'pod_lstm_modes20_assin.dat'
#diffdatanum = False

#time_sum = 5000
bolaverage = True
#sum_simple = time_sum - yanchi

# ==================== 【训练集划分】 ====================
#train_simple_num_original = 4000  # 原始训练样本数（增强前）

# ==================== 【数据增强参数设置】 ====================
#AUGMENT_ENABLED = True
#AUGMENT_RATIO = 8
#NOISE_RANGE_MIN = 0.01
#NOISE_RANGE_MAX = 0.05
#CURRICULUM_LEARNING = True
#TEMPORAL_CORRELATION = True
#MODE_DEPENDENT = True

# ==================== 【XLBB 标识符 - 统一所有文件名】 ====================
XLBB = modelFileName

os.makedirs('./results_asspre/'+ XLBB + '/', exist_ok=True)
# ==================== 【递归预测测试参数 - 修复可采样范围】 ====================
#RECURSIVE_TEST_ENABLED = True       
#RECURSIVE_GUIDE_LENGTH = 10         # 真值引导长度（=窗口长度）
#RECURSIVE_PREDICT_LENGTH_TRAIN = 3800    # 训练监控时预测步数
#RECURSIVE_TEST_INTERVAL = 5        # 每多少轮计算一次
#RECURSIVE_TEST_SAMPLE_RATIO = 0.00001   # 测试样本比例
#RECURSIVE_TEST_SAMPLE_MIN = 1           # 最少测试样本数

# --- 最终评估时的完整测试（训练结束后执行） ---
#FINAL_EVAL_GUIDE_LENGTH = 256       # 最终评估引导长度
#FINAL_EVAL_PREDICT_LENGTH = 'all'   # 最终评估预测步数（'all'=预测到数据结束）

#%% ==================== 【数据准备部分】 ====================
# 输入文件路径
input_f='normalized_modecoeff_M' + Mach + '_AOA' + alpha + '.dat'

file_path = './dataSet/POD/' + input_f

if not os.path.exists(file_path):
    raise FileNotFoundError(f"数据文件未找到：{file_path}")

dataSetcp = np.loadtxt(file_path).T
print(f" 归一化数据加载成功！形状：{dataSetcp.shape}")
print(f" 数据范围：[{dataSetcp.min():.4f}, {dataSetcp.max():.4f}]")

dataSetall = torch.from_numpy(dataSetcp).type(torch.FloatTensor)

sum_simple2,_=dataSetcp.shape
sum_simple2=sum_simple2-yanchi
sum_simple=sum_simple2
# ==================== 【构建滑动窗口数据】 ====================
if longtimepre != 0:
    sum_simple=longtimepre

X = torch.zeros([sum_simple, yanchi, input_channels])
Y = torch.zeros([sum_simple, yanchi, output_channels])
for i in range(sum_simple2): 
    X[i,:,:] = dataSetall[i:i+yanchi, 0:input_channels]    
    Y[i,:,:] = dataSetall[i+1:i+1+yanchi, 0:output_channels]

#X_train_original = X[0:train_simple_num_original,:,:].clone()
#Y_train_original = Y[0:train_simple_num_original,:,:].clone()

#print(f"原始 X_train 形状：{X_train_original.shape}")
#%% ==================== 【归一化设置】 ====================
isXnorm = 0
isYnorm = 0
average_XY = 0
optimthorgy = 'AdamW'

torch.manual_seed(randomSeed)
X_norm = 0
Y_norm = 0


#%% ==================== 【预测评估 - 方式二：自回归预测】 ====================
print("\n" + "="*60)
print("开始自回归预测评估...")
print("="*60)






Timedata = np.ones((sum_simple, 2))
Timedata[:,0] = list(range(sum_simple))
Timedata[:,1] = Timedata[:,1] * 0.0001
for inum in range(sum_simple):
    Timedata[inum,0] = Timedata[inum,1] * Timedata[inum,0]

Datareal = dataSetcp[yanchi:, 0:input_channels_load2]

# ==================== 【可配置的引导长度】 ====================
#pre_c_num = FINAL_EVAL_GUIDE_LENGTH

X_pre = X.clone()
outdata2 = np.zeros((sum_simple, input_channels_load2))

output2 = modelpre2(X_pre[0:pre_c_num])
output2_np = output2.detach().numpy()

for inum in range(pre_c_num):
    outdata2[inum,:] = output2_np[inum, -1, :]

X_pre[pre_c_num,:,:] = torch.from_numpy(output2_np[-1,:,:]).type(torch.FloatTensor)
for inum in range(pre_c_num, sum_simple-1, 1):
    output3 = modelpre2(X_pre[inum:inum+1])
    X_pre[inum+1,:,:] = output3[:,:,:]
    outdata2[inum,:] = output3.detach().numpy()[0, -1, :]

inum = inum + 1
output3 = modelpre2(X_pre[inum:inum+1])
outdata2[inum,:] = output3.detach().numpy()[0, -1, :]

if longtimepre != 0:
    outputData2 = np.concatenate((Timedata[:,0:1], outdata2), axis=1)
    zhuangtai='M' + Mach + '_AOA' + alpha + '_YC' + str(yanchi) + '_PCN' + str(pre_c_num) + '_LT' + str(longtimepre)
    np.savetxt('./results_asspre/' + XLBB + '/' + zhuangtai + '.dat', 
               outputData2, fmt='%.15e')
    sys.exit()
print(f" 真实值范围：[{Datareal.min():.4f}, {Datareal.max():.4f}]")
print(f" 预测值范围：[{outdata2.min():.4f}, {outdata2.max():.4f}]")

dataerror2 = Datareal - outdata2
errorpre2 = np.sum(np.abs(dataerror2)) / np.sum(np.abs(Datareal))
errorpre2 = round(errorpre2, 6)
print(f" 自回归预测相对误差：{errorpre2}")
print(f" 引导长度：{pre_c_num} 步")

outputData2 = np.concatenate((Timedata[:,0:1], Datareal, outdata2), axis=1)
zhuangtai='M' + Mach + '_AOA' + alpha + '_YC' + str(yanchi) + '_PCN' + str(pre_c_num) + '_e' + str(errorpre2)
np.savetxt('./results_asspre/' + XLBB + '/' + zhuangtai + '.dat', 
           outputData2, fmt='%.15e')

print("\n" + "="*60)
print(" 训练和评估完成！")
print("="*60)

# ==================== 【可视化：真值 vs 预测值对比 - 文件名包含 XLBB】 ====================
print("\n" + "="*60)
print("生成预测结果对比图...")
print("="*60)

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'Prediction vs Ground Truth ({XLBB})', fontsize=16, fontweight='bold')
    
    for j in range(min(3, input_channels_load2)):
        ax = axes[j//2, j%2]
        
        ax.plot(range(sum_simple), Datareal[:, j], label='Ground Truth', 
                linewidth=1.5, color='blue', alpha=0.8)
        ax.plot(range(sum_simple), outdata2[:, j], label='Prediction', 
                linewidth=1.5, color='red', linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Normalized Coefficient', fontsize=10)
        ax.set_title(f'Mode {j+1}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        modal_error = np.sum(np.abs(Datareal[:, j] - outdata2[:, j])) / np.sum(np.abs(Datareal[:, j])) * 100
        ax.text(0.02, 0.98, f'Error: {modal_error:.2f}%', transform=ax.transAxes, 
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax = axes[2, 0]
    ax.scatter(Datareal.flatten(), outdata2.flatten(), alpha=0.2, s=1, c='blue', edgecolors='none')
    
    min_val = min(Datareal.min(), outdata2.min())
    max_val = max(Datareal.max(), outdata2.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit (y=x)')
    
    ax.set_xlabel('Ground Truth (Normalized)', fontsize=10)
    ax.set_ylabel('Prediction (Normalized)', fontsize=10)
    ax.set_title('Scatter Plot: All Modes', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    correlation = np.corrcoef(Datareal.flatten(), outdata2.flatten())[0, 1]
    ax.text(0.02, 0.98, f'Correlation: {correlation:.4f}', transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax = axes[2, 1]
    errors = (Datareal - outdata2).flatten()
    ax.hist(errors, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    
    ax.set_xlabel('Error (Ground Truth - Prediction)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Error Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    ax.text(0.02, 0.98, f'Mean: {errors.mean():.2e}\nStd: {errors.std():.2e}', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    save_path = f'./results_asspre/{XLBB}/prediction_comparison_{zhuangtai}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" 对比图已保存：{save_path}")
    
    plt.show()
    
except ImportError:
    print("⚠️ 未安装 matplotlib，跳过可视化")

# ==================== 【输出详细数值对比】 ====================
print("\n📋 前 20 个时间步详细对比（模态 1，归一化空间）：")
print(f"{'Step':<6} {'Real':>18} {'Pred':>18} {'Error':>18} {'RelErr(%)':>12}")
print("-" * 75)
for i in range(min(20, sum_simple)):
    real_val = Datareal[i, 0]
    pred_val = outdata2[i, 0]
    abs_err = abs(real_val - pred_val)
    rel_err = abs_err / (abs(real_val) + 1e-8) * 100
    print(f"{i:<6} {real_val:>18.6f} {pred_val:>18.6f} {abs_err:>18.6f} {rel_err:>12.2f}")

detail_file = f'./results_asspre/{XLBB}/detailed_comparison_{zhuangtai}.csv'
with open(detail_file, 'w') as f:
    f.write("Step,Mode,Real_Value,Predicted_Value,Absolute_Error,Relative_Error_Percent\n")
    for j in range(min(5, input_channels_load2)):
        for i in range(sum_simple):
            real_val = Datareal[i, j]
            pred_val = outdata2[i, j]
            abs_err = abs(real_val - pred_val)
            rel_err = abs_err / (abs(real_val) + 1e-8) * 100
            f.write(f"{i},{j+1},{real_val:.6f},{pred_val:.6f},{abs_err:.6f},{rel_err:.4f}\n")

print(f" 详细对比数据已保存：{detail_file}")

print("\n" + "="*60)
print(" 可视化对比完成！")
print("="*60)
