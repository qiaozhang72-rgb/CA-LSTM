# lstm 模型主体
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from utils import data_generator as data_generator
import time

# ==================== 【数据增强模块】 ====================
class PODNoiseAugmenter:
    """POD 系数噪声增强器（支持课程学习）"""
    
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
    
        print(f"🔄 开始数据增强：原始 {n_original} 个样本 → 增强 {n_augmented} 个样本")
    
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
    
        print(f"✅ 数据增强完成！最终形状：{X_final.shape}")
    
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
            bias=bolaverage,
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
alpha='6'

randomSeed = 40
EPOCH = 1000
LR = 1e-3
input_channels = 20
output_channels = 20
hidden_num = 128
num_layers = 4
batch_size = 64

#fileName = 'pod_lstm_modes20_assin.dat'
diffdatanum = False
yanchi = 10
time_sum = 5000
bolaverage = True
sum_simple = time_sum - yanchi

# ==================== 【训练集划分】 ====================
train_simple_num_original = 4000  # 原始训练样本数（增强前）

# ==================== 【数据增强参数设置】 ====================
AUGMENT_ENABLED = True
AUGMENT_RATIO = 8
NOISE_RANGE_MIN = 0.01
NOISE_RANGE_MAX = 0.05
CURRICULUM_LEARNING = True
TEMPORAL_CORRELATION = True
MODE_DEPENDENT = True

# ==================== 【XLBB 标识符 - 统一所有文件名】 ====================
XLBB = 'M' + Mach + '-AOA' + alpha + '-SZ-' + str(num_layers) + '-' + str(hidden_num) + '-AR' + str(AUGMENT_RATIO) + '-C' + str(yanchi) + '-TN' + str(int(train_simple_num_original/1000)) + 'K'

# ==================== 【递归预测测试参数 - 修复可采样范围】 ====================
RECURSIVE_TEST_ENABLED = True       
RECURSIVE_GUIDE_LENGTH = 10         # 真值引导长度（=窗口长度）
RECURSIVE_PREDICT_LENGTH_TRAIN = 3800    # 训练监控时预测步数
RECURSIVE_TEST_INTERVAL = 5        # 每多少轮计算一次
RECURSIVE_TEST_SAMPLE_RATIO = 0.00001   # 测试样本比例
RECURSIVE_TEST_SAMPLE_MIN = 1           # 最少测试样本数

# --- 最终评估时的完整测试（训练结束后执行） ---
FINAL_EVAL_GUIDE_LENGTH = 256       # 最终评估引导长度
FINAL_EVAL_PREDICT_LENGTH = 'all'   # 最终评估预测步数（'all'=预测到数据结束）

#%% ==================== 【数据准备部分】 ====================
# 输入文件路径
input_f='normalized_modecoeff_M' + Mach + '_AOA' + alpha + '.dat'

file_path = './dataSet/POD/' + input_f

if not os.path.exists(file_path):
    raise FileNotFoundError(f"数据文件未找到：{file_path}")

dataSetcp = np.loadtxt(file_path).T
print(f"✅ 归一化数据加载成功！形状：{dataSetcp.shape}")
print(f"✅ 数据范围：[{dataSetcp.min():.4f}, {dataSetcp.max():.4f}]")

dataSetall = torch.from_numpy(dataSetcp).type(torch.FloatTensor)

# ==================== 【构建滑动窗口数据】 ====================
X = torch.zeros([sum_simple, yanchi, input_channels])
Y = torch.zeros([sum_simple, yanchi, output_channels])
for i in range(sum_simple): 
    X[i,:,:] = dataSetall[i:i+yanchi, 0:input_channels]    
    Y[i,:,:] = dataSetall[i+1:i+1+yanchi, 0:output_channels]

X_train_original = X[0:train_simple_num_original,:,:].clone()
Y_train_original = Y[0:train_simple_num_original,:,:].clone()

print(f"✅ 原始 X_train 形状：{X_train_original.shape}")

#%% ==================== 【应用数据增强】 ====================
if AUGMENT_ENABLED:
    print("\n" + "="*60)
    print("🚀 开始数据增强（带课程学习）")
    print("="*60)
    
    augmenter = PODNoiseAugmenter(
        noise_range=(NOISE_RANGE_MIN, NOISE_RANGE_MAX),
        temporal_correlation=TEMPORAL_CORRELATION,
        mode_dependent=MODE_DEPENDENT,
        curriculum_learning=CURRICULUM_LEARNING,
        random_seed=randomSeed
    )
    
    X_train, Y_train = augmenter.augment_dataset(
        X_train_original, Y_train_original, 
        augment_ratio=AUGMENT_RATIO,
        epoch=0,
        total_epochs=EPOCH
    )
    
    print(f"\n✅ 数据增强后统计：")
    print(f"   X_train 形状：{X_train.shape}")
    print(f"   增强倍数：{len(X_train) / len(X_train_original):.1f}x")
    
    # ==================== 【保存增强数据 - 文件名包含增强信息】 ====================
    augment_info = f'AR{AUGMENT_RATIO}_N{int(NOISE_RANGE_MIN*1000)}-{int(NOISE_RANGE_MAX*1000)}'
    save_path = f'./dataSet/POD/augmented_data_{XLBB}_{augment_info}.npz'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path,
             X_train=X_train.numpy(),
             Y_train=Y_train.numpy(),
             X_train_original=X_train_original.numpy(),
             Y_train_original=Y_train_original.numpy(),
             augment_ratio=AUGMENT_RATIO,
             noise_range=(NOISE_RANGE_MIN, NOISE_RANGE_MAX),
             temporal_correlation=TEMPORAL_CORRELATION,
             mode_dependent=MODE_DEPENDENT,
             curriculum_learning=CURRICULUM_LEARNING)
    print(f"\n💾 增强数据已保存到：{save_path}")
    
    train_simple_num = len(X_train)
else:
    print("\n⚠️  数据增强已关闭，使用原始数据")
    X_train = X_train_original
    Y_train = Y_train_original
    train_simple_num = train_simple_num_original

print("\n" + "="*60)
print("✅ 数据准备完成！可以开始训练")
print("="*60)

#%% ==================== 【归一化设置】 ====================
isXnorm = 0
isYnorm = 0
average_XY = 0
optimthorgy = 'AdamW'

torch.manual_seed(randomSeed)
X_norm = 0
Y_norm = 0

# ==================== 【确保保存目录存在】 ====================
os.makedirs('./model_trained/' + XLBB + '/', exist_ok=True)
os.makedirs('./results_train/'+ XLBB + '/', exist_ok=True)

# ==================== 【实例化网络】 ====================
model = LSTM(input_size=input_channels, output_size=output_channels, 
           hidden_size=hidden_num, num_layers=num_layers)

# ==================== 【定义优化器和学习率策略】 ====================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=30, cooldown=10, min_lr=1e-5)

#%%循环训练模块

loss_func = nn.MSELoss()
test_loss_func = nn.MSELoss()

train_losses=[]
test_losses=[]  # 现在是多训练样本的递归预测平均误差
total_loss=0
lossMin=10
lossMintrain=10

# ==================== 【创建噪声增强器用于训练时动态加噪】 ====================
if AUGMENT_ENABLED:
    train_augmenter = PODNoiseAugmenter(
        noise_range=(NOISE_RANGE_MIN, NOISE_RANGE_MAX),
        temporal_correlation=TEMPORAL_CORRELATION,
        mode_dependent=MODE_DEPENDENT,
        curriculum_learning=CURRICULUM_LEARNING,
        random_seed=randomSeed + 1
    )
else:
    train_augmenter = None

# ==================== 【递归预测误差计算函数 - 修复可采样范围】 ====================
def compute_recursive_error_multi_sample(model, X_data, guide_length, predict_length, 
                                         sample_ratio=0.1, min_samples=10, device='cpu'):
    """
    计算多个训练样本的递归预测平均误差（修复可采样范围）
    Args:
        model: LSTM 模型
        X_data: 完整滑动窗口数据 [n_samples, window_size, n_modes]
        guide_length: 真值引导步数
        predict_length: 递归预测步数（'all'=预测到数据结束）
        sample_ratio: 对多少比例的可采样样本进行测试
        min_samples: 最少测试多少个样本
        device: 计算设备
    Returns:
        avg_recursive_error: 平均递归预测相对误差
        n_tested: 实际测试的样本数
        n_valid: 可采样样本总数
    """
    model.eval()
    
    n_total_samples = len(X_data)
    
    # ==================== 【修复】计算最大可采样索引 ====================
    if predict_length == 'all':
        max_predict_length = n_total_samples - guide_length - 1
    else:
        max_predict_length = predict_length
    
    # 最大可采样索引 = 总样本数 - 引导长度 - 预测长度 - 1
    # 因为样本 idx 对应时刻 [idx, idx+19]，预测需要访问到 idx+19+guide+predict
    max_sample_idx = n_total_samples - guide_length - max_predict_length - 1
    max_sample_idx = max(0, max_sample_idx)  # 确保非负
    
    # 可采样样本数
    n_valid_samples = max_sample_idx + 1
    
    if n_valid_samples < min_samples:
        print(f"⚠️  警告：可采样样本数 ({n_valid_samples}) 少于最少样本数 ({min_samples})")
        n_valid_samples = min(min_samples, n_total_samples)
        max_sample_idx = n_valid_samples - 1
    
    # 计算需要测试的样本数
    n_test_samples = max(min_samples, int(n_valid_samples * sample_ratio))
    n_test_samples = min(n_test_samples, n_valid_samples)
    
    # ==================== 【修复】只在可采样范围内随机选择 ====================
    np.random.seed(int(time.time()) % 10000)
    test_indices = np.random.choice(n_valid_samples, n_test_samples, replace=False)
    
    recursive_errors = []
    
    for idx in test_indices:
        # 获取初始窗口
        initial_window = X_data[idx:idx+1].to(device)  # [1, 20, 20]
        
        # 获取真值序列（用于对比）
        true_sequence = []
        total_steps = guide_length + max_predict_length
        
        for t in range(total_steps):
            if idx + t + 1 < len(X_data):
                true_sequence.append(X_data[idx + t + 1, -1, :].numpy())
            else:
                break
        
        if len(true_sequence) < guide_length + 10:  # 至少预测 10 步
            continue
        
        true_sequence = np.array(true_sequence)
        
        # 递归预测
        predictions = []
        history = initial_window.clone()
        
        with torch.no_grad():
            # 引导阶段（用真值）
            for t in range(min(guide_length, len(true_sequence))):
                output = model(history)
                if output.dim() == 3:
                    output = output[:, -1, :]
                predictions.append(output[0].cpu().numpy())
                
                # 用真值更新历史
                true_next = torch.FloatTensor(X_data[idx + t + 1, :, :]).unsqueeze(0).to(device)
                history = torch.cat([history[:, 1:, :], true_next[:, -1:, :]], dim=1)
            
            # 递归阶段（用预测值）
            for t in range(max_predict_length):
                if len(predictions) >= len(true_sequence):
                    break
                output = model(history)
                if output.dim() == 3:
                    output = output[:, -1, :]
                predictions.append(output[0].cpu().numpy())
                
                # 用预测值更新历史
                history = torch.cat([history[:, 1:, :], output.unsqueeze(1)], dim=1)
        
        if len(predictions) < guide_length + 10:
            continue
        
        predictions = np.array(predictions)
        
        # 计算递归阶段的误差
        recursive_predictions = predictions[guide_length:]
        recursive_true = true_sequence[guide_length:]
        
        if len(recursive_true) > 0:
            recursive_error = np.sum(np.abs(recursive_predictions - recursive_true)) / (np.sum(np.abs(recursive_true)) + 1e-8)
            recursive_errors.append(recursive_error)
    
    if len(recursive_errors) > 0:
        avg_recursive_error = np.mean(recursive_errors)
    else:
        avg_recursive_error = 10.0
    
    return avg_recursive_error, len(recursive_errors), n_valid_samples

# 计算可采样样本数
if RECURSIVE_PREDICT_LENGTH_TRAIN == 'all':
    max_predict = train_simple_num_original - RECURSIVE_GUIDE_LENGTH - 1
else:
    max_predict = RECURSIVE_PREDICT_LENGTH_TRAIN

n_valid_samples = max(0, train_simple_num_original - RECURSIVE_GUIDE_LENGTH - max_predict - 1)
n_recursive_test_samples = max(RECURSIVE_TEST_SAMPLE_MIN, int(n_valid_samples * RECURSIVE_TEST_SAMPLE_RATIO))
n_recursive_test_samples = min(n_recursive_test_samples, n_valid_samples)

print("\n" + "="*60)
print(f"📊 递归测试设置：")
print(f"   引导长度：{RECURSIVE_GUIDE_LENGTH} 步")
print(f"   预测长度：{RECURSIVE_PREDICT_LENGTH_TRAIN} 步")
print(f"   测试间隔：{RECURSIVE_TEST_INTERVAL} 轮")
print(f"   可采样样本：{n_valid_samples} 个")
print(f"   测试样本比例：{RECURSIVE_TEST_SAMPLE_RATIO*100:.4f}%")
print(f"   每次测试样本数：{n_recursive_test_samples} 个")
print("="*60)

for epoch in range(EPOCH):
    model.train()
    total_loss = 0
    
    indices = torch.randperm(train_simple_num)
    
    for i in range(0, train_simple_num, batch_size):
        batch_idx = indices[i:i+batch_size]
        
        if diffdatanum == True:
            x, y = X_train[batch_idx], Y_train[batch_idx]
        else:
            x = X_train[batch_idx]
            y = Y_train[batch_idx]
        
        # ==================== 【训练时动态加噪】 ====================
        if train_augmenter is not None and AUGMENT_ENABLED:
            x_numpy = x.numpy()
            y_numpy = y.numpy()
            
            x_aug_list = []
            for j in range(len(x)):
                x_aug, _ = train_augmenter.augment_sample(
                    x_numpy[j], y_numpy[j], epoch, EPOCH
                )
                x_aug_list.append(x_aug)
            x = torch.FloatTensor(np.array(x_aug_list))
        
        optimizer.zero_grad()
        output = model(x)
        loss = loss_func(output, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        
    train_losses.append(total_loss/train_simple_num)
    
    # ==================== 【递归预测误差作为 test_loss - 多样本平均】 ====================
    model.eval()
    
    if RECURSIVE_TEST_ENABLED and (epoch + 1) % RECURSIVE_TEST_INTERVAL == 0:
        print(f"\n🔍 计算递归预测误差（可采样{n_valid_samples}个中的{n_recursive_test_samples}个）...")
        test_loss1, n_tested, n_valid = compute_recursive_error_multi_sample(
            model, X_train_original, 
            guide_length=RECURSIVE_GUIDE_LENGTH,
            predict_length=RECURSIVE_PREDICT_LENGTH_TRAIN,
            sample_ratio=RECURSIVE_TEST_SAMPLE_RATIO,
            min_samples=RECURSIVE_TEST_SAMPLE_MIN,
            device='cpu'
        )
        test_losses.append(test_loss1)
        print(f"✅ 递归测试完成：可采样{n_valid}个，实际测试{n_tested}个样本，平均误差={test_loss1:.6f}")
    else:
        # 不计算递归误差的轮次，用上一轮的值
        if len(test_losses) > 0:
            test_losses.append(test_losses[-1])
        else:
            test_losses.append(10.0)
        test_loss1 = test_losses[-1]
    
    scheduler3.step(train_losses[-1])
    
    print('Epoch: ', epoch, '| lr:%.2e' % optimizer.state_dict()['param_groups'][0]['lr'], 
          '| train loss:%.2e' % (total_loss/train_simple_num),
          '| recursive test loss:%.2e' % test_loss1)
   
    total_loss = 0
    # ==================== 【保存模型 - 文件名包含 XLBB】 ====================
    if test_loss1 < lossMin:
        torch.save([model.state_dict(), input_channels, output_channels,
                    hidden_num, num_layers, bolaverage, diffdatanum],
                   f'./model_trained/{XLBB}/{randomSeed}_{num_layers}_{hidden_num}_{optimthorgy}_{yanchi}_{XLBB}.pt01')
        lossMin = test_loss1
        print(f"  💾 保存最佳递归测试模型 (recursive_test_loss={test_loss1:.2e})")
        
    if train_losses[-1] < lossMintrain:
        torch.save([model.state_dict(), input_channels, output_channels,
                    hidden_num, num_layers, bolaverage, diffdatanum],
                   f'./model_trained/{XLBB}/{randomSeed}_{num_layers}_{hidden_num}_{optimthorgy}_{yanchi}_{XLBB}.pt02')
        lossMintrain = train_losses[-1]
        
    if (epoch + 1) % 200 == 0:
        torch.save([model.state_dict(), input_channels, output_channels,
                    hidden_num, num_layers, bolaverage, diffdatanum],
                   f'./model_trained/{XLBB}/{randomSeed}_{num_layers}_{hidden_num}_{optimthorgy}_{epoch+1}_{yanchi}_{XLBB}.pt')

#%% ==================== 【保存最终模型】 ====================
torch.save([model.state_dict(), input_channels, output_channels,
            hidden_num, num_layers, bolaverage, diffdatanum],
           f'./model_trained/{XLBB}/{randomSeed}_{num_layers}_{hidden_num}_{optimthorgy}_{yanchi}_{XLBB}_fin.pt')

# 保存损失历史
hisI = len(test_losses)
history = np.zeros([5, hisI])
history[0,:] = range(hisI)
history[1,:] = train_losses[:hisI]
history[2,:] = test_losses 
np.savetxt(f'./results_train/{XLBB}/lossHistory_{XLBB}_{num_layers}_{randomSeed}_{hidden_num}_{optimthorgy}_{isXnorm}_{yanchi}',
           history.T, fmt='%.15e')

# ==================== 【可视化：训练损失曲线 - 文件名包含 XLBB】 ====================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', label='Train Loss', linewidth=1.5)
plt.plot(test_losses, 'r-', label='Recursive Test Loss', linewidth=1.5)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.title(f'Training and Recursive Test Loss Curves ({XLBB})', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle=':')

loss_data = np.column_stack([range(len(train_losses)), train_losses, test_losses])
np.savetxt(f'./results_train/{XLBB}/loss_data_{XLBB}.csv', loss_data, fmt='%.15e', 
           header='Epoch,Train_Loss,Recursive_Test_Loss')

plt.subplot(1, 2, 2)
window = 50
train_smooth = np.convolve(train_losses, np.ones(window)/window, mode='valid')
test_smooth = np.convolve(test_losses, np.ones(window)/window, mode='valid')
plt.plot(train_smooth, 'b-', label='Train (smoothed)', linewidth=1.5, alpha=0.7)
plt.plot(test_smooth, 'r-', label='Test (smoothed)', linewidth=1.5, alpha=0.7)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss (smoothed)', fontsize=12)
plt.title(f'Loss Curves (Moving Average, window={window})', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig(f'./results_train/{XLBB}/training_curves_{XLBB}.png', dpi=300, bbox_inches='tight')
print(f"✅ 训练曲线已保存：./results_train/{XLBB}/training_curves_{XLBB}.png")
plt.show()

#%% ==================== 【预测评估 - 方式一：单步预测】 ====================
print("\n" + "="*60)
print("开始单步预测评估...")
print("="*60)

modelFileName = f'{randomSeed}_{num_layers}_{hidden_num}_{optimthorgy}_{yanchi}_{XLBB}.pt01'
modelPara, input_channels_load, output_channels_load, \
hidden_num_load, num_layers_load, bolaverage_load, diffdatanum_load = torch.load('./model_trained/' + XLBB + '/' + modelFileName)

modelpre = LSTM(input_size=input_channels_load, output_size=output_channels_load, 
           hidden_size=hidden_num_load, num_layers=num_layers_load)
modelpre.load_state_dict(modelPara)
modelpre.eval()

output1 = modelpre(X)
output1_np = output1.detach().numpy()
outdata = output1_np[:, -1, :]

Datareal = dataSetcp[yanchi:, 0:input_channels_load]
dataerror = Datareal - outdata
errorpre = np.sum(np.abs(dataerror)) / np.sum(np.abs(Datareal))
errorpre = round(errorpre, 6)

print(f"✅ 真实值范围：[{Datareal.min():.4f}, {Datareal.max():.4f}]")
print(f"✅ 预测值范围：[{outdata.min():.4f}, {outdata.max():.4f}]")
print(f"✅ 单步预测相对误差：{errorpre}")

Timedata = np.ones((sum_simple, 2))
Timedata[:,0] = list(range(sum_simple))
Timedata[:,1] = Timedata[:,1] * 0.005 * 6
for inum in range(sum_simple):
    Timedata[inum,0] = Timedata[inum,1] * Timedata[inum,0]

outputData = np.concatenate((Timedata[:,0:1], Datareal, outdata), axis=1)
np.savetxt(f'./results_train/{XLBB}/train_in_out_{XLBB}_{num_layers_load}_{randomSeed}_{hidden_num_load}_{optimthorgy}_{isXnorm}_{errorpre}_{yanchi}', 
           outputData, fmt='%.15e')

#%% ==================== 【预测评估 - 方式二：自回归预测】 ====================
print("\n" + "="*60)
print("开始自回归预测评估...")
print("="*60)

#modelFileName = f'{randomSeed}_{num_layers}_{hidden_num}_{optimthorgy}_{yanchi}_{XLBB}.pt02'
modelFileName = '40_4_128_AdamW_1000_10_M0.71-AOA6-SZ-4-128-AR8-C10-TN4K.pt'
modelPara, input_channels_load2, output_channels_load2, \
hidden_num_load2, num_layers_load2, bolaverage_load2, diffdatanum_load2 = torch.load('./model_trained/' + XLBB + '/' + modelFileName)
#modelPara, input_channels_load2, output_channels_load2, \
#hidden_num_load2, num_layers_load2, bolaverage_load2, diffdatanum_load2 = torch.load('./model_trained/' + XLBB + '/' + modelFileName)
modelpre2 = LSTM(input_size=input_channels_load2, output_size=output_channels_load2, 
           hidden_size=hidden_num_load2, num_layers=num_layers_load2)
modelpre2.load_state_dict(modelPara)
modelpre2.eval()

Timedata = np.ones((sum_simple, 2))
Timedata[:,0] = list(range(sum_simple))
Timedata[:,1] = Timedata[:,1] * 0.005 * 6
for inum in range(sum_simple):
    Timedata[inum,0] = Timedata[inum,1] * Timedata[inum,0]

Datareal = dataSetcp[yanchi:, 0:input_channels_load2]

# ==================== 【可配置的引导长度】 ====================
#pre_c_num = FINAL_EVAL_GUIDE_LENGTH
pre_c_num = 512
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

print(f"✅ 真实值范围：[{Datareal.min():.4f}, {Datareal.max():.4f}]")
print(f"✅ 预测值范围：[{outdata2.min():.4f}, {outdata2.max():.4f}]")

dataerror2 = Datareal - outdata2
errorpre2 = np.sum(np.abs(dataerror2)) / np.sum(np.abs(Datareal))
errorpre2 = round(errorpre2, 6)
print(f"✅ 自回归预测相对误差：{errorpre2}")
print(f"✅ 引导长度：{pre_c_num} 步")

outputData2 = np.concatenate((Timedata[:,0:1], Datareal, outdata2), axis=1)
np.savetxt('./results_train/' + XLBB + '/train_assin_' + modelFileName + str(errorpre2) + '.dat', 
           outputData2, fmt='%.15e')

print("\n" + "="*60)
print("✅✅✅ 训练和评估完成！✅✅✅")
print("="*60)

# ==================== 【可视化：真值 vs 预测值对比 - 文件名包含 XLBB】 ====================
print("\n" + "="*60)
print("📊 生成预测结果对比图...")
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
    
    save_path = f'./results_train/{XLBB}/prediction_comparison_{XLBB}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存：{save_path}")
    
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

detail_file = f'./results_train/{XLBB}/detailed_comparison_{XLBB}.csv'
with open(detail_file, 'w') as f:
    f.write("Step,Mode,Real_Value,Predicted_Value,Absolute_Error,Relative_Error_Percent\n")
    for j in range(min(5, input_channels_load2)):
        for i in range(sum_simple):
            real_val = Datareal[i, j]
            pred_val = outdata2[i, j]
            abs_err = abs(real_val - pred_val)
            rel_err = abs_err / (abs(real_val) + 1e-8) * 100
            f.write(f"{i},{j+1},{real_val:.6f},{pred_val:.6f},{abs_err:.6f},{rel_err:.4f}\n")

print(f"✅ 详细对比数据已保存：{detail_file}")

print("\n" + "="*60)
print("✅✅✅ 可视化对比完成！✅✅✅")
print("="*60)

#%% ==================== 【生成训练配置总结文件】 ====================
config_summary = f"""
========================================
训练配置总结 ({XLBB})
========================================
时间：{time.strftime('%Y-%m-%d %H:%M:%S')}

【模型参数】
- 随机种子：{randomSeed}
- LSTM 层数：{num_layers}
- 隐藏层维度：{hidden_num}
- 学习率：{LR}
- 批次大小：{batch_size}
- 训练轮数：{EPOCH}

【数据增强】
- 增强开关：{AUGMENT_ENABLED}
- 增强倍数：{AUGMENT_RATIO}x
- 噪声范围：{NOISE_RANGE_MIN} - {NOISE_RANGE_MAX}
- 课程学习：{CURRICULUM_LEARNING}
- 时序相关：{TEMPORAL_CORRELATION}
- 模态依赖：{MODE_DEPENDENT}

【递归测试】
- 递归测试：{RECURSIVE_TEST_ENABLED}
- 引导长度：{RECURSIVE_GUIDE_LENGTH} 步
- 预测长度：{RECURSIVE_PREDICT_LENGTH_TRAIN} 步
- 测试间隔：{RECURSIVE_TEST_INTERVAL} 轮
- 测试样本比例：{RECURSIVE_TEST_SAMPLE_RATIO*100:.4f}%
- 每次测试样本数：{n_recursive_test_samples} 个
- 最少测试样本：{RECURSIVE_TEST_SAMPLE_MIN} 个

【最终评估】
- 评估引导长度：{FINAL_EVAL_GUIDE_LENGTH} 步
- 评估预测长度：{FINAL_EVAL_PREDICT_LENGTH}

【数据划分】
- 原始训练样本：{train_simple_num_original} 个
- 增强后训练样本：{train_simple_num} 个
- 窗口长度：{yanchi} 步
- 总时间步：{time_sum} 步

【最终结果】
- 最佳递归测试误差：{lossMin:.6f}
- 单步预测误差：{errorpre:.6f}
- 自回归预测误差：{errorpre2:.6f}
========================================
"""

with open(f'./results_train/{XLBB}/config_summary_{XLBB}.txt', 'w', encoding='utf-8') as f:
    f.write(config_summary)

print(f"✅ 配置总结已保存：./results_train/{XLBB}/config_summary_{XLBB}.txt")
print("\n" + "="*60)