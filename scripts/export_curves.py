import os
import glob
import pickle
import matplotlib.pyplot as plt

print("🔍 步骤 1: 开始寻找【最长】的训练日志...")
base_dir = "runs/rsl_exteroceptive_simple_mybot_v3"
if not os.path.exists(base_dir):
    print(f"❌ 报错: 找不到基础目录 {base_dir}，请确认你在这个目录下运行。")
    exit()

run_dirs = glob.glob(os.path.join(base_dir, "*"))
if not run_dirs:
    print(f"❌ 报错: 在 {base_dir} 里没有找到任何实验文件夹。")
    exit()

longest_run = None
max_data_length = -1
best_data = []

# 自动找出包含数据最多（训练时间最长）的文件夹
for rd in run_dirs:
    pkl_file = os.path.join(rd, "metrics.pkl")
    if os.path.exists(pkl_file):
        try:
            data_blocks = []
            with open(pkl_file, 'rb') as f:
                while True:
                    try:
                        data_blocks.append(pickle.load(f))
                    except EOFError:
                        break
            
            # 如果这个文件夹的数据块比之前的多，就把它作为"最长记录"
            if len(data_blocks) > max_data_length:
                max_data_length = len(data_blocks)
                longest_run = rd
                best_data = data_blocks
        except Exception as e:
            continue

if longest_run is None:
    print("❌ 报错: 没有找到任何包含有效 metrics.pkl 的文件夹！")
    exit()

print(f"✅ 找到数据最丰富的训练文件夹: {longest_run}")
print(f"💡 该文件夹共包含 {max_data_length} 组数据 (对应更长的训练时间)。")

# 强制锁定提取 'rew_total/mean' (总奖励)，不再误提惩罚项！
target_reward_key = 'train/episode/rew_total/mean'

steps = []
rewards = []

for block in best_data:
    if target_reward_key in block and 'timesteps' in block:
        steps.append(block['timesteps'])
        rewards.append(block[target_reward_key])
        
if not steps:
    print(f"❌ 报错: 在数据中没有找到 {target_reward_key} 指标！")
    exit()

print("\n" + "="*40)
print(f"📊 【报告任务 4 数据】")
print(f"▶ 总训练步数: {steps[-1]} 步")
print(f"▶ 最终平均总奖励 (Total Reward): {rewards[-1]:.2f}")
print("="*40 + "\n")

# 开始画图
plt.figure(figsize=(10,6), dpi=300)
plt.plot(steps, rewards, label='Total Mean Reward', linewidth=2, color='#2ca02c') # 换成代表上升的绿色
plt.title('Training Curve - Total Reward')
plt.xlabel('Steps')
plt.ylabel('Reward Value')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('training_reward_curve.png', bbox_inches='tight')
print("✅ 【报告任务 3 图片】已成功生成: training_reward_curve.png (请在Tabby左侧SFTP下载)")
