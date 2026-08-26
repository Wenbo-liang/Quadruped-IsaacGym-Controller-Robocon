import os
import glob
import pickle
import matplotlib.pyplot as plt

base_dir = "runs/rsl_exteroceptive_simple_mybot_v3"
runs = sorted(glob.glob(os.path.join(base_dir, "*")), key=os.path.getctime)

def load_data(run_path):
    pkl_path = os.path.join(run_path, "metrics.pkl")
    if not os.path.exists(pkl_path):
        return None, None
    raw_data = []
    with open(pkl_path, 'rb') as f:
        while True:
            try:
                raw_data.append(pickle.load(f))
            except EOFError:
                break
    if not raw_data or not isinstance(raw_data[0], dict):
        return None, None
    
    rk = 'train/episode/rew_total/mean'
    sk = 'timesteps'
    rewards, steps = [], []
    for i, row in enumerate(raw_data):
        if rk in row and row[rk] is not None:
            rewards.append(row[rk])
            steps.append(row.get(sk, i))
    return steps, rewards

# 提取修改后实验和原始长实验
latest_run = runs[-1]
longest_run = max(runs[:-1], key=lambda p: len(glob.glob(os.path.join(p, "*"))))

s_old, r_old = load_data(longest_run)
s_new, r_new = load_data(latest_run)

# 动态对齐横坐标上限
max_step_target = max(s_new) if s_new else None

if s_old and r_old and max_step_target:
    filtered_old = [(s, r) for s, r in zip(s_old, r_old) if s <= max_step_target]
    s_old_aligned = [p[0] for p in filtered_old]
    r_old_aligned = [p[1] for p in filtered_old]
else:
    s_old_aligned, r_old_aligned = s_old, r_old

plt.figure(figsize=(10, 6), dpi=300)
if s_old_aligned and r_old_aligned:
    plt.plot(s_old_aligned, r_old_aligned, label='Original (Smoothness Penalty = -0.1)', color='#2ca02c', alpha=0.8, linewidth=2.0)
if s_new and r_new:
    plt.plot(s_new, r_new, label='Modified (Smoothness Penalty = -1.0)', color='#d62728', linewidth=2.0)

plt.title('Reward Comparison (Aligned to Initial Training Phase)')
plt.xlabel('Timesteps')
plt.ylabel('Total Mean Reward')
plt.xlim(0, max_step_target if max_step_target else None)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', framealpha=0.9)
plt.savefig('task5_comparison_curve_aligned.png', bbox_inches='tight')
print(f"\n✅ 等步长对齐对比图已生成: task5_comparison_curve_aligned.png (对齐步数上限: {max_step_target})")
