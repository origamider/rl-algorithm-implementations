import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# 概要
# Bandit問題を自分なりに書いて理解してみる!
# ε-greedy法を利用する!
# 状況
# 10個のスロットがあって、それぞれ当たる確率、外れる確率がバラバラ。当たったらコイン1枚,外れたら0枚。
# この時、できるだけたくさんのコインを集めるようにPlayerが行動すると、どの程度コインを得られるか。
# Playerは、試行回数runs,各試行についてsteps回行動する。そして各stepsにおける当たる確率を平均する。

# Bandit
class Slots:
    def __init__(self,arms=10):
        self.rates = np.random.rand(arms) # 10個のスロットそれぞれの当たる確率をランダムに作成
    
    # スロットを回す処理
    def play(self,slot_id):
        rate = self.rates[slot_id]
        tmp = np.random.rand()
        if tmp < rate: # 当たった場合
            return 1
        else:
            return 0

# Agent
class Player:
    def __init__(self, epsilon, num_slots=10):
        self.epsilon = epsilon # 探索を選ぶ確率(他の台もチャレンジする)
        self.Q = np.zeros(num_slots)
        self.n = np.zeros(num_slots)

    def update(self,slot_id,reward):
        self.n[slot_id] += 1
        self.Q[slot_id] += (reward-self.Q[slot_id])/self.n[slot_id]
    
    def get_action(self):
        tmp = np.random.rand()
        if tmp < self.epsilon: # 探索
            return np.random.randint(0,len(self.Q))
        else: # 現時点で最もQの高い台を選択(貪欲)
            return np.argmax(self.Q)

runs = 200
steps = 1000
epsilon = 0.3
total_rewards = []
all_rates = np.zeros((runs,steps))

for run in range(runs):
    slots = Slots()
    player = Player(epsilon=epsilon)
    total_reward = 0
    rates = []
    for step in range(steps):
        slot_id = player.get_action()
        reward = slots.play(slot_id)
        total_reward += reward
        player.update(slot_id,reward)
        total_rewards.append(total_reward)
        rates.append(total_reward/(step+1))
    all_rates[run] = rates

avg_rates = np.average(all_rates,axis=0) # 各stepについて平均を取るよ~

# 合計獲得メダル数
# plt.xlabel('繰り返し回数')
# plt.ylabel('合計獲得メダル数')
# plt.plot(total_rewards)
# plt.show()

# 当たった確率
plt.title(f'ε={epsilon}の時')
plt.xlabel('繰り返し回数')
plt.ylabel('当たった確率')
plt.plot(avg_rates)
plt.show()
