# rl-algorithm-implementations

いろんな強化学習のアルゴリズムを実装してます。

## 概要
色んな強化学習アルゴリズムをPyTorchで実装したり、OpenAI Gymnasiumを使った強化学習環境でシミュレーションしたりしています。

## 参考書籍
- 『ゼロから作るDeep Learning ❹ 強化学習編』
- 『ゼロから作るDeep Learning ❺ ―生成モデル編』

## 実装済みのアルゴリズム
  * モンテカルロ法 (MC Control / Evaluation)
  * Q-Learning
  * SARSA (On-policy / Off-policy)
  * 動的計画法 (Value Iteration / Policy Iteration)
  * DQN
  * Double DQN
  * DDPG (連続値制御)

## 検証環境
Gymnasium
* `CartPole-v1`
* `MountainCar-v0`
* `Pendulum-v1`
* `Acrobot-v1`
* `LunarLander-v3`
* `Blackjack-v1`

## 今後の取り組み (To-Do)
* 分布型強化学習、PPO、SACなどのより発展的なアルゴリズムの実装
* 世界モデル（World Models）の実装