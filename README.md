# rl-algorithm-implementations

強化学習のアルゴリズムを、ライブラリのブラックボックスに頼らず基礎からスクラッチで実装・検証しているリポジトリです。

## 概要
書籍『ゼロから作るDeep Learning ❹ 強化学習編』での学習をきっかけに、基礎的なテーブルデータのアルゴリズムから深層強化学習までをPyTorchで実装しています。
現在は、OpenAI Gymnasiumの様々な環境（離散値・連続値制御）を、自分で書いたアルゴリズムで攻略する実験を行っています。

単に動くコードを書くのではなく、「数式や論文の理論が、コードのどこに該当するのか」を理解した上で実装することを心がけています。

## 実装済みのアルゴリズム
  * モンテカルロ法 (MC Control / Evaluation)
  * Q-Learning
  * SARSA (On-policy / Off-policy)
  * 動的計画法 (Value Iteration / Policy Iteration)
  * DQN
  * Double DQN
  * DDPG (連続値制御)

## 検証環境
Gymnasiumを利用し、色んな環境で強化学習アルゴリズムを動かしています。
* `CartPole-v1`
* `MountainCar-v0`
* `Pendulum-v1`
* `Acrobot-v1`
* `LunarLander-v3`
* `Blackjack-v1`

## 今後の取り組み (To-Do)
* 実装済みのアルゴリズムのコードリファクタリング（共通処理の切り出しなど）
* 分布型強化学習、PPO、SACなどのより発展的なアルゴリズムの実装
* いずれ世界モデル（World Models）などの最先端の概念も、理解したいと思っています！