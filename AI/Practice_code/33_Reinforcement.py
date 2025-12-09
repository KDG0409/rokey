# MDP의 한 사이클
# def mdp_cycle():
#     """
#     t=0: 초기 상태
#     """
#     state = env.reset()  # S_0

#     while not done:
#         # 1. 현재 상태 관찰
#         print(f"현재 상태: {state}")

#         # 2. 정책에 따라 행동 선택
#         action = policy(state)
#         # π(a|s) 주어진 상황에서 어떤 행동을 취할 확률 : 정책(π)
#         print(f"선택한 행동: {action}")

#         # 3. 환경과 상호작용
#         next_state, reward, done, info = env.step(action)
#         print(f"보상: {reward}")
#         print(f"다음 상태: {next_state}")

#         # 4. 상태 업데이트
#         # 행동을 취해서 다음 단계, 즉 다음 상태(next_state)로 전이했으니깐
#         # 다음 상태가 현재 상태로 업데이트 됨
#         state = next_state

#     print("에피소드 종료!")

# V(s)

# V(s) = "이 상태에서 시작해서 끝까지 얻을 보상의 평균"

# def V(state, policy, gamma=0.9):
#     """
#     state(주어진 상태)에서 시작해서 policy(정책)를 따라갈 때
#     미래의 모든 보상을 할인(gamma=0.9 할인율)하여 합산한 기댓값
#     """
#     total_reward = 0
#     current_state = state
#     discount = 1.0

#     while not done:
#         action = policy(current_state)
#         next_state, reward = env.step(action)
#         total_reward += discount * reward
#         discount *= gamma
#         current_state = next_state

#     return total_reward

# V(s) 와 Q(s, a) 관계

# 상태의 가치 V(s)는 각 행동의 가치를 정책 확률로 가중 평균한 것

# 특정 상태에서의 Q-value
# state = [0.02, 0.5, 0.03, -0.8]  # 약간 오른쪽으로 기운 상태

# Q(state, action=0) = 45.2  # 왼쪽으로 밀면 45.2의 가치
# Q(state, action=1) = 78.5  # 오른쪽으로 밀면 78.5의 가치

# # 최적 행동 선택
# best_action = argmax(Q(state, a)) = 1  # 오른쪽!

# # V(state) 계산 (π가 각 행동을 50%씩 선택한다면)
# V(state) = 0.5 * 45.2 + 0.5 * 78.5 = 61.85

# 최적 가치 함수 (벨만방정식)

# 최적가치는 V*(s) 는 가장 좋은 행동(a) 선택했을 때 가치인데, 정책에 대한 기대값(합)이 아니라 최대값(max) 사용함
# 최적가치 = 즉각적인 보상 R(s,a,s')와 미래가치 (gamma V*(s')) 가중평균한 누적합의 최대화하는 값(max)

# def bellman_optimality_equation(state, V, gamma=0.9):
#     """
#     최적 가치 계산
#     """
#     max_value = float('-inf')

#     # 모든 가능한 행동에 대해
#     for action in possible_actions(state):
#         action_value = 0

#         # 각 행동의 결과로 가능한 모든 다음 상태에 대해
#         for next_state, prob in get_next_states(state, action):
#             reward = get_reward(state, action, next_state)
#             action_value += prob * (reward + gamma * V[next_state])

#         # 최대 가치 업데이트
#         max_value = max(max_value, action_value)

#     return max_value

# Q-learning

# 상태 s에서 행동 a를 했을 때 얻을 수 있는 예상되는(기대되는) 총 보상
# Q Table

# 모든 (상태-행동) (state - action) 쌍을 Q-Value 저장

# alpha(학습률) : [0,1] 사이 값

# 0.1 : 새로운 정보를 10%만 반영 (천천히 학습)
# 1 : 새로운 정보를 100%만 반영 (빠르게 학습, 불안정)
# gamma(감마, 할인계수 discount factor)

# 미래 보상 중요도
# [0,1] 사이 값
# gamma = 0 즉각적인 보상만 중요
# gamma = 0.9 먼 미래도 중요
# r (reward 보상)

# 행동 a를 취했을 얻을 수 있는 즉각적인 보상
# maxQ(s', a')

# 다음 상태에서 가능한 최대 Q value

# # 예시: FrozenLake
# current_state = 0
# action = 2  # RIGHT
# next_state = 1
# reward = 0  # 골이 아니므로 보상 0

# # 현재 Q-values
# Q_table[0, 2] = 0.5  # Q(s=0, a=RIGHT) = 0.5

# # 다음 상태의 최대 Q-value
# Q_table[1, :] = [0.3, 0.4, 0.6, 0.7]
# max_next_Q = 0.7  # max Q(s'=1, a')
# # 다음 상태 가치

# # 하이퍼파라미터
# alpha = 0.1  # 학습률
# gamma = 0.9  # 할인 계수

# # Q-Learning 업데이트
# target = reward + gamma * max_next_Q
# target = 0 + 0.9 * 0.7 = 0.63  # 목표값

# TD_error = target - Q_table[0, 2]
# TD_error = 0.63 - 0.5 = 0.13  # 오차

# Q_table[0, 2] = Q_table[0, 2] + alpha * TD_error
# Q_table[0, 2] = 0.5 + 0.1 * 0.13 = 0.513  # 업데이트된 값

# print(f"Q(0, RIGHT): 0.5 → 0.513 (개선됨!)")

"""
Q-Learning 알고리즘:

1. Q-table 초기화
   모든 Q(s, a) = 0 (또는 작은 랜덤 값)

2. 반복 (각 에피소드마다):
   a. 환경 초기화 (s ← 초기 상태)

   b. 에피소드 종료까지 반복:
      i.   ε-greedy로 행동 a 선택
      ii.  행동 실행 → (s', r) 관찰
      iii. Q-value 업데이트:
           Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
      iv.  s ← s' (다음 상태로 이동)

   c. ε 감소 (탐색 줄이기)

3. 수렴할 때까지 반복
"""

"""
Q-Learning (Table):
┌─────────┬─────┬─────┬─────┬─────┐
│ State   │  L  │  D  │  R  │  U  │
├─────────┼─────┼─────┼─────┼─────┤
│ State 0 │ 0.1 │ 0.5 │ 0.8 │ 0.2 │
│ State 1 │ 0.3 │ 0.4 │ 0.6 │ 0.7 │
│   ...   │ ... │ ... │ ... │ ... │
│State 15 │ 0.0 │ 0.0 │ 0.0 │ 0.0 │
└─────────┴─────┴─────┴─────┴─────┘

DQN (Neural Network):
     State
       │
       ▼
┌──────────────┐
│ Input Layer  │
│ (state dim)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Hidden Layer 1│
│  (128 units) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Hidden Layer 2│
│  (128 units) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Output Layer  │
│ (n_actions)  │
└──────┬───────┘
       │
       ▼
  [Q(s,a₀), Q(s,a₁), Q(s,a₂), Q(s,a₃)]
"""

"""
CartPole DQN 예시:

Input: [cart_pos, cart_vel, pole_angle, pole_vel]
       4차원 벡터

       ↓ (4 → 128)

Hidden 1: [h1, h2, h3, ..., h128]
          128개 뉴런
          ReLU 활성화

       ↓ (128 → 128)

Hidden 2: [h1, h2, h3, ..., h128]
          128개 뉴런
          ReLU 활성화

       ↓ (128 → 2)

Output: [Q(s, 왼쪽), Q(s, 오른쪽)]
        2개 Q-values
"""

