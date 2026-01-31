import torch
import torch.optim as optim
import random
import numpy as np
from collections import deque
from model import FraudDQN
from env import FraudEnv

# Hiperparametreler
GAMMA = 0.99
LEARNING_RATE = 0.0005
MEMORY_SIZE = 10000     # Veri büyüdüğü için hafızayı artırdık
BATCH_SIZE = 128        # Daha hızlı öğrenme
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.998   # Keşif süresini biraz uzattık
TARGET_UPDATE = 10
EPISODES = 900         # Veri seti büyüdüğü için 900 tur idealdir

def train():
    # DOSYA LİSTESİ: Hem Kaggle verisi hem senin verin
    dosya_listesi = ["Bank_Transaction_Fraud_Detection.csv", "BankaVerileri.xlsx"]
    
    env = FraudEnv(dosya_listesi) 
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    policy_net = FraudDQN(state_dim, action_dim)
    target_net = FraudDQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = deque(maxlen=MEMORY_SIZE)
    
    epsilon = EPSILON_START
    
    print(f"Eğitim başlıyor... Toplam Veri: {env.n_steps} işlem.")

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    action = policy_net(state_t).argmax().item()
            
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, float(done)))
            
            state = next_state
            total_reward += reward
            
            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states_t = torch.FloatTensor(np.array(states))
                actions_t = torch.LongTensor(actions).unsqueeze(1)
                rewards_t = torch.FloatTensor(rewards)
                next_states_t = torch.FloatTensor(np.array(next_states))
                dones_t = torch.FloatTensor(dones)
                
                current_q = policy_net(states_t).gather(1, actions_t)
                next_q = target_net(next_states_t).max(1)[0].detach()
                target_q = rewards_t + (GAMMA * next_q * (1 - dones_t))
                
                loss = torch.nn.functional.mse_loss(current_q.squeeze(), target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if (episode + 1) % 50 == 0:
            print(f"Tur: {episode+1}/{EPISODES}, Ortalama Ödül: {total_reward:.1f}, Epsilon: {epsilon:.2f}")

    torch.save(policy_net.state_dict(), "fraud_dqn_model.pth")
    print("Mükemmel! Model her iki veri setiyle de eğitildi ve 'fraud_dqn_model.pth' güncellendi.")

if __name__ == "__main__":
    train()