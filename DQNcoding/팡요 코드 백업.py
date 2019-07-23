#https://github.com/seungeunrho/minimalRL/blob/52aaa800e7bad30920a21a62f04a17d25663245c/dqn.py

del gym.envs.registry.env_specs['UDN-v0']

import gym
import gym_UDN
import collections   #replay buffer에서 쓰일 deque를 import하기 위함 double ended que?
import random

import torch    #pytorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer():            #3:35. et=(st,at,rt,st+1)인 튜플 e를 buffer에 저장
    def __init__(self):
        self.buffer = collections.deque()   #maxlen이상이면 FIFO으로 빠져나감
        self.batch_size = 32      #replay buffer에서 sampling할때 필요
        self.size_limit = 50000   #buffer의 최대 크기, DQN 논문에서는 백만
     
    def put(self, data):          #replay buffer에 데이터를 넣는 것, FIFO, 들어와서 다 차면 왼쪽으로 나감
        self.buffer.append(data)
        if len(self.buffer) > self.size_limit:
            self.buffer.popleft()
    
    def sample(self, n):
        return random.sample(self.buffer, n)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):    #Q network, torch.nn 모듈을 상속받음
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 64)  
        self.fc2 = nn.Linear(64, 64) 
        self.fc3 = nn.Linear(64, 2)  #action이 2개(cartpole 왼쪽 오른쪽), 우리는 2*(BSnum-1)개 output되야함...

    def forward(self, x):       #forward함수로 action을 할 수 있다
        x = F.relu(self.fc1(x))  #input 4개에서 64개로 fully connected, Relu
        x = F.relu(self.fc2(x))  #64개에서 64개로 fully connected, Relu
        x = self.fc3(x)          #64개에서 2개로 output, 여기서는 Relu안 넣음. q value가 음수일 수도 있으므로
        #원래 value function 마지막 단에서는 relu를 사용하지 않습니다
        return x
      
    def sample_action(self, obs, epsilon): #sample action은 e-greedy를 위해, 그냥 epsilon값만 정하면 e-greedy로 돌아간다
        out = self.forward(obs)
        coin = random.random() #0~1사이의 랜덤값
        if coin < epsilon:   #if 조건 실행될 확률 = epilson값
            return random.randint(0,1)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, gamma, optimizer, batch_size):   #한 episode끝날때마다 train 함수가 호출
    #episode끝날때마다 for loop횟수만큼 update
    for i in range(10):  #반복 10번은 한번만 update하기 그래서 그냥 10번 했다네.. #replaybuffer에서 열 번 update
        batch = memory.sample(batch_size) 
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        #여기까지 mini batch를 구성하는 작업
        #####연산속도를 위해 batch(예시는 size 32인 batch)로 처리#######
        for transition in batch: #for loop 돌면서 위의 list 값들을 append시키고 tensor로 만듬, transition은 tupletype의 변수
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a]) #s는 이미 array형태, s와 a타입을 맞추기 위해 [a]로 쓴다. 안맞추면 torch에서 dimension안맞는다고 알림
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        #s:state, a:action, r:reward, s_prime:next state
        s,a,r,s_prime,done_mask = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                  torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                  torch.tensor(done_mask_lst)
        q_out = q(s) #여기 예시에서 s는 [32(batch_size),4(원래 s dimension)] 사이즈
        #s가 input으로 들어가서, q의 output은 Shape가 [32,2]
        q_a = q_out.gather(1,a)  #a는 [32,1], q_out.gather는 실제로 취한 action의 q값만 골라낸다. 여기서 a는 일종의 index
        #gather(1,a)에서 1은 2번째 축에서 고르라는 것
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1) #q대신 q-target을 호출, q_learning이니까 max를 취함
        #q_target은 [32,2]. max(1)[0]을 하면 shape이 32, unsqueeze를 하면 [32,1], 차원을 맞춰야 연산 된다
        target = r + gamma * max_q_prime * done_mask #TD-target(다음 state 추측치)
        #done_mask는 마지막 state일때는 0이므로, 마지막 state일때는 gamma만 남음
        loss = F.smooth_l1_loss(target, q_a)  #loss=(target-q_a), 원래 (optimal-q_a)인데 optimal을 모르니까 td-target 대입.
        #smooth_l1_loss는 pytorch제공함수, -1~1사이면은 제곱에 비례, 아니면 선형 비례?
        
        optimizer.zero_grad() #초기화
        loss.backward() #호출하면 gradient가 backpropagate하면서 구해짐
        optimizer.step() #호출하면 gradient로 update됨
#episode가 한번 끝날때마다 샘플 320개(batch_size32*for loop 횟수10)를 사용하여 update
# ->sample이 그 이하이면 학습이 제대로 안되니까 충분히 sample을 만든뒤 학습해야함


def main():
    #env = gym.make('CartPole-v1')   #space만 맞추면 어떤 env등 작동하는 코드일듯?
    env = gym.make('UDN-v0')
    q = Qnet()                      #q network 선언
    q_target = Qnet()               #target q network 선언 
    q_target.load_state_dict(q.state_dict())  #q의 weight를 q_target으로 load해옴 (load_state_dict는 torch 함수)
    #q.state_dict() q의 모든 weight값이 dict형태로 저장되어 있음
    memory = ReplayBuffer()

    avg_t = 0
    gamma = 0.98
    batch_size = 32
    optimizer = optim.Adam(q.parameters(), lr=0.0005)  #q.parameter를 update, 이때 q-target은 update안함!

    for n_epi in range(10000):  #에피소드를 10000으로 일단,
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #엡실론을 Linear annealing from 8% to 1%,
                                                     #따라서 exploration을 처음에 많이 하다가 점점 줄인다
        s = env.reset()
            #s가 state 받고
        for t in range(600):
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)     #sample_action:e-greedy,
            s_prime, r, done, info = env.step(a)       #a를 environment에 넣어서 다음 state(s_prime), 다음 reward, done 얻음
            done_mask = 0.0 if done else 1.0           #game이 끝나면 0이고 안끝났으면 1, TD-target에서 쓰기 위해(15:30)
            memory.put((s,a,r/200.0,s_prime, done_mask))  #memory를 계속 저장
            s = s_prime 

            if done:
                break
        avg_t += t

        if memory.size()>2000:       #memory가 충분히 쌓이면 train함수 호출(충분히 쌓이고 시작해야함) 
            train(q, q_target, memory, gamma, optimizer, batch_size)

        if n_epi%20==0 and n_epi!=0:  #20에피소드마다 최근 에피소드 평균 timestep, target network를 update
            q_target.load_state_dict(q.state_dict())
            print("# of episode :{}, Avg timestep : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(
                                                            n_epi, avg_t/20.0, memory.size(), epsilon*100))
            avg_t = 0
    env.close()

if __name__ == '__main__':
    main()