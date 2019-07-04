## objective: minimalizing BS energy consumption

# Python Environment Design
## RL 모델은 MCTS 고려
#### 고려해야 할 사항 :
UDN, MS(Mass Base Station), BS, UE,

* about BS 

분포: PPP
Status: active(=on) or sleep(=off), active or stan
power consumption : constant p.c(power consumption), transmission p.c(power consumption)
BS가 p.c를 다르게 가지는 경우 or BS의 p.c가 uniform 한 경우
BS association rule, on/off machanism

* about UE 

초기분포 : PPP or uniform
움직임 : random walk (non-uniform/direction), velocity-도플러 효과 고려해야할까?

#### constant in UDN modeling
3.5Ghz & 28GHz(bandwidth 800MHz)
BS lambda(밀도)

#### varaibles in UDN modeling

h(height)= suppose most BS(microBS especially) has average height of building in environment
r(distancd)
bandwidth
data rate -> SIR(or SINR, considering noise)
*SIR
-Interference 
  Los/NLoS
  Coverage area(distance up -> data rate down)
-noise(\sigma ^2 )
height of BS, UE velocity, UE demand, 

##RL modeling
MDP-state(BS on/off), action(on/off), reward, transition probability matrix, \gamma(discount factor)
Energy consumption(E_c) should be minimal & SIR should be upper than \beta(threshold) 
(SIR > \beta in coverage)
number of BS=n개
E_c = Energy consumption function.=\SIGMA(E_cK)
E_c1, .... E_cn = n번쨰 BS의 Energy consumption

r4

BS association rule : 신호 센거 잡는 걸로 가정
각 BS에서 받는 신호는 구분할 수 있나?
BS는 각 UE에 대해 다른 출력으로 신호를 주고받나?

transition 과정에서 disadvantage??


