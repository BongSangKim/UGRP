## objective: minimalizing BS energy consumption

# UDN 통신 모델 by RL

#### 고려해야 할 사항 :
UDN, MS(Mass Base Station), BS, UE,

* about BS 

분포: PPP
Status: active or sleep(=off)
power consumption : constant p.c(power consumption), transmission p.c(power consumption)
BS가 p.c를 다르게 가지는 경우 or BS의 p.c가 uniform 한 경우

* about UE 

초기분포 : PPP or uniform
움직임 : random walk (non-uniform/direction)

#### varaibles in UDN modeling

h(height)= suppose most BS(microBS especially) has average height of building in environment

data rate -> SIR(or SINR, considering noise)
*SIR
-Interference 
  Los/NLoS
  Coverage area(distance up -> data rate down)
-nosie(\sigma ^2 )
