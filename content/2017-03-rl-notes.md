Title: Reforcement Learning Notes
Date: 2017-03-09
Slug: rl-notes
Category: Machine Learning


# Learning note on [Markov Decision Process](https://www.youtube.com/watch?v=lfHX2hHRMVQ).

##### Markov Property

$$ \mathbb{P}[S_{t+1}|S_t] = \mathbb{P}[S_{t+1}|S_1, ..., S_t] $$

##### Markov Process or Markov Chain

$\langle S, P \rangle$

$S$ is a finite set of states.

$P$ is a state transition probability matrix. $P_{ss'} = \mathbb{P}[S_{t+1}=s'|S_t=s]$

### Markov Reward Process

$\langle S, P, R, \gamma \rangle$

$S$ is a finite set of states.

$P$ is a state transition probability matrix. $P_{ss'} = \mathbb{P}[S_{t+1}=s'|S_t=s]$

$R$ is a reward function, $R_s = \mathbb{E}[R_{t+1}|S_t=s]$

$\gamma$ is a discount factor, $\gamma \in [0, 1]$

##### Value Function

value function $v(s)$ gives the long-term value of state $s$

$$ G_t = R_{t+1} + \gamma R_{t+2} + ... = \sum_{k=0}^\infty\gamma^k R_{t+k+1} $$

$$
\begin{eqnarray}
v(s) &=& \mathbb{E}[G_t|S_t=s] \\
&=& \mathbb{E}[R_{t+1} + \gamma(R_{t+2}+\gamma R_{t+3}+...)|S_t=s] \\
&=& \mathbb{E}[R_{t+1} + \gamma G_{t+1}|S_t=s] \\
&=& \mathbb{E}[R_{t+1} + \gamma v(S_{t+1})|S_t=s]
\end{eqnarray}
$$

##### Bellman Equation for MRP

$$
\begin{eqnarray}
v(s) &=& \mathbb{E}[R_{t+1} + \gamma v(S_{t+1})|S_t=s] \\
&=& R_s + \gamma \sum_{s' \in S}P_{ss'}v(s')
\end{eqnarray}
$$

$$ v = R + \gamma P v $$

### Markov Decision Process

$\langle S, A, P, R, \gamma \rangle$

$S$ is a finite set of states

$A$ is a finite set of actions

$P$ is state transition probability matrix, $P_{ss'}^a = \mathbb{P}[S_{t+1}=s'|S_t=s,A_t=a]$

$R$ is a reward function, $R_s^a = \mathbb{E}[R_{t+1}|S_t=s,A_t=a]$

$\gamma$ is a discount factor, $\gamma \in [0, 1]$

##### Policy

A *policy* $\pi$ is a distribution over actions given states

$$ \pi(a|s) = \mathbb{P}[A_t=a|S_t=s] $$

$$ A_t \sim \pi(\cdot|s), \forall t \gt 0 $$

Given $M = \langle S, A, P, R, \gamma \rangle$ and policy $\pi$

$S_1, S_2, ...$ is a Markov process $\langle S, P^\pi \rangle$

$S_1, R_2, S_2, R_3, ...$ is a Markov reward process $\langle S, P^\pi, R^\pi, \gamma \rangle$

$$ P_{ss'}^\pi = \sum_{a \in A}\pi(a|s)P_{ss'}^a $$

$$ R_s^\pi = \sum_{a \in A}\pi(a|s)R_s^a $$

*state-value* function $v_\pi(s)$

$$ v_\pi(s) = \mathbb{E}_\pi[G_t|S_t=s] $$

*action-value* function $q_\pi(s, a)$

$$ q_\pi(s, a) = \mathbb{E}_\pi[G_t|S_t=s, A_t=a] $$

##### Bellman Equation for value function

$$ v_\pi(s) = \mathbb{E_\pi}[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t=s] $$

$$ q_\pi(s, a) = \mathbb{E_\pi}[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1})|S_t=s,A_t=a] $$

<img src="/images/2017/rl-notes/rl_mdp_v-300x116.png" alt="" width="300" height="116" class="aligncenter size-medium wp-image-228" />

$$ v_\pi(s) =  \sum_{a \in A}\pi(a|s)q_\pi(s, a) $$

<img src="/images/2017/rl-notes/rl_mdp_q-300x122.png" alt="" width="300" height="122" class="aligncenter size-medium wp-image-227" />

$$ q_\pi(s, a) = R_s^a + \gamma \sum_{s' \in S}P_{ss'}^a v_\pi(s') $$

<img src="/images/2017/rl-notes/rl_mdp_v-1-300x155.png" alt="" width="300" height="155" class="aligncenter size-medium wp-image-230" />

$$ v_\pi(s) = \sum_{a \in A}\pi(a|s)(R_s^a + \gamma \sum_{s' \in S}P_{ss'}^a v_\pi(s')) $$

$$ v_\pi = R^\pi + \gamma P^\pi v_\pi $$

<img src="/images/2017/rl-notes/rl_mdp_q-1-300x145.png" alt="" width="300" height="145" class="aligncenter size-medium wp-image-229" />

$$ q_\pi(s, a) = R_s^a + \gamma \sum_{s' \in S}P_{ss'}^a \sum_{a' \in A}\pi(a'|s')q_\pi(s', a') $$

##### Optimal Value Function

$$ v_\ast(s) = \max_{\pi}v_\pi(s) $$

$$ q_\ast(s, a) = \max_{\pi}q_\pi(s, a) $$

policy ordering

$$ \pi \gt \pi' \quad if v_\pi(s) \ge v_{\pi'}(s), \forall s $$

There exists an optimal policy $\pi_\ast$ that $\pi_\ast \ge \pi, \forall \pi$
All optimal policies achieve the optimal value function, $v_{\pi_\ast}(s) = v_\ast(s)$
All optimal policies achieve the optimal action value function, $q_{\pi_\ast}(s, a) = q_\ast(s, a)$

$$
\begin{eqnarray}
\pi_\ast(a|s) =
\begin{cases}
1 \quad if a = \arg\max_{a \in A}q_\ast(s, a) \\
0 \quad otherwise
\end{cases}
\end{eqnarray}
$$

<img src="/images/2017/rl-notes/rl_mdp_ov-300x117.png" alt="" width="300" height="117" class="aligncenter size-medium wp-image-234" />

$$ v_\ast(s) = \max_{a}q_\ast(s, a) $$

<img src="/images/2017/rl-notes/rl_mdp_oq-300x127.png" alt="" width="300" height="127" class="aligncenter size-medium wp-image-232" />

$$ q_\ast(s, a) = R_s^a + \gamma \sum_{s' \in S}P_{ss'}^av_\ast(s') $$

<img src="/images/2017/rl-notes/rl_mdp_ov-1-300x147.png" alt="" width="300" height="147" class="aligncenter size-medium wp-image-235" />

$$ v_\ast(s) = \max_{a}R_s^a + \gamma \sum_{s' \in S}P_{ss'}^av_\ast(s') $$

<img src="/images/2017/rl-notes/rl_mdp_oq-1-300x148.png" alt="" width="300" height="148" class="aligncenter size-medium wp-image-233" />

$$ q_\ast(s, a) = R_s^a + \sum_{s' \in S}P_{ss'}^a\max_{a'}q_\ast(s', a') $$

### Summary

$$ v_\pi(s) =  \sum_{a \in A}\pi(a|s)q_\pi(s, a) $$

$$ q_\pi(s, a) = R_s^a + \gamma \sum_{s' \in S}P_{ss'}^a v_\pi(s') $$

$$ v_\ast(s) = \max_{a \in A}q_\ast(s, a) $$

$$ q_\ast(s, a) = R_s^a + \gamma \sum_{s' \in S}P_{ss'}^av_\ast(s') $$

# Learning note on [Planning by Dynamic Programming](https://www.youtube.com/watch?v=Nd1-UUMVfz4).

##### Bellman Expectation Equation

$v_\pi(s)$, $q_\pi(s, a)$

$$ v_\pi(s) = \sum_{a \in A}\pi(a|s)q_\pi(s, a) $$

$$ q_\pi(s, a) = R_s^a + \sum_{s' \in S}P_{ss'}^a v_\pi(s') $$

##### Bellman Optimality Equation

$v_\ast(s) $, $q_\ast(s, a)$

$$ v_\ast(s) = \max_{a \in A}q_\ast(s, a) $$

$$ q_\ast(s, a) = R_s^a + \gamma \sum_{s' \in S}P_{ss'}^a v_\ast(s') $$

### Iterative Policy Evaluation

Problem: evaluate a given policy $\pi$

Solution: iterative application of Bellman Expectation Equation

$v_1 \to v_2 \to ... \to v_\pi$

<img src="/images/2017/rl-notes/rl2-ipe-300x157.png" alt="" width="300" height="157" class="aligncenter size-medium wp-image-271" />

$$ v_{k+1}(s) = \sum_{a \in A}\pi(s, a)(R_s^a + \gamma \sum_{s' in S}P_{ss'}^a v_k(s')) $$

### Policy Iteration

1. Given an initial policy $\pi$
2. Evaluate the policy $\pi$

  $$ v_\pi(s) = \mathbb{E_\pi}[R_{t+1}+\gamma R_{t+1} + ...|S_t=s] $$

3. Improve the policy by acting greedily with respect $v_\pi$

  $$ \pi' = greedy(v_\pi) $$

4. Repeat step 2 and 3 until $\pi$ converges to $\pi^\ast$

<img src="/images/2017/rl-notes/rl-pi-300x174.png" alt="" width="300" height="174" class="aligncenter size-medium wp-image-275" />

for deterministic policy $a = \pi(s)$

$$ \pi'(s) = \arg\max_{a \in A}q_\pi(s, a) $$

$$ q_\pi(s, \pi'(s)) = \arg\max_{a \in A}q_\pi(s, a) \ge q_\pi(s, \pi(s)) = v_\pi(s) $$

$$
\begin{eqnarray}
v_\pi(s) &\le& q_\pi(s, \pi'(s)) = \mathbb{E_{\pi'}}[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s] \\
&\le& \mathbb{E_{\pi'}}[R_{t+1}+\gamma q_\pi(S_{t+1}, \pi'(S_{t+1}))|S_t=s] \\
&\le& \mathbb{E_{\pi'}}[R_{t+1}+\gamma R_{t+2} + \gamma^2 q_\pi(S_{t+2}. \pi'(S_{t+2}))|S_t=s] \\
&\le& \mathbb{E_{\pi'}}[R_{t+1}+\gamma R_{t+2} +...|S_t=s] = v_{\pi'}(s)
\end{eqnarray}
$$

$$ v_\pi(s) \le v_{\pi'}(s) $$

if improvements stop or converges

$$ q_\pi(s, \pi'(s)) = \max_{a \in A}q_\pi(s, a) = q_\pi(s, \pi(s)) = v_\pi(s) $$

$$ v_\pi(s) = \max_{a \in A}q_\pi(s, a) $$

so $v_\pi(s)$ satisfies Bellman Optimality Equation

$$ v_\pi(s) = v_\ast(s) $$

### Value Iteration

Problem: find optimal policy $\pi$
Solution: iterative application of Bellman Optimality Equaltion
$v_1 \to v_2 \to ... \to v_\ast $
Using synchronous backups

  - at each iteration k+1
  - for all states $s \in S$
  - update $v_{k+1}(s) = v_k(s')$

No explicit policy
Intermediate value function $v_k$ may not correspond to any policy

<img src="/images/2017/rl-notes/rl2-ipe-300x157.png" alt="" width="300" height="157" class="aligncenter size-medium wp-image-271" />

$$ v_{k+1}(s) = \max_{a \in A}R_s^a + \gamma \sum_{s' \in S}P_{ss'}^a v_k(s') $$

### Summary

|Problem|Bellman Equation|Algorithm|
|-----|-----|-----|
|Prediction|Bellman Expectation Equation|Iterative Policy Evaluation|
|Control|Bellman Expectation Equation + Greedy Policy Improvement|Policy Iteration|
|Control|Bellman Optimiality Equation|Value Iteration|

# Learning note on [Model-Free Prediction](https://www.youtube.com/watch?v=PnHCvfgC_ZA).

Model-Free Prediction is about estimating the value function of an *unknown* MDP.

### Monte-Carlo Reinforcement Learning

learn $v_\pi$ from **complete** episodes of experience under policy $\pi$.

$$ S_1, A_1, R_2, ..., S_k \sim \pi $$

*return* is the total discount reward

$$ G_t = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{T-1}R_T $$

$$ v_\pi(s) = \mathbb{E_\pi}[G_t|S_t=s] $$

Monte-Carlo policy evaluation uses *empirical mean* return instead of *expected*  return.

- $N(s) \gets N(s) + 1$
- $S(s) \gets S(s) + G_t$
- $V(s) = S(s) / N(s)$
- $V(s) \to v_\pi(s)$ as $N(s) \to \infty$

update $S(s)$ with return $G_t$

$$ N(S_t) \gets N(S_t) + 1 $$

$$ V(S_t) \gets V(S_t) + \frac{1}{N(S_t)}(G_t - V(S_t)) $$

$$ V(S_t) \gets V(S_t) + \alpha(G_t - V(S_t)) $$

### Temporal-Difference Learning

learn directly from episodes of experience. episodes can be **incomplete** using *bootstrapping* and updates a guess towards guess.

- learn $v_\pi$ online from experience under policy $\pi$
- $V(S_t) \gets V(S_t) + \alpha(G_t - V(S_t))$
- replace *actual* return $G_t$ with *estimated* return $R_{t+1}+\gamma V(S_{t+1})$
- $V(S_t) \gets V(S_t) + \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$
- $R_{t+1}+\gamma V(S_{t+1})$ is called TD target
- $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is called TD error

Driving Home Example: MC vs. TD

<img src="/images/2017/rl-notes/rl3-mcvstd-1024x503.png" alt="" width="660" height="324" class="aligncenter size-large wp-image-299" />

### Differences between MC and TD

Return $G_t$ is unbiased estimate of $v_\pi(S_t)$ while True TD target $R_{t+1}+\gamma v_\pi(S_{t+1})$ is a biased estimate.

TD target is much lower variance than Return

- Return depends on **many** random actions, transitions, rewards
- TD target depends on **one** random action, transition, reward

MC has high variance and zero bias

- Good convergence properties
- Not every sensitive to  initial value
- Very simple to understand and use
- MC doesn't exploit Markov property, usually more efficient in non-Markov environments

TD has low variance and some bias

- Usually more efficient than MC
- TD(0) converges to $v_\pi(s)$
- More sensitive to initial value, because bootstrapping
- TD exploits Markov property, usually more efficient in Markov environments

TD and MC both converge: $V(s) \to v_\pi(s)$ as $experience \to \infty$

##### Monte-Carlo Backup

$$ V(S_t) \gets V(S_t) + \alpha(G_t - V(S_t)) $$

<img src="/images/2017/rl-notes/rl3-mc-1024x520.png" alt="" width="660" height="335" class="aligncenter size-large wp-image-301" />

##### Temporal-Difference Backup

$$ V(S_t) \gets V(S_t) + \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(S_t)) $$

<img src="/images/2017/rl-notes/rl3-td-1024x521.png" alt="" width="660" height="336" class="aligncenter size-large wp-image-302" />

##### Dynamic Programming Backup

$$ V(S_t) \gets \mathbb{E_\pi}[R_{t+1} + \gamma V(S_{t+1})] $$

<img src="/images/2017/rl-notes/rl3-dp-1024x532.png" alt="" width="660" height="343" class="aligncenter size-large wp-image-303" />

**Bootstrapping**: update involves an estimate

- MC doesn't boostrap
- DP, TD bootstraps

**Sampling**: update samples an expectation

- DP doesn't sample
- MC, TD samples

<img src="/images/2017/rl-notes/rl3-view.png" alt="" width="996" height="705" class="aligncenter size-full wp-image-304" />

### $TD(\lambda)$

n-step return

<img src="/images/2017/rl-notes/rl3-nstep.png" alt="" width="954" height="609" class="aligncenter size-full wp-image-306" />

- $n = 1$, $G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})$, TD
- $n = 2$, $G_t^{(2)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2})$
- $n = \infty$, $G_t^{(\infty)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{T-1} R_{T}$, MC

$$ G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n}) $$

$$ V(S_t) \gets V(S_t) + \alpha (G_t^{(n)} - V(S_t)) $$

$\lambda$ return combines all n-step return with weight $(1-\lambda)\lambda^{n-1}$

<img src="/images/2017/rl-notes/rl3-lambda.png" alt="" width="589" height="687" class="aligncenter size-full wp-image-307" />

<img src="/images/2017/rl-notes/rl3-lambda-weight-1024x389.png" alt="" width="660" height="251" class="aligncenter size-large wp-image-308" />

$$ G_t^\lambda = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1}G_t^{(n)} $$

$$ V(S_t) \gets V(S_t) + \alpha(G_t^\lambda - V(S_t)) $$

Forward view of $TD(\lambda)$

<img src="/images/2017/rl-notes/rl3-lambda-forward-1024x280.png" alt="" width="660" height="180" class="aligncenter size-large wp-image-310" />

- Update value function towards the $\lambda$-return
- Forward-view looks into the future to compute $G_t^\lambda$
- Like MC, can only be computed from complete episodes

Backward view of $TD(\lambda)$

<img src="/images/2017/rl-notes/rl3-lambda-backward-1024x400.png" alt="" width="660" height="258" class="aligncenter size-large wp-image-313" />

Eligibility traces

$$ E_0(s) = 0 $$

$$ E_t(s) = \gamma \lambda E_{t-1}(s) + 1(S_t=s) $$

$$ \delta_t = R_{t+1} +\gamma V(S_{t+1}) - V(S_t) $$

$$ V(s) \gets V(s) + \alpha \delta_t E_t(s) $$

The sum of offline updates is identical for forward-view and backward view for $TD(\lambda)$

$$ \sum_{t=1}^{T}\alpha \delta_t E_t(s) = \sum_{t=1}^T \alpha (G_t^\lambda - V(S_t))1(S_t=s) $$


# Learning note on [Model Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4).

Model-Free Control is about optimizing the value function of an *unknown* MDP.

### On-policy Monte-Carlo Control

On-policy: Learn about policy $\pi$ from experience sampled from $\pi$. Off-policy: Learn about policy $\pi$ from experience sampled from $\mu$.

Greedy policy improvement over $Q(s, a)$ is model free

$$ \pi'(s) = \arg\max_{a \in A}Q(s, a) $$

$\epsilon$-Greedy Exploration

$$
\pi'(a|s) =
\begin{cases}
\frac{\epsilon}{m} + 1 - \epsilon, & a^\ast = \arg\max_{a \in A}Q(s, a) \\\\
\frac{\epsilon}{m}, & otherwise
\end{cases}
$$

<img src="/images/2017/rl-notes/rl4-mcc.png" alt="" width="860" height="509" class="aligncenter size-full wp-image-328" />

**Every Episode**

- Policy evaluation: Monte-Carlo policy evaluation, $Q \approx q_\pi$
- Policy improvement: $\epsilon$-greedy improvement

### On-policy TD Control

**SARSA**

<img src="/images/2017/rl-notes/rl4-sarsa-239x300.png" alt="" width="239" height="300" class="aligncenter size-medium wp-image-329" />

$$ Q(S,A) \gets Q(S,A) + \alpha(R+\gamma Q(S',A') - Q(S,A)) $$

<img src="/images/2017/rl-notes/rl-sarsa2.png" alt="" width="807" height="488" class="aligncenter size-full wp-image-330" />

**Every time step**

- Policy evaluation: Sarsa, $Q \approx q_\pi$
- Policy improvement: $\epsilon$-greedy improvement

<img src="/images/2017/rl-notes/rl4-sarsa-algo-1024x381.png" alt="" width="660" height="246" class="aligncenter size-large wp-image-331" />

### Sarsa$(\lambda)$

n-Step Sarsa

- $n=1$, $q_t^{(1)} = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$, Sarsa
- $n=2$, $q_t^{(2)} = R_{t+1} + \gamma R_{t+1} + \gamma^s Q(S_{t+2}, A_{t+2})$
- $n=\infty$, $q_t^{(\infty)} = R_{t+1} +\gamma R_{t+2} + ... + \gamma^{T-1}R_T$

$$ q_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1}R_{t+n} + \gamma^n Q(S_{t+n}, A_{t+n}) $$

$$ Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha (q_t^{(n)} - Q(S_t, A_t)) $$

$\lambda$ return as TD$(\lambda)$

<img src="/images/2017/rl-notes/rl4-sarsa-lambda.png" alt="" width="527" height="528" class="aligncenter size-full wp-image-333" />

$$ q_t^\lambda = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1}q_t^{(n)} $$

Forward View

$$ Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha (q_t^{\lambda} - Q(S_t, A_t)) $$

Backward View use **eligibility traces**

$$ E_0(s, a) = 0 $$

$$ E_t(s, a) = \gamma \lambda E_{t-1}(s, a) + 1(S_t=s, A_t=a) $$

$Q(s, a)$ is updated for every state $s$ and action $a$

$$ \delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) $$

$$ Q(s, a) \gets Q(s, a) + \alpha \delta_t E_t(s, a) $$

<img src="/images/2017/rl-notes/rl4-sarsa-lambda-algo-1024x582.png" alt="" width="660" height="375" class="aligncenter size-large wp-image-334" />

Sarsa $(\lambda)$ makes reward information flow backward to the path it follows

<img src="/images/2017/rl-notes/rl4-info-1024x332.png" alt="" width="660" height="214" class="aligncenter size-large wp-image-336" />

### Off-policy Learning

targe policy $\pi$, behave policy $\mu$, with importance sampling

$$ V(S_t) \gets V(S_t) + \alpha (\frac{\pi(A_t|S_t)}{\mu(A_t|S_t)}(R_{t+1} + \gamma V(S_{t+1})) - V(S_t)) $$

**Q-Learning**

- Consider off-policy learning of action-value $Q(s,a)$
- No importance sampling required
- Next action is chosen using behavior policy $A_{t+1} \sim \mu(\cdot|S_t)$
- But consider alternative successor action $A' \sim \pi(\cdot|S_t)$
- Update $Q(S_t, A_t)$ towards value of alternative action

$$ Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, A') - Q(S_t, A_t)) $$

policy $\pi$ is **greedy** w.r.t $Q(s,a)$, policy $\mu$ is $\epsilon$-greedy w.r.t $Q(s,a)$

$$ \pi(S_{t+1}) = \arg\max_{a'}Q(S_{t+1}, a') $$

$$ R_{t+1} + \gamma Q(S_{t+1}, A') = R_{t+1} + \max_{a'}\gamma Q(S_{t+1}, a') $$

<img src="/images/2017/rl-notes/rl4-q.png" alt="" width="301" height="280" class="aligncenter size-full wp-image-337" />

$$ Q(S,A) \gets Q(S,A) + \alpha(R + \gamma \sum_{a'}Q(S', a') - Q(S,A)) $$

<img src="/images/2017/rl-notes/rl4-q-algo-1024x347.png" alt="" width="660" height="224" class="aligncenter size-large wp-image-338" />

### Summary

<img src="/images/2017/rl-notes/rl4-summary.png" alt="" width="1154" height="689" class="aligncenter size-full wp-image-341" />

<img src="/images/2017/rl-notes/rl4-summary-2.png" alt="" width="1160" height="460" class="aligncenter size-full wp-image-342" />

### References

- [RL Course by David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
