# Centralised CEM-MPC for Counterfactual Joint-Action Planning

The planner in our current proposal can be understood very simply:

> **The world model answers “what happens next?”, while CEM-MPC repeatedly asks the world model “what happens if all agents do this instead?” and searches for the best joint-action sequence.**

Formally,

```math
\boxed{
\text{current multi-agent state}
\rightarrow
\text{sample joint-action plans}
\rightarrow
\text{world-model rollouts}
\rightarrow
\text{score plans}
\rightarrow
\text{refine plans}
\rightarrow
\text{execute}
}
```

The phrase **centralised CEM-MPC** is not one special algorithm with one canonical implementation. It combines three independent ideas:

```math
\underbrace{\text{Centralised}}_{\text{who is planned for?}}
+
\underbrace{\text{CEM}}_{\text{how is the best plan searched?}}
+
\underbrace{\text{MPC}}_{\text{how is the plan executed?}}.
```

CEM originated as a general stochastic optimisation method that adapts a sampling distribution towards high-performing samples; it was later widely used for trajectory optimisation in learned model-based control, including PETS and PlaNet-style systems. citeturn2search0turn1search3turn0search0 LeWorldModel follows the same basic latent-MPC philosophy: roll out candidate action sequences through the learned latent dynamics, minimise a terminal latent goal cost, execute only part of the plan, and replan. citeturn5view0

For **our paper**, the key point is that CEM does not merely consume predictions from the world model. It repeatedly searches for unusual actions that the model predicts will work well. This makes CEM an especially clean tool for exposing the exact problem we care about:

```math
\boxed{
\text{Can the learned multi-agent model correctly evaluate}
\;
\textbf{counterfactual joint actions}
\;
\text{proposed by the planner?}
}
```

This matches the planning-centric gap already identified in the supplied literature review. fileciteturn0file0 fileciteturn0file1

## Core Idea

### Start without the terminology

Suppose we have two agents.

At the current state,

```math
Z_t=(z_t^1,z_t^2),
```

we could ask them to execute:

```math
A_t=(a_t^1,a_t^2).
```

But we do not know which joint action is best.

So instead of immediately acting, we **imagine** several possibilities:

```text
Plan 1: Agent 1 → left,  Agent 2 → right
Plan 2: Agent 1 → left,  Agent 2 → left
Plan 3: Agent 1 → right, Agent 2 → right
...
```

A world model predicts what each plan would cause:

```math
(Z_t,A_t)
\xrightarrow{F_\theta}
\hat Z_{t+1}.
```

For a horizon \(H\), we recursively simulate:

```math
\hat Z_{t+1}
=
F_\theta(Z_t,A_t),
```

```math
\hat Z_{t+2}
=
F_\theta(\hat Z_{t+1},A_{t+1}),
```

```math
\cdots
```

```math
\hat Z_{t+H}
=
F_\theta(\hat Z_{t+H-1},A_{t+H-1}).
```

This is exactly the role played by learned latent dynamics in systems such as PlaNet and LeWM: rather than executing candidate controls in the real environment, the controller evaluates hypothetical futures through the learned model. citeturn0search0turn5view0

We give each imagined future a cost. For goal-directed control, the simplest form is

```math
J(A_{t:t+H-1})
=
d(\hat Z_{t+H},Z_g).
```

Then ideally,

```math
A^*
=
\arg\min_{A_{t:t+H-1}}
J(A_{t:t+H-1}).
```

The difficulty is that \(A\) may contain dozens or hundreds of continuous variables. We therefore need an optimiser.

That optimiser is **CEM**.

After finding a plan, we **do not normally execute the entire sequence**. We execute only the first action,

```math
A_t^*,
```

observe the real next state,

```math
Z_{t+1},
```

and solve the planning problem again.

That is **MPC**.

Finally, if

```math
A_t
=
(a_t^1,\ldots,a_t^N)
```

contains actions for **all agents** and the optimisation is performed jointly using the team state, the planner is **centralised**.

So:

```math
\boxed{
\text{Centralised CEM-MPC}
=
\text{joint multi-agent planning}
+
\text{CEM trajectory optimisation}
+
\text{receding-horizon control}.
}
```

This distinction is useful because MAZero, for example, is also learned-model multi-agent planning, but uses a centralised model with MCTS rather than CEM trajectory optimisation. Its authors explicitly identify the growth of the multi-agent joint action space as a central planning difficulty. citeturn9search0

## Taxonomy

There are several independent decisions hidden inside the words “world-model planner”. Separating them makes our method much easier to position.

| Axis | Main choices | Our current choice |
|---|---|---|
| **Dynamics source** | Exact simulator / analytical model / learned world model | **Learned latent world model** |
| **Representation** | Physical state / observation / latent state | **Latent or structured vector latent** |
| **Planning execution** | Open-loop / receding horizon MPC | **MPC** |
| **Trajectory optimiser** | Random shooting / CEM / MPPI / gradients / MCTS | **CEM** |
| **Agent coordination** | Independent / distributed / centralised | **Centralised** |
| **Action proposal** | Full joint distribution / factorised distribution / structured distribution | Start **factorised** |
| **Objective** | Reward / hand-designed cost / value / goal distance / learned energy | Initially **goal/task cost** |
| **Model uncertainty** | Deterministic / probabilistic / ensemble | Initially **deterministic** |

PlaNet is a canonical example of learned latent dynamics plus online trajectory optimisation; PETS combines probabilistic learned dynamics with CEM-based MPC; and LeWM uses latent-space CEM/MPC with a goal-matching cost. citeturn0search0turn1search3turn5view0

### Open-loop planning versus MPC

These are easy to confuse.

In **open-loop trajectory optimisation**, we solve once:

```math
A_{t:t+H-1}^*
=
\arg\min_A J(A),
```

then execute:

```math
A_t^*,A_{t+1}^*,\ldots,A_{t+H-1}^*.
```

If the model was wrong at step two, too bad.

MPC instead does:

```math
\text{plan}
\rightarrow
\boxed{\text{execute only first action}}
\rightarrow
\text{observe}
\rightarrow
\text{plan again}.
```

Or occasionally the first \(K\ll H\) actions are executed before replanning. LeWM explicitly adopts this receding-horizon strategy because autoregressive model error grows with rollout horizon. citeturn5view0

This is particularly attractive for a learned world model:

```math
\boxed{
\text{prediction error accumulates}
\quad\Longrightarrow\quad
\text{reobserve and correct frequently}.
}
```

### Centralised versus distributed MPC

In **centralised MPC**, one optimiser receives a representation of the full relevant team state and produces all agents' actions:

```math
Z_t
=
(z_t^1,\ldots,z_t^N),
```

```math
A_t^*
=
(a_t^{1*},\ldots,a_t^{N*}).
```

In **distributed MPC**, agents instead solve local optimisation problems while exchanging information or repeatedly negotiating coupled variables. Classical multi-agent MPC literature contains joint, prioritised, distributed, and conflict-based variants precisely because solving one large central optimisation becomes costly as team size and coupling increase. citeturn1academia36turn8search1

For the present paper, I strongly favour **centralised MPC** because decentralisation is not the research question. Centralisation gives us the cleanest possible scientific instrument for asking:

> **If the planner has all necessary information, is the learned world model itself good enough to evaluate joint interventions?**

That isolates world-model fidelity from communication, local observability, consensus optimisation, and decentralised execution.

### Centralised does not mean “monolithic model”

This is especially important for our method.

We can have:

```math
\text{relational / factorised world model}
+
\text{centralised planner}.
```

For example,

```math
m_t^i
=
\sum_{j\neq i}
\phi
(
z_t^i,z_t^j,a_t^i,a_t^j
),
```

```math
\hat z_{t+1}^i
=
f(z_t^i,a_t^i,m_t^i),
```

while CEM still chooses:

```math
A_{t:t+H-1}
=
\{
a_{t:t+H-1}^1,
\dots,
a_{t:t+H-1}^N
\}
```

jointly.

So the architecture can exploit **local interaction structure**, while planning remains globally coordinated. That separation is central to the research direction developed in the supplied review. fileciteturn0file0

## Mathematical Formulation

Let there be \(N\) cooperative agents. Agent \(i\) has latent state

```math
z_t^i\in\mathbb R^{d_z}
```

and continuous action

```math
a_t^i\in\mathbb R^{d_a}.
```

Define the team representation and joint action as

```math
Z_t
=
(z_t^1,\ldots,z_t^N),
```

```math
A_t
=
(a_t^1,\ldots,a_t^N).
```

For homogeneous agents we may share the transition parameters across agents:

```math
\hat z_{t+1}^i
=
F_\theta
\left(
z_t^i,
a_t^i,
\{z_t^j,a_t^j\}_{j\neq i}
\right).
```

The essential multi-agent difference is therefore:

```math
\boxed{
a_t^j
\text{ can alter }
z_{t+1}^i
\quad
\text{for }
i\neq j.
}
```

This is the dependency our proposed relational predictor is designed to represent. fileciteturn0file1

### Planning variable

The planner does not optimise only the next action.

It optimises a **joint action sequence**:

```math
\mathbf A
=
(A_t,A_{t+1},\ldots,A_{t+H-1}).
```

Since

```math
A_h
=
(a_h^1,\ldots,a_h^N),
```

the continuous optimisation dimension is

```math
\boxed{
D_{\text{plan}}
=
N\,d_a\,H.
}
```

For example, with

```math
N=4,\qquad d_a=2,\qquad H=10,
```

CEM is searching an

```math
D_{\text{plan}}=80
```

dimensional continuous variable.

This is different from saying CEM explicitly enumerates an exponential number of joint actions. It does not. It performs stochastic optimisation in an increasingly high-dimensional continuous space. Exact enumeration of \(m\) discrete actions per agent over horizon \(H\), by comparison, would contain

```math
m^{NH}
```

sequences. MAZero's motivation similarly notes that multi-agent planning faces an expansive joint action space and therefore needs special search machinery. citeturn9search0

### World-model rollout

For candidate \(k\),

```math
\mathbf A^{(k)}
=
(A_t^{(k)},\ldots,A_{t+H-1}^{(k)}),
```

initialise

```math
\hat Z_t^{(k)}
=
Z_t.
```

Then recursively apply:

```math
\hat Z_{t+h+1}^{(k)}
=
F_\theta
\left(
\hat Z_{t+h}^{(k)},
A_{t+h}^{(k)}
\right),
```

for

```math
h=0,\ldots,H-1.
```

PlaNet, DINO-WM, V-JEPA 2-AC and LeWM all instantiate the general principle of using predicted future representations to support test-time action optimisation, although their representations, models and costs differ. citeturn0search0turn3search0turn4academia24turn5view0

### Cost function

For our simplest cooperative goal-reaching task:

```math
J(\mathbf A)
=
d
\left(
\hat Z_{t+H}(\mathbf A),
Z_g
\right).
```

For assigned agent goals:

```math
J(\mathbf A)
=
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
\hat z_{t+H}^i
-
z_g^i
\right\|_2^2.
```

Another valid formulation includes trajectory costs:

```math
J(\mathbf A)
=
\sum_{h=1}^{H}
c(\hat Z_{t+h},A_{t+h-1})
+
c_T(\hat Z_{t+H}).
```

LeWM uses the particularly simple terminal latent-distance form, optimising the predicted terminal latent state towards the encoded goal observation. citeturn5view0

The exact cost is conceptually separate from CEM:

```math
\boxed{
\text{CEM does not care what }J\text{ means.}
}
```

It only requires that candidate trajectories can be assigned scalar scores. This is why CEM can optimise hand-designed physical costs, predicted rewards, latent distances, value-bootstrapped objectives, or eventually a learned **energy function**. Differentiable CEM work has even examined CEM in energy-based structured prediction and continuous-control settings. citeturn1search2

That gives us a clean future bridge to energy-based modelling:

```math
J(\mathbf A)
\quad\longrightarrow\quad
E_\psi(Z_t,\mathbf A,Z_g).
```

The CEM planner itself would not need to change.

## How the Cross-Entropy Method Works

Here is the intuitive version:

> **Guess many plans → keep the good ones → move the sampling distribution towards them → repeat.**

That is CEM.

Rubinstein's original formulation presents CEM as a stochastic optimisation method based on adapting a probability distribution by minimising a cross-entropy/KL objective towards a distribution concentrated around desirable solutions. citeturn2search0

### Initial distribution

For continuous actions, define a distribution over complete action sequences:

```math
q_0(\mathbf A)
=
\mathcal N
(
\mathbf A;
\mu_0,\Sigma_0
).
```

Frequently,

```math
\mu_0=0
```

and \(\Sigma_0\) is chosen broad enough to cover the permitted action range.

In a simple implementation we use diagonal covariance:

```math
\Sigma
=
\operatorname{diag}(\sigma^2).
```

The optimisation variable is still the **entire joint trajectory**.

### Sample candidate plans

At CEM iteration \(r\):

```math
\mathbf A^{(k)}
\sim
q_r(\mathbf A),
\qquad
k=1,\ldots,K.
```

So we might generate:

```text
Candidate 1:
 Agent 1: → → ↑ ...
 Agent 2: ↑ ← ↑ ...

Candidate 2:
 Agent 1: ← → ↓ ...
 Agent 2: → → ↑ ...

...
Candidate K
```

Each is a complete hypothetical team plan.

### Roll out every candidate

For each candidate,

```math
\mathbf A^{(k)},
```

we query the world model:

```math
Z_t
\rightarrow
\hat Z_{t+1}^{(k)}
\rightarrow
\hat Z_{t+2}^{(k)}
\rightarrow\cdots\rightarrow
\hat Z_{t+H}^{(k)}.
```

Then calculate

```math
J_k
=
J(\mathbf A^{(k)}).
```

This repeated model evaluation is why latent models can be attractive: test-time trajectory optimisation may require thousands of model evaluations, so compact latent predictors can be considerably cheaper than heavyweight observation-generative models. LeWM specifically emphasises planning efficiency as one motivation for its compact latent representation. citeturn5view0

### Select elites

Suppose lower cost is better.

Sort candidates:

```math
J_{(1)}
\le
J_{(2)}
\le
\cdots
\le
J_{(K)}.
```

Take the best \(M\), or equivalently an elite fraction

```math
\rho
=
\frac{M}{K}.
```

Call the elite set

```math
\mathcal E_r.
```

Only these candidates are used to update the sampling distribution.

### Fit the next distribution

For a Gaussian CEM implementation:

```math
\mu_{r+1}
=
\frac{1}{M}
\sum_{\mathbf A^{(k)}\in\mathcal E_r}
\mathbf A^{(k)},
```

and

```math
\Sigma_{r+1}
=
\frac{1}{M}
\sum_{\mathbf A^{(k)}\in\mathcal E_r}
(\mathbf A^{(k)}-\mu_{r+1})
(\mathbf A^{(k)}-\mu_{r+1})^\top.
```

For diagonal CEM this reduces to independent variances for each action coordinate.

This update is not arbitrary. For a chosen parametric family \(q_\eta\), CEM can be viewed as fitting the sampling distribution to an elite/rare-event distribution by minimising

```math
D_{\mathrm{KL}}
(
p_{\text{elite}}
\|
q_\eta
),
```

which is equivalent to maximising the elite samples' log-likelihood under \(q_\eta\). citeturn2search0turn2search1

So:

```math
\boxed{
\text{CEM gradually concentrates probability mass around low-cost plans.}
}
```

### Repeat

Run several CEM iterations:

```math
q_0
\rightarrow
q_1
\rightarrow
q_2
\rightarrow
\cdots
\rightarrow
q_R.
```

Conceptually:

```text
Iteration 0
[very broad possible actions]

        ↓ evaluate

Iteration 1
[more probability near good actions]

        ↓ evaluate

Iteration 2
[more concentrated]

        ↓

Final plan
```

CEM has been widely used in learning-based MPC because it is derivative-free and can optimise non-convex objectives, although ordinary random-sampling CEM becomes inefficient in high-dimensional control. Hybrid CEM/gradient optimisation and iCEM were developed partly to address this sampling burden; iCEM adds mechanisms such as temporally correlated samples and memory and reported substantially lower sampling requirements in its experiments. citeturn1search0turn1search1

### Then MPC closes the loop

Once CEM finishes:

```math
\mathbf A^*
=
(A_t^*,A_{t+1}^*,\ldots,A_{t+H-1}^*).
```

Do **not** execute everything.

Execute:

```math
\boxed{A_t^*}
```

meaning simultaneously:

```math
a_t^{1*},
a_t^{2*},
\dots,
a_t^{N*}.
```

Observe the real next state,

```math
Z_{t+1}^{\text{real}},
```

then restart CEM:

```math
Z_{t+1}^{\text{real}}
\xrightarrow{\text{CEM}}
A_{t+1:t+H}^*.
```

That is the receding-horizon MPC loop used in latent-planning systems such as LeWM. citeturn5view0

The complete controller is therefore:

```text
while episode not finished:

    observe current team state Z_t

    initialise CEM distribution q(A)

    repeat R CEM iterations:

        sample K joint action sequences

        for every candidate:
            roll out learned world model for H steps
            compute candidate cost

        choose best M candidates

        update CEM distribution from elites

    execute first joint action A_t

    receive true next observation

    replan
```

## Why Centralisation Matters in Multi-Agent Control

Here is the key distinction for our paper.

### Independent planning

An independent planner would solve:

```math
a_{t:t+H}^{i*}
=
\arg\min
J_i
\left(
a_{t:t+H}^i
\right)
```

for each agent separately.

This effectively assumes something like:

```math
z_{t+1}^i
\approx
f(z_t^i,a_t^i).
```

But that fails when:

```math
a_t^j
\rightarrow
z_{t+1}^i.
```

Examples include collision avoidance, cooperative transport, formation control, pushing the same object, covering landmarks, or agents physically blocking one another. Multi-agent control literature therefore frequently has to reason about coupled actions rather than solving completely independent controllers. citeturn1academia36turn8search1

### Centralised planning

Our planner instead searches:

```math
\boxed{
\mathbf A^*
=
\arg\min_{\mathbf A}
J
\left(
F_\theta^{(H)}(Z_t,\mathbf A)
\right).
}
```

Here

```math
\mathbf A
=
\begin{bmatrix}
a_t^1 & \cdots & a_t^N\\
a_{t+1}^1 & \cdots & a_{t+1}^N\\
\vdots & & \vdots\\
a_{t+H-1}^1 & \cdots & a_{t+H-1}^N
\end{bmatrix}.
```

That means the optimiser can discover coordination such as:

```text
Agent 1 moves left
while
Agent 2 moves right
```

because the **pair** is evaluated as a single candidate future.

This is what we need.

### Factorised CEM is still centralised

This distinction is subtle but very important.

Suppose we use:

```math
q(\mathbf A)
=
\prod_{h=0}^{H-1}
\prod_{i=1}^{N}
q_{h,i}(a_{t+h}^i).
```

It looks independent because each action is sampled from its own Gaussian.

But after sampling, we assemble:

```math
\mathbf A^{(k)}
=
(
A_t^{(k)},
\ldots,
A_{t+H-1}^{(k)}
)
```

and score it jointly:

```math
J_k
=
J
\left(
F_\theta^{(H)}
(
Z_t,
\mathbf A^{(k)}
)
\right).
```

So:

```math
\boxed{
\text{factorised proposal}
\neq
\text{decentralised planning}.
}
```

The search distribution may be factorised, but the **evaluation and elite selection are globally coupled**.

I would call this:

> **Centralised CEM with factorised action proposals.**

That is probably the cleanest baseline for our paper.

### Why not start with full covariance?

A full Gaussian over

```math
D=N d_a H
```

variables can explicitly represent correlations such as

```math
a_t^1
\leftrightarrow
a_t^2
```

and temporal correlations such as

```math
a_t^1
\leftrightarrow
a_{t+1}^1.
```

But its covariance contains \(O(D^2)\) entries, making estimation increasingly expensive and sample-hungry.

A diagonal/factorised distribution contains only \(O(D)\) means and variances.

This is the reasonable initial trade-off:

```math
\boxed{
\text{simple proposal distribution}
+
\text{interaction-aware joint scoring}.
}
```

If this later proves insufficient, iCEM-style temporal correlations or structured covariance are natural extensions; iCEM specifically shows that sample reuse and temporal correlation can improve CEM efficiency in high-dimensional control. citeturn1search1

## Why This Planner Is Almost Perfect for Our Research Question

This is where CEM becomes more than an implementation detail.

It actually gives us the **experimental mechanism** for studying counterfactual multi-agent world models.

### CEM deliberately changes the action distribution

Suppose offline data came from behaviour policy \(\beta\):

```math
A_t
\sim
\beta(A_t\mid Z_t).
```

The world model is trained on:

```math
(Z_t,A_t,Z_{t+1})
\sim
\mathcal D_\beta.
```

If agents behaved cooperatively in the dataset, their actions may be strongly correlated.

For example, the dataset frequently contains:

```math
(a_A^1,a_A^2),
```

but almost never:

```math
(a_B^1,a_A^2).
```

CEM does not respect the behaviour distribution.

It samples alternative action combinations:

```math
\tilde A_t
=
(\tilde a_t^1,\ldots,\tilde a_t^N).
```

As optimisation proceeds, CEM moves its proposal distribution towards whatever the **learned model believes** is advantageous. CEM-based planners are therefore sensitive to model accuracy away from the observed training distribution; PETS explicitly addressed uncertainty in learned model-based planning, while LeWM notes that longer rollouts increase model bias and prediction error. citeturn1search3turn5view0

This gives our paper's central phenomenon:

```math
\boxed{
A_t\sim p_{\text{data}}
\qquad
\text{during training}
}
```

but

```math
\boxed{
A_t\sim q_{\text{CEM}}
\qquad
\text{during planning}.
}
```

And in the multi-agent case:

```math
q_{\text{CEM}}
```

contains **new combinations of agents' actions**.

This is exactly the counterfactual joint-action gap emphasised in the supplied review. fileciteturn0file0 fileciteturn0file1

### This makes plan ranking more important than MSE

CEM never asks:

> “Was the prediction numerically perfect?”

It asks:

> “Which candidates are the best?”

Suppose the real costs are:

```math
J(A)=1.0,
\qquad
J(B)=2.0.
```

The model predicts:

```math
\hat J(A)=1.5,
\qquad
\hat J(B)=2.7.
```

Numerically imperfect.

But ordering is correct:

```math
A<B.
```

CEM is still happy.

Now suppose:

```math
\hat J(A)=1.1,
\qquad
\hat J(B)=0.9.
```

The numerical errors might even look small, but the **ranking is reversed**.

CEM selects \(B\).

That is disastrous for control.

Therefore:

```math
\boxed{
\text{CEM fundamentally converts world-model prediction into a ranking problem.}
}
```

This gives a stronger justification for the metric we had already proposed:

```math
\rho_{\text{plan}}
=
\operatorname{Spearman}
\left(
\hat J(A^{(k)}),
J_{\text{oracle}}(A^{(k)})
\right).
```

The supplied literature review had already identified plan-ranking fidelity as more planning-relevant than ordinary one-step error. fileciteturn0file1

### We can make the metric even more CEM-specific

CEM literally selects an **elite set**.

Let:

```math
\mathcal E_{\text{WM}}
=
\text{top }M\text{ plans according to learned WM},
```

and use the exact VMAS simulator to obtain:

```math
\mathcal E_{\text{oracle}}
=
\text{true top }M\text{ plans}.
```

Then define:

```math
\boxed{
\text{Elite Agreement}
=
\frac{
|\mathcal E_{\text{WM}}
\cap
\mathcal E_{\text{oracle}}|
}{
M
}.
}
```

This quantity directly asks:

> **Does the world model tell CEM to keep the same candidates that the real environment would keep?**

We can also define **selection regret**:

```math
R_{\text{select}}
=
J_{\text{oracle}}
\left(
A_{\text{WM}}^*
\right)
-
\min_{k}
J_{\text{oracle}}
\left(
A^{(k)}
\right).
```

This is arguably even cleaner than next-state MSE.

And VMAS is particularly useful here because it provides a fast vectorised 2D physics simulator, giving us an exact simulator against which counterfactual candidate plans can be evaluated. citeturn10academia13

So the fact that VMAS has known cheap dynamics is **not a weakness for this experiment**.

It gives us an oracle.

```math
\boxed{
\text{VMAS simulator}
=
\text{counterfactual ground truth}
}
```

against which the learned planner can be diagnosed.

### Oracle CEM-MPC becomes an essential baseline

Run exactly the same CEM algorithm twice.

**Oracle:**

```math
\mathbf A_{\text{oracle}}^*
=
\operatorname{CEM}
(
F_{\text{VMAS}},
J
).
```

**Learned:**

```math
\mathbf A_{\text{WM}}^*
=
\operatorname{CEM}
(
F_\theta,
J
).
```

Same:

```math
K,\quad H,\quad R,\quad M,\quad J.
```

The only difference is:

```math
F_{\text{VMAS}}
\quad\text{vs}\quad
F_\theta.
```

Then:

```math
\boxed{
\Delta_{\text{planning}}
=
J_{\text{closed-loop}}^{\text{oracle MPC}}
-
J_{\text{closed-loop}}^{\text{learned MPC}}.
}
```

This cleanly measures:

> **How much of oracle multi-agent predictive control can be recovered from learned offline dynamics?**

That is a much stronger interpretation than simply saying:

> “Our model gets good VMAS reward.”

## Minimal Implementation and Positioning

For the first paper I would keep the controller almost boringly simple.

### World model

Compare:

```math
\textbf{Independent}
\qquad
\hat z_{t+1}^i
=
f(z_t^i,a_t^i),
```

```math
\textbf{Joint / Concatenated}
\qquad
\hat Z_{t+1}
=
f(Z_t,A_t),
```

and

```math
\textbf{Relational}
```

```math
m_t^i
=
\sum_{j\neq i}
\phi
(
z_t^i,z_t^j,
a_t^i,a_t^j
),
```

```math
\hat z_{t+1}^i
=
f(z_t^i,a_t^i,m_t^i).
```

All three use the **same CEM-MPC**.

That is crucial:

```math
\boxed{
\text{hold the planner constant}
\quad\Longrightarrow\quad
\text{differences come from the world model}.
}
```

### Planner

Use the same factorised Gaussian proposal:

```math
q(\mathbf A)
=
\prod_{h=0}^{H-1}
\prod_{i=1}^{N}
\mathcal N
\left(
a_{t+h}^i;
\mu_{h,i},
\operatorname{diag}(\sigma_{h,i}^2)
\right).
```

At every control step:

```math
\boxed{
\begin{aligned}
&\text{sample joint plans}\\
&\downarrow\\
&\text{roll out learned relational WM}\\
&\downarrow\\
&\text{score complete team future}\\
&\downarrow\\
&\text{select elites}\\
&\downarrow\\
&\text{update }(\mu,\sigma)\\
&\downarrow\\
&\text{execute first joint action}\\
&\downarrow\\
&\text{replan}.
\end{aligned}
}
```

### Sensible first planning grid

I would treat the following as **engineering starting points, not literature-prescribed constants**:

| Parameter | Initial values to test |
|---|---:|
| Horizon \(H\) | \(5,10,15\) |
| Candidates \(K\) | \(256,512,1024\) |
| CEM iterations \(R\) | \(3,5\) |
| Elite fraction | \(5\%-10\%\) |
| Actions executed before replanning | **1** |
| Proposal | diagonal Gaussian |
| Agents | \(2-4\) |

For example,

```math
N=4,\quad
d_a=2,\quad
H=10,
```

gives an 80-dimensional plan.

With

```math
K=512,\qquad R=5,
```

each real control step evaluates

```math
512\times5
=
2560
```

candidate trajectories, corresponding to roughly

```math
2560\times10
=
25\,600
```

world-model transition steps before batching.

This is precisely why keeping the learned model small and batched matters.

If standard CEM becomes the compute bottleneck, **iCEM is the first upgrade I would test**, rather than inventing a new planner. Its published changes—sample memory and temporally correlated actions—were explicitly designed to make CEM more sample-efficient for real-time model-based planning. citeturn1search1

### The core evaluation should now become

```math
\boxed{
\text{logged prediction}
}
```

```math
\downarrow
```

```math
\boxed{
\text{counterfactual prediction}
}
```

```math
\downarrow
```

```math
\boxed{
\text{CEM plan ranking / elite agreement}
}
```

```math
\downarrow
```

```math
\boxed{
\text{closed-loop MPC control}
}
```

Specifically, I would report:

| Level | Metric |
|---|---|
| Logged dynamics | next-step MSE |
| Counterfactual dynamics | oracle error under held-out joint actions |
| Candidate ranking | Spearman \(\rho\) |
| **CEM-specific ranking** | **elite-set agreement** |
| Plan quality | oracle selection regret |
| Final control | success / return |
| Oracle gap | learned MPC vs simulator MPC |

The potential headline result is then not merely:

```math
\text{Relational WM gets lower MSE}.
```

It is:

```math
\boxed{
\begin{array}{c}
\text{Independent / Joint / Relational models have similar}\\
\text{logged prediction error,}
\\[2mm]
\textbf{but}
\\[2mm]
\text{the relational model better preserves the oracle}\\
\text{ranking and elite set under counterfactual joint actions,}
\\[2mm]
\textbf{therefore}
\\[2mm]
\text{CEM-MPC obtains better closed-loop coordination.}
\end{array}
}
```

That is considerably more interesting.

### What CEM contributes scientifically

CEM itself should **not** be claimed as a methodological contribution. CEM is established, with roots in Rubinstein's stochastic optimisation work and extensive later use in model-based control. citeturn2search0turn1search3turn0search0

Centralised learned-model planning in MARL is also not new: MAZero already combines a centralised learned model with explicit MCTS search. citeturn9search0

Our positioning is narrower:

> **We use a simple centralised sampling-based planner as a probe of whether multi-agent latent world models support reliable counterfactual joint-action reasoning.**

More precisely:

```math
\boxed{
\textbf{Contribution}
\neq
\text{a new CEM algorithm}
}
```

```math
\boxed{
\textbf{Contribution}
=
\text{understanding which learned dynamics structure makes}
}
```

```math
\boxed{
\text{joint-action trajectory optimisation faithful enough for control.}
}
```

And CEM is almost ideal for this because its elite-selection mechanism makes **plan-ranking failure directly observable**.

There is also a natural later connection to the energy-based direction. CEM requires only a scalar objective, so the current hand-designed or latent-distance score

```math
J(\mathbf A)
```

can eventually become a learned energy

```math
E_\psi
(
Z_t,
\mathbf A,
Z_g
).
```

CEM remains unchanged:

```math
\boxed{
\mathbf A^*
=
\arg\min_{\mathbf A}
E_\psi(Z_t,\mathbf A,Z_g).
}
```

CEM has already been studied as an optimiser for energy-based structured prediction, while energy-based latent dynamics have also been explored for control in work such as CLOUD. citeturn1search2turn0search1 My recommendation, however, would be to **first establish the deterministic relational-world-model + CEM result**. Otherwise, transition modelling, energy modelling and planning all change simultaneously and the counterfactual question becomes much harder to isolate.

The simplest final picture for the current paper is therefore:

```math
\boxed{
Z_t
\xrightarrow{\text{CEM samples}}
\mathbf A^{(1)},\ldots,\mathbf A^{(K)}
}
```

```math
\boxed{
(Z_t,\mathbf A^{(k)})
\xrightarrow{\text{Relational World Model}}
\hat Z_{t+1:t+H}^{(k)}
}
```

```math
\boxed{
\hat Z^{(k)}
\xrightarrow{\text{goal cost}}
\hat J_k
}
```

```math
\boxed{
\hat J_1,\ldots,\hat J_K
\xrightarrow{\text{CEM elite selection}}
\mathbf A^*
}
```

```math
\boxed{
\text{execute }A_t^*
\rightarrow
\text{observe real }Z_{t+1}
\rightarrow
\text{repeat}
}
```

and the scientific question underneath it remains exactly the focused one we started with:

> **When CEM proposes joint actions that were not observed together in the offline data, does an interaction-aware world model preserve the correct ordering of those counterfactual futures well enough for MPC to control the team?**

### Key references

**Rubinstein, 1999 — *The Cross-Entropy Method for Combinatorial and Continuous Optimization*.** The foundational optimisation treatment of CEM and its KL/cross-entropy interpretation. citeturn2search0

**Chua et al., NeurIPS 2018 — *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS).* ** A canonical learned-dynamics + CEM-MPC system and an important reference for model uncertainty in sampling-based MPC. citeturn1search3

**Hafner et al., ICML 2019 — *Learning Latent Dynamics for Planning from Pixels (PlaNet).* ** Canonical latent dynamics + online trajectory optimisation/MPC reference. citeturn0search0

**Bharadhwaj et al., L4DC 2020 — *Model-Predictive Control via Cross-Entropy and Gradient-Based Optimization*.** Clearly describes standard CEM trajectory optimisation and its difficulties in high-dimensional action spaces. citeturn1search0

**Amos & Yarats, ICML 2020 — *The Differentiable Cross-Entropy Method*.** Useful both for differentiable CEM and for the connection between CEM and energy-based optimisation. citeturn1search2

**Pinneri et al., CoRL 2021 — *Sample-efficient Cross-Entropy Method for Real-time Planning*.** Introduces iCEM and shows how temporal correlation and sample reuse can substantially improve CEM efficiency. citeturn1search1

**Liu et al., ICLR 2024 — *Efficient Multi-agent Reinforcement Learning by Planning (MAZero).* ** Important positioning paper showing that centralised learned-model multi-agent planning already exists, using MCTS rather than CEM. citeturn9search0

**Zhou et al., ICML 2025 — *DINO-WM*.** Modern reward-free feature-space world modelling with test-time action-sequence optimisation. citeturn3search0

**Assran et al., 2025 — *V-JEPA 2*.** Demonstrates action-conditioned latent world modelling and goal-directed robotic planning after large-scale self-supervised pretraining. citeturn4academia24

**Maes et al., 2026 — *LeWorldModel*.** The closest methodological inspiration for our deliberately simple latent predictor + CEM/MPC approach; its planner minimises terminal latent goal distance and uses receding-horizon execution. citeturn5view0

**Bettini et al., 2022 — *VMAS*.** Provides the fast vectorised multi-agent physics environment that, in our study, can serve not only as a benchmark but as an **oracle for counterfactual CEM plan evaluation**. citeturn10academia13