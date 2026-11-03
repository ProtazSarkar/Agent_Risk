# Agentic Risk Analysis Architecture

## 1. System Overview

The **Agentic Risk Analysis** framework provides active behavioral shielding and safety assurance for autonomous agent systems. Instead of treating agent interactions as atomic single-turn prompts, the framework:
1. Decomposes multi-step goals into structured **Directed Acyclic Graphs (Task DAGs)** with alternative routes.
2. Extracts **psycholinguistic textual markers** and **graph topological features** across tasks.
3. Evaluates task necessity and eliminates superfluous branches via a **Necessity Filter**.
4. Applies a high-throughput **XGBoost Risk Shield** for fast inference-time classification (Safe, Warning, Blocked).
5. Formulates multi-step agent trajectories using **Markov Decision Chains** and sets dynamic audit thresholds using **Stackelberg Security Games**.

```
+-------------------------------------------------------------+
|                         User Prompt                         |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                       Planner Layer                         |
|   (APIPlanner / LocalPlanner with fine-tuned LoRA Llama 3)   |
+-------------------------------------------------------------+
                              |
                              v [Structured JSON PlanDAG]
+-------------------------------------------------------------+
|                      Necessity Filter                       |
|          (Prunes dead branches & redundant tasks)           |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                  Feature Extractor Layer                    |
|  - Psycholinguistic Markers (Urgency, Deception, Privilege) |
|  - Graph Topology (In/Out-Degree, Centrality, Roots, Leaves)|
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                    XGBoost Risk Shield                      |
|       Predicts task risk probability & safety verdicts       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                   Decision Theory Shield                    |
|  - Markov Chain: Cumulative trajectory compromise modeling   |
|  - Stackelberg Game: Dynamic leader-follower thresholding   |
+-------------------------------------------------------------+
```

---

## 2. Mathematical Formulations

### 2.1 Multi-Route Task DAG
An agent plan $\mathcal{P}$ consists of multiple execution routes $\mathcal{R}_1, \dots, \mathcal{R}_k$. Each route is a directed acyclic graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where:
- $\mathcal{V} = \{t_1, t_2, \dots, t_n\}$ are atomic task nodes.
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ are causal dependency edges where $(t_i, t_j) \in \mathcal{E}$ indicates $t_i$ must finish before $t_j$ starts.

### 2.2 Necessity Score $\mathcal{N}(t)$
For any task $t \in \mathcal{V}$, let $\mathcal{L} \subset \mathcal{V}$ be the set of terminal sink nodes ($d_{out}(v) = 0$).
$$\mathcal{N}(t) = \begin{cases}
0.05 & \text{if } \forall l \in \mathcal{L}, \nexists \text{ path } t \rightsquigarrow l \\
\min\left(1.0, \, b(t) + 0.3 \cdot \frac{|\text{Descendants}(t)|}{|\mathcal{V}| - 1}\right) & \text{otherwise}
\end{cases}$$
where $b(t) = 0.8$ if $t$ is labeled critical, and $0.5$ otherwise.

### 2.3 Markov Trajectory Model
Execution over route $\mathcal{R}$ is modeled across states $S = \{\text{Safe}, \text{Warning}, \text{Compromised}, \text{Goal}\}$ with transition matrix $\mathbf{T}$.
- $\text{Compromised}$ is an absorbing hazard state: $\mathbf{T}_{2,2} = 1$.
- State progression follows $\mathbf{v}_{k+1} = \mathbf{v}_k \mathbf{T}$.

### 2.4 Stackelberg Shield Policy
The defender acts as the leader setting detection threshold $\theta \in [0, 1]$. An adversary observes $\theta$ and chooses an attack profile. The defender solves:
$$\max_{\theta} \, \mathbb{E}[U_D(\theta)] = \sum_{i=1}^M \left[ \mathbb{I}(p_i \ge \theta) \cdot U_{\text{block}}(y_i) + \mathbb{I}(p_i < \theta) \cdot U_{\text{allow}}(y_i) \right]$$
where $U_{\text{block}}(0) = -c_{\text{FA}}$, $U_{\text{allow}}(1) = -c_{\text{breach}}$, and $U_{\text{allow}}(0) = r_{\text{success}}$.

