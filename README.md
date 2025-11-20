# ChaosRL – PPO Ball Balance Example with Zero Dependencies

A small experiment where an agent learns to balance a ball on a tilting platform.

Built mostly for fun and education — no external ML frameworks, and no performance optimization yet.  
Everything runs on top of a custom **autodiff engine** and a **minimal PPO implementation** written from scratch in C#.

![training preview](screenshot.gif)

---

### ✨ What’s inside
- Simple **continuous control task** – tilt a panel to keep a ball centered  
- **Autodiff engine** (scalar-based, tensor-based, dynamic graph)  
- **PPO** implementation with entropy bonus, value function, clipping, etc.  
- All code runs in **Unity / C#**, single-threaded  

---

### 👾 How to run
- Open the **Dojo** scene  
- There’s an **Academy** object that contains the main PPO implementation and config  
- An **ArenaSpawner** creates agents on play  
- Press **Play**  
- The policy should converge around **200k-300k steps** and total reward should reach 500-1000 on average
  _Note: there’s a chance the policy will initialize poorly and struggle to learn. 
   If rewards stay below 100, just restart the policy. 
   Usually it happens when *Entropy Loss* below 1.0_

---

### 💻 Code to look at
- **Value.cs** simple implementaion of AutoDiff, easy to understand 
- **Tensor.cs** data oriented implementation of AutoDiff
- **Academy.cs** PPO implementation

---

### 🧠 Roadmap
- ~~Tensor calss with better data layout~~ (**200x-400x** speed up of MatMul)
- Backend with **vectorized ops (CPU)**
- **Multithreading** support for simulation and training
- **Compute shader** backend (GPU)
- Core plumbing: Agents, Academy, Save/Load, Configs, Telemetry

---

### ❓Current issues 
- Performance on CPU is still way behind libs like PyTorch
- Broadcasting is very limited for now and supports only shape mismatch. For example, **Add** two tensors (2, 5, 10) and (5,10) will work, because the tail of the shape is identical. Operations on scalars tensors also will work becuase scalar has shape (1) which can match anything.
- Because of limited broadcasting **Normalize()** method will work only on whole tensor, dim=0 or dim=last_dimention 
- **ExpandLast** works only for last dim. I need to add general **Expand** then **Normalize** in any dimention will be easy

---

### 💡 Notes
This is not a production ready code. It’s a learning playground to understand how PPO and autodiff actually work under the hood. 
