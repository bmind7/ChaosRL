# ChaosRL – PPO Ball Balance Example with Zero Dependencies

A small experiment where an agent learns to balance a ball on a tilting platform.

Built mostly for fun and education — no external ML frameworks.  
Everything runs on top of a custom **autodiff engine** and a **minimal PPO implementation** written from scratch in C#.

![training preview](screenshot.gif)

---

### ✨ What’s inside
- Simple **continuous control task** – tilt a panel to keep a ball centered  
- **Autodiff engine** (scalar-based, tensor-based, dynamic graph)  
- **PPO** implementation with entropy bonus, value function, clipping, etc.  
- All code runs in **Unity / C# / Burst** 

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
- **Value.cs** simple implementation of AutoDiff, easy to understand 
- **Tensor.cs** data oriented implementation of AutoDiff
- **Academy.cs** PPO implementation

---

### 🧠 Roadmap
- ~~Tensor class with better data layout~~ (**200x-400x** speedup of MatMul)
- ~~Backend with vectorized ops (CPU)~~ (**50x** speedup)
- ~~Multithreading support for simulation and training~~ (**8x** speedup)
- **K-Blocking** support for MatMul with fused Mult + Add operations (that should close the gap with PyTorch)
- Refactor agent code to support batching during inference
- **Compute shader** backend (GPU)
- Core plumbing: Agents, Academy, Save/Load, Configs, Telemetry

---

### Benchmarks

**ChaosRL** MatMul
| Matrix Size           | Avg Time (ms) | Std Dev (ms) | GFLOPS |
| --------------------- | ------------- | ------------ | ------ |
| 64×64 @ 64×64         | 0.078         | 0.008        | 20.28  |
| 128×128 @ 128×128     | 0.260         | 0.021        | 48.39  |
| 256×256 @ 256×256     | 1.004         | 0.059        | 100.28 |
| 512×512 @ 512×512     | 3.481         | 0.177        | 231.32 |
| 1024×1024 @ 1024×1024 | 27.856        | 0.971        | 231.28 |
| 2048×2048 @ 2048×2048 | 202.621       | 3.737        | 254.36 |

**PyTorch** MatMul
| Matrix Size           | Avg Time (ms) | Std Dev (ms) | GFLOPS |
| --------------------- | ------------- | ------------ | ------ |
| 64×64 @ 64×64         | 0.343         | 0.200        | 4.58   |
| 128×128 @ 128×128     | 0.328         | 0.072        | 38.37  |
| 256×256 @ 256×256     | 0.651         | 0.157        | 154.71 |
| 512×512 @ 512×512     | 2.340         | 0.286        | 344.21 |
| 1024×1024 @ 1024×1024 | 12.508        | 0.263        | 515.05 |
| 2048×2048 @ 2048×2048 | 98.482        | 1.084        | 523.34 |

*Note: to benchmark build with **IL2CPP** backend and **MatMultBenchmark** scene*

---

### ❓Current issues 
- Broadcasting is very limited for now and supports only shape mismatch. For example, **Add** two tensors (2, 5, 10) and (5,10) will work, because the tail of the shape is identical. Operations on scalars tensors also will work because scalar has shape (1) which can match anything.
- Because of limited broadcasting **Normalize()** method will work only on whole tensor, dim=0 or dim=last_dimention 
- **ExpandLast** works only for last dim. I need to add general **Expand** then **Normalize** in any dimension will be easy

---

### 💡 Notes
This is not a production ready code yet. It’s a learning playground to understand how PPO and autodiff actually work under the hood. 
