# Optimal Sample Selection Algorithm Documentation

## 1. Problem Definition (问题定义)

### 1.1 问题描述

从 `n` 个样本（编号 0..n-1）中找到**最少数量**的 k-组（k-group），使得每个 j-子集都被至少 T 个 k-组"覆盖"。

**覆盖的定义**：k-组与 j-子集的公共元素数 ≥ s。

### 1.2 符号说明

| 符号 | 含义 | 取值范围 |
|------|------|----------|
| `n` | 样本总数 | 7 ≤ n ≤ 25 |
| `k` | 每组的大小（k-group） | 4 ≤ k ≤ 7 |
| `j` | 需要被覆盖的子集大小（j-subset） | s ≤ j ≤ k |
| `s` | 覆盖所需的最小交集大小 | 3 ≤ s ≤ 7 |
| `T` | 每个 j-子集至少被覆盖的次数 | T ≥ 1 |
| `C(a,b)` | 组合数，从 a 个中选 b 个 | - |

### 1.3 数学形式化

给定集合 S，|S| = n，求解：
```
minimize |G|
subject to: 对于所有 J ⊆ S，|J| = j，
             |{G ∈ G : |G ∩ J| ≥ s}| ≥ T
```

### 1.4 直观理解

想象你有 n 个不同的物品，要把它们分成若干"小组"，每个小组有 k 个物品。你的目标是：**用最少的组数**，保证**任意 j 个物品的组合**，都至少有 T 个小组"包含"它们中的至少 s 个。

**应用场景**：
- 测试用例生成（确保所有输入组合都被测试到）
- 传感器网络部署（确保所有区域都被覆盖）
- 药物筛选实验（确保所有组合都被验证）

---

## 2. Algorithm Overview (算法总览)

### 2.1 三层加速策略

| 层 | 技术 | 效果 | 说明 |
|----|------|------|------|
| 1 | 候选剪枝 | 减少 5~144 倍 | 每步只生成能覆盖当前"最紧迫" j-子集的候选 |
| 2 | 位掩码编码 | 速度提升约 10 倍 | 用 uint32 表示集合，交集 = 按位与 |
| 3 | Numba JIT 并行 | 速度提升 10~100 倍 | 内层循环编译为机器码，多核并行 |

### 2.2 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│  输入：n_samples, k, j, s, T, time_limit, seed             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 1: 预处理 (Preprocessing)                              │
│  - 生成所有 C(n,j) 个 j-子集                                  │
│  - 转换为位掩码数组 (j_masks)                                │
│  - 计算理论下界 (lb)                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 2: 贪心构造循环 (Greedy Loop)                          │
│  - 多次重启 (restarts)                                       │
│  - 前 5 次：纯随机重启                                         │
│  - 之后：80% LNS 扰动 + 20% 纯随机                             │
│  - 每次：贪心构造 → 冗余压缩 → 更新最优解                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 3: 局部搜索优化 (Local Search)                          │
│  - 使用剩余时间预算 (> 5 秒)                                  │
│  - Swap 操作：移除一个组，替换为新组，删除另一个冗余组         │
│  - 净效果：解大小 -1                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 4: 验证 (Verification)                                 │
│  - 检查所有 j-子集是否被覆盖 ≥ T 次                            │
│  - 返回结果和统计信息                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Foundations (数学基础)

### 3.1 位掩码编码 (Bitmask Encoding)

**概念**：将集合转换为 uint32 整数，第 i 位为 1 表示元素 i 在集合中。

**示例**：
```python
样本集合 S = {0, 1, 2, 3, 4, 5, 6}
子集 A = {0, 2, 4} → 二进制 0b10101 → 十进制 21

位位置：  6 5 4 3 2 1 0
         ┌──────────┐
         │0 1 0 1 0 1│
         └──────────┘
```

**核心操作**：

| 操作 | 位运算 | 数学含义 |
|------|--------|---------|
| 集合大小 | `popcount(A)` | \|A\| |
| 集合交集 | `A & B` | A ∩ B |
| 交集大小 | `popcount(A & B)` | \|A ∩ B\| |
| 元素添加 | `A | (1 << i)` | A ∪ {i} |

**优势**：
- 空间：32 位存储 25 个元素的集合（压缩率 88%）
- 时间：O(1) 获取交集大小（硬件指令 vs O(min(|A|,|B|)) 循环）

### 3.2 计数下界 (Counting Lower Bound)

**公式**：
```
lb = ⌈ C(n,j) × T / max_cov ⌉
```

其中：
```
max_cov = Σ C(k,t) × C(n-k, j-t)   (t 从 s 到 min(j,k))
```

**推导过程**：

1. **总覆盖需求**：C(n,j) 个 j-子集，每个需覆盖 T 次 → C(n,j) × T 次
2. **单组能力**：一个 k-group 最多覆盖 max_cov 个 j-subset
3. **下界**：至少需要 ceil(总需求 / 单组能力) 个 k-group

**示例计算** (n=7, k=6, j=5, s=5, T=1)：

```
C(7,5) = 21 个 j-subset
max_cov = C(6,5) = 6  (因为 s=5，只能有 5 个交集)
lb = ⌈21 × 1 / 6⌉ = 4

实际最优解 = 6（下界较宽松）
```

**为什么下界宽松？**

下界假设每个 k-group 都能达到理论最大覆盖能力 (max_cov)，但实际中：
- 不同 k-group 之间会有重叠覆盖
- 某些 j-subset 可能难以同时覆盖
- 组合约束限制了最优性

---

## 4. Core Functions (核心函数)

### 4.1 _popcount: 集合大小计算

**功能**：计算 uint32 整数中二进制 1 的个数（即集合大小）。

**算法**：Kernighan 算法
```python
def _popcount(x):
    c = 0
    while x:
        x = x & (x - 1)  # 消去最低位的 1
        c += 1
    return c
```

**时间复杂度**：O(1 的个数)，最多 7 次循环（n≤25，子集最多 25 个元素）。

**示例**：
```python
_popcount(0b1011) = 3  # 集合 {0,1,3}，大小为 3
```

### 4.2 _gains_parallel: 并行增益计算

**功能**：对每个候选 k-组，计算它能覆盖多少个"当前未覆盖"的 j-子集。

**输入**：
- `cand_masks`：候选 k-组的位掩码数组 [候选数]
- `uncov_j_masks`：未覆盖 j-子集的位掩码数组 [未覆盖数]
- `s`：覆盖所需的最小交集大小

**输出**：`gains[i]` = 第 i 个候选覆盖的未覆盖 j-子集数量

**核心逻辑**：
```python
for each candidate G:
    gain = 0
    for each uncovered j-subset J:
        if popcount(G & J) >= s:  # 覆盖条件
            gain += 1
    gains[i] = gain
```

**并行优化**：使用 `prange` 在多核上并行计算每个候选的增益。

**数学本质**：对每个候选 G，计算集合 {J : |G ∩ J| ≥ s} 的基数。

### 4.3 _update_cover: 覆盖计数更新

**功能**：将新选定的 k-组加入解后，更新所有 j-子集的覆盖计数。

**核心逻辑**：
```python
for each j-subset J:
    if popcount(best_mask & J) >= s:  # 被覆盖
        cover_count[J] += 1
```

**原地修改**：直接更新 `cover_count` 数组，避免额外空间。

### 4.4 _covered_indices: 覆盖下标提取

**功能**：返回被指定 k-组覆盖的所有 j-子集的下标。

**输入**：
- `g_mask`：一个 k-组的位掩码
- `j_masks`：全部 j-子集的位掩码数组
- `s`：最小交集阈值

**输出**：`int32` 数组，被 `g_mask` 覆盖的 j-子集下标列表

**示例**：
```python
covered = _covered_indices(G, j_masks, s=5)
# covered = [0, 3, 7, 15]  # G 覆盖了第 0, 3, 7, 15 个 j-subset
```

### 4.5 _filter_cover_all: 全覆盖筛选

**功能**：筛选能覆盖"全部"指定 j-子集的候选 k-组。

**应用场景**：局部搜索 swap 操作中的关键过滤。

**输入**：
- `cand_masks`：候选 k-组的位掩码列表
- `under_indices`：当前欠覆盖 j-子集的下标列表

**输出**：`bool` 数组，`True` 表示该候选覆盖了全部欠覆盖 j-子集

**核心逻辑**：
```python
for each candidate G:
    ok = True
    for each under-covered J in under_indices:
        if popcount(G & J) < s:  # 未能覆盖
            ok = False
    result[i] = ok
```

**数学含义**：对候选 G，检查 {J ∈ under_indices : |G ∩ J| ≥ s} 是否等于 under_indices。

---

## 5. Helper Functions (辅助函数)

### 5.1 _mask: 集合转位掩码

**功能**：将元素下标的有序元组转换为 uint32 位掩码。

**示例**：
```python
_mask((0, 2, 4)) == 0b10101 == 21
```

**代码**：
```python
def _mask(sub):
    m = 0
    for x in sub:
        m |= (1 << x)  # 第 x 位设为 1
    return m
```

### 5.2 preprocess: 预处理

**功能**：生成所有 C(n,j) 个 j-子集及其对应的 uint32 位掩码。

**执行时机**：在 `solve()` 开始时调用一次，后续整个算法都复用。

**输出**：
- `j_subsets`：j-子集的下标元组列表 [C(n,j)]
- `j_masks`：j-子集的位掩码数组 [C(n,j)]

**示例** (n=4, j=2)：
```python
j_subsets = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
j_masks   = [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100]
```

**时间复杂度**：O(C(n,j) × j)

### 5.3 candidates_for: 候选生成（剪枝）

**功能**：生成所有与指定 j-子集有至少 s 个公共元素的 k-子集。

**这是算法最关键的剪枝步骤**。贪心算法每步只需覆盖某个特定的 j-子集，因此只需考虑"能覆盖它"的候选，而不是全部 C(n,k) 个 k-组。

**生成方式**：枚举从 j_sub 中取 t 个元素、从其余 n-j 个元素中取 k-t 个元素的所有组合，t 从 s 到 min(j,k)。

**候选数量公式**：
```
Σ C(|j_sub|, t) × C(n-|j_sub|, k-t)   (t 从 s 到 min(j,k))
```

**典型值**（n=25, k=7）：
| 参数 | 全量 C(n,k) | 候选数 | 剪枝倍数 |
|------|-------------|--------|---------|
| j=7, s=5 | 480,700 | ~3,340 | **144 倍** |
| j=6, s=3 | 480,700 | ~93,100 | **5 倍** |

**为什么剪枝效果好？**

因为要求 k-group 与 j-subset 至少有 s 个公共元素，这大大缩小了候选空间。

**示例**：
```python
cands = candidates_for(J_target, n=25, k=7, s=5)
# cands 包含所有与 J_target 有 ≥5 个公共元素的 7-子集
```

### 5.4 counting_lower_bound: 下界计算

**功能**：用计数论证计算解大小的理论下界。

**公式**：
```
lb = ⌈ C(n,j) × T / max_cov ⌉

max_cov = Σ C(k,t) × C(n-k, j-t)   (t 从 s 到 min(j,k))
```

**推导**：
1. 总覆盖需求：C(n,j) × T 次
2. 单组最大覆盖：max_cov 个 j-subset
3. 最少组数：ceil(总需求 / 单组能力)

**注意**：此下界通常比真实最优解小 2~3 倍（比较宽松）。

**示例**：
```python
lb = counting_lower_bound(n=7, k=6, j=5, s=5, T=1)
# lb = 4 (实际最优解 = 6)
```

---

## 6. Greedy Construction Algorithm (贪心构造算法)

### 6.1 函数签名

```python
def greedy_once(n, k, j, s, T, j_subsets, N_j, j_masks, rng,
                t_deadline=float('inf'), 
                init_masks=None, init_tuples=None)
```

### 6.2 贪心策略（每步选最优 k-组）

**核心循环**：

```python
while True:
    # Step 1: 找出覆盖次数最少（最紧迫）的 j-子集
    uncov = {J : cover_count[J] < T}
    if uncov 为空：break  # 所有 j-subset 都已满足
    
    # Step 2: 随机选一个最紧迫的 j-subset J*（随机打破平局）
    J* = random.choice(uncov)
    
    # Step 3: 生成能覆盖 J* 的候选 k-groups
    cands = candidates_for(J*, n, k, s)
    
    # Step 4: 并行计算每个候选的"增益"（能覆盖几个未满足的 j-subset）
    for each G in cands:
        gains[G] = |{J ∈ uncov : |G ∩ J| ≥ s}|
    
    # Step 5: 选增益最大的候选，加入解
    G_best = argmax(gains)
    solution.append(G_best)
    
    # Step 6: 更新覆盖计数
    for each J covered by G_best:
        cover_count[J] += 1
```

### 6.3 算法特点

| 特点 | 说明 | 优势 |
|------|------|------|
| 贪心策略 | 每步解决"最紧迫"的约束 | 快速找到可行解 |
| 随机性 | 平局时随机选择 | 避免陷入局部最优 |
| 剪枝 | 只考虑能覆盖 J* 的候选 | 减少搜索空间 |
| 并行加速 | 增益计算用 numba prange | 速度提升 10~100 倍 |

### 6.4 LNS 热启动 (init_masks 参数)

**功能**：从已有的部分解开始，跳过这些组已覆盖的 j-subset，只需补全剩余未覆盖的部分。

**应用场景**：LNS 扰动重启时的初始解。

**示例**：
```python
# 保留 30% 的当前最优解
keep_idx = [0, 5, 10, 15, 20]  # 保留 5 个组
init_masks = [best_masks[i] for i in keep_idx]
init_tuples = [best_tuples[i] for i in keep_idx]

# 从部分解重启
m, t = greedy_once(..., init_masks=init_masks, init_tuples=init_tuples)
```

**数学本质**：从部分解 G_partial 开始，继续构造完整解。

### 6.5 截止时间处理 (t_deadline 参数)

**功能**：若超过截止时间，切换为"快速完成"模式：对剩余每个未覆盖 j-subset 直接选第一个有效候选，不再追求最优，确保返回的解是合法的完整解。

**触发条件**：`time.perf_counter() >= t_deadline`

**快速完成逻辑**：
```python
for each uncovered J:
    cands = candidates_for(J, n, k, s)
    for G in cands:
        if popcount(G & J) >= s:  # 找到第一个能覆盖 J 的候选
            solution.append(G)
            update_cover(G)
            break  # 不追求最优，快速完成
```

---

## 7. Minimization Algorithm (冗余压缩算法)

### 7.1 函数签名

```python
def minimize_solution(selected_masks, selected_tuples,
                      j_masks, N_j, s, T)
```

### 7.2 核心思想

**问题**：贪心算法生成的解可能有冗余组，能否删除一些组后仍然合法？

**冗余判定**：
```
Gᵢ 是冗余的 ⇔ 删除 Gᵢ 后，所有被它覆盖的 j-subset 的 cover_count 仍 > T
```

**直观理解**：如果删除 Gᵢ 后，所有 j-subset 仍然被覆盖 ≥ T 次，说明 Gᵢ 是多余的，可以安全删除。

### 7.3 算法流程

```python
# Step 1: 计算所有 k-组的覆盖缓存和总 cover_count
cover_count = 0
cov_cache = {}  # 缓存每个 k-group 覆盖的 j-subset 下标

for each G in solution:
    cov = _covered_indices(G, j_masks, s)  # G 覆盖的 j-subset 下标
    cov_cache[G] = cov
    for each J in cov:
        cover_count[J] += 1

# Step 2: 迭代删除冗余组
changed = True
while changed:
    changed = False
    for i from len(solution)-1 down to 0:  # 倒序遍历
        Gᵢ = solution[i]
        cov = cov_cache[Gᵢ]
        
        # 冗余判定：如果所有被 Gᵢ 覆盖的 j-subset 的 cover_count 都 > T
        if all(cover_count[J] > T for J in cov):
            # 安全删除
            for each J in cov:
                cover_count[J] -= 1
            solution.pop(i)  # 删除 Gᵢ
            changed = True  # 标记有变化，继续迭代
```

### 7.4 为什么要倒序遍历？

**原因**：`solution.pop(i)` 会改变数组长度，正序遍历会导致跳过元素。

**示例**：
```python
solution = [G₁, G₂, G₃, G₄]

# 正序遍历问题：
i=0: 删除 G₁ → solution = [G₂, G₃, G₄]
i=1: 当前是 G₂，删除 G₂ → solution = [G₃, G₄]
i=2: 当前是 G₄（原 i=3 的元素）→ 跳过了 G₃！

# 倒序遍历正确：
i=3: 检查 G₄
i=2: 检查 G₃
i=1: 检查 G₂
i=0: 检查 G₁
```

### 7.5 为什么要 while changed 循环？

**原因**：删除一个组可能使其他组变得不冗余（或更冗余），需要重新评估。

**示例**：
```python
初始解：[G₁, G₂, G₃, G₄, G₅] 共 5 个组

第一轮迭代：
  - G₅ 冗余 → 删除 → [G₁, G₂, G₃, G₄]
  - G₃ 不冗余（因为 G₅ 删除后，某些 J 的 cover_count 降至 T）

第二轮迭代：
  - 重新检查 G₁, G₂, G₃, G₄
  - G₄ 现在冗余 → 删除 → [G₁, G₂, G₃]

第三轮迭代：
  - 无冗余组可删除 → changed=False → 循环结束
```

### 7.6 时间复杂度

```
O(passes × m × avg_coverage)

其中：
  - passes = 迭代轮数（通常 2~5 轮）
  - m = 解的大小（初始组数）
  - avg_coverage = 平均每个 k-group 覆盖的 j-subset 数量
```

---

## 8. Local Search Optimization (局部搜索优化)

### 8.1 函数签名

```python
def local_search_swap(masks, tuples, j_subsets, j_masks,
                      N_j, n, k, j, s, T, rng, t_deadline=float('inf'))
```

### 8.2 核心思想

**目标**：在贪心构造完成后，进一步压缩解的大小。

**Swap 操作**：找到一对组合（移除 Gᵢ，替换为 G_new），使得解中另一个组 Gⱼ 变成冗余，从而净减少一个组。

**每次成功的 swap 使解大小 -1**。

### 8.3 详细流程

```python
improved = True
while improved and 时间 不足：
    improved = False
    
    # Step 1: 重建当前解的覆盖状态
    cover_count = 0
    cov_cache = {}
    for each G in solution:
        cov = _covered_indices(G, j_masks, s)
        cov_cache[G] = cov
        for each J in cov:
            cover_count[J] += 1
    
    # Step 2: 随机打乱组的顺序，逐个尝试
    order = shuffle([0, 1, ..., m-1])
    
    for i in order:
        Gᵢ = solution[i]
        cov_i = cov_cache[Gᵢ]
        
        # Step 3: 临时移除 Gᵢ，找现在"欠覆盖"的 j-subset 集合 U
        reduced = cover_count.copy()
        reduced[cov_i] -= 1  # 移除 Gᵢ 的影响
        U = {J : reduced[J] < T}  # 欠覆盖的 j-subset
        
        if U 为空：
            # Gᵢ 本身就是冗余的 → 直接删除
            solution.pop(i)
            improved = True
            break
        
        # Step 4: 生成以 U 中某个 j-subset 为种子的候选 k-groups
        seed_J = random.choice(U)
        cands = candidates_for(seed_J, n, k, s)
        
        # Step 5: 筛选能覆盖 U 中**全部** j-subset 的候选
        valid_cands = [G for G in cands if all(popcount(G & J) >= s for J in U)]
        
        # Step 6: 对每个有效候选 G_new，检查是否能触发 swap
        for G_new in valid_cands:
            # 计算：移除 Gᵢ + 添加 G_new 后的覆盖状态
            test_count = reduced.copy()
            test_count[cov_new] += 1
            
            # 检查是否有另一个组 Gⱼ 变得冗余
            for j_idx in range(len(solution)):
                if j_idx == i:
                    continue
                Jⱼ = solution[j_idx]
                cov_J = cov_cache[Jⱼ]
                if all(test_count[J] > T for J in cov_J):
                    # 执行 swap：用 G_new 替换 Gᵢ，删除 Gⱼ → 解大小 -1
                    solution[i] = G_new
                    solution.pop(j_idx)
                    improved = True
                    break
            
            if improved:
                break
        
        if improved:
            break  # 找到 swap 后，重新从头开始扫描
```

### 8.4 为什么 Swap 有效？

**直观理解**：

```
初始解：[G₁, G₂, G₃, G₄, G₅] 共 5 个组

Swap 操作：
  1. 移除 G₁ → 某些 J 变得欠覆盖（U = {J₃, J₇}）
  2. 找到 G_new = [1,3,5,7,9,11,13] 能覆盖 U 中全部 j-subset
  3. 检查：添加 G_new 后，G₄ 变得冗余（它覆盖的所有 J 仍有 cover_count > T）
  4. 执行 swap：[G_new, G₂, G₃, G₄, G₅] → [G_new, G₂, G₃, G₅] 净减少 1 组
```

**数学本质**：在解空间 G 的邻域中搜索更优解，通过"交换 + 删除"操作实现局部最优。

### 8.5 适用场景

- **时间预算有剩余** (> 5 秒) 的中等规模实例（如 n=15）
- **不适用**：用满时间的大规模实例（如 n=25，无剩余时间）

---

## 9. Verification Function (验证函数)

### 9.1 函数签名

```python
def verify(selected_masks, j_masks, N_j, s, T) -> bool
```

### 9.2 功能

验证一个解是否合法（所有 j-子集均被覆盖至少 T 次）。

### 9.3 核心逻辑

```python
cover_count = 0
for each G in solution:
    cov = _covered_indices(G, j_masks, s)
    cover_count[cov] += 1

return all(cover_count[J] >= T for J in range(N_j))
```

### 9.4 为什么独立验证？

**原因**：不依赖算法过程中的中间状态，是一个独立的、无偏的正确性检验。

**用途**：确认最终解的合法性（应始终为 True）。

---

## 10. Main Solver (主求解器)

### 10.1 函数签名

```python
def solve(n_samples: list, k: int, j: int, s: int,
          T: int = 1, time_limit: float = 30.0,
          seed: int = 42, verbose: bool = True) -> (result, info)
```

### 10.2 整体流程

```python
# Step 1: 预处理
n = len(n_samples)
j_subsets, j_masks = preprocess(n, j)  # 生成所有 j-subset
N_j = len(j_subsets)
lb = counting_lower_bound(n, k, j, s, T)  # 计算理论下界

# Step 2: 小规模 exact / 大规模增强型 heuristic
best_masks, best_tuples = None, None
restarts = 0
no_improve = 0

while time 未耗尽:
    restarts += 1
    
    # 决定：纯随机重启 or LNS 扰动？
    if 连续无改进 ≥ 5 次 and random() < 0.8:
        # LNS 扰动：保留 30% 的当前最优解，重启贪心
        n_keep = max(1, int(len(best) * 0.7))
        keep_idx = random.choice(len(best), n_keep)
        init_masks = [best[i] for i in keep_idx]
        init_tuples = [best[i] for i in keep_idx]
        m, t = greedy_once(..., init_masks=init_masks, init_tuples=init_tuples)
    else:
        # 纯随机重启
        m, t = greedy_once(...)
    
    # 冗余压缩
    m, t = minimize_solution(m, t, j_masks, N_j, s, T)
    
    # 更新最优解
    if best 为空 or len(m) < len(best):
        best = m, t
        no_improve = 0
    else:
        no_improve += 1
    
    # 收敛检测
    if len(best) <= lb:
        break  # 达到下界，停止
    if no_improve >= max(200, N_j // 100):
        break  # 无改进次数过多，停止

# Step 3: 局部搜索优化（使用剩余时间）
if 剩余时间 > 5 秒 and len(best) > lb:
    before_ls = len(best)
    best = local_search_swap(best, ..., t_deadline=剩余时间)
    输出：减少了 {before_ls - len(best)} 个组

# Step 4: 验证并返回
ok = verify(best, j_masks, N_j, s, T)
result = [original_sample_IDs for each G in best]
info = {solution_size, lower_bound, gap, restarts, time, valid}

return result, info
```

### 10.3 参数说明

| 参数 | 含义 | 默认值 | 说明 |
|------|------|--------|------|
| `n_samples` | 样本 ID 列表（长度 n） | 必需 | 支持任意可排序元素 |
| `k` | 每组大小 | 4 ≤ k ≤ 7 | 必需 |
| `j` | j-子集大小 | s ≤ j ≤ k | 必需 |
| `s` | 最小交集阈值 | 3 ≤ s ≤ 7 | 必需 |
| `T` | 覆盖次数要求 | 1 | 可选 |
| `time_limit` | 总时间预算（秒） | 30 | 可选 |
| `seed` | 随机种子 | 42 | 可选，固定后每次结果相同 |
| `verbose` | 是否打印进度 | True | 可选 |

### 10.4 返回结果

```python
result : list[tuple]，选出的 k-组列表，每个元素是原始样本 ID 的有序元组

info : dict，包含以下字段：
  - solution_size : 解的组数
  - lower_bound   : 计数下界（解大小 ≥ 此值）
  - gap           : solution_size - lower_bound（与下界的差距）
  - restarts      : 总重启次数
  - time          : 实际耗时（秒）
  - valid         : 解是否通过验证（bool，应始终为 True）
```

### 10.5 重启策略详解

| 阶段 | 策略 | 说明 |
|------|------|------|
| 前 5 次 | 纯随机重启 | 每次用不同的随机种子，探索不同区域 |
| 之后 | 80% LNS 扰动 + 20% 纯随机 | 保持多样性，避免陷入局部最优 |

**为什么前 5 次纯随机？**

因为还没有最优解可供 LNS 扰动，前 5 次重启的目的是快速找到一个可行解。

**为什么 80% LNS + 20% 随机？**

- LNS 扰动：利用已有解的信息，在邻域中搜索更优解
- 纯随机重启：保持多样性，避免陷入局部最优

---

## 11. Warmup Mechanism (预热机制)

### 11.1 为什么需要预热？

**问题**：Numba 第一次调用 `@njit` 函数时需要编译，耗时约 10~30 秒。

**解决方案**：在模块导入时预编译全部 numba JIT 函数，确保第一次 `solve()` 调用不会出现长时间延迟。

### 11.2 实现

```python
def _warmup():
    dummy_c = np.array([np.uint32(0b111)], dtype=np.uint32)
    dummy_j = np.array([np.uint32(0b110), np.uint32(0b101)], dtype=np.uint32)
    dummy_cc = np.zeros(2, dtype=np.int32)
    dummy_under = np.array([0, 1], dtype=np.int32)
    
    # 触发全部 @njit 函数的编译
    _ = _gains_parallel(dummy_c, dummy_j, np.int32(2))
    _update_cover(np.uint32(0b111), dummy_j, dummy_cc, np.int32(2))
    _ = _covered_indices(np.uint32(0b111), dummy_j, np.int32(2))
    _ = _filter_cover_all(dummy_c, dummy_j, dummy_under, np.int32(2))

if _HAS_NUMBA:
    _warmup()
```

### 11.3 缓存机制

**Numba 的 cache=True 选项**：
- 编译结果存入磁盘缓存（.pyc 文件）
- 下次导入直接加载，无需重新编译
- 启动几乎即时

---

## 12. Algorithm Complexity Analysis (算法复杂度分析)

### 12.1 预处理阶段

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 生成 j-子集 | O(C(n,j) × j) | O(C(n,j) × j) |
| 转换为位掩码 | O(C(n,j) × j) | O(C(n,j)) |
| 计算下界 | O(j) | O(1) |

### 12.2 贪心构造阶段

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 单次贪心 | O(m × C(n,k) 剪枝 × 并行) | m 为解的大小，剪枝后候选数约 3,000~93,000 |
| 并行加速 | O(m × C(n,k) 剪枝 / 核数) | 多核并行 |

### 12.3 冗余压缩阶段

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| minimize_solution | O(passes × m × avg_coverage) | passes 通常 2~5 轮 |

### 12.4 局部搜索阶段

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| local_search_swap | O(passes × m² × avg_coverage) | 每轮扫描所有组，检查 swap 可能性 |

### 12.5 总体复杂度

```
总时间复杂度 = O(预处理 + restarts × (贪心 + 冗余压缩 + 局部搜索))

典型运行时间：
  - n=10: < 1 秒
  - n=15: 10~60 秒
  - n=25: 1200 秒（heuristic 时间上限）
```

---

## 13. Practical Examples (实用示例)

### 13.1 示例 1：小规模问题

```python
from algorithm import solve

# n=7, k=6, j=5, s=5, T=1
# 已知最优解 = 6 个组

samples = list(range(1, 8))  # [1, 2, 3, 4, 5, 6, 7]
result, info = solve(samples, k=6, j=5, s=5, T=1, time_limit=60, seed=42)

print(f"Solution size: {info['solution_size']}")  # 应该等于 6
print(f"Groups: {result}")
```

### 13.2 示例 2：多覆盖 (T=2)

```python
# n=7, k=6, j=5, s=5, T=2
# 每个 j-subset 需要被覆盖 2 次

samples = list(range(1, 8))
result, info = solve(samples, k=6, j=5, s=5, T=2, time_limit=120, seed=42)

print(f"Solution size: {info['solution_size']}")  # 应该 > 6（因为 T=2）
print(f"Gap: {info['gap']}")
```

### 13.3 示例 3：自定义样本 ID

```python
# 样本不是连续的整数，可以是任意可排序对象

samples = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
result, info = solve(samples, k=6, j=5, s=5, T=1, time_limit=60, seed=42)

print(f"Result: {result}")
# 输出：[('A', 'B', 'C', 'D', 'E', 'F'), ('A', 'B', 'C', 'D', 'E', 'G'), ...]
```

### 13.4 示例 4：时间控制

```python
# 限制运行时间为 10 秒

samples = list(range(1, 26))  # n=25
result, info = solve(samples, k=7, j=6, s=5, T=1, time_limit=10, seed=42)

print(f"Time: {info['time']}s")  # 应该接近 10 秒
print(f"Solution size: {info['solution_size']}")
```

---

## 14. Key Insights (关键洞察)

### 14.1 为什么贪心有效？

**贪心策略**：每步选择"增益最大"的 k-group，即能覆盖最多未满足 j-subset 的组。

**数学保证**：贪心算法是集合覆盖问题的**对数近似算法**，解的质量为 O(ln N) 倍最优解。

**实际效果**：通过剪枝和并行加速，贪心算法能在合理时间内找到接近最优的解。

### 14.2 为什么 LNS 有效？

**LNS 扰动**：保留 30% 的当前最优解，重新构造剩余部分。

**优势**：
- 利用已有解的信息（exploitation）
- 探索新的解空间区域（exploration）
- 平衡探索与利用，避免陷入局部最优

### 14.3 为什么 Swap 操作有效？

**Swap 操作**：移除 Gᵢ，添加 G_new，同时删除冗余的 Gⱼ。

**数学本质**：在解空间 G 的邻域中搜索更优解，通过"交换 + 删除"操作实现局部最优。

**适用场景**：时间预算有剩余的中等规模实例。

### 14.4 为什么需要冗余压缩？

**问题**：贪心算法倾向于生成冗余覆盖，因为每步只考虑当前最优，不考虑全局最优。

**解决方案**：贪心构造后，删除所有可以安全删除的组。

**效果**：通常能减少 10~30% 的组数。

---

## 15. Summary (总结)

### 15.1 算法核心思想

1. **位掩码编码**：用 uint32 表示集合，快速计算交集大小
2. **贪心构造**：每步选择"增益最大"的 k-group，快速找到可行解
3. **剪枝优化**：只考虑能覆盖当前 j-subset 的候选，减少搜索空间
4. **并行加速**：用 Numba JIT 编译为机器码，多核并行
5. **冗余压缩**：删除所有可以安全删除的组，压缩解的大小
6. **LNS 扰动**：保留部分最优解，重启贪心，平衡探索与利用
7. **局部搜索**：Swap 操作进一步压缩解的大小

### 15.2 适用场景

| 场景 | 适用性 | 说明 |
|------|--------|------|
| 小样本 (n≤10) | ⭐⭐⭐⭐⭐ | 快速找到最优解 |
| 中等样本 (10<n≤15) | ⭐⭐⭐⭐⭐ | 10~60 秒内找到高质量解 |
| 大样本 (n>15) | ⭐⭐⭐⭐ | 1200 秒内找到更高质量的近似最优解 |
| 多覆盖 (T>1) | ⭐⭐⭐⭐ | 解的大小随 T 线性增长 |

### 15.3 性能对比

| 算法 | 时间复杂度 | 解质量 | 适用规模 |
|------|-----------|--------|---------|
| 穷举搜索 | O(C(n,k)×m) | 最优 | n≤8 |
| 贪心算法 | O(m×C(n,k) 剪枝) | 近似 | n≤25 |
| 模拟退火 | O(迭代次数×评估) | 较好 | n≤20 |
| 遗传算法 | O(种群大小×迭代次数) | 较好 | n≤30 |
| **本算法** | **O(restarts×(贪心 + 冗余 + 局部搜索))** | **高质量** | **n≤25** |

---

## 16. References (参考)

1. **Covering Design**: 组合数学中的经典问题，研究如何用最少的 k-subset 覆盖所有 j-subset。
2. **Greedy Set Cover**: 贪心算法是集合覆盖问题的标准近似算法，近似比为 O(ln N)。
3. **Numba JIT**: Python 的即时编译库，将 Python 代码编译为机器码，速度提升 10~100 倍。
4. **Large Neighborhood Search (LNS)**：组合优化中的启发式算法，通过扰动和重构搜索解空间。
5. **Kernighan's Algorithm**: 计算整数中二进制 1 的个数的经典算法，时间复杂度 O(1 的个数)。
