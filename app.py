"""
app.py — GUI for the Optimal Sample Selection System
=====================================================

两个界面：
  S1  主计算界面：填写参数 → Execute 计算 → Store 保存 / Print 导出
  S2  记录浏览界面：查看已保存的计算结果 → Display / Delete

存储：所有结果保存在同目录的 results.json 文件中。

技术说明：
  - GUI 框架：tkinter（Python 内置，无需安装）
  - 计算线程：solve() 在后台线程运行，避免界面卡死
  - 线程通信：后台线程通过 queue.Queue 向主线程传递进度和结果，
              主线程通过 after(100ms) 轮询队列更新界面
"""

import contextlib
import json
import os
import queue
import random
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog

from algorithm import DEFAULT_TIME_LIMIT, solve

# ── file-based storage ────────────────────────────────────────────────────────

_RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")


def _load() -> list:
    """
    从 results.json 读取所有已保存记录，返回记录列表。

    若文件不存在则返回空列表（首次运行时的正常情况）。
    若文件存在但内容损坏（非法 JSON），同样返回空列表，避免程序崩溃。

    返回
    ----
    list，每个元素是一个 dict，包含 id / params / samples / info / groups 字段
    """
    if not os.path.exists(_RESULTS_FILE):
        return []
    with open(_RESULTS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save(records: list) -> None:
    """
    将记录列表完整写回 results.json（覆盖写入）。

    每次 Store 或 Delete 操作都调用此函数：先读出全部记录，
    修改列表后再整体写回，保持文件格式为格式化的 JSON（indent=2）。

    参数
    ----
    records : list，要保存的全部记录（替换文件中原有内容）
    """
    with open(_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _next_run(m, n, k, j, s) -> int:
    """
    计算指定参数组合的下一个运行编号（1-based）。

    扫描已有记录，统计与当前参数 (m,n,k,j,s) 完全相同的记录数量，
    加 1 后作为新记录的运行编号 x（体现在文件名 m-n-k-j-s-x-y 的 x 部分）。

    参数
    ----
    m, n, k, j, s : 整数，当前计算的五个主要参数

    返回
    ----
    int，下一个运行编号（如首次运行返回 1，第二次返回 2）
    """
    return sum(
        1 for r in _load()
        if (r["params"]["m"] == m and r["params"]["n"] == n
            and r["params"]["k"] == k and r["params"]["j"] == j
            and r["params"]["s"] == s)
    ) + 1


# ── stdout redirector (used during background solve) ─────────────────────────

class _Writer:
    """
    将 print() 输出重定向到线程安全队列的伪文件对象。

    solve() 函数通过 print() 输出进度信息（如 "restart 5: 25 groups"）。
    在后台线程中，我们用 contextlib.redirect_stdout(_Writer(queue)) 将
    这些 print() 调用的输出捕获并放入队列，由主线程取出后显示在 GUI 文本框里。

    这样做是必要的，因为 tkinter 不允许从非主线程直接修改界面控件。
    """

    def __init__(self, q: queue.Queue):
        """
        参数
        ----
        q : queue.Queue，消息队列，write() 会向其中放入 ("log", 文本) 元组
        """
        self._q = q

    def write(self, s: str):
        """
        接收 print() 写入的字符串，封装为消息放入队列。

        空字符串（如 print() 产生的换行符）被过滤掉，避免无意义的队列操作。
        """
        if s:
            self._q.put(("log", s))

    def flush(self):
        """
        满足文件接口要求的空方法（队列写入是即时的，无需刷新缓冲）。
        """
        pass


# ── shared text-file export ───────────────────────────────────────────────────

def _write_txt(path: str, groups, info: dict, p: dict, samples: list) -> None:
    """
    将一次计算结果导出为易读的纯文本文件。

    文件内容格式：
      第 1 行：标题
      第 2 行：参数（m, n, k, j, s, T）
      第 3 行：使用的样本 ID 列表
      第 4 行：统计信息（组数、下界、gap、是否合法、耗时）
      分隔线
      之后每行：一个 k-group（样本 ID 列表格式）

    参数
    ----
    path    : str，导出文件的完整路径
    groups  : list[tuple 或 list]，k-group 列表（原始样本 ID）
    info    : dict，solve() 返回的统计信息字典
    p       : dict，参数字典（含 m, n, k, j, s, T）
    samples : list，本次使用的 n 个样本 ID
    """
    with open(path, "w", encoding="utf-8") as f:
        method = info.get("method", "heuristic")
        optimal = info.get("optimal", False)
        f.write("Optimal Sample Selection Result\n")
        f.write(f"Parameters: m={p['m']}  n={p['n']}  k={p['k']}  "
                f"j={p['j']}  s={p['s']}  T={p['T']}\n")
        f.write(f"Samples:    {samples}\n")
        f.write(f"Groups: {info['solution_size']}  "
                f"lb={info['lower_bound']}  gap={info['gap']}  "
                f"valid={info['valid']}  method={method}  "
                f"optimal={optimal}  time={info['time']}s\n")
        f.write("-" * 44 + "\n")
        for g in groups:
            f.write(f"{list(g)}\n")


# ── Screen 1 — main computation view ─────────────────────────────────────────

class Screen1(tk.Frame):
    """
    S1 主计算界面。

    界面布局（从上到下）：
      标题 → 参数输入区（6 个参数，2 列网格） → 样本选择模式 →
      操作按钮（Execute / Store / Clear） → 输出日志区 →
      底部按钮（Print / DB）

    状态管理：
      _result    : 最近一次成功计算的结果，None 表示尚未计算
      _computing : 是否正在后台计算（防止重复点击 Execute）
      _q         : 后台线程与主线程通信的队列
    """

    _DEFAULTS = dict(m="45", n="9", k="6", j="5", s="5", T="1")
    """六个参数的默认显示值（启动时预填入输入框）。"""

    def __init__(self, master):
        """
        初始化 Screen1，创建所有实例变量并构建 UI。

        参数
        ----
        master : App 实例（主窗口），用于调用 show_screen2() 跳转界面
        """
        super().__init__(master)
        self._result = None       # (groups, info, params, samples) after solve
        self._computing = False
        self._q = queue.Queue()
        self._build()
        self._poll()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self):
        """
        构建 S1 的全部 UI 控件并布局。

        使用 tkinter pack 布局管理器（从上到下堆叠）。
        参数输入区内部使用 grid 布局（2×3 表格，标签+输入框交替排列）。
        所有控件引用保存为实例变量（_vars, _exec_btn, _store_btn, _log 等），
        以便后续在事件处理方法中访问和修改。
        """
        # title
        tk.Label(self,
                 text="An Optimal Samples Selection System",
                 font=("Helvetica", 16, "bold")).pack(pady=(12, 4))

        # parameters
        pf = ttk.LabelFrame(self, text="Parameters")
        pf.pack(fill=tk.X, padx=14, pady=4)

        fields = [
            ("m  (45–54):", "m"), ("k  (4–7):",  "k"),
            ("n  (7–25):",  "n"), ("j  (s–k):",  "j"),
            ("T  (≥ 1):",   "T"), ("s  (3–7):",  "s"),
        ]
        self._vars = {}
        for idx, (label, key) in enumerate(fields):
            row, col = divmod(idx, 2)
            tk.Label(pf, text=label, width=11, anchor="e").grid(
                row=row, column=col * 2, padx=8, pady=4, sticky="e")
            v = tk.StringVar(value=self._DEFAULTS[key])
            tk.Entry(pf, textvariable=v, width=8).grid(
                row=row, column=col * 2 + 1, padx=6, pady=4, sticky="w")
            self._vars[key] = v

        # selection mode
        mf = ttk.LabelFrame(self, text="Sample Selection")
        mf.pack(fill=tk.X, padx=14, pady=4)

        self._mode = tk.StringVar(value="random")
        rb_row = tk.Frame(mf)
        rb_row.pack(fill=tk.X, padx=6, pady=4)
        tk.Radiobutton(rb_row, text="Random n from m",
                       variable=self._mode, value="random",
                       command=self._on_mode).pack(side=tk.LEFT, padx=8)
        tk.Radiobutton(rb_row, text="Input n samples manually",
                       variable=self._mode, value="manual",
                       command=self._on_mode).pack(side=tk.LEFT, padx=8)

        # manual entry row (hidden initially, shown inside mf when "manual" selected)
        self._manual_row = tk.Frame(mf)
        tk.Label(self._manual_row,
                 text="Sample IDs (comma-separated):").pack(side=tk.LEFT, padx=6)
        self._manual_var = tk.StringVar()
        tk.Entry(self._manual_row, textvariable=self._manual_var,
                 width=46).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        # action buttons
        bf = tk.Frame(self)
        bf.pack(pady=6)
        self._exec_btn = ttk.Button(bf, text="Execute",
                                    command=self._execute, width=10)
        self._exec_btn.pack(side=tk.LEFT, padx=6)
        self._store_btn = ttk.Button(bf, text="Store",
                                     command=self._store, width=10,
                                     state=tk.DISABLED)
        self._store_btn.pack(side=tk.LEFT, padx=6)
        ttk.Button(bf, text="Clear",
                   command=self._clear, width=10).pack(side=tk.LEFT, padx=6)

        # output log
        ttk.Label(self, text="Output:").pack(anchor="w", padx=14)
        self._log = scrolledtext.ScrolledText(
            self, height=14, state=tk.DISABLED,
            font=("Courier", 10), wrap=tk.WORD)
        self._log.pack(fill=tk.BOTH, expand=True, padx=14, pady=(2, 4))

        # bottom bar
        bb = tk.Frame(self)
        bb.pack(fill=tk.X, padx=14, pady=(0, 10))
        ttk.Button(bb, text="Print",
                   command=self._print, width=10).pack(side=tk.LEFT)
        ttk.Button(bb, text="DB",
                   command=lambda: self.master.show_screen2(),
                   width=10).pack(side=tk.RIGHT)

    # ── mode toggle ───────────────────────────────────────────────────────────

    def _on_mode(self):
        """
        切换样本输入模式时显示或隐藏手动输入行。

        当用户选择 "Input n samples manually" 单选按钮时，
        在 Sample Selection 区域内展开手动输入框；
        切换回 "Random n from m" 时收起输入框。

        使用 pack() / pack_forget() 动态控制控件的可见性，
        不删除控件，因此输入内容在切换时被保留。
        """
        if self._mode.get() == "manual":
            self._manual_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        else:
            self._manual_row.pack_forget()

    # ── parameter parsing & validation ───────────────────────────────────────

    def _parse_params(self) -> dict:
        """
        从 UI 输入框读取并验证六个参数，返回整数参数字典。

        验证规则：
          m : 45 ≤ m ≤ 54
          n : 7 ≤ n ≤ 25，且 n ≤ m
          k : 4 ≤ k ≤ 7
          j : s ≤ j ≤ k
          s : 3 ≤ s ≤ 7
          T : T ≥ 1

        返回
        ----
        dict，键为参数名（字符串），值为整数

        异常
        ----
        ValueError，若任意参数不合法，消息描述具体原因（由调用方显示弹窗）
        """
        try:
            p = {k: int(v.get().strip()) for k, v in self._vars.items()}
        except ValueError:
            raise ValueError("All parameters must be integers.")
        if not 45 <= p["m"] <= 54:
            raise ValueError("m must be in 45–54.")
        if not 7 <= p["n"] <= 25:
            raise ValueError("n must be in 7–25.")
        if not 4 <= p["k"] <= 7:
            raise ValueError("k must be in 4–7.")
        if not 3 <= p["s"] <= 7:
            raise ValueError("s must be in 3–7.")
        if not p["s"] <= p["j"] <= p["k"]:
            raise ValueError("j must satisfy s ≤ j ≤ k.")
        if p["T"] < 1:
            raise ValueError("T must be ≥ 1.")
        if p["n"] > p["m"]:
            raise ValueError("n cannot exceed m.")
        return p

    def _parse_samples(self, p: dict) -> list:
        """
        根据当前模式生成 n 个样本 ID 列表（1-based，范围 1..m）。

        Random 模式：从 range(1, m+1) 中无放回随机抽取 n 个，排序后返回。
        Manual 模式：解析用户输入的逗号分隔整数，验证数量、唯一性和范围。

        参数
        ----
        p : dict，由 _parse_params() 返回的参数字典

        返回
        ----
        list[int]，长度为 n 的已排序样本 ID 列表

        异常
        ----
        ValueError，若手动输入不合法（数量错误、重复、越界）
        """
        m, n = p["m"], p["n"]
        if self._mode.get() == "random":
            return sorted(random.sample(range(1, m + 1), n))
        raw = self._manual_var.get().strip()
        try:
            ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            raise ValueError("Sample IDs must be comma-separated integers.")
        if len(ids) != n:
            raise ValueError(
                f"Expected {n} sample IDs, got {len(ids)}.")
        if len(set(ids)) != n:
            raise ValueError("Sample IDs must be distinct.")
        if any(x < 1 or x > m for x in ids):
            raise ValueError(f"Sample IDs must be in the range 1–{m}.")
        return sorted(ids)

    # ── execute ───────────────────────────────────────────────────────────────

    def _execute(self):
        """
        点击 Execute 按钮的响应函数：验证输入后在后台线程启动 solve()。

        执行流程：
        1. 若已在计算中则直接返回（防止重复点击）。
        2. 调用 _parse_params() 和 _parse_samples() 验证输入；失败则弹窗提示。
        3. 清空日志、禁用 Execute 和 Store 按钮、标记计算状态。
        4. 启动后台 daemon 线程运行 worker()。

        worker() 内部：
        - 用 contextlib.redirect_stdout 把 solve() 的 print() 输出重定向到队列。
        - 计算完成后向队列放入 ("done", 结果元组)；
          发生异常则放入 ("err", 错误信息字符串)。
        """
        if self._computing:
            return
        try:
            p = self._parse_params()
            samples = self._parse_samples(p)
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return

        self._clear_log()
        self._result = None
        self._store_btn.config(state=tk.DISABLED)
        self._computing = True
        self._exec_btn.config(state=tk.DISABLED, text="Running…")
        self._log_write(f"Samples: {samples}\n\n")

        def worker():
            try:
                with contextlib.redirect_stdout(_Writer(self._q)):
                    groups, info = solve(
                        samples, p["k"], p["j"], p["s"],
                        T=p["T"], time_limit=DEFAULT_TIME_LIMIT, verbose=True)
                self._q.put(("done", (groups, info, p, samples)))
            except Exception as exc:
                self._q.put(("err", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    # ── queue polling (runs on main thread every 100 ms) ─────────────────────

    def _poll(self):
        """
        每 100 毫秒轮询一次后台线程队列，将消息更新到 GUI。

        消息类型：
          ("log",  str)              : 进度文本，追加到日志框
          ("done", (groups,info,...)) : 计算完成，显示结果、恢复按钮状态
          ("err",  str)              : 计算出错，弹出错误对话框

        此函数通过 self.after(100, self._poll) 实现循环调用（不阻塞主线程）。
        这是 tkinter 与后台线程通信的标准模式：后台线程只写队列，
        主线程只读队列，避免多线程直接操作控件引发的竞态条件。
        """
        try:
            while True:
                tag, data = self._q.get_nowait()
                if tag == "log":
                    self._log_write(data)
                elif tag == "done":
                    groups, info, p, samples = data
                    self._result = (groups, info, p, samples)
                    self._computing = False
                    self._exec_btn.config(state=tk.NORMAL, text="Execute")
                    self._store_btn.config(state=tk.NORMAL)
                    self._log_write(
                        f"\n── {info['solution_size']} groups ──\n")
                    for g in groups:
                        self._log_write(f"  {list(g)}\n")
                elif tag == "err":
                    messagebox.showerror("Compute Error", data)
                    self._computing = False
                    self._exec_btn.config(state=tk.NORMAL, text="Execute")
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # ── store ─────────────────────────────────────────────────────────────────

    def _store(self):
        """
        点击 Store 按钮：将当前计算结果追加保存到 results.json。

        命名规则：m-n-k-j-s-x-y
          x = 该参数组合的第几次运行（调用 _next_run() 确定）
          y = 本次解的 k-group 数量

        保存内容：参数、样本列表、统计信息、全部 k-group。
        保存后弹出确认对话框显示记录 ID。
        """
        if self._result is None:
            messagebox.showwarning("Nothing to Store", "Run Execute first.")
            return
        groups, info, p, samples = self._result
        x = _next_run(p["m"], p["n"], p["k"], p["j"], p["s"])
        rec_id = (f"{p['m']}-{p['n']}-{p['k']}-{p['j']}-"
                  f"{p['s']}-{x}-{info['solution_size']}")
        record = {
            "id":      rec_id,
            "params":  p,
            "samples": samples,
            "info":    info,
            "groups":  [list(g) for g in groups],
        }
        recs = _load()
        recs.append(record)
        _save(recs)
        messagebox.showinfo("Stored", f"Saved as  {rec_id}")

    # ── print to file ─────────────────────────────────────────────────────────

    def _print(self):
        """
        点击 Print 按钮：弹出"另存为"对话框，将当前结果导出为 .txt 文件。

        使用 filedialog.asksaveasfilename() 让用户选择保存路径和文件名，
        默认文件名格式为 result_m-n-k-j-s.txt。
        实际写入由 _write_txt() 完成。
        """
        if self._result is None:
            messagebox.showwarning("Nothing to Print", "Run Execute first.")
            return
        groups, info, p, samples = self._result
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=(f"result_{p['m']}-{p['n']}-"
                         f"{p['k']}-{p['j']}-{p['s']}.txt"))
        if not path:
            return
        _write_txt(path, groups, info, p, samples)
        messagebox.showinfo("Saved", f"Saved to:\n{path}")

    # ── clear ─────────────────────────────────────────────────────────────────

    def _clear(self):
        """
        点击 Clear 按钮：清空日志、重置所有参数为默认值、清除手动输入。

        _result 置为 None，Store 按钮重新禁用。
        不切换当前的样本选择模式（Random/Manual 保持不变）。
        """
        self._clear_log()
        self._result = None
        self._store_btn.config(state=tk.DISABLED)
        for k, v in self._DEFAULTS.items():
            self._vars[k].set(v)
        self._manual_var.set("")

    # ── log helpers ───────────────────────────────────────────────────────────

    def _log_write(self, text: str):
        """
        向只读日志文本框追加一段文本，并自动滚动到底部。

        tkinter 的 ScrolledText 默认只读（state=DISABLED），
        写入时需临时开启（NORMAL），写完再关回（DISABLED），
        防止用户手动编辑日志内容。
        """
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, text)
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    def _clear_log(self):
        """
        清空日志文本框的全部内容。

        "1.0" 是 tkinter Text 控件的起始位置标记（第 1 行第 0 个字符）。
        """
        self._log.config(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)
        self._log.config(state=tk.DISABLED)


# ── Screen 2 — saved results browser ─────────────────────────────────────────

class Screen2(tk.Frame):
    """
    S2 数据库浏览界面。

    界面布局：
      标题 → 记录列表（可滚动 Listbox）→ 操作按钮（Display / Delete / Back）

    记录以 "m-n-k-j-s-x-y" 格式显示在列表中。
    Display 操作会弹出独立的 Toplevel 窗口展示详细内容，不替换主界面。
    """

    def __init__(self, master):
        """
        初始化 Screen2，构建 UI。

        参数
        ----
        master : App 实例
        """
        super().__init__(master)
        self._build()

    def _build(self):
        """
        构建 S2 的全部 UI 控件：标题、带滚动条的 Listbox、三个操作按钮。
        """
        tk.Label(self, text="Database Resources",
                 font=("Helvetica", 14, "bold")).pack(pady=(12, 6))

        # scrollable record list
        lf = tk.Frame(self)
        lf.pack(fill=tk.BOTH, expand=True, padx=14)
        sb = tk.Scrollbar(lf)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._lb = tk.Listbox(lf, yscrollcommand=sb.set,
                               font=("Courier", 11), height=20,
                               selectmode=tk.SINGLE)
        self._lb.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self._lb.yview)

        # buttons
        bf = tk.Frame(self)
        bf.pack(pady=10)
        ttk.Button(bf, text="Display",
                   command=self._display, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Button(bf, text="Delete",
                   command=self._delete, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Button(bf, text="Back",
                   command=lambda: self.master.show_screen1(),
                   width=10).pack(side=tk.LEFT, padx=6)

    def refresh(self):
        """
        重新从 results.json 加载并刷新列表显示。

        每次从 S1 切换到 S2 时（show_screen2()）都调用此方法，
        确保显示的是最新的保存记录（包括刚刚 Store 的新记录）。
        """
        self._lb.delete(0, tk.END)
        for r in _load():
            self._lb.insert(tk.END, r["id"])

    def _selected(self) -> dict | None:
        """
        获取列表中当前选中的记录。

        若无选中项，弹出提示框并返回 None。
        通过列表下标从 _load() 返回的列表中取对应记录，
        确保与显示顺序一致。

        返回
        ----
        dict 或 None，选中的记录字典或 None（无选中）
        """
        sel = self._lb.curselection()
        if not sel:
            messagebox.showwarning("No Selection", "Please select a record.")
            return None
        recs = _load()
        idx = sel[0]
        return recs[idx] if idx < len(recs) else None

    def _display(self):
        """
        点击 Display 按钮：在独立弹窗中展示选中记录的详细内容。

        弹窗（Toplevel）包含：
          - 顶部：参数信息（m, n, k, j, s, T）
          - 中部：统计信息（组数、下界、gap、耗时、是否合法）
          - 主体：可滚动文本区域，每行显示一个 k-group
          - 底部：Print（导出）和 Close（关闭）按钮

        使用 Toplevel 而非替换主界面，允许同时查看多条记录。
        """
        rec = self._selected()
        if rec is None:
            return
        p, info = rec["params"], rec["info"]

        win = tk.Toplevel(self)
        win.title(rec["id"])
        win.geometry("520x540")

        tk.Label(win,
                 text=(f"m={p['m']}  n={p['n']}  k={p['k']}  "
                       f"j={p['j']}  s={p['s']}  T={p['T']}"),
                 font=("Helvetica", 11, "bold")).pack(pady=(10, 2))
        tk.Label(win,
                 text=(f"{info['solution_size']} groups  |  "
                       f"lb={info['lower_bound']}  gap={info['gap']}  "
                       f"time={info['time']}s  valid={info['valid']}  "
                       f"method={info.get('method', 'heuristic')}  "
                       f"optimal={info.get('optimal', False)}"),
                 font=("Helvetica", 10)).pack(pady=(0, 6))

        txt = scrolledtext.ScrolledText(win, font=("Courier", 10))
        txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        for g in rec["groups"]:
            txt.insert(tk.END, f"{g}\n")
        txt.config(state=tk.DISABLED)

        bbf = tk.Frame(win)
        bbf.pack(pady=6)
        ttk.Button(bbf, text="Print",
                   command=lambda: self._print_record(rec),
                   width=10).pack(side=tk.LEFT, padx=6)
        ttk.Button(bbf, text="Close",
                   command=win.destroy,
                   width=10).pack(side=tk.LEFT, padx=6)

    def _print_record(self, rec: dict):
        """
        将指定记录导出为 .txt 文件（从 Display 弹窗的 Print 按钮调用）。

        默认文件名为记录的 ID（如 45-9-6-5-5-1-14.txt）。
        实际写入由 _write_txt() 完成。

        参数
        ----
        rec : dict，要导出的记录字典（含 params, info, groups, samples）
        """
        p, info = rec["params"], rec["info"]
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"{rec['id']}.txt")
        if not path:
            return
        _write_txt(path, rec["groups"], info, p, rec.get("samples", []))
        messagebox.showinfo("Saved", f"Saved to:\n{path}")

    def _delete(self):
        """
        点击 Delete 按钮：删除选中记录并刷新列表。

        删除前弹出确认对话框，防止误操作。
        实现方式：过滤掉目标 id 的记录后整体写回 results.json。
        """
        rec = self._selected()
        if rec is None:
            return
        if not messagebox.askyesno("Delete", f"Delete  {rec['id']} ?"):
            return
        recs = [r for r in _load() if r["id"] != rec["id"]]
        _save(recs)
        self.refresh()


# ── application root ──────────────────────────────────────────────────────────

class App(tk.Tk):
    """
    应用程序主窗口（继承自 tk.Tk）。

    管理两个界面（Screen1 和 Screen2）的切换：
    任意时刻只有一个界面可见，通过 pack() / pack_forget() 实现切换。
    Screen1 和 Screen2 在初始化时都创建，切换时只控制显示状态，
    不销毁和重建控件，避免状态丢失。
    """

    def __init__(self):
        """
        初始化主窗口：设置标题、尺寸，创建两个子界面，显示 S1。
        """
        super().__init__()
        self.title("An Optimal Samples Selection System")
        self.geometry("820x700")
        self.resizable(True, True)
        self.s1 = Screen1(self)
        self.s2 = Screen2(self)
        self.show_screen1()

    def show_screen1(self):
        """
        切换到 S1（主计算界面）。

        隐藏 S2（pack_forget），显示 S1（pack）。
        fill=BOTH + expand=True 使 S1 充满整个窗口。
        """
        self.s2.pack_forget()
        self.s1.pack(fill=tk.BOTH, expand=True)

    def show_screen2(self):
        """
        切换到 S2（数据库浏览界面），并刷新记录列表。

        隐藏 S1，显示 S2，同时调用 s2.refresh() 确保列表是最新的。
        """
        self.s1.pack_forget()
        self.s2.pack(fill=tk.BOTH, expand=True)
        self.s2.refresh()


if __name__ == "__main__":
    App().mainloop()
