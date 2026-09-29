# `factoropt` 目录约定（Task B）

这里是 Task B 的业务运行库：

- `task_b.py`：路由、episode、特征构造、预测打分为组合结算。
- `semantic.py`：track 规范化与 factor card 校验。
- `task_b_frozen.py`：frozen-code 协议本体——窗口构造、源码冻结、rolling 重训与审计。
- `agentic/`：Task B 的 frozen adapter，以及 `loop`/`tools`/`adapters` 三个
  re-export shim，指向仓库顶层的 `shared/agentic`。
- `__init__.py`：只导出当前主轨用到的 Task B API。

provider-neutral 的 Chat Completions 循环、受限工具执行器和 `TaskAdapter` 协议在
`shared/agentic/`，task_b 与 task_c 都按包名 import，不再按文件路径加载对方的
`factoropt`。Task B 不依赖 `task_a/`。

`task_b_frozen.py` 会在运行时按路径加载 Task C 的 `task_c_code_runner.py` 和
`task_c.py`，复用同一套受限执行器与公开字段校验，这是有意保留的唯一跨任务依赖。
