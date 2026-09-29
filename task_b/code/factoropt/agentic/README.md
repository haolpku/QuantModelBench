# `factoropt.agentic`（Task B）

Task B 在共享工具循环之上的适配层。provider-neutral 的实现已经提到仓库顶层的
`shared/agentic/`，这里只留三样东西：

| 文件 | 作用 |
|---|---|
| `frozen_adapter.py` | `TaskBFrozenCodeAdapter`：frozen-code 契约的 `prompt` / `prompt_version` / `schema_path` / `validate` / `score`。它读取 `task_b/prompt/prompt_b_frozen_code.md`，并在 `agentic_calls=0` 时直接回放 payload 里已冻结的源码，不经过任何工具调用。 |
| `loop.py`、`tools.py`、`adapters.py` | re-export shim，转出 `shared/agentic/` 的 `AgenticConfig`、`run_agentic_case`、`ToolDispatcher`、`TaskAdapter`、`resolved_task_variant` 等。保留它们是为了让旧 import 路径和 bundle manifest 不用改。 |

`prompt_b_frozen_code.md` 的正文 hash 会写进 revision audit，所以改 prompt 等于改实验身份。

## 共享实现

`shared/agentic/` 提供与任务无关的 Chat Completions 循环：请求重试、payload 摘要、
工具 schema、子进程中的 `python_exec`、artifact 落盘和逐次调用审计。它不认识任何任务名、
validator 或业务规则；任务差异化全部通过 `TaskAdapter` 协议注入。

`shared/agentic/protocol.py` 提供：

- `TaskAdapter`：`prompt` / `prompt_version` / `schema_path` / `validate` / `score`；
- `resolved_task_variant(task_name, payload)`：显式 `payload["task_variant"]` 优先，
  否则按 `horizon_months` 推导。

Task B 与 Task C 各自带一份同名 `factoropt` 包（保证各自脚本自包含），所以共享代码放在
顶层的 `shared/`，两个任务都按包名 import，不依赖 `sys.path` 上谁先被加载。
