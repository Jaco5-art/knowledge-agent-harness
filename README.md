# 证据约束的知识抽取 Agent Harness

个人求职工程项目：从关系抽取与验证实验，扩展到有状态执行、受控工具调用、语义门控和可追溯评估。采用面向生产的工程设计；未证明达到生产部署可靠性。

## 最终状态

截至本次收尾，停止新增付费实验。原版固定 100 篇 Re-DocRED 结果作为历史基准；升级版专项实验单列。升级版 Re-DocRED 适配层已完成离线验收，但升级版 100 篇真实批量评估未执行，批量入口未交付。

## 能力与技术栈

- Python、Pydantic：输入输出契约、角色专用 schema、编号约束。
- OpenAI API、FAISS：结构化抽取、检索、验证、纠错；替代工具可通过注册表接入。
- Harness、SQLite：会话持久化、版本检查、重试、调用预算、人工复核、纠错循环控制。
- 上下文预算与证据重叠合并：可选启用；文本可无损重建不等于模型判断不变。
- 语义目录：别名解析、关系规范化、实体类型校验、歧义转复核。
- LangGraph / OpenAI Agents SDK：统一 CLI 可选择的确定性调度器，共用业务策略与应用层存储。
- MCP 2.x：本地 stdio 历史查询服务与客户端；固定作用域、参数校验、只读 SQLite。
- JSON/JSONL：调用轨迹、usage 覆盖率、实验报告与错误分析。

## 使用现有环境

在已有 v0.3.1 项目根目录运行；不要求重新创建环境。离线示例：

```powershell
.\.venv\Scripts\python.exe .\project.py run --runtime harness
.\.venv\Scripts\python.exe .\project.py run --runtime langgraph
.\.venv\Scripts\python.exe .\project.py run --runtime sdk
```

上述命令各自创建新示例会话；从输出取得 session_id 后，使用相同 --db、--tenant 和 --session 恢复指定会话。完成会话再次运行不会增加工具调用。可用 --max-steps 1 暂停。SDK 与 LangGraph 每次调度默认最多 200 步，达到上限后可能保留 running 状态，应按会话恢复，不把它当成完成。

```powershell
.\.venv\Scripts\python.exe .\project.py demo
.\.venv\Scripts\python.exe .\mcp_history.py self-test
```

历史查询实际使用：

```powershell
.\.venv\Scripts\python.exe .\mcp_history.py query --db .\business_history.sqlite --tenant YOUR_TENANT --scope YOUR_SCOPE --revision business-v1 --subject ENTITY_A --predicate invested_in --object ENTITY_B
```

必须使用数据库实际的作用域和实体 ID；示例占位符不是默认凭证。`serve` 子命令用于 MCP 客户端启动服务。查询结果标记为未验证参考数据，不能自动成为事实证据。该 MCP 接口是独立可调用工具，未声称模型会自主选择它，也未将日常语义工作流的直接历史查询替换为 MCP 传输。

## 代码入口

| 范围 | 入口 |
|---|---|
| 日常会话 | project.py run / harness.__main__ |
| 调度选择 | harness/runtime_engines.py |
| 上下文预算与压缩 | harness/context.py、context_adapter.py |
| 业务语义输出 | harness/semantic_workflow.py |
| 历史查询 | harness/history_query.py、mcp_history.py |
| 原基准实验 | knowledge_agents/formal_experiment.py |
| 新编号适配层 | redocred_harness_adapter.py |

`project.py run` 输出 Harness 接受的事实；业务语义门控由独立 workflow / business 入口执行，不能把原始接受结果说成已经过语义门控。

## 阅读顺序

1. EXPERIMENT_RESULTS.md：原版成绩与升级实验分开呈现。
2. INTERVIEW_NOTES.md：简历与面试表述。
3. HANDOFF.md：范围、已知缺陷和归档说明。
4. evidence/：实际上传的验收报告及原始实验输出。

## 不要误用

不要声称升级版在 100 篇上的准确率；不要把 10 篇合成样本的 100% 完全匹配推广到真实业务；不要称为自主 LLM 规划或原生框架崩溃恢复；不要把只读 MCP hint 当成安全实现。付费实验脚本保留供复现，本次不再运行。
