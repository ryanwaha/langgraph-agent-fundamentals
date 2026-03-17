# Architecture Plan: JDE ↔ LangGraph Redesign

## 1. JDE 與 LangGraph 的定位

**MSG**：執行軌跡，給 ReAct loop 用，ephemeral。每次 `ainvoke` 只帶 `[HumanMessage(current)]`，不積累歷史。

**JDE**：對話記憶，SSOT，跨 session 持久。`Q + A + sum`，精煉有意義。`dag_context` 展平後只進 system message，**絕不進 MSG**。

兩者以 `thread_id` 鬆耦合。

**現狀問題**：DAG 生命週期（load → build_context → invoke → append → write）責任落在 TG bot 前端，應移出。

---

## 2. DAG 生命週期責任歸屬

**目標**：bot 層只做 `graph.ainvoke(msg, thread_id)` → 拿 answer。

**`dag_graph` 的存活位置**：進 LangGraph `State`（Option A）。

- JDE 設計為 object-first，避免重複讀取，memory 與檔案維持一致
- 無 checkpointer（見點 3），無序列化問題，Option A 成立
- 節點間共享同一物件實例，符合 JDE 設計意圖

**節點責任分配**：

```
call_model（第一次進入）
  └─ load(jsonl_path(thread_id)) → dag_graph 進 State
     build_dag_context(dag_graph) → 注入 system message

summarize
  └─ 從 state.messages 提取 tool trace
     寫入 per-thread append-only log → 取得 log_ref（行號）
     LLM 產生 sum（含行為摘要）
     append_node(q, a, sum, log_ref) + write_session
```

---

## 3. Checkpointer

**結論：不加。**

此 agent 不負責複雜長鏈任務，無回溯執行狀態的需求。JDE 已處理跨 session 記憶。Checkpointer 提供的 ReAct 崩潰恢復與 time travel 在此場景無使用價值。

---

## 4. Agent 行為紀錄（Log）

**結構**：per-thread append-only JSONL，路徑 `logs/{thread_id}.jsonl`。

**`log_ref`**：JDE node 的獨立欄位，值為行號（integer，nullable）。不與 `node.id` 共用——兩系統生命週期不是 1:1，顯式 ref 比 ID 對齊更穩健。

**顆粒度**：tool name + input + 成敗。不存 raw response content（語意已壓縮進 `sum`）。

```json
{
  "node_id": "ULID_A",
  "ts": "2026-03-17T...",
  "tools": [
    {"name": "tavily_search", "input": {"query": "..."}, "ok": true}
  ]
}
```

細節待實作時定。

---

## 5. 待討論

- Thread 進程設計
