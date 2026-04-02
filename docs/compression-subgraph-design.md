# 壓縮 Subgraph — 設計文件 v2

> **性質**：實作參照文件，記錄現行設計決策與權衡。
> **對應規格原稿**：`NexCortex-Stage0-LangGraph壓縮Subgraph設計規格.md`（Stage 0 初版）
> **日期**：2026-04-02

---

## §1 與初版規格的主要差異

| 面向 | 初版規格（Stage 0） | 現行實作（v2） | 決策動機 |
|------|--------------------|--------------|---------| 
| 主模型 | `qwen3.5:4b`（本地 Ollama） | `gemini-3.1-flash-lite-preview`（Gemini API） | 4B 模型評估可靠性約 5.5/10；語意壓縮不應省這段成本 |
| L1 標籤系統 | KEEP / COMPRESS / DROP（三值） | KEEP / DROP（二值） | COMPRESS 引入「可從前提推導者」主觀判斷，L1+L2 兩層損失疊加；推理步驟一律 KEEP 更安全 |
| 路由訊號 | `annotated`（帶標籤文本字串） | `keep_segments`（KEEP 段落）+ `meta`（RouterMeta） | 分離「過濾結果」與「內容特徵」；L2 無需解析標籤，直接接收乾淨段落 |
| 路由維度 | short / long（長度） | short / long + RouterMeta 三維度 | 以正交維度描述內容特徵，取代固化的會話類型枚舉（見 §3） |
| L2 長度限制 | ≤ 120 字 | 無硬限制，品質準則驅動 | 長度是壓縮品質的結果，不是輸入約束 |
| SHORT_THRESHOLD | 800 chars | 300 chars | 實際語料校準；多數短回答 < 300 chars |
| Fallback 機制 | 空 segments → raw a[:2000] | L1 失敗 → `route="short"`（強制短路） | 原方案把 raw 文字送進 L2_LONG prompt，造成 prompt/content 矛盾 |

---

## §2 架構圖

```
run_compression(q, a, node_id)
        │
        ▼
   [router]
   len(a) < 300? ──── yes ────► [l2_compress] ──► sum
        │                              ▲
        │ no                           │ (L1 例外時強制)
        ▼                              │
  [l1_annotate]                        │
   KEEP/DROP 標記                      │
   RouterMeta 三維度                   │
        │                              │
        │ keep_segments + meta         │
        ▼                              │
   [l2_compress] ─────────────────────►
        │
        ▼
       sum
```

---

## §3 RouterMeta — 正交三維度設計

### 設計動機

初版規格以長度作為唯一路由訊號（short/long）。實作中討論過增加內容類型枚舉（factual / code / search / decision…），但這會碰到 NexCortex 修訂十四原則的問題：

> **離散分類是 top-down 強制詮釋**，強迫每個節點套入預設類型，實際語料卻是連續混合的。

解法：三個**正交**、**三值**的維度，讓「類型」在查詢時湧現，而非 ETL 時固化。

```python
class RouterMeta(BaseModel):
    density:   Literal["sparse", "dense", "null"]   # 命題數量
    structure: Literal["flat",   "causal", "null"]  # 推理結構
    modality:  Literal["prose",  "code",   "null"]  # 內容形式
```

| 維度 | sparse/flat/prose | dense/causal/code | null |
|------|-------------------|-------------------|------|
| density | 1-2 個獨立命題 | 3+ 個獨立命題 | 無法判斷 |
| structure | 命題並列，無依賴 | 含因果鏈或條件分支 | 無法判斷 |
| modality | 純自然語言 | 含程式碼或形式符號 | 無法判斷 |

RouterMeta 由 L1 同一 call 輸出，作為 L2 的內容特徵信號（非路由決策，而是壓縮策略提示）。

---

## §4 State Schema

```python
class CompressionState(TypedDict):
    node_id:       str   # logging 用，不影響邏輯
    q:             str   # 原始問題
    a:             str   # 原始答案
    route:         str   # "short" | "long"，由 router 設定
    keep_segments: str   # KEEP-only 段落，以 \n\n 連接；short path 為 ""
    meta:          dict  # RouterMeta.model_dump()；short path 為 {}
    sum:           str   # 最終輸出
```

> **設計注意**：`keep_segments` 取代初版的 `annotated`。差異是：`annotated` 是帶 `[KEEP]`/`[COMPRESS]`/`[DROP]` 標籤的原始文本，L2 需要解析標籤；`keep_segments` 是已過濾的乾淨段落，L2 直接壓縮，無歧義。

---

## §5 L1 節點設計

### 二值標籤準則

**KEEP（滿足任一條件即 KEEP）**
- 事實命題：具體聲明、數值、名稱、版本、識別符
- 推理步驟與因果機制（即使在理論上可再推導）
- 分支條件與限制：結論的適用邊界
- 負向決策：嘗試了 X、因為 Y 放棄、改用 Z
- 結論與決策

> ⚠ 不確定時，預設 KEEP

**DROP（三個條件必須同時成立）**
- 元對話或格式引導語（問候、確認、鋪墊句）
- 常識性背景，且與本答案結論無因果連結
- 與其他段落語意完全重疊（非補充，是重複）

### 為何廢除 COMPRESS

初版的 COMPRESS 類別允許「可從前提推導的推理步驟」被壓縮。問題：

1. **主觀性**：什麼「可從前提推導」依賴閱讀者的背景知識，4B 模型判斷不穩定
2. **損失疊加**：L1 壓縮一次，L2 再壓縮一次，推理鏈在兩層累積損失
3. **語意安全性**：推理步驟即使冗長，也是因果鏈的組成部分，刪除可能斷鏈

解法：所有推理步驟一律 KEEP，決策壓力集中在 L2。

### 失敗保護

```python
# L1 例外 → 強制 short path（而非送 raw text 進 L2_LONG）
except Exception:
    return {"keep_segments": "", "meta": {}, "route": "short"}

# 全 DROP 時 → 保留全文，記 warning
if not keep_segments:
    keep_segments = "\n\n".join(paragraphs)
    logger.warning("all paragraphs DROPped, preserving all")
```

---

## §6 L2 節點設計

### 品質準則（取代字數限制）

四條必要條件，全部滿足才算合格輸出：

1. **命題完整性** — 每個 KEEP 段落的核心命題必須在摘要中有對應陳述，不得遺漏
2. **結構保真** — 因果連接詞必須保留；`A → B → C` 不可簡化為 `C`
3. **具體值逐字** — 數值、名稱、識別符、程式碼片段禁止改寫或換算
4. **零幻覺** — 不含任何原文中缺席的資訊

### 路徑分支

| 條件 | 使用 prompt | 輸入 |
|------|------------|------|
| `route == "long"` 且 `keep_segments` 非空 | `L2_LONG` | `keep_segments` + `meta` |
| 其他（short path 或 L1 失敗） | `L2_SHORT` | `a[:3000]` |

---

## §7 模型層設計

### 為何切換至 Gemini

- Qwen 3.5 4B 實測評估 ~5.5/10：KEEP/DROP 判斷在邊界案例不穩定，structured output 偶爾不合規
- 語意壓縮是 knowledge graph 的核心品質瓶頸，不應以節省成本為由犧牲可靠性
- Gemini Flash Lite 具備足夠推理能力，API 延遲可接受（壓縮為非同步背景任務）

### thinking_budget

```python
# include_thoughts 故意不傳（defaults False）
# thinking token 若出現在 structured output 回應中，會破壞 JSON parsing
return init_chat_model(
    name,
    model_provider="google_genai",
    thinking_budget=512,   # 最小推理預算，給邊界案例用
)
```

### Ollama 介面保留

Ollama 分支保留在 `llm.py`，可透過 `COMPRESS_MODEL=ollama/<model>` 重新啟用。當前設定：

```env
COMPRESS_MODEL=google_genai/gemini-3.1-flash-lite-preview
COMPRESS_FALLBACK_MODEL=   # 停用
COMPRESS_THINKING_BUDGET=512
```

---

## §8 公開介面

```python
from agent.compression.graph import run_compression

# 保證返回非空字串
sum_text: str = await run_compression(
    q="使用者問題",
    a="模型回答",
    node_id="01KMPKDT3F...",  # optional, for logging
)
```

失敗保底層次：
1. L2 正常 → `L2Output.sum`
2. L2 例外 → `{"sum": ""}` → caller 觸發保底
3. 保底 → `q[:100] + "..."`（最終兜底，永不返回空字串）

---

## §9 已知限制與後續工作

| 限制 | 說明 | 優先度 |
|------|------|--------|
| `split("\n\n")` 對 markdown code block 不安全 | code block 中若有空行會被切斷 | 中 |
| `a[:4000]` 截斷 | 超長答案的後半段完全不進 L1 | 中 |
| RouterMeta 對 L2 的影響尚未量化 | 三維度作為提示信號，但 L2 有多少程度使用它尚不確定 | 低 |
| 命題保留率無自動驗證 | 需人工或自動化評估 | 低 |

---

## §10 評估標準（繼承初版 §9）

| 評估維度 | 通過標準 |
|----------|----------|
| 命題覆蓋率 | `sum` 保留原始節點中所有不可壓縮命題 |
| 結構保真 | 因果鏈未被截斷或壓平 |
| 具體值保留 | 數值、名稱、識別符逐字出現在 `sum` 中 |
| 無幻覺 | `sum` 不包含原文中不存在的事實 |

> 初版規格的「sum 字數 ≤ 原文 20%」已廢除，理由見 §1。

---

*文件日期：2026-04-02*
*對應初版規格：`NexCortex-Stage0-LangGraph壓縮Subgraph設計規格.md`*
*對應實作：`src/agent/compression/graph.py`, `src/agent/compression/prompts.py`, `src/agent/llm.py`*
