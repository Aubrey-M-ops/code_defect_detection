## Environment Preparation

### 1. Install dependencies 

```makefile
make install
```

### 2.  Dataloading

```makefile
make data
```

### 3. Train Baseline

```makefile
make train-baseline
```

## 💡 AST OPTIMIZER

### Code 👉 AST Structure

```json
{
  "type": "translation_unit",
  "start_point": [
    0,
    0
  ],
  "end_point": [
    145,
    0
  ],
  "text": "static av_cold int vdadec_init(AVCodecContext *avctx)\n\n{\n\n    VDADecoderContext *ctx = avctx->priv_d",
  "children": [
    {
      "type": "function_definition",
      "start_point": [
        0,
        0
      ],
      "end_point": [
        144,
        1
      ],
      ...
}
```

### Augment Text ( AST Sequence + Code)

---

### **Old Format**

```text
[CODE]
<code>
[AST]
function_definition assignment_expression binary_expression binary_expression assignment_expression call_expression if_statement unary_expression return_statement call_expression for_statement assignment_expression binary_expression assignment_expression call_expression if_statement unary_expression assignment_expression break_statement if_statement assignment_expression call_expression return_statement
```

---

### **New Format (Structured)**

```text
[CODE]
<code>
[AST-CF] if_statement for_statement if_statement if_statement
[AST-OP] assignment_expression assignment_expression call_expression return_statement call_expression assignment_expression assignment_expression call_expression assignment_expression break_statement assignment_expression call_expression return_statement
[AST-STATS] binary_expression:3 unary_expression:2
```

---

### **3. Performance Comparison**

| Metric                      | Old Method                           | New Method                           | Improvement               |
| --------------------------- | ------------------------------------ | ------------------------------------ | ------------------------- |
| **Control-flow nodes**      | Mixed together                       | Explicitly separated (4 nodes)       | ✅ Clear semantics         |
| **Noisy nodes**             | Includes `function_definition`, etc. | Removed                              | ✅ Reduced interference    |
| **Repeated expressions**    | Listed individually                  | Aggregated as counts                 | ✅ Information compression |
| **Structural organization** | Single flat sequence                 | Three-part structure                 | ✅ Easier to learn         |
| **Tunable parameters**      | Only `max_nodes`                     | `max_control_flow`, `max_operations` | ✅ More flexible control   |


## RESULT

### Baseline - CODEBERT【 Fine-tune】

The running results are logged in `reults/baseline_1.json`, `reults/baseline_2.json`, `reults/baseline_3.json`

| Metric    | Run 1  | Run 2  | Run 3  | **Mean ± Std**      |
| --------- | ------ | ------ | ------ | ------------------- |
| Accuracy  | 0.6340 | 0.6274 | 0.6351 | **0.6321 ± 0.0042** |
| Precision | 0.6164 | 0.5952 | 0.6138 | **0.6085 ± 0.0115** |
| Recall    | 0.5378 | 0.5904 | 0.5546 | **0.5610 ± 0.0269** |
| **F1**    | 0.5745 | 0.5928 | 0.5827 | **0.5833 ± 0.0092** |
| **MCC**   | 0.2578 | 0.2494 | 0.2610 | **0.2560 ± 0.0060** |
| Loss      | 0.6190 | 0.6054 | 0.6143 | **0.6129 ± 0.0069** |


### AST OPTIMIZATION - CODEBERT + AST STRUCTURE【 Fine-tune】

The running results are logged in `reults/ast_1.json`, `reults/ast_2.json`, `reults/ast_3.json`

| Metric    | Run 1  | Run 2  | Run 3  | **Mean ± Std**      |
| --------- | ------ | ------ | ------ | ------------------- |
| Accuracy  | 0.6402 | 0.6398 | 0.6409 | **0.6403 ± 0.0005** |
| Precision | 0.6187 | 0.6159 | 0.6130 | **0.6159 ± 0.0023** |
| Recall    | 0.5649 | 0.5737 | 0.5920 | **0.5769 ± 0.0113** |
| **F1**    | 0.5906 | 0.5941 | 0.6024 | **0.5957 ± 0.0049** |
| **MCC**   | 0.2717 | 0.2717 | 0.2753 | **0.2729 ± 0.0017** |
| Loss      | 0.6299 | 0.6197 | 0.6215 | **0.6237 ± 0.0045** |

### Performance Comparison and Improvement Analysis

Based on the experimental results above, we observe a **consistent and meaningful improvement** after introducing the `structured AST information` into the CodeBERT fine-tuning process.

#### Overall Performance Gain
Compared with the baseline CodeBERT model, the **AST-optimized model achieves better performance across all key evaluation metrics**, especially those that are more reliable under class imbalance.

- **F1-score** improves from **0.5833 ± 0.0092** to **0.5957 ± 0.0049**, showing a clear and stable gain.
- **MCC** increases from **0.2560 ± 0.0060** to **0.2729 ± 0.0017**, indicating a more balanced and robust classification performance.
- **Accuracy** also improves from **0.6321 ± 0.0042** to **0.6403 ± 0.0005**, while maintaining low variance across runs.


