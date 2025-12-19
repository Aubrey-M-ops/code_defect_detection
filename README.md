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

```json
{
    AST_TOKENS: "translation_unit function_definition storage_class_specifier static type_identifier ERROR identifier function_declarator identifier parameter_list ( parameter_declaration type_identifier pointer_declarator * identifier ) compound_statement { declaration type_identifier ......",
    CODE: "\nstatic av_cold int ffat_close_encoder(AVCodecContext *avctx)\n\n{\n\n    ATDecodeContext *at = avctx->priv_data;\n\n    AudioConverterDispose(at->converter);\n\n    av_frame_unref(&at->new_in_frame);\n\n    av_frame_unref(&at->in_frame);\n\n    ff_af_queue_close(&at->afq);\n\n    return 0;\n\n}\n"
}
```



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