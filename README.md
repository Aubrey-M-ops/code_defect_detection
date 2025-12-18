## 1. Install dependencies 

```makefile
make install
```

## 2.  Dataloading

```makefile
make data
```

## 3. Train Baseline

```makefile
make train-baseline
```

## RESULT
### Baseline - CODEBERT Fine-tune

The running results are logged in `reults/baseline_1.json`, `reults/baseline_2.json`, `reults/baseline_3.json`

| Metric    | Run 1  | Run 2  | Run 3  | **Mean ± Std**      |
| --------- | ------ | ------ | ------ | ------------------- |
| Accuracy  | 0.6340 | 0.6274 | 0.6351 | **0.6321 ± 0.0042** |
| Precision | 0.6164 | 0.5952 | 0.6138 | **0.6085 ± 0.0115** |
| Recall    | 0.5378 | 0.5904 | 0.5546 | **0.5610 ± 0.0269** |
| **F1**    | 0.5745 | 0.5928 | 0.5827 | **0.5833 ± 0.0092** |
| **MCC**   | 0.2578 | 0.2494 | 0.2610 | **0.2560 ± 0.0060** |
| Loss      | 0.6190 | 0.6054 | 0.6143 | **0.6129 ± 0.0069** |
