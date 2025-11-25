# Flow Tiny Recursive Model

## Setup
```bash
git clone https://github.com/Azuma413/FlowTRM.git
cd FlowTRM
uv sync
```
- make datasets
```bash
uv run dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000
``` 
## train
mlp_t=Trueだと24GB超える．
```bash
run_name="pretrain_mlp_t_sudoku"
uv run pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=False arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```