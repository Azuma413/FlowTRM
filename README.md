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
- original mlp
mlp_t=Trueだと24GB超えるので注意
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
- flow trm
```bash
run_name="pretrain_flow_trm_sudoku"
uv run pretrain.py \
arch=rf_trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=False arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```
- pusht
```bash
uv run train_pusht.py
```
```bash
uv run eval_pusht_gym.py
```

## Recursive Flow TRM (RF-TRM)

RF-TRMは、TRMの再帰的推論構造をFlow Matching (Rectified Flow) のODEソルバーとして再解釈したマルチモーダル模倣学習モデルです。

### 特徴
- **Dynamic Vector Field**: 思考状態 $z$ の更新に伴い、行動生成のためのベクトル場 $v(y, z, t)$ が動的に最適化されます。
- **Single Tiny Network**: 軽量な単一ネットワークを再帰的に呼び出すことで、パラメータ数を抑えつつ複雑な推論を実現します。
- **Multimodal Generation**: 初期ノイズ $y_0$ と思考 $z$ の相互作用により、マルチモーダルな行動分布を表現可能です。

### アルゴリズム
1. **Inference**: ノイズ $y_0$ から開始し、思考 $z$ を更新しながら Flow Matching のステップを進めることで、徐々に正解アクション $y_1$ へと近づけます。
2. **Training**: Unrolled Flow Matching により、全ステップでのベクトル場学習を行います (Deep Supervision)。

