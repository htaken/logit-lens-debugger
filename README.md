# logit-lens-debugger

Hugging Face の causal LM（Qwen / Llama）に **Logit Lens** を適用し、「各 Transformer layer の時点で、モデルが次トークンとして何を高く評価していたか」を layer ごとに確認できる診断ツールです。

音声認識 → LLM 整形・要約 → 議事録生成のようなパイプラインで「固有名詞が書き換わった」「数値が変わった」「原文にない情報が補完された」といった問題を、入出力ログより一段深いところから調べるための **LLM デバッガの土台** として設計しています。特定案件に依存しない Python パッケージで、他の LLM アプリケーションにも外付けできます。

> **重要な前提**
> 本ツールが出力するのは「各中間表現（residual stream）を語彙空間へ射影したときの token prediction」です。
> モデルの「思考」や「考えていたこと」を観測しているわけではなく、内部推論過程そのものでもありません。
> 詳しくは [何を解析していて、何を解析していないか](#何を解析していて何を解析していないか) を参照してください。

---

## Logit Lens とは

decoder-only Transformer は次の流れで次トークンの logits を計算します。

```
token embedding → decoder block 1 → … → decoder block N → final norm → lm_head (unembedding) → logits
```

通常は最後の hidden state だけが `lm_head` に渡されますが、途中の layer の hidden state も同じ次元を持つため、そのまま final norm と `lm_head` に通して「その layer 時点の語彙分布」を読むことができます。これが Logit Lens（nostalgebraist, 2020）です。

```
hidden state (layer i, 対象 token position)
    ↓ final normalization   （final-norm モード時のみ）
    ↓ lm_head / unembedding
    ↓ vocabulary logits
    ↓ softmax / ranking
    ↓ top-k tokens
```

layer を横に並べると、次のような推移（trajectory）が観察できます。

```
layer 21  → ' __'      p=0.20
layer 22  → ' Tokyo'   p=0.97
layer 23  → ' Tokyo'   p=0.73
layer 24  → ' Tokyo'   p=0.72   ← 最終 layer = モデルの実際の出力分布
```

## 何を解析していて、何を解析していないか

**解析しているもの**

- 指定した token position の residual stream を、各 layer 時点で語彙空間に射影したときの top-k token とその token id / logit / probability / rank
- 追跡対象 token（target）の各 layer における rank と probability
- 各 layer の分布のエントロピー（confidence の目安）
- 2 つのテキスト間で top-1 予測が分岐し始める layer（`compare`）

**解析していないもの・注意点**

- **モデルの「思考」や内部推論過程そのものではありません。** 中間 layer の表現が最終 layer と同じ基底で書かれている保証はなく、特に小規模モデルの前半〜中盤 layer では意味の取れない token（コード断片や別言語の断片）が top-1 に来ることが普通にあります。これは Logit Lens の既知の限界で、学習済み射影を使う tuned lens などで緩和されます。
- attention がどの入力トークンを参照したか、どの入力が出力に寄与したかは分かりません（attribution / attention 解析は別手法です）。
- 表示される probability は「その layer の表現をそのまま最終出力として扱った場合」の値です。モデルがその時点でその token を「出力しようとしていた」ことを意味しません。
- 結果は checkpoint・tokenizer・prompt 形式に強く依存します。例: `Qwen/Qwen2.5-0.5B` に `The capital of Japan is` を与えると最終出力の top-1 は `' __'`（穴埋め問題形式）で、`' Tokyo'` は 3 位です。greedy 生成でも `' ____\nA.'` が出ることを確認しており、ツールの誤りではなくモデルの実際の挙動です。`Q: What is the capital of Japan?\nA:` の形式にすると `' Tokyo'` が p=0.72 で top-1 になります。

## インストール

要件: Python 3.11+ / PyTorch 2.1+ / Transformers 4.56+（5.x で動作確認）。可視化（HTML 出力）を使う場合のみ Plotly が必要です。

```bash
# uv
uv sync --group dev --extra viz
uv run logit-lens --help

# pip
pip install -e ".[viz]"
logit-lens --help
```

## 使い方

### CLI

```bash
uv run logit-lens analyze \
  --model Qwen/Qwen2.5-0.5B \
  --text $'Q: What is the capital of Japan?\nA:' \
  --token-index -1 \
  --top-k 5 \
  --json outputs/qa.json \
  --html outputs/qa.html
```

出力例（Apple M2 / MPS / bf16 での実行結果を抜粋）:

```
Model: Qwen/Qwen2.5-0.5B  (device=mps:0, lens=final-norm)
Tokens (11): [0]'Q'  [1]':'  [2]' What'  [3]' is'  [4]' the'  [5]' capital'  [6]' of'  [7]' Japan'  [8]'?\n'  [9]'A'  [10]':'  <-- analyzed
Predicting the token after index 10 ':'; tracked target: none

Layer 0 (embeddings)
   1. ':'                      p=1.0000  logit=109.00
   2. ','                      p=0.0000  logit=85.00
   ...

Layer 21
   1. '\\"",\n'                p=0.1966  logit=18.62
   2. '(*)('                   p=0.1531  logit=18.38
   3. ';break'                 p=0.0929  logit=17.88

Layer 22
   1. ' Tokyo'                 p=0.9716  logit=22.62
   2. '东京'                    p=0.0157  logit=18.50
   3. 'Tok'                    p=0.0021  logit=16.50

Layer 24
   1. ' Tokyo'                 p=0.7245  logit=18.12
   2. ' Kyoto'                 p=0.0463  logit=15.38
   3. ' Osaka'                 p=0.0233  logit=14.69

Top-1 trajectory (final-norm lens) for the token after index 10 ':'
  layer   0  ':'                      p=1.000
  ...
  layer  19  ';break'                 p=0.568
  layer  20  ' whiteColor'            p=0.281
  layer  21  '\\"",\n'                p=0.197
  layer  22  ' Tokyo'                 p=0.972
  layer  23  ' Tokyo'                 p=0.726
  layer  24  ' Tokyo'                 p=0.724
```

主なオプション:

| オプション | 意味 |
|---|---|
| `--model, -m` | Hub の model id またはローカルパス（既定: `Qwen/Qwen2.5-0.5B-Instruct`） |
| `--text, -t` | 解析するテキスト（必須） |
| `--token-index, -i` | 射影する position。負数は末尾から。予測はその **次の** token について（既定: `-1`） |
| `--top-k, -k` | layer ごとに保持する候補数（既定: 10） |
| `--lens-mode` | `final-norm`（既定）または `raw` |
| `--target-token` | 各 layer での rank / probability を追跡する token。省略時は `--token-index` の次に実際に存在する token |
| `--device` / `--dtype` | `cuda` / `mps` / `cpu`、`bf16` / `fp16` / `fp32` / `auto`（既定: 自動選択。GPU/MPS は bf16、CPU は fp32） |
| `--show` | layer ごとに表示する行数（既定: top-k 全部） |
| `--json` / `--html` | 結果の JSON 保存 / Layer × Token ヒートマップの HTML 保存 |

その他のコマンド:

```bash
# 2 つのテキストで top-1 予測が分岐する layer を調べる
uv run logit-lens compare --model Qwen/Qwen2.5-0.5B \
  --text-a $'Q: What is the capital of Japan?\nA:' \
  --text-b $'Q: What is the capital of France?\nA:'

# 保存済み JSON からモデルをロードせずに可視化を再生成する
uv run logit-lens visualize outputs/qa.json --html outputs/qa_heatmap.html --kind heatmap
uv run logit-lens visualize outputs/qa.json --html outputs/qa_traj.html --kind trajectory

# 保存済み JSON から layer instability を計算する（モデル不要）
uv run logit-lens metrics outputs/qa.json --from-layer 1 --json outputs/qa_metrics.json

# 原文と出力の diff から怪しい token position を自動選択して解析する
uv run logit-lens diff --model Qwen/Qwen2.5-0.5B \
  --source "売上は150万円でした" --output "売上は1500万円でした" \
  --context 1 --json outputs/diff.json --html-dir outputs/diff_html

# 出力の各 token を anomaly スコアでスキャンする
uv run logit-lens scan --model Qwen/Qwen2.5-0.5B \
  --prompt "売上は150万円でした。要約して。" --output "売上は1500万円でした。" \
  --source "売上は150万円でした" --top 20 --json outputs/anomaly.json
```

### Python API

```python
from logit_lens_debugger import LogitLensAnalyzer

analyzer = LogitLensAnalyzer(
    model_name="Qwen/Qwen2.5-0.5B",
    device="mps",          # 省略可: cuda → mps → cpu の順で自動選択
    lens_mode="final-norm" # or "raw"
)

result = analyzer.analyze(
    text="Q: What is the capital of Japan?\nA:",
    token_index=-1,
    top_k=10,
)

result.input_tokens              # ['Q', ':', ' What', ' is', ..., 'A', ':']
result.token_index               # 10  (負の index は正規化される)
result.num_layers                # 24  (len(result.layers) == 25; layer 0 は embedding 出力)

layer = result.layers[22]
layer.layer_index                # 22
layer.top_tokens[0]              # TokenPrediction(token=' Tokyo', token_id=26194, logit=22.6, probability=0.97, rank=1)
layer.entropy                    # その layer の分布のエントロピー

[l.top1.token for l in result.layers]   # top-1 の trajectory
result.save_json("outputs/qa.json")     # LogitLensResult.load_json() で復元可能
```

JSON の形（抜粋）:

```json
{
  "model_name": "Qwen/Qwen2.5-0.5B",
  "lens_mode": "final-norm",
  "text": "Q: What is the capital of Japan?\nA:",
  "input_tokens": ["Q", ":", " What", "..."],
  "input_token_ids": [48, 25, 3555, "..."],
  "token_index": 10,
  "top_k": 5,
  "target_token": null,
  "target_token_id": null,
  "layers": [
    {
      "layer_index": 22,
      "entropy": 0.21,
      "target": null,
      "top_tokens": [
        {"token": " Tokyo", "token_id": 26194, "logit": 22.62, "probability": 0.9716, "rank": 1}
      ]
    }
  ]
}
```

#### 生成された token の「理由」を調べる（target 追跡）

生成結果 `j` 番目の token を調べたいときは、prompt と生成結果を連結したテキストを渡し、`token_index=j-1` を指定します。`target_token` を省略すると、その次に実在する token（= 調べたい token）が自動的に追跡対象になり、各 layer での rank と probability が `layer.target` に入ります。

```python
result = analyzer.analyze("Q: What is the capital of Japan?\nA: Tokyo", token_index=-2, top_k=10)
result.target_token                     # ' Tokyo'
for layer in result.layers:
    t = layer.target
    print(layer.layer_index, t.rank, round(t.probability, 4))

# 「本来出てほしかった token」を指定して追跡することもできる
analyzer.analyze(text, token_index=-2, target_token=" Osaka")
```

#### 2 つのテキストの比較

```python
cmp = analyzer.compare(text_a, text_b, token_index=-1, top_k=10)
cmp.top1_agreement          # layer ごとに top-1 が一致したか (list[bool])
cmp.first_divergence_layer  # 最初に分岐した layer（None なら全 layer 一致）
cmp.divergence_layers       # 分岐した layer の一覧
```

#### Layer instability（予測の不安定さ）

`result.trajectory()` の top-1 が layer をまたいでどれだけ入れ替わるかを 1 つのスコアにまとめます。モデル・tokenizer は不要で、`LogitLensResult`（JSON から復元したものでも可）だけから計算できる純関数です。

```python
metrics = result.instability()          # from_layer=1（既定）, late_fraction=1/3

# もしくはモジュール関数として
from logit_lens_debugger import compute_instability
metrics = compute_instability(result, from_layer=1, late_fraction=1 / 3)

metrics.instability_score   # [0, 1] にクリップ済み
metrics.settle_layer        # これ以降ずっと最終 layer と同じ top-1 になる最初の layer
```

| フィールド | 内容 |
|---|---|
| `top1_switches` | 隣接 layer 間で top-1 の token が変わった回数 |
| `oscillations` | 一度離れた token に top-1 が戻った回数（`A→B→A→B` なら 2） |
| `distinct_top1` | top-1 に現れた token の種類数 |
| `settle_layer` | それ以降ずっと最終 layer と同じ top-1 になる最初の layer |
| `late_switches` / `late_start_layer` | 後半（既定: 末尾 1/3）で起きた switch の回数と、その境界となる layer |
| `instability_score` | 上記を `[0, 1]` に正規化したスコア |
| `target_first_top1_layer` / `target_first_topk_layer` | target が初めて top-1 / top-k に入った layer（target 追跡時のみ、それ以外は `None`） |
| `target_rank_log_mean` / `target_rank_log_std` | 各 layer の `log2(rank)` の平均・母標準偏差（target 追跡時のみ） |
| `target_final_rank` / `target_final_prob` | 最終 layer での target の rank / probability（target 追跡時のみ） |

CLI からはモデルをロードせず、保存済み JSON だけで計算できます。

```bash
uv run logit-lens metrics outputs/qa.json --from-layer 1 --json outputs/qa_metrics.json
```

> **注意**: 小規模モデルの前半〜中盤 layer では top-1 が不安定なのが常態です。`instability_score` は「どの position を詳しく調べるか」の優先順位付けのための heuristics であり、それ単独で問題の原因を示す証拠ではありません。

#### 原文と生成結果の diff から対象 position を自動選択する

音声認識のテキスト化 → LLM で要約・整形、のようなパイプラインでは「どの token を調べればよいか」が最初の問題です。`analyzer.analyze_diff()` は原文（`source`）と出力（`output`）を **モデルの tokenizer で** それぞれ tokenize し、token 文字列の列同士を `difflib` で比較して、変化した箇所だけを解析対象にします。

```python
result = analyzer.analyze_diff(
    source="売上は150万円でした",
    output="売上は1500万円でした",
    top_k=10,
    context=1,       # 変化箇所の前後 n token も解析対象に加える
)

for span in result.diff_spans:
    span.kind, span.source_text, span.output_text   # "replace", "150", "1500"

for pos in result.positions:
    pos.token_index      # 解析した residual stream position
    pos.target_token      # そこで実際に出力された token（"1500" 側の断片）
    pos.diff_span         # この position の元になった DiffSpan
```

Qwen の tokenizer は数字を 1 桁ずつ token 化するため（`1500` → `1`,`5`,`0`,`0`）、素の diff だと `0` が 1 個 insert されただけに見えます。`expand_spans=True`（既定）は同じ文字クラス（数字 / かな / 漢字。英字は対象外 -- 実際の tokenizer では単語の先頭 token が空白付き（`" sales"` のような形）になるため、英字まで対象にすると文全体まで広がってしまいます）の隣接 token をまとめて span を数値全体に広げます。ただし空白始まりの token（新しい単語の先頭）をまたいでは広げません（`" 2024"` の直後の `" 150"` が別の数値として保たれるのはこのためです）。

`prompt` を省略すると `source` がそのまま「モデルに実際に渡した文脈」として使われます（近似: 「モデルは原文を見てこの出力をした」とみなす）。実際に渡したプロンプトが別にある場合は `prompt=` で指定してください。

```python
result = analyzer.analyze_diff(
    source=transcript,
    output=llm_output,
    prompt=f"次の文字起こしを要約してください:\n{transcript}",
)
```

`prompt` と `output` は **別々に tokenize してから連結** します。連結後の文字列をまとめて tokenize すると、境界で token が結合されて実際の生成 token 列とずれることがあるためです（`output` を生成時の id 列で渡せば生成ログを厳密に再現できます）。

もっと直接的に位置を指定したい場合は次の 2 つも使えます。

```python
# prompt + 生成結果（id 列 or 文字列）から、生成された token ごとに解析する
analyzer.analyze_generation(prompt="the capital of japan is", output="tokyo osaka")

# 既に分かっている position 集合を 1 回の forward でまとめて解析する
analyzer.analyze_positions("the capital of japan is tokyo", positions=[2, 4])
```

いずれも `MultiPositionResult` を返します（1 つの文脈に対する複数 position 分の結果を、`input_tokens` を重複させずにまとめたもの）。

| フィールド | 内容 |
|---|---|
| `text` / `input_tokens` / `input_token_ids` | 解析した文脈全体（prompt + output） |
| `prompt_length` | `input_token_ids` のうち output が始まる index |
| `positions` | `PositionResult` のリスト（`token_index` / `layers` / `target_token` / `diff_span`） |
| `diff_spans` | `analyze_diff` が検出した `DiffSpan` の一覧（`analyze_positions` / `analyze_generation` では空） |
| `output_tokens` | `input_tokens[prompt_length:]` |
| `to_results()` | 各 position を独立した `LogitLensResult` に展開する（既存の可視化・`instability()` をそのまま使うため） |

```python
for single in result.to_results():
    print(single.instability().instability_score)
    export_heatmap_html(single, f"outputs/position_{single.token_index}.html")
```

JSON の形（抜粋）:

```json
{
  "prompt_length": 4,
  "positions": [
    {
      "token_index": 5,
      "target_token": "1500",
      "diff_span": {"kind": "replace", "source_text": "150", "output_text": "1500",
                     "source_range": [2, 3], "output_range": [2, 3]},
      "layers": ["... LayerResult と同じ形 ..."]
    }
  ],
  "diff_spans": ["... DiffSpan ..."]
}
```

CLI からも同じことができます（`--html-dir` で position ごとにヒートマップを書き出せます）。

```bash
uv run logit-lens diff --source "sales were 150 yen" --output "sales were 1500 yen" \
  --context 1 --json outputs/diff.json --html-dir outputs/diff_html
```

> **制約**: diff は token 列ベースなので、出力が原文を大幅に言い換えている場合は span がほぼ全域になり絞り込みの効果が薄れます。文字オフセットベースの diff は未実装です。

#### 自動 anomaly detection（怪しい token のスコアリング）

出力の各 token を「lens の trajectory がどれだけ不安定か」「原文と比べてどれだけ新規か」という observable な signal でスコアリングし、調べるべき候補に優先順位を付けます。既定では出力の全 token が対象、`only_diff=True`（CLI: `--only-diff`）を指定すると機能 1 の diff span に絞れます。

| signal | 種別 | 意味 |
|---|---|---|
| `low_final_prob` | lens 系 | 最終 layer でも target の確率が低い（`1 - p_final`） |
| `final_rank` | lens 系 | 最終 layer で target が top-1 でない（rank の log スケール、既定で rank 1024 が上限） |
| `final_entropy` | lens 系 | 最終 layer の分布のエントロピー（`log(vocab_size)` で正規化） |
| `late_emergence` | lens 系 | target が top-k に入るのが最終層直前までかかった（機能 3 の `target_first_topk_layer`） |
| `instability` | lens 系 | 機能 3 の `instability_score` |
| `flip_at_end` | lens 系 | 最終 1〜2 layer で top-1 が入れ替わった |
| `in_diff` | テキスト系 | 機能 1 の diff span 内にある |
| `numeric` | テキスト系 | 数字 token（`token_class(tok) == "digit"`） |
| `novel` | テキスト系 | 原文の token 列に一度も現れない content token（記号・機能語・単独のかな粒子は除外） |
| `entity_like` | テキスト系 | `novel` かつ 先頭大文字 / 漢字 / カタカナのみ |
| `entity` | テキスト系 | 外部 NER から渡された `entity_spans` に含まれる |

各 signal は `[0, 1]` に収まる heuristic で、**評価できない signal は 0 ではなく省略され**、合成スコアは省略された signal を除いた重み付き平均になります（例: target を追跡できない position では `low_final_prob` / `final_rank` / `late_emergence` が丸ごと省略されます）。既定の重みは lens 系 60% / テキスト系 40% です。

```python
from logit_lens_debugger import DEFAULT_ANOMALY_WEIGHTS
DEFAULT_ANOMALY_WEIGHTS
# {"low_final_prob": 0.15, "final_rank": 0.10, "final_entropy": 0.05, "late_emergence": 0.15,
#  "instability": 0.10, "flip_at_end": 0.05,
#  "in_diff": 0.15, "numeric": 0.05, "novel": 0.10, "entity_like": 0.05, "entity": 0.05}
```

`AnomalyConfig` で重みと閾値を変更できます（`weights` / `flag_threshold` / `score_threshold` / `from_layer` / `late_fraction` / `rank_saturation_log2`）。JSON に含めて再現可能です。

```python
from logit_lens_debugger import AnomalyConfig

report = analyzer.detect_anomalies(
    prompt="次の文字起こしを要約してください:\n" + transcript,
    output=llm_output,
    source=transcript,             # 省略すると novel/in_diff は評価されない
    top_k=10,
    config=AnomalyConfig(score_threshold=0.6),
)

for anomaly in report.top(10):
    anomaly.token_index      # 解析した residual stream position
    anomaly.output_index     # 出力 token 列での index（token_index が prompt 内なら None）
    anomaly.token            # 実際に生成された token
    anomaly.score            # 合成スコア
    anomaly.signals          # 評価できた signal のみを含む dict
    anomaly.flags            # flag_threshold 以上だった signal 名（DEFAULT_ANOMALY_WEIGHTS の順）

report.flagged()            # score >= config.score_threshold の position だけ
report.save_json("outputs/anomaly.json")   # AnomalyReport.load_json() で復元可能
```

`only_diff=True` は `source` が必須で（省略すると `ValueError`）、機能 1 と同じ `context` / `chunk_size` を受け取ります。`source` を渡しつつ `only_diff=False`（既定）の場合は、全 token を解析しつつ diff span も計算して `in_diff` を有効にします。

**NER との連携（hook）**: 「原文にない固有名詞」「entity の変更」の検出そのものはこのパッケージのスコープ外です。外部の NER 結果があれば `entity_spans`（出力 token index の `[start, end)` 範囲のリスト）を渡すと `entity` signal に反映されます。

```python
# 外部 NER が output の 2 番目の token を固有名詞と判定した場合
analyzer.detect_anomalies(prompt, output, source=source, entity_spans=[(1, 2)])
```

CLI からは `scan` コマンドで同じことができます。

```bash
# prompt.txt = モデルに実際に渡した文字列（指示 + 文字起こし）、summary.txt = LLM の出力
uv run logit-lens scan --model Qwen/Qwen2.5-0.5B \
  --prompt-file prompt.txt --output-file summary.txt --source-file transcript.txt \
  --top 20 --threshold 0.5 --json outputs/anomaly.json
```

```
Model: Qwen/Qwen2.5-0.5B  (lens=final-norm)
Context: 512 tokens (prompt 480, output 32)

Anomaly report: 32 positions scored, 4 with score >= 0.50
  pos out# token             score  final_p settle  flags
  501    21 大阪               0.81    0.312      22  novel,entity_like,low_final_prob
  ...
```

> **注意**: anomaly スコアは「どの position を先に見るか」の優先順位付けであり、それ自体が問題の証拠ではありません。中間 layer の予測（`late_emergence` / `instability`）は特に小規模モデルで不安定なのが常態で、単独では判断材料になりません。`novel` / `entity_like` は文字クラスに基づく粗い近似であり、本物の NER ではありません。

#### 可視化

```python
from logit_lens_debugger.visualization import (
    export_heatmap_html,      # Layer × Token ヒートマップ（色 = probability）
    export_trajectory_html,   # layer ごとの top-1 probability の折れ線（top-1 が変わった layer に token ラベル）
    format_trajectory,        # テキスト表
)

export_heatmap_html(result, "outputs/heatmap.html")
print(format_trajectory(result))
```

ヒートマップの行は「各 layer の top-k の和集合」から、ピーク確率が高い順に最大 30 token（`max_tokens`）を選び、ピークを迎えた layer の順に並べます。target token は top-k 外の layer でも probability が分かるので行が全て埋まります。

## token_index と layer 番号の意味

- `token_index = i` は **position i の residual stream を射影する** という意味です。得られる予測は **token i+1** についてのものです。負の index は末尾から数えます。
- 生成 token `j` の理由を調べるには `token_index = j - 1`（上記 target 追跡を参照）。
- `layer 0` は embedding 出力（最初の decoder block への入力）、`layer i` は i 番目の decoder block 通過後の residual stream です。N 層モデルなら N+1 行になります。
- `layer 0` は embedding と `lm_head` の重みが共有されているため、final-norm モードでは入力 token 自身が p≈1.0 になります。これは自明な結果で、解析上の意味はほぼありません。

## Lens モード

Logit Lens には実装上の流儀があります。本ツールは両方を実装し、**既定は `final-norm`** です。

| mode | 処理 | 特徴 |
|---|---|---|
| `final-norm`（既定） | hidden → `model.norm`（RMSNorm）→ `lm_head` | 標準的な流儀。最終 layer の結果がモデル本来の出力分布と **完全に一致** し（テストで検証）、layer 間で logit のスケールが揃う |
| `raw` | hidden → `lm_head` | norm を挟まない射影。residual stream のノルムは layer ごとに大きく変わるため、layer 間の logit / probability の比較には向かない |

CLI では `--lens-mode raw`、API では `LogitLensAnalyzer(..., lens_mode="raw")` または `analyzer.lens_mode = "raw"` で切り替えます。

**実装メモ（`output_hidden_states=True` を使っていない理由）**
Transformers 4.5x 以降では `output_hidden_states=True` で返る tuple の最後の要素が final norm 適用後の `last_hidden_state` で上書きされます（`check_model_inputs(tie_last_hidden_states=True)`）。これをそのまま使うと最終 layer で `raw` が raw でなくなり、`final-norm` では norm が二重適用されます。本ツールは各 decoder block と embedding に forward hook を張って **norm 前の出力** を取得し、hook 内で対象 position だけをスライスします（`adapters/base.py`）。テスト `test_hidden_states_match_transformers_and_last_is_pre_norm` でこの性質を固定しています。

## アーキテクチャ

```
CLI (cli.py) / Python
        │
        ▼
LogitLensAnalyzer (analyzer.py)      analyze() / compare()  … token index 正規化、target 解決、結果組み立て
        │
        ▼
LogitLensEngine (engine.py)          hidden → (final norm) → lm_head → softmax → top-k / target rank / entropy
        │
        ▼
ModelAdapter (adapters/)             tokenize / get_hidden_states / apply_final_norm / project_to_vocab / decode_token
   ├── HFCausalLMAdapter             HF decoder-only 共通実装（hook による hidden state 取得）
   ├── QwenAdapter                   model_type: qwen2, qwen3
   └── LlamaAdapter                  model_type: llama
        │
        ▼
Hugging Face Transformers

出力: LogitLensResult (types.py) ── to_dict / to_json ── CLI テキスト (visualization/trajectory.py) ── Plotly HTML (visualization/heatmap.py)
```

モデル固有のコードは `adapters/` に閉じており、engine / analyzer / visualization は `ModelAdapter` インターフェースだけを使います。

主要ファイル:

| ファイル | 役割 |
|---|---|
| `src/logit_lens_debugger/types.py` | `TokenPrediction` / `LayerResult` / `LogitLensResult` / `ComparisonResult` / `LensMode`（dataclass、JSON 往復可） |
| `src/logit_lens_debugger/adapters/base.py` | `ModelAdapter` ABC と `HFCausalLMAdapter`、device / dtype 解決 |
| `src/logit_lens_debugger/adapters/qwen.py`, `llama.py` | ファミリー別 adapter（`supported_model_types` を宣言） |
| `src/logit_lens_debugger/adapters/registry.py` | `config.model_type` → adapter の解決、`load_adapter` / `register_adapter` |
| `src/logit_lens_debugger/engine.py` | Logit Lens 本体（position ごとに別 target も指定可能） |
| `src/logit_lens_debugger/analyzer.py` | 高レベル API（`analyze_positions` / `analyze_generation` / `analyze_diff` を含む） |
| `src/logit_lens_debugger/metrics.py` | `compute_instability()` → `InstabilityMetrics`（torch 非依存、JSON からも計算可） |
| `src/logit_lens_debugger/diff.py` | `diff_token_sequences()` / `select_positions()` / `token_class()`（torch 非依存、diff から対象 position を選ぶ） |
| `src/logit_lens_debugger/anomaly.py` | `score_position()` / `detect_anomalies()` → `AnomalyReport`（torch 非依存、diff / instability を合成して token をスコアリング） |
| `src/logit_lens_debugger/visualization/` | ヒートマップ・trajectory（Plotly）とテキスト整形 |
| `src/logit_lens_debugger/cli.py` | typer CLI（`analyze` / `compare` / `diff` / `scan` / `metrics` / `visualize`） |
| `tests/` | 極小ランダム初期化モデルによるオフラインテスト + opt-in の実モデル統合テスト |
| `examples/basic_analysis.py` | API の最小サンプル |

## 対応モデル

| Adapter | `model_type` | 状態 |
|---|---|---|
| `QwenAdapter` | `qwen2`（Qwen2 / Qwen2.5）, `qwen3`（Qwen3 dense） | `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-0.5B-Instruct` で動作確認。同一構造の 1.5B / 3B / 7B、Qwen3-0.6B なども同 adapter で動く想定 |
| `LlamaAdapter` | `llama`（Llama 2 / 3、SmolLM2 など） | 極小ランダム初期化モデルでテスト済み。実 checkpoint は未検証 |

`model.model.{embed_tokens, layers, norm}` と `model.lm_head` という標準レイアウトの HF decoder-only モデルなら、`HFCausalLMAdapter` を継承して `supported_model_types` を宣言し `register_adapter` に登録するだけで追加できます。

```python
from logit_lens_debugger import HFCausalLMAdapter, register_adapter

@register_adapter
class MistralAdapter(HFCausalLMAdapter):
    supported_model_types = ("mistral",)
```

## メモリ・性能

- 全 layer × 全 position × 全 vocabulary の logits は保持しません。hook の中で対象 position だけを抜き出し、layer ごとに `lm_head` を通して top-k を取ったら直ちに logits を解放します。
- forward は decoder 部分（`model.model`）のみ実行し、全 position に対する `lm_head` の計算を省いています。
- `torch.inference_mode()` 下で実行。既定 dtype は CUDA / MPS で bf16、CPU で fp32。softmax と top-k は fp32 で計算します。
- `LogitLensEngine.run()` は複数 position を一度の forward で処理できます（将来の自動 anomaly 抽出用）。
- 実測: `Qwen/Qwen2.5-0.5B`（24 層、bf16、Apple M2 16 GB / MPS）で 1 テキストの解析は数秒、モデルロード込みで約 10 秒。MPS bf16 と CPU fp32 の結果は一致することを確認しています。

## テスト

```bash
uv run pytest                                          # オフライン。極小ランダム初期化の Qwen2 / Llama で約 3 秒
LLD_RUN_INTEGRATION=1 uv run pytest -m integration     # 実モデル（Qwen/Qwen2.5-0.5B、約 1 GB をダウンロード）
uv run ruff check src tests examples && uv run ruff format --check src tests examples
```

オフラインテストで検証している主な性質:

- final-norm モードの最終 layer の logits がモデル本来の `logits` と一致する
- 取得した hidden state が Transformers の `output_hidden_states` と一致し、最終 layer が norm **前** であること
- top-k の並び・rank・probability（softmax との一致）・entropy の範囲・vocab を超える top-k のクランプ
- target 追跡の rank / probability の整合性、複数 position 解析と単一 position 解析の一致
- hook が実行後に残らないこと、registry のディスパッチ、JSON 往復、CLI の各コマンド、可視化の構造

## 現在の制約

- 1 回の解析は 1 テキスト・batch size 1 です（ただし `analyze_positions` / `analyze_generation` / `analyze_diff` なら、同じテキストの複数 position を 1 回の forward でまとめて解析できます）。
- chat template は適用しません。Instruct モデルを解析する場合は、実際にモデルへ渡した形式（template 適用後の文字列）を `--text` に渡してください。
- 中間 layer の予測は解釈しづらいことが多く（前述）、tuned lens のような学習済み射影は未実装です。
- Qwen の tokenizer は数字を 1 桁ずつ token 化します（`150` → `1`,`5`,`0`）。数値の書き換えを追う場合は複数 position をまたいで見る必要があります（`analyze_diff` は同じ文字クラスの隣接 token をまとめる `expand_spans` でこれに対応しています）。
- diff（`analyze_diff` / `logit-lens diff`）は tokenizer の token 列ベースの比較です。出力が原文の大幅な言い換え・順序入替の場合は span がほぼ全域になり、絞り込みの効果が薄れます。文字オフセットベースの diff は未実装です。
- Llama は実 checkpoint での検証を行っていません。MoE や非標準レイアウト（`model.model.norm` / `lm_head` 以外に norm や head があるモデル）は未対応です。
- 可視化は HTML export のみで、ライトテーマ固定です。

## 次に実装すると良い機能

設計上は以下を後付けしやすくしています。

1. ~~Input / Output diff → 対象 position の自動選択~~ → 実装済み（`diff.py`、`analyzer.analyze_diff()` / `analyze_generation()` / `analyze_positions()`、CLI `diff` コマンド）
2. ~~自動 anomaly detection~~ → 実装済み（`anomaly.py`、`analyzer.detect_anomalies()`、CLI `scan` コマンド）
3. ~~Layer instability スコア~~ → 実装済み（`metrics.py`）
4. **Prompt comparison の拡張**（9.4）: `compare()` を position 対応付け付きに拡張し、top-1 一致だけでなく layer ごとの KL divergence や target rank 差を出す。
5. **Model comparison**（9.5）: 2 つの `LogitLensAnalyzer` の結果を同じ `ComparisonResult` 形式で並べる。
6. **chat template オプション**、**tuned lens**、**batch 対応**。

## 参考

- nostalgebraist, "interpreting GPT: the logit lens" (2020)
- Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens" (2023)
