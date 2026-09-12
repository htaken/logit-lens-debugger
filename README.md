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
- 2 つのテキスト間の layer ごとの比較（`compare`）: top-1 一致・top-k 集合の Jaccard overlap・KL / Jensen-Shannon divergence（全語彙分布ベース）・共通 target token の rank / probability の差
- 同じテキストに対する 2 モデル間の layer ごとの比較（`compare-models`）: 層数・語彙・tokenization の違いを吸収して同じ形で並べる
- 原文と LLM 出力の token 列 diff から「調べるべき position」を選ぶこと（`diff`）、生成 token ごとの anomaly スコア（`scan`）、top-1 trajectory の不安定さ指標（`metrics`）。いずれも観測値から計算する heuristics です

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
# 2 つのテキストを layer ごとに比較する（top-1 一致・top-k overlap・KL/JS・target rank）
uv run logit-lens compare --model Qwen/Qwen2.5-0.5B \
  --text-a $'Q: What is the capital of Japan?\nA:' \
  --text-b $'Q: What is the capital of France?\nA:' \
  --kl-threshold 0.2 --json outputs/cmp.json --html outputs/cmp.html
```

出力例（Qwen2.5-0.5B / MPS bf16、抜粋）:

```
layer  A top-1 (p)                B top-1 (p)                same  overlap   KL(A‖B)   KL(B‖A)       JS  target rank A->B
    0  ':' (1.000)                ':' (1.000)                yes     1.000    0.0000    0.0000   0.0000  -
   17  '\');?></' (0.292)         '\');?></' (0.278)         yes     1.000    0.0088    0.0090   0.0022  -
   18  ' whiteColor' (0.172)      ' whiteColor' (0.795)      yes     0.667    1.7972    1.2216   0.3031  -
   19  ';break' (0.568)           ' whiteColor' (0.588)      NO      0.667    1.2270    1.7055   0.2942  -
   22  ' Tokyo' (0.972)           ' Paris' (0.652)           NO      0.000   12.6959   16.1735   0.6901  -
   24  ' Tokyo' (0.724)           ' Paris' (0.792)           NO      0.111    9.8412   10.3566   0.6443  -

Top-1 first diverges at layer 8, last differs at layer 24 (7 of 25 layers differ).
JS first >= 0.2 at layer 18.
```

top-1 の分岐（layer 8）は中間層のノイズで、分布そのものが離れ始めるのは layer 18 からです。KL / JS を見ると「Japan / France の違いが表現に現れる深さ」が top-1 だけを見るよりはっきり分かります。

`--no-kl` を付けると KL/JS 列は `-` になり（全語彙分布を保持しないため計算コストがゼロになる）、フッタは `JS first >= ...: no distributions.` になります。`--json` は拡張された `ComparisonResult`（`layers` / `alignment` / `kind` / `metadata` を含む）を保存し、`--html` は layer ごとの KL/JS 折れ線と top-1 分岐位置のマーカーを描画します。

```bash
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

# 同じテキストを 2 つのモデルで比較する（既定は逐次ロード。語彙が同一なら KL/JS も出る）
uv run logit-lens compare-models --model-a Qwen/Qwen2.5-0.5B --model-b Qwen/Qwen2.5-0.5B-Instruct \
  --text $'Q: What is the capital of Japan?\nA:' --json outputs/model_cmp.json

# 保存済みの analyze 結果 2 つをモデルなしで比較する（KL/JS は不可）
uv run logit-lens compare-results outputs/a.json outputs/b.json
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

Qwen の tokenizer は数字を 1 桁ずつ token 化するため（`1500` → `1`,`5`,`0`,`0`）、素の diff だと `0` が 1 個 insert されただけに見えます。`expand_spans=True`（既定）は、変化した token に隣接する同じ文字クラス（数字 / かな / 漢字）の token をまとめて、span を数値全体に広げます。

- 英字は拡張の対象外です。HF の tokenizer では単語の先頭 token に空白が付く（`" sales"`）ため、英字まで対象にすると span が文全体に広がってしまいます。
- 空白始まりの token（新しい単語の先頭）をまたいでは広げません。`" 2024"` の直後の `" 150"` が別の数値として保たれるのはこのためです。
- insert / delete で片側が空の場合は、変化した内容と同じ文字クラスの隣接 token だけを吸収します。数字を消した隣の「は」を巻き込んで replace に見せることはありません。

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

出力の各 token を「lens の trajectory がどれだけ不安定か」「原文と比べてどれだけ新規か」という observable な signal でスコアリングし、調べるべき候補に優先順位を付けます。既定では出力の全 token が対象、`only_diff=True`（CLI: `--only-diff`）を指定すると前節の diff で検出した span に絞れます。

| signal | 種別 | 意味 |
|---|---|---|
| `low_final_prob` | lens 系 | 最終 layer でも target の確率が低い（`1 - p_final`） |
| `final_rank` | lens 系 | 最終 layer で target が top-1 でない（rank の log スケール、既定で rank 1024 が上限） |
| `final_entropy` | lens 系 | 最終 layer の分布のエントロピー（`log(vocab_size)` で正規化） |
| `late_emergence` | lens 系 | target が top-k に入るのが最終層直前までかかった（Layer instability の `target_first_topk_layer` / 層数） |
| `instability` | lens 系 | Layer instability の `instability_score`（次節） |
| `flip_at_end` | lens 系 | 最終 1〜2 layer で top-1 が入れ替わった |
| `in_diff` | テキスト系 | 原文との diff で検出した span 内にある |
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

`only_diff=True` は `source` が必須で（省略すると `ValueError`）、`analyze_diff` と同じ `context` / `chunk_size` を受け取ります。`source` を渡しつつ `only_diff=False`（既定）の場合は、全 token を解析しつつ diff span も計算して `in_diff` を有効にします。

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

出力例（Qwen2.5-0.5B。prompt `次の文を要約してください: 売上は150万円でした。\n要約:`、output `売上は1500万円でした。`、source `売上は150万円でした。`、抜粋）:

```
Anomaly report: 11 positions scored, 1 with score >= 0.50

  pos  out# token             score  final_p settle  flags
   27     6 '0'               0.579    0.004     17  low_final_prob,late_emergence,instability,in_diff,numeric
   21     0 '売'               0.491    0.021     24  low_final_prob,late_emergence,instability,flip_at_end
   24     3 '1'               0.489    0.822     23  late_emergence,instability,flip_at_end,in_diff,numeric
   25     4 '5'               0.415    0.931     22  late_emergence,instability,in_diff,numeric
   26     5 '0'               0.384    0.993     19  late_emergence,instability,in_diff,numeric
```

余分に増えた 2 個目の `0`（出力 token 6）が最上位に来ています。最終層でもその確率は 0.004（top-1 は `万`）で、モデルはこの桁を出したがっていなかったことが分かります。原文にある `1`,`5`,`0` は最終層の確率が高く、diff に入っているために順位が上がっているだけです。

> **注意**: anomaly スコアは「どの position を先に見るか」の優先順位付けであり、それ自体が問題の証拠ではありません。中間 layer の予測（`late_emergence` / `instability`）は特に小規模モデルで不安定なのが常態で、単独では判断材料になりません。`novel` / `entity_like` は文字クラスに基づく粗い近似であり、本物の NER ではありません。

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

#### 2 つのテキストの比較

```python
cmp = analyzer.compare(
    text_a, text_b,
    token_index=-1,        # text_a の position（負数は末尾から）
    token_index_b=None,    # text_b の position。省略時は token_index と同じ
    top_k=10,
    target_token=None,     # 両方のテキストで共通に追跡する token（省略時は各テキストの次 token）
    keep_log_probs=True,   # KL / JS 計算用に全語彙分布を保持する（既定 on）
)

cmp.top1_agreement          # layer ごとに top-1 が一致したか (list[bool])
cmp.first_divergence_layer  # 最初に分岐した layer（None なら全 layer 一致）
cmp.divergence_layers       # 分岐した layer の一覧
cmp.kind                    # "prompt"（次節のモデル間比較では "model"）
cmp.alignment                # [(layer_a, layer_b), ...]  prompt モードでは恒等対応
```

`cmp.layers[i]`（`LayerComparison`、`alignment[i]` に対応）には layer ごとの詳細指標が入ります。

```python
lc = cmp.layers[10]
lc.layer_a, lc.layer_b     # 比較した layer index（prompt モードでは同じ値）
lc.top1_same                # 既存の top-1 一致
lc.topk_overlap              # top-k 集合の Jaccard 係数
lc.kl_a_b, lc.kl_b_a, lc.js  # KL(A‖B) / KL(B‖A) / Jensen-Shannon divergence（自然対数）。
                             # keep_log_probs=False や JSON から復元した結果では None
lc.target_rank_a, lc.target_rank_b        # 共通 target の rank
lc.target_prob_a, lc.target_prob_b        # 共通 target の probability
lc.target_rank_log_delta    # log2(rank_b) - log2(rank_a)
lc.target_logprob_delta     # ln(p_b) - ln(p_a)

cmp.first_layer_kl_above(0.1, direction="js")   # JS が閾値を超える最初の layer（direction: a_b/b_a/js/max）
cmp.first_layer_target_rank_differs()           # 共通 target の rank が最初に食い違う layer
```

`keep_log_probs=True`（既定）は 1 position あたり `(層数+1) × vocab_size` の float32 を CPU に保持します。Qwen2.5-0.5B（25 層、vocab 151936）なら約 15 MB で、`compare()` が返る前に解放されるので実害はありません。KL/JS が不要なら `keep_log_probs=False` を渡すと、その分の計算とメモリ確保自体を省けます（`lc.kl_a_b` などは `None` になります）。

#### モデル間比較（Model comparison）

同じ入力テキストを **2 つの異なるモデル**（例: `Qwen2.5-0.5B` vs `Qwen2.5-1.5B`、Qwen vs Llama）に通し、`analyzer.compare()` と同じ `ComparisonResult`（`kind="model"`）で layer ごとに並べます。

```python
from logit_lens_debugger import compare_model_names

cmp = compare_model_names(
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B",
    text="Q: What is the capital of Japan?\nA:",
    token_index=-1,         # または char_index=... （下記）
    top_k=10,
    target_token=None,      # str のみ許可。int の token id はモデルごとに語彙が違うため TypeError
    keep_log_probs=True,    # 語彙が同一の場合のみ KL/JS に使われる
    sequential=True,        # 既定: A を解析・解放してから B をロードする
)
```

`sequential=True`（既定）は A をロード → 解析 → `del` + `gc.collect()`（CUDA/MPS ならキャッシュも解放）→ B をロード、の順で進みます。M2 16GB のような環境で 1.5B クラスのモデルを 2 つ同時にロードするのは厳しいため、既定では 2 つのモデルが同時にメモリへ載ることはありません。同一語彙判定（後述）はこのとき互いのモデルを見ずに、それぞれの `vocab_fingerprint()` だけで行います。両方のモデルを同時にロードしたい場合（例: 実験的に速度を優先する、または同じ tokenizer インスタンスを共有させて `vocab_identical` の高速パスを使わせたい場合）は `sequential=False` を渡します。既にロード済みの 2 つの `LogitLensAnalyzer` を比較したいだけなら `compare_models(analyzer_a, analyzer_b, text, ...)` を直接使えます（`compare_model_names` はモデル名からのロードを足しただけの薄いラッパです）。

2 つのモデルを比べるときに生じる 3 つの不一致は、次のように吸収します。

| 不一致 | 対応 |
|---|---|
| 層数が違う | 相対深さ（`layer / num_layers`）で最も近い層同士を対応付ける（`align_layers()`、補間はしない） |
| 語彙が違う | `token_id` ではなく **decode した文字列** で `top1_same` / `topk_overlap` を比較する。KL/JS は語彙が完全に一致するときだけ計算し、それ以外は `None` |
| tokenization が違い position がずれる | 既定の末尾（`token_index=-1`）なら通常は両者で揃う。文中の特定位置を見たい場合は **`char_index`**（文字オフセット）を指定すると、各モデルの tokenizer が「その文字を含む token」に変換する（fast tokenizer なら `offset_mapping` で厳密、低速 tokenizer は累積デコード長からの近似） |

```python
cmp.kind                      # "model"
cmp.alignment                  # [(layer_a, layer_b), ...]  層数が違えば恒等対応にならない
cmp.metadata["same_vocab"]     # True なら KL/JS が使え、token_id で比較されている
cmp.metadata["model_a"], cmp.metadata["model_b"]
cmp.metadata["char_index"]     # 指定した場合のみ非 None
```

語彙が同一かどうかは `vocab_identical(adapter_a, adapter_b)` で判定します（同じ `tokenizer` オブジェクトを共有しているか、そうでなければ `adapter.vocab_fingerprint()`— 語彙全体の sha256 — が一致するか）。

CLI からも同じことができます。

```bash
uv run logit-lens compare-models \
  --model-a Qwen/Qwen2.5-0.5B --model-b Qwen/Qwen2.5-1.5B \
  --text $'Q: What is the capital of Japan?\nA:' \
  --json outputs/model_cmp.json --html outputs/model_cmp.html
```

```
A: Qwen/Qwen2.5-0.5B  (24 layers, token_index=10, analyzed ':')
B: Qwen/Qwen2.5-1.5B  (28 layers, token_index=10, analyzed ':')
vocab: identical (KL/JS available)

 A ly  B ly  A top-1 (p)                B top-1 (p)                same  overlap   KL(A‖B)   KL(B‖A)       JS  target rank A->B
    0     0  ' __'                (0.010)  ' __'                (0.010)  yes     1.000    0.0000    0.0000   0.0000  -
   12    14  ' Tokyo'              (0.340)  ' Tokyo'              (0.560)  yes     0.700    0.4100    0.3800   0.1900  -
   24    28  ' Tokyo'              (0.720)  ' Tokyo'              (0.910)  yes     0.900    0.0900    0.1100   0.0500  -
```

`--token-index` と `--char-index` は排他です。モデルの読み込みを両方同時に行いたい場合は `--concurrent` を付けます（既定は逐次ロード）。語彙が異なるモデル同士（Qwen vs Llama など）を比べると `vocab: different (compared by decoded text; KL/JS n/a)` と表示され、KL/JS 列は `-` になります。

モデルをロードせず、`analyze --json` で保存済みの 2 つの結果だけを比べることもできます（KL/JS は保存されていないため常に利用不可）。

```bash
uv run logit-lens compare-results outputs/qwen_0.5b.json outputs/qwen_1.5b.json --json outputs/model_cmp.json
```

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
LogitLensAnalyzer (analyzer.py)      analyze() / analyze_positions() / analyze_diff() / detect_anomalies() / compare()
        │                            compare_models()  … token index 正規化、target 解決、結果組み立て
        ▼
LogitLensEngine (engine.py)          hidden → (final norm) → lm_head → softmax → top-k / target rank / entropy
        │                            run_detailed(keep_log_probs=True) で全語彙 log-prob も保持（比較用）
        ▼
ModelAdapter (adapters/)             tokenize / get_hidden_states / apply_final_norm / project_to_vocab / decode / char_to_token
   ├── HFCausalLMAdapter             HF decoder-only 共通実装（hook による hidden state 取得）
   ├── QwenAdapter                   model_type: qwen2, qwen3
   └── LlamaAdapter                  model_type: llama
        │
        ▼
Hugging Face Transformers

結果型 (types.py): LogitLensResult / MultiPositionResult / AnomalyReport / ComparisonResult ── to_dict / to_json で JSON 往復
torch 非依存の後処理: diff.py（差分 → position）/ metrics.py（instability）/ anomaly.py（スコアリング）── 保存済み JSON から再計算可能
比較: comparison.py（KL / JS / top-k overlap / 層対応付け / 語彙判定。KL / JS の計算だけ torch を使う）
出力: CLI テキスト (visualization/trajectory.py, comparison.py) ── Plotly HTML (visualization/heatmap.py, trajectory.py, comparison.py)
```

モデル固有のコードは `adapters/` に閉じており、engine / analyzer / visualization は `ModelAdapter` インターフェースだけを使います。

主要ファイル:

| ファイル | 役割 |
|---|---|
| `src/logit_lens_debugger/types.py` | `TokenPrediction` / `LayerResult` / `LogitLensResult` / `ComparisonResult` / `LayerComparison` / `LensMode`（dataclass、JSON 往復可） |
| `src/logit_lens_debugger/adapters/base.py` | `ModelAdapter` ABC と `HFCausalLMAdapter`、device / dtype 解決 |
| `src/logit_lens_debugger/adapters/qwen.py`, `llama.py` | ファミリー別 adapter（`supported_model_types` を宣言） |
| `src/logit_lens_debugger/adapters/registry.py` | `config.model_type` → adapter の解決、`load_adapter` / `register_adapter` |
| `src/logit_lens_debugger/engine.py` | Logit Lens 本体（position ごとに別 target も指定可能） |
| `src/logit_lens_debugger/analyzer.py` | 高レベル API（`analyze_positions` / `analyze_generation` / `analyze_diff` / `compare_models()` / `compare_model_names()` を含む） |
| `src/logit_lens_debugger/metrics.py` | `compute_instability()` → `InstabilityMetrics`（torch 非依存、JSON からも計算可） |
| `src/logit_lens_debugger/diff.py` | `diff_token_sequences()` / `select_positions()` / `token_class()`（torch 非依存、diff から対象 position を選ぶ） |
| `src/logit_lens_debugger/anomaly.py` | `score_position()` / `detect_anomalies()` → `AnomalyReport`（torch 非依存、diff / instability を合成して token をスコアリング） |
| `src/logit_lens_debugger/comparison.py` | `compare_results()` / `compare_layer()` / KL・JS divergence / `topk_overlap()` / `align_layers()` / `vocab_identical()`（`analyzer.compare()` / `compare_models()` の実体。KL/JS のみ torch 使用） |
| `src/logit_lens_debugger/visualization/` | ヒートマップ・trajectory・comparison（Plotly）とテキスト整形 |
| `src/logit_lens_debugger/visualization/comparison.py` | `format_comparison()`（テキスト表）と `format_model_comparison_header()`、`build_comparison_figure()` / `export_comparison_html()`（layer ごとの KL/JS 折れ線） |
| `src/logit_lens_debugger/cli.py` | typer CLI（`analyze` / `compare` / `compare-models` / `compare-results` / `diff` / `scan` / `metrics` / `visualize`） |
| `tests/` | 極小ランダム初期化モデルによるオフラインテスト + opt-in の実モデル統合テスト |
| `examples/basic_analysis.py` | API の最小サンプル |
| `examples/compare_models.py` | 2 モデル比較（`compare_model_names`、逐次ロード）のサンプル |

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
- `LogitLensEngine.run()` は複数 position を一度の forward で処理します（`analyze_positions` / `analyze_diff` / `scan` が利用）。position が多い場合は `chunk_size` で forward を分割し、layer ごとに作る `[positions, vocab]` の logits のピークを抑えられます。
- `run_detailed(keep_log_probs=True)` は比較のために全語彙 log-prob を CPU に退避します。1 position あたり `(層数+1) × vocab` の float32 で、Qwen2.5-0.5B なら約 15 MB です。既定の `run()` では保持しません。
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
- diff の span 拡張（数字 / かな / 漢字、空白始まり token の単語境界、insert / delete の吸収規則）と position 選択、複数 position 解析と単一解析の一致（float は batched matmul の丸め差を許容）
- instability 指標の厳密値（switch / oscillation / settle layer / late switch）、anomaly signal の単調性・省略規則・重みの正規化
- KL / JS の手計算との一致と非負性、top-k overlap、層対応付け（`align_layers`）と語彙判定（`vocab_identical`）、旧形式 `ComparisonResult` JSON の読み込み、2 モデル比較の逐次ロード順

## 現在の制約

- 1 回の解析は 1 テキスト・batch size 1 です（ただし `analyze_positions` / `analyze_generation` / `analyze_diff` なら、同じテキストの複数 position を 1 回の forward でまとめて解析できます）。
- chat template は適用しません。Instruct モデルを解析する場合は、実際にモデルへ渡した形式（template 適用後の文字列）を `--text` に渡してください。
- 中間 layer の予測は解釈しづらいことが多く（前述）、tuned lens のような学習済み射影は未実装です。
- Qwen の tokenizer は数字を 1 桁ずつ token 化します（`150` → `1`,`5`,`0`）。数値の書き換えを追う場合は複数 position をまたいで見る必要があります（`analyze_diff` は同じ文字クラスの隣接 token をまとめる `expand_spans` でこれに対応しています）。
- diff（`analyze_diff` / `logit-lens diff`）は tokenizer の token 列ベースの比較です。出力が原文の大幅な言い換え・順序入替の場合は span がほぼ全域になり、絞り込みの効果が薄れます。文字オフセットベースの diff は未実装です。
- Llama は実 checkpoint での検証を行っていません。MoE や非標準レイアウト（`model.model.norm` / `lm_head` 以外に norm や head があるモデル）は未対応です。
- 可視化は HTML export のみで、ライトテーマ固定です。
- モデル間比較（`compare_models` / `compare_model_names`）の層対応付け（`align_layers`）は相対深さに基づく最近傍のヒューリスティックで、補間は行いません。KL/JS は語彙が完全に一致するモデル同士でしか計算できません（`vocab_identical` が False なら常に `None`）。
- `char_to_token`（`char_index` による position 指定）は fast tokenizer なら `offset_mapping` で厳密ですが、低速 tokenizer や `ToyTokenizer` では累積デコード長からの近似になります。

## 次に実装すると良い機能

設計上は以下を後付けしやすくしています。

1. ~~Input / Output diff → 対象 position の自動選択~~ → 実装済み（`diff.py`、`analyzer.analyze_diff()` / `analyze_generation()` / `analyze_positions()`、CLI `diff` コマンド）
2. ~~自動 anomaly detection~~ → 実装済み（`anomaly.py`、`analyzer.detect_anomalies()`、CLI `scan` コマンド）
3. ~~Layer instability スコア~~ → 実装済み（`metrics.py`）
4. ~~Prompt comparison の拡張~~ → 実装済み（`comparison.py`、`analyzer.compare()` に `token_index_b` / KL・JS / top-k overlap / target rank 差、CLI `compare` の拡張オプション）
5. ~~Model comparison~~ → 実装済み（`analyzer.compare_models()` / `compare_model_names()`、`comparison.align_layers()` / `vocab_identical()`、CLI `compare-models` / `compare-results` コマンド）
6. **chat template オプション**（[#1](https://github.com/htaken/logit-lens-debugger/issues/1)）、**tuned lens**（[#2](https://github.com/htaken/logit-lens-debugger/issues/2)）、**batch 対応**（[#3](https://github.com/htaken/logit-lens-debugger/issues/3)）。着手の条件と要件は各 issue を参照。

## 参考

- nostalgebraist, "interpreting GPT: the logit lens" (2020)
- Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens" (2023)
