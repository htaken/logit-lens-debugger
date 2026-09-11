# 追加機能 1〜5 の要件と実装方針

README「次に実装すると良い機能」の 1〜5（`prompt.md` の要件 9.1〜9.5）を実装するための設計メモ。

| # | 機能 | 要件番号 | 依存 |
|---|---|---|---|
| 1 | Input / Output diff → 対象 position の自動選択 | 9.1 | なし |
| 2 | 自動 anomaly detection | 9.2 | 1, 3 |
| 3 | Layer instability スコア | 9.3 | なし |
| 4 | Prompt comparison の拡張（position 対応付け・KL・target rank 差） | 9.4 | なし（engine 拡張） |
| 5 | Model comparison | 9.5 | 4 |

推奨実装順: **3 → 1 → 2 → 4 → 5**（純関数から始めて、モデル依存部分を後に回す）。

---

## 0. 現状の土台と、そのまま使えるもの

- `LogitLensEngine.run(input_ids, positions, top_k, target_token_id)` は複数 position を 1 forward で処理できる。ただし `target_token_id` は全 position 共通の 1 個なので、**position ごとに別 target** を持てるよう拡張が必要（機能 1, 2）。
- `LayerResult.entropy` と `layer.target.rank / probability` は既に記録されている（機能 2, 3 の材料）。
- `ComparisonResult` は top-1 一致のみ。`from_dict` / `load_json` がない（機能 4, 5 で追加）。
- engine は top-k 取得直後に logits を解放する。KL を出すには「要求された position だけ」全語彙分布を保持する口が必要（機能 4）。
- `visualize` コマンドは JSON からモデル無しで再描画できる。新しい指標も **JSON から再計算できる純関数** にすると同じ運用ができる。
- テスト基盤: 極小ランダム初期化の Qwen2（3 層）/ Llama（2 層）と whitespace 区切りの `ToyTokenizer`。`VOCAB` に `sales were 150 1500 yen` が入っており、diff のテストにそのまま使える。

---

## 1. 共通方針

### 1.1 追加モジュール

```
src/logit_lens_debugger/
├── diff.py               # 機能 1: token 列 diff と position 選択（torch 非依存）
├── metrics.py            # 機能 3: LogitLensResult → InstabilityMetrics（torch 非依存）
├── anomaly.py            # 機能 2: diff + metrics を合成してスコアリング（torch 非依存）
├── comparison.py         # 機能 4, 5: 比較ロジック（analyzer.compare の実体を移す。KL のみ torch 使用）
├── types.py              # 新 dataclass の追加
├── analyzer.py           # analyze_positions / analyze_generation / analyze_diff / compare_models
├── engine.py             # run_detailed()（分布保持）と position 別 target
├── cli.py                # diff / scan / metrics / compare-models / compare-results、compare の拡張
└── visualization/
    └── comparison.py     # 任意: layer ごとの KL・一致率の折れ線
```

### 1.2 設計原則

- **モデル固有コードは `adapters/` に閉じたまま。** 新モジュールは `ModelAdapter` か `LogitLensResult` にしか依存しない。
- **`diff.py` / `metrics.py` / `anomaly.py` の中核は torch を import しない純関数。** 保存済み JSON から再計算でき、`ToyTokenizer` だけでテストできる。
- **後方互換。** 既存 public API と JSON スキーマは変えない。追加はすべて default 付き optional フィールド。
- **メモリ原則を維持。** 全語彙分布を保持するのは「比較のため明示的に要求した position」だけで、CPU fp32、`(層数+1) × V`。Qwen2.5-0.5B なら 25 × 151936 × 4 B ≈ 15 MB / position。
- **解釈上の注記を必ず付ける。** 小規模モデルの前半〜中盤 layer で top-1 が不安定なのは常態であり、スコアは「調べる優先順位を付けるための heuristics」で「原因の証拠」ではない。README の該当セクションを各機能で更新する。

### 1.3 追加する型（`types.py`）

```python
@dataclass(frozen=True)
class DiffSpan:                      # 機能 1
    kind: Literal["replace", "insert", "delete"]
    source_range: tuple[int, int]    # source token index [start, end)
    output_range: tuple[int, int]    # output token index [start, end)
    source_text: str
    output_text: str

@dataclass
class PositionResult:                # 機能 1, 2: 1 position 分（input_tokens は持たない）
    token_index: int
    layers: list[LayerResult]
    target_token: str | None
    target_token_id: int | None
    diff_span: DiffSpan | None = None

@dataclass
class MultiPositionResult:           # 機能 1, 2: 同一文脈・複数 position
    model_name: str
    lens_mode: LensMode
    text: str
    input_tokens: list[str]
    input_token_ids: list[int]
    prompt_length: int               # output token が始まる位置
    top_k: int
    positions: list[PositionResult]
    diff_spans: list[DiffSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_results(self) -> list[LogitLensResult]: ...   # 既存の可視化を再利用するため

@dataclass(frozen=True)
class InstabilityMetrics: ...        # 機能 3（§2 参照）

@dataclass
class PositionAnomaly: ...           # 機能 2（§4 参照）
@dataclass
class AnomalyReport: ...

@dataclass(frozen=True)
class LayerComparison: ...           # 機能 4（§5 参照）

@dataclass
class ComparisonResult:              # 既存 5 フィールドは維持し、以下を追加
    ...
    layers: list[LayerComparison] = field(default_factory=list)
    alignment: list[tuple[int, int]] = field(default_factory=list)   # (layer_a, layer_b)
    kind: Literal["prompt", "model"] = "prompt"
    metadata: dict[str, Any] = field(default_factory=dict)
    # from_dict / load_json を追加
```

`MultiPositionResult` を新設する理由: 議事録級の長い文脈で position ごとに `LogitLensResult` を作ると `input_tokens` が position 数だけ重複する。`to_results()` で必要な時だけ展開する。

---

## 2. 機能 3: Layer instability スコア（要件 9.3）

### 要件

- `LogitLensResult` から次を算出する。モデルは不要で、JSON からも計算できる。

| 指標 | 定義 |
|---|---|
| `top1_switches` | 隣接 layer 間で top-1 の token_id が変わった回数 |
| `oscillations` | 一度離れた token に top-1 が戻った回数（`150 → 1500 → 150 → 1500` は 2） |
| `distinct_top1` | top-1 に現れた token の種類数 |
| `settle_layer` | その層以降ずっと最終層の top-1 と一致する最初の層。最終層で初めて一致なら `num_layers` |
| `late_switches` | 後半（既定: 末尾 1/3 の layer）での switch 回数 |
| `instability_score` | `top1_switches / (評価 layer 数 - 1)` を基本に、oscillation を加点して [0, 1] にクリップ |
| target あり | `target_first_top1_layer`, `target_first_topk_layer`, `target_rank_log_mean`, `target_rank_log_std`, `target_final_rank`, `target_final_prob` |

- `from_layer`（既定 1）で評価開始層を指定する。layer 0 は embedding 出力で入力 token 自身が p≈1 になる自明な層なので既定で除外する。
- CLI: `analyze` の末尾に 1 行サマリを追加、`logit-lens metrics result.json` で JSON から表示。

### 実装方針

- `metrics.py` に `compute_instability(result, from_layer=1, late_fraction=1/3) -> InstabilityMetrics`。`LogitLensResult.instability(**kw)` を便宜メソッドとして追加。
- `InstabilityMetrics` は frozen dataclass、`to_dict` 付き。`LogitLensResult.to_dict` には含めない（派生値であり再計算できる）。CLI の `--json` では `metadata["instability"]` に書き込む選択肢もあるが、真の値は常に再計算で得られるようにする。
- rank の統計は `log2(rank)` で取る（rank は桁が大きくばらつくため）。
- テスト: top-1 列を指定して `LayerResult` を組み立てる helper を `tests/` に作り、switches / oscillations / settle_layer / late_switches を厳密値で検証。`from_layer` の効果、target なし時に target 系が `None` になることを確認。

---

## 3. 機能 1: Input / Output diff → 対象 position の自動選択（要件 9.1）

### 要件

- 入力
  - `source`: 原文（文字起こし結果など）
  - `output`: LLM 出力。文字列、または生成時の token id 列（あればこちらが最も正確）
  - `prompt`: モデルに実際に渡した文字列（指示 + 原文）。省略時は `source` を文脈として使う
- 出力: 差分 span の一覧と、解析対象 position（`token_index`）+ target token の組、その lens 結果（`MultiPositionResult`）。
- 差分の単位は **モデルの tokenizer による token 列**。`source` と `output` をそれぞれ tokenize し、token 文字列の列同士を `difflib.SequenceMatcher(autojunk=False)` で比較する。
  - 解析単位が token なので変換が不要
  - 日本語のように空白がない文でも動く
  - fast tokenizer の `offset_mapping` に依存しない（`ToyTokenizer` でテスト可能）
- Qwen は数字を 1 桁ずつ token 化するので、`150 → 1500` は `0` 1 個の `insert` になる。**span を連続する同一文字クラス（数字 / 英字 / かな / 漢字）の token 全体に拡張**する（`expand_spans=True`、既定 on）。
- `context`（既定 0）で span の前後 n token も解析対象に加える。
- `delete`（原文にあり出力にない）は output 側に position がない。削除箇所直後の output token を対象に含める（`include_after_delete=True`、既定 on。「省略した直後に何を出そうとしていたか」を見る）。
- `equal` は対象外。

### 実装方針

**`diff.py`（純 Python）**

```python
def diff_token_sequences(
    source_tokens: list[str], output_tokens: list[str], expand_spans: bool = True
) -> list[DiffSpan]: ...

def select_positions(
    spans: list[DiffSpan], prompt_length: int, output_length: int,
    context: int = 0, include_after_delete: bool = True,
) -> list[tuple[int, DiffSpan | None]]:
    """output token j を調べる position は prompt_length + j - 1、target は ids[prompt_length + j]。"""
```

**`engine.py`**

- `run()` の `target_token_id: int | Sequence[int | None] | None` を position ごとに受けられるようにする。実装は `logits[:, target_id]` を `logits.gather(-1, ids[:, None])` に置き換えるだけ。`None` の position は target を出さない。
- 単一 int を渡した場合の挙動は現状と同一（既存テストで固定）。

**`analyzer.py`**

```python
def analyze_positions(self, text_or_ids, positions, top_k=10, targets=None) -> MultiPositionResult
def analyze_generation(self, prompt: str, output: str | list[int], top_k=10,
                       positions: Sequence[int] | None = None) -> MultiPositionResult
def analyze_diff(self, source: str, output: str | list[int], prompt: str | None = None,
                 top_k=10, context=0, expand_spans=True) -> MultiPositionResult
```

- 文脈の id 列は **`prompt` と `output` を別々に tokenize して連結**する。`prompt + output` を一括 tokenize すると境界で token が結合され、生成時の id 列と食い違うことがある。`output` を id 列で渡せば生成ログをそのまま再現できる。
- `prompt_length = len(prompt_ids)` を `MultiPositionResult` に記録する。
- forward は 1 回。hook 内で対象 position だけを抽出する既存の仕組みで、長い transcript でも hidden state の保持量は position 数に比例する。

**CLI**

```bash
logit-lens diff --source "売上は150万円でした" --output "売上は1500万円でした" \
  [--prompt "..." | --prompt-file prompt.txt] [--context 1] [--json out.json] [--html-dir out/]
```

出力: span テーブル（kind / 原文 / 出力 / 対象 position）→ position ごとの要約（target の最終 p と rank、target が初めて top-1 になった layer、`settle_layer`、`instability_score`）。`--html-dir` は `to_results()` で展開した各 position のヒートマップを書き出す。

**テスト**

- `sales were 150 yen` vs `sales were 1500 yen`: `replace` 1 件、output_range が `1500` を指す、position は `prompt_length + 2`、target が `1500`。
- 数字拡張: 1 桁 token を模した vocab を追加して `1 5 0` vs `1 5 0 0` が span 全体に広がること。
- `prompt` あり / なしでの `prompt_length` と position のずれ。
- `delete` の直後 token が含まれること、`context` の前後拡張、`equal` のみなら空。
- `run()` の position 別 target: 単一 int と `[id] * P` で結果が一致、`None` 混在で該当 position の `target` が `None`。

### 制約

- 出力が原文の大幅な言い換え・順序入替だと diff がほぼ全域になる。その場合は機能 2 のスコアで絞る。
- 文字オフセットベースの diff（`offset_mapping` 利用）は v2 候補。

---

## 4. 機能 2: 自動 anomaly detection（要件 9.2）

### 要件

- 入力: `(prompt, output, source?)`、または既存の `MultiPositionResult`。既定は **output の全 token** を対象にし、`--only-diff` で機能 1 の span 内に限定できる。
- 各 output token について次の signal を [0, 1] で算出する。

| 種別 | signal | 意味 |
|---|---|---|
| lens 系 | `low_final_prob` | `1 - p_final(target)`。最終層でも自信がない |
| lens 系 | `final_rank` | 最終層で target が top-1 でない（rank の log スケール） |
| lens 系 | `final_entropy` | 最終層エントロピー / `log V` |
| lens 系 | `late_emergence` | target が初めて top-k に入る相対深さ（最終層直前まで出てこない token を拾う） |
| lens 系 | `instability` | 機能 3 の `instability_score` |
| lens 系 | `flip_at_end` | 最終 1〜2 層で top-1 が入れ替わった |
| テキスト系 | `in_diff` | 機能 1 の span 内 |
| テキスト系 | `numeric` | 数字 token |
| テキスト系 | `novel` | source の token 列に一度も現れない content token（記号・空白・機能語相当を文字クラスで除外） |
| テキスト系 | `entity_like` | novel かつ 先頭大文字 / カタカナ・漢字連続 |

- 合成スコア `anomaly_score = Σ w_i · s_i`（重みは `AnomalyConfig` で変更可、既定値を README に明記）。スコア順に並べ、`threshold` または `top_n` でレポートする。
- 要件 9.2 の「原文にない固有名詞」「entity の変更」「hallucination 候補」は `novel` / `entity_like` / `in_diff` の heuristics で近似する。**NER 本体はスコープ外**。外部 NER を使う場合は `entity_spans: list[tuple[int, int]]`（output の文字範囲）を注入できる hook を用意する。
- 出力: `AnomalyReport(positions: list[PositionAnomaly], config, model_name, ...)`、JSON 保存可、CLI テーブル。

```python
@dataclass
class PositionAnomaly:
    token_index: int
    token: str                       # 実際に生成された token
    score: float
    signals: dict[str, float]        # 上表の各 signal
    flags: list[str]                 # 閾値を超えた signal 名
    diff_span: DiffSpan | None
    instability: InstabilityMetrics
```

### 実装方針

- `anomaly.py`（純 Python）に `score_position(position_result, text_flags, config)` と `detect_anomalies(multi_result, source_tokens=None, config=None) -> AnomalyReport`。
- 全 token 解析は engine の複数 position 一括 forward を使う。層ごとに `[P, V]` の logits を作るので、P が数百なら層あたり数十〜百 MB になる。`chunk_size`（既定 64）で positions を分割し、複数回 forward する口を `analyze_positions` に付ける（forward の再計算コストと logits のピークメモリのトレードオフ。既定はメモリ優先）。
- `AnomalyConfig` は dataclass（重み・閾値・`late_fraction`・`chunk_size`）。JSON に含めて再現可能にする。
- CLI: `logit-lens scan --prompt ... --output ... [--source ...] [--top 20] [--only-diff] [--json report.json]`。テーブル列: position / token / score / flags / final p / settle layer。
- テスト
  - 手作り `PositionResult` でスコアの単調性（final p が低いほど高い、target が遅く出るほど高い）。
  - `novel` / `numeric` / `entity_like` の判定。`novel` は `ToyTokenizer` で `osaka` が source に無ければ立つ。
  - `chunk_size` を 1 と全体にして結果が一致すること（分割の正しさ）。
  - config の重みを 0 にするとその signal が寄与しないこと。

### 注意

- README に「anomaly スコアは調査の優先順位付けであり、中間 layer の予測は特に小規模モデルでは不安定なので単独では判断しない」と明記する。

---

## 5. 機能 4: Prompt comparison の拡張（要件 9.4）

### 要件

- 現状は同一モデル・同じ `token_index`・top-1 一致のみ。以下を追加する。
  - **position 対応付け**: `token_index_a` / `token_index_b` を個別に指定できる（既定は両方 -1）。prompt A/B の長さが違っても末尾なら揃うので、まずこれで十分。文字オフセット指定は機能 5 と共通化（§6）。
  - **layer ごとの指標** `LayerComparison`

    | フィールド | 内容 |
    |---|---|
    | `layer_a`, `layer_b` | 比較した layer index（機能 5 では異なりうる） |
    | `top1_same` | 既存の top-1 一致 |
    | `topk_overlap` | top-k 集合の Jaccard 係数 |
    | `kl_a_b`, `kl_b_a`, `js` | 全語彙分布の KL（両方向）と JS divergence。分布が無い場合は `None` |
    | `target_rank_a/b`, `target_prob_a/b` | 共通 target の rank / p |
    | `target_rank_log_delta`, `target_logprob_delta` | target の差分 |

  - **分岐 layer の定義を追加**: 既存の top-1 分岐に加え、`first_layer_kl_above(threshold)`、`first_layer_target_rank_differs()` をメソッドとして提供。
  - **target**: `target_token` を指定すれば両方で同じ token を追跡（「本来出てほしかった token」）。省略時は各 text の次 token（存在すれば）。
- 既存の `top1_agreement` / `first_divergence_layer` / `last_divergence_layer` / `divergence_layers` は維持。

### 実装方針

**`engine.py`**

```python
@dataclass
class EngineOutput:
    layers: list[list[LayerResult]]              # 既存 run() の戻り値
    log_probs: torch.Tensor | None               # [L+1, P, V] CPU float32、keep_log_probs=True のとき

def run_detailed(self, input_ids, positions, top_k=10, target_token_id=None,
                 keep_log_probs=False) -> EngineOutput
def run(self, ...) -> list[list[LayerResult]]:   # run_detailed(...).layers の薄いラッパ
```

- `keep_log_probs=True` のときだけ層ごとの `log_probs` を `.cpu()` に退避してから解放する。既定 off なので既存のメモリ特性は変わらない。

**`comparison.py`**

- `compare_results(res_a, res_b, log_probs_a=None, log_probs_b=None, alignment=None) -> ComparisonResult`。`analyzer.compare()` はこれを呼ぶだけにする。
- KL / JS は `torch` で計算（`exp(lp_a) * (lp_a - lp_b)` の和）。JSON から復元した結果同士では `kl=None` になり、`topk_overlap` と target 系は再計算できる。
- `ComparisonResult` に `from_dict` / `load_json` を追加。

**CLI `compare` の追加オプション**

```bash
logit-lens compare --text-a ... --text-b ... [--token-index-b -1] [--target-token " Tokyo"] \
  [--kl/--no-kl] [--json cmp.json] [--html cmp.html]
```

テーブルに `KL(A‖B)` / `overlap` / `target rank A→B` 列を追加。`--html` は layer を横軸に KL と top-1 一致を描く折れ線（`visualization/comparison.py`、任意）。

**テスト**

- 同一 text: 全層で `kl=0`、`topk_overlap=1`、既存の一致テストと整合。
- `run_detailed(keep_log_probs=True).log_probs` の `topk` が `layers` の top-k と一致し、`exp().sum() ≈ 1`。
- 手計算の KL（2 層ぶんを `log_softmax` から算出）と一致。
- `token_index_b` を個別指定した場合の position。
- `ComparisonResult` の JSON 往復。

---

## 6. 機能 5: Model comparison（要件 9.5）

### 要件

- 同じ入力テキストを 2 つのモデル（例: `Qwen2.5-0.5B` vs `Qwen2.5-1.5B`、Qwen vs Llama）で解析し、機能 4 と同じ `ComparisonResult`（`kind="model"`）で並べる。
- 3 つの不一致を吸収する。

| 不一致 | 方針 |
|---|---|
| 層数が違う | **相対深さ**（`layer / num_layers`）で A の各層に B の最近傍層を対応付け、`alignment: list[(layer_a, layer_b)]` として記録。両方の生 index を `LayerComparison` に持つ |
| 語彙が違う | `token_id` ではなく **decode した文字列** で比較（`top1_same`, `topk_overlap` は文字列集合）。KL は語彙が同一（vocab size と全 id→文字列が一致）のときだけ計算し、それ以外は `None` |
| tokenization が違い position がずれる | 既定の末尾（`-1`）なら両者で揃う。文中を見る場合は **`char_index`（文字オフセット）** を指定し、各 tokenizer で「その文字を含む token」に変換する |

- target は文字列で受け、各 tokenizer で resolve する（複数 piece なら先頭 piece。既存の警告と同じ扱い）。
- 2 モデルを同時にロードすると M2 16 GB のような環境では厳しい場合があるため、**逐次ロード**（A を解析 → 解放 → B を解析）を既定にする。JSON 保存済みの結果同士を比較する経路（`compare-results a.json b.json`）も用意し、KL 以外はすべて算出できるようにする。

### 実装方針

- `adapters/base.py` の `ModelAdapter` に `char_to_token(text, char_index) -> int` を追加。`HFCausalLMAdapter` は fast tokenizer の `return_offsets_mapping` を使い、非 fast tokenizer や `ToyTokenizer` では「token を順に decode して累積文字数で近似」に fallback。
- `comparison.py`
  - `align_layers(num_layers_a, num_layers_b) -> list[tuple[int, int]]`（相対深さの最近傍）。
  - `vocab_identical(adapter_a, adapter_b) -> bool`（同一 tokenizer オブジェクト、または vocab size と `get_vocab()` の一致）。
  - `compare_results` に文字列比較モードを追加（`compare_by="token_id" | "text"`。`kind="model"` では `"text"`）。
- `analyzer.py`: モジュール関数 `compare_models(model_a, model_b, text, token_index=-1, char_index=None, top_k=10, target_token=None, lens_mode=..., device=..., dtype=..., sequential=True) -> ComparisonResult`。`sequential=True` のときは A を解析して結果と log_probs を保持したまま `del` してから B をロードする。`LogitLensAnalyzer` インスタンス 2 つを渡す形も受け付ける。
- CLI

```bash
logit-lens compare-models --model-a Qwen/Qwen2.5-0.5B --model-b Qwen/Qwen2.5-1.5B \
  --text $'Q: What is the capital of Japan?\nA:' [--char-index 30] [--target-token " Tokyo"] [--json]
logit-lens compare-results a.json b.json [--html]
```

出力テーブルは `layer A (rel) | layer B | A top-1 | B top-1 | same | overlap | KL | target rank A/B`。

- テスト
  - tiny qwen（3 層 → 4 行）と tiny llama（2 層 → 3 行）で `align_layers` の対応が期待どおり（`[(0,0),(1,1),(2,1),(3,2)]` など、端点が必ず対応すること）。
  - 同一 `ToyTokenizer` なら `vocab_identical=True` で KL が出る。語彙をずらした別 `ToyTokenizer` を作り、`kl=None` かつ文字列比較は動くこと。
  - `char_index` → token index の fallback 実装が whitespace tokenizer で正しいこと。
  - `compare-results` が JSON だけで動くこと。

---

## 7. 変更予定ファイル一覧

| ファイル | 変更 | 機能 |
|---|---|---|
| `src/logit_lens_debugger/types.py` | `DiffSpan`, `PositionResult`, `MultiPositionResult`, `InstabilityMetrics`, `PositionAnomaly`, `AnomalyReport`, `AnomalyConfig`, `LayerComparison` 追加。`ComparisonResult` 拡張と `from_dict` / `load_json` | 全部 |
| `src/logit_lens_debugger/metrics.py` | 新規。`compute_instability` | 3 |
| `src/logit_lens_debugger/diff.py` | 新規。`diff_token_sequences`, `select_positions`, 文字クラス判定 | 1, 2 |
| `src/logit_lens_debugger/anomaly.py` | 新規。`score_position`, `detect_anomalies` | 2 |
| `src/logit_lens_debugger/comparison.py` | 新規。`compare_results`, `align_layers`, `vocab_identical`, KL / JS | 4, 5 |
| `src/logit_lens_debugger/engine.py` | position 別 target、`EngineOutput` と `run_detailed(keep_log_probs)` | 1, 4 |
| `src/logit_lens_debugger/analyzer.py` | `analyze_positions` / `analyze_generation` / `analyze_diff` / `compare` の委譲 / `compare_models` | 1, 2, 4, 5 |
| `src/logit_lens_debugger/adapters/base.py` | `char_to_token` 追加、`vocab_size` / `get_vocab` 相当のアクセサ | 5 |
| `src/logit_lens_debugger/cli.py` | `diff`, `scan`, `metrics`, `compare-models`, `compare-results` 追加。`compare` にオプション追加。`analyze` にサマリ行 | 全部 |
| `src/logit_lens_debugger/visualization/comparison.py` | 任意。KL・一致率の折れ線 | 4, 5 |
| `src/logit_lens_debugger/__init__.py` | 新 API の export | 全部 |
| `tests/test_metrics.py`, `test_diff.py`, `test_anomaly.py`, `test_comparison.py` | 新規 | 3, 1, 2, 4/5 |
| `tests/conftest.py` | 手作り `LogitLensResult` builder、1 桁数字 vocab、語彙の異なる `ToyTokenizer` | 全部 |
| `tests/test_engine.py`, `test_analyzer.py`, `test_cli.py` | 既存挙動の互換テスト追加 | 1, 4 |
| `examples/diff_analysis.py`, `examples/compare_models.py` | 新規 | 1, 5 |
| `README.md` | 各機能の使い方・解釈上の注意・JSON 形の追記、「現在の制約」「次に実装すると良い機能」の更新 | 全部 |

---

## 8. 仮決めした判断（変えたい場合は先に相談）

| 論点 | 仮決め | 理由 |
|---|---|---|
| diff の単位 | モデル tokenizer の token 列 | 解析単位と一致、日本語対応、`ToyTokenizer` でテスト可能。文字オフセット版は v2 |
| 複数 position の結果型 | `MultiPositionResult` 新設 + `to_results()` | 長文で `input_tokens` の重複を避けつつ既存可視化を再利用 |
| KL の既定 | `compare` では on | 1 position あたり約 15 MB で実害がない。`analyze` 系では off |
| anomaly の既定重み | lens 系 0.6 / テキスト系 0.4 の均等配分から開始 | 実データで調整する前提。config を JSON に残す |
| 2 モデル比較のロード | 逐次ロードが既定、`sequential=False` で同時 | M2 16 GB で 1.5B クラス 2 つを同時に載せるのは厳しい |
| NER | スコープ外、span 注入 hook のみ | 依存を増やさない。パッケージの汎用性を保つ |
| 層数が違うモデルの対応付け | 相対深さの最近傍 | 単純で説明しやすい。補間は行わない |
