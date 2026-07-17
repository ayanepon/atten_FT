# 実験コード一式

このディレクトリは、論文実験で使用したコードをGitHubに公開しやすい形でまとめたものです。

主に以下を含みます。

- 提案手法のAttention更新量抽出コード
- LoRA fine-tuningコード
- AUC / AUPRC / TPRを計算する解析コード
- 比較手法のコード
  - AttenMIA-style
  - LoRA-Leak-style
  - Min-K / Min-K++

大きなモデルcheckpointや生成済みの実験結果CSVは含めていません。

## ディレクトリ構成

```text
anonymous_github_experiment_code/
  README.md
  README_JA.md
  requirements.txt
  configs/
  data/
  models/
  results/
  scripts/
  supplement/
  src/
    train/
    proposed/
    baselines/
    analysis/
```

## データ

全実験で同じMIMIR hard splitを使用します。

以下の3つのCSVを配置してください。

```text
data/mimir_hardsplit/
  mimir_wikipedia_pt_member.csv
  mimir_wikipedia_ft_nonmember.csv
  mimir_wikipedia_unseen_nonmember.csv
```

各ファイルの意味は以下です。

```text
mimir_wikipedia_pt_member.csv
  Pythia等の事前学習に含まれているとみなすPTデータ

mimir_wikipedia_ft_nonmember.csv
  LoRA fine-tuningに使用するFTデータ

mimir_wikipedia_unseen_nonmember.csv
  事前学習にもFTにも使用しないUnseenデータ
```

データをGitHubに載せられない場合は、CSV本体は含めず、上記の場所に手動で配置してください。

## FTモデル

FT済みモデルcheckpointはGitHubには含めていません。

必要な場合は、以下に配置してください。

```text
models/
```

または、`src/train/` 以下の学習コードを使って作成してください。

主な学習コード:

```text
src/train/train_mimir_wikipedia_hardsplit_lora.py
src/train/train_mimir_wikipedia_hardsplit_lora_pythia410m.py
src/train/train_mimir_wikipedia_hardsplit_lora_gptneo27b.py
```

## 提案手法

提案手法の主要コードは以下です。

```text
src/proposed/mimir_hardsplit_attention_common.py
src/proposed/experiment4_mimir_hardsplit_stopping_condition.py
```

モデル別のfixed-20実験用コード:

```text
src/proposed/experiment4_gptneo27b_fixed20_common.py
src/proposed/experiment4_gptneo27b_fixed20_ft.py
src/proposed/experiment4_gptneo27b_fixed20_pt.py
src/proposed/experiment4_gptneo27b_fixed20_unseen.py

src/proposed/experiment4_pythia410m_fixed20_common.py
src/proposed/experiment4_pythia410m_fixed20_ft.py
src/proposed/experiment4_pythia410m_fixed20_pt.py
src/proposed/experiment4_pythia410m_fixed20_unseen.py
```

20/50/100ステップおよびearly stoppingの停止条件比較は、以下を使います。

```text
src/proposed/run_pythia1b_stopping_conditions.py
src/proposed/run_pythia410m_stopping_conditions.py
src/proposed/run_gptneo27b_stopping_conditions.py
```

内部では以下の共通コードを呼び出します。

```text
src/proposed/experiment4_mimir_hardsplit_stopping_condition.py
```

## 解析コード

AUC、AUPRC、TPR@FPR、10回平均、比較手法との検定などを行うコードです。

```text
src/analysis/analyze_mimir_fixed_steps_repeated_auc.py
src/analysis/compare_fixedstep_proposed_baselines_strict.py
src/analysis/compare_proposed_attenmia_loraleak_10runs.py
src/analysis/run_strict_fixed20_3model_comparison_10runs.py
src/analysis/evaluate_loss_direction_selected_3model.py
```

`run_strict_fixed20_3model_comparison_10runs.py` は、提案手法、AttenMIA、LoRA-Leak、Initial loss、Loss decreaseを同じ分割で比較します。
`evaluate_loss_direction_selected_3model.py` は、Pythia-1B、Pythia-410M、GPT-Neo-2.7Bのloss系のみをfold内方向選択で評価します。

## 追加supplementコード

論文実験で使用した追加コードは以下に入れています。

```text
supplement/
```

ここには、元のpaper pipeline、orchestration用コード、robustness check、reviewer follow-up実験、CPU-only testなどが含まれます。
整理済みのトップレベルコードと混ざらないよう、別ディレクトリとして保持しています。

まず以下を確認してください。

```text
supplement/README.md
supplement/PAPER_ALIGNMENT.md
supplement/STRUCTURE.md
```

## 比較手法

比較手法は `src/baselines/` にまとめています。

```text
src/baselines/run_attenmia_official_mimir_hardsplit.py
src/baselines/run_attenmia_official_mimir_hardsplit_gptneo27b.py
src/baselines/run_lora_leak_official_mimir_hardsplit.py
src/baselines/run_lora_leak_official_mimir_hardsplit_gptneo27b.py
src/baselines/compare_mink_strict_fixedstep_10runs.py
```

## 実行手順

基本的には以下の順番です。

```bash
pip install -r requirements.txt
bash scripts/00_prepare_splits.sh
bash scripts/01_train_pythia410m.sh
bash scripts/01_train_gptneo27b.sh
bash scripts/02_extract_fixed20_gptneo27b.sh
bash scripts/03_analyze_gptneo27b.sh
bash scripts/04_run_baselines.sh
```

停止条件比較を実行する場合:

```bash
PYTHONPATH=src/proposed python src/proposed/run_pythia1b_stopping_conditions.py
PYTHONPATH=src/proposed python src/proposed/run_pythia410m_stopping_conditions.py
PYTHONPATH=src/proposed python src/proposed/run_gptneo27b_stopping_conditions.py
```

loss系を含む比較手法評価:

```bash
python src/analysis/run_strict_fixed20_3model_comparison_10runs.py
python src/analysis/evaluate_loss_direction_selected_3model.py
```

絶対パスを使う環境では、環境変数でモデル・出力先を上書きしてください。

## GitHubに含めないもの

以下はGitHubには含めない方針です。

```text
__pycache__/
*.pyc
.DS_Store
models/ 以下の大きいcheckpoint
results/ 以下の大きい結果CSV
一時的なplot
```

このため、`models/` と `results/` には `.gitkeep` のみを置いています。

## 注意

- FTをpositive classとして扱います。
- AUCは結果を見て反転していません。
- 提案手法のElastic Net特徴量選択はfold内でのみ行います。
- 全モデル・全比較手法で同じMIMIR hard splitを使用します。
