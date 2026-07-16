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
  - G-DriftMIA
  - GDS

大きなモデルcheckpointや生成済みの実験結果CSVは含めていません。

## ディレクトリ構成

```text
github_experiment_code_package/
  README.md
  README_JA.md
  requirements.txt
  configs/
  data/
  models/
  results/
  scripts/
  scripts_abs/
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

Pythia-1Bについては、以下の共通コードにPythia-1Bのcheckpointパスを指定して実行します。

```text
src/proposed/experiment4_mimir_hardsplit_stopping_condition.py
```

## 解析コード

AUC、AUPRC、TPR@FPR、10回平均、比較手法との検定などを行うコードです。

```text
src/analysis/analyze_mimir_fixed_steps_repeated_auc.py
src/analysis/compare_fixedstep_proposed_baselines_strict.py
src/analysis/compare_proposed_attenmia_loraleak_10runs.py
```

## 比較手法

比較手法は `src/baselines/` にまとめています。

```text
src/baselines/run_attenmia_official_mimir_hardsplit.py
src/baselines/run_lora_leak_official_mimir_hardsplit.py
src/baselines/compare_mink_strict_fixedstep_10runs.py
```

追加で実装した比較手法:

```text
src/baselines/run_g_driftmia_mimir_hardsplit.py
src/baselines/run_g_driftmia_pythia1b_mimir_hardsplit.py
src/baselines/run_g_driftmia_pythia410m_mimir_hardsplit.py
src/baselines/run_g_driftmia_gptneo27b_mimir_hardsplit.py

src/baselines/run_gds_mimir_hardsplit.py
src/baselines/run_gds_pythia1b_mimir_hardsplit.py
src/baselines/run_gds_pythia410m_mimir_hardsplit.py
src/baselines/run_gds_gptneo27b_mimir_hardsplit.py
```

## 実行手順

基本的には以下の順番です。

```bash
pip install -r requirements.txt
bash scripts/00_prepare_splits.sh
bash scripts/01_train_gptneo27b.sh
bash scripts/02_extract_fixed20_gptneo27b.sh
bash scripts/03_analyze_gptneo27b.sh
bash scripts/04_run_baselines.sh
```

絶対パスを使う環境では、`scripts_abs/` 以下のファイルを使用してください。

## G-DriftMIAの実行

```bash
PYTHONPATH=src/baselines python src/baselines/run_g_driftmia_pythia1b_mimir_hardsplit.py
PYTHONPATH=src/baselines python src/baselines/run_g_driftmia_pythia410m_mimir_hardsplit.py
PYTHONPATH=src/baselines python src/baselines/run_g_driftmia_gptneo27b_mimir_hardsplit.py
```

## GDSの実行

```bash
PYTHONPATH=src/baselines python src/baselines/run_gds_pythia1b_mimir_hardsplit.py
PYTHONPATH=src/baselines python src/baselines/run_gds_pythia410m_mimir_hardsplit.py
PYTHONPATH=src/baselines python src/baselines/run_gds_gptneo27b_mimir_hardsplit.py
```

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

