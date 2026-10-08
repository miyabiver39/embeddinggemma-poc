# リリースの手順と、SBOM・脆弱性検査

## 概要

| 契機 | 内容 | 結果の置き場所 |
|---|---|---|
| タグ `v*` の push | 版付きイメージの公開、全イメージの SBOM 作成と脆弱性検査 | GitHub のリリース(本文と添付ファイル) |
| 毎週月曜 06:17(日本時間) | main の最新イメージ(`:cpu` など)の SBOM 作成と脆弱性検査 | GitHub Actions の実行結果(サマリーと成果物。90 日保存) |
| 手動実行(Actions の「security」→ Run workflow) | 指定した版、または main の最新イメージの検査。`release` を有効にすると既存リリースの添付を更新 | 同上(`release` 有効時はリリースも) |

ワークフローは `.github/workflows/security.yml`、タグの push 時に呼び出す設定は `.github/workflows/build.yml` の `security` ジョブです。

## リリースの作り方

1. `pyproject.toml` の `version` と `src/vmsembed/__init__.py` の `__version__` を新しい版に合わせ、main に取り込みます。
2. タグを付けて push します。

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

3. GitHub Actions の「build」が、テスト、5 種類のイメージのビルドと公開、SBOM の作成と脆弱性検査の順に実行します。
   すべて完了すると、タグと同名のリリースが作成されます(変更履歴は GitHub が自動で作成します)。

公開されるイメージのタグは `<版>-<種類>` です(例: `ghcr.io/miyabiver39/embeddinggemma-poc:0.2.0-cpu`)。
`:cpu` などの種類だけのタグは main の最新を指し、リリースのたびには変わりません。

## 添付ファイル

イメージの種類(`slim` / `cpu` / `cuda` / `rocm` / `intel`)ごとに、次の 4 ファイルを添付します。

| ファイル | 内容 |
|---|---|
| `sbom-<種類>.spdx.json` | SBOM(SPDX 形式)。OS パッケージと Python パッケージを含む |
| `sbom-<種類>.cdx.json` | SBOM(CycloneDX 形式) |
| `vulnerabilities-<種類>.json` | 脆弱性検査の結果(grype の JSON 形式・全件) |
| `vulnerabilities-<種類>.txt` | 脆弱性検査の結果(一覧表・全件) |

リリース本文には、イメージごとの深刻度別の件数(括弧内は修正版が公開されている件数)と、Critical / High の一覧を記載します。
検査したイメージはダイジェスト付きで記録するため、後からタグが付け替わっても、どのイメージを検査したかを特定できます。

## 使用ツール

- SBOM: [syft](https://github.com/anchore/syft)
- 脆弱性検査: [grype](https://github.com/anchore/grype)(syft の SBOM を入力にするため、イメージのダウンロードは 1 回で済みます)

どちらも実行のたびに公式のインストーラで最新版を入れます。脆弱性データベースの形式が grype の版に依存するためです。
使用した版と脆弱性データベースの作成日時は、リリース本文に記載します。

## 結果の読み方と対応の方針

- 件数は「イメージに含まれるパッケージに、既知の脆弱性がある」ことを示すもので、本アプリの使い方で悪用できるかどうかは個別の判断が必要です。
  GPU 系のイメージは CUDA / ROCm / oneAPI のライブラリを多数含むため、件数が多くなる傾向があります。
- 2026-10-08 の初回検査(手動実行)では、Critical 0 件、High 3 件(いずれも venv 内の setuptools 78.1.0 とその同梱物。pip / setuptools を更新した後の再検査では 5 イメージとも High 0 件、修正版のある脆弱性 0 件)、
  Medium 約 1,150 件でした。Medium の大半は Ubuntu の ffmpeg(universe 区画のため、通常の Ubuntu では修正の提供が遅れがち)で、
  同じ 89 件の脆弱性が ffmpeg の 9 つのライブラリのパッケージで重複して数えられています。表の「脆弱性 ID の種類数」で実数を確認してください。
  ffmpeg は細工された動画を読み込んだ場合に影響し得るため、出どころの分からない映像は取り込まないことを推奨します。
- 修正版が公開されているものは、次のいずれかで対応します。
  - OS パッケージ: イメージを再ビルドすると、Ubuntu の更新が取り込まれます(タグを打ち直す、または main に push)。
  - Python パッケージ: Dependabot(`.github/dependabot.yml`)の提案、または `docker/Dockerfile` の版の指定で更新します。
    PyTorch と transformers は動作確認した版に固定しているため、更新する前に動作を確認してください(AGENTS.md)。
- 過去のリリースを最新の脆弱性データベースで検査し直す場合は、手動実行で `tag` に版(例 `v0.2.0`)を、`release` を有効にして実行します。
  リリース本文の検査結果の節と添付ファイルが更新されます(変更履歴などの他の記載は残ります)。

## 注意

- 検査ツールのインストーラは GitHub 上のスクリプトを実行します。GitHub Actions の中だけで実行し、利用者の環境には影響しません。
- 本ソフトウェアは開発・検証用途で、認証などのセキュリティ機能を持ちません。脆弱性検査の結果が 0 件でも、信頼できるネットワークの外に公開しないでください(README 参照)。
