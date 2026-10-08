# 引き継ぎメモ(次の担当者・AI エージェント向け)

作成: 2026-10-08 / 更新: 2026-10-08(コンテナの起動確認、Intel イメージの修正、取り込みの受付強化・監視フォルダ、リリース時の SBOM・脆弱性検査。すべて main に取り込み済み)
先に `AGENTS.md`(方針・規約)と `docs/design.md`(設計)を読んでください。このメモは「今どこまで進んでいて、何が未確認で、次に何をするか」だけをまとめます。

## 1. 依頼の要点(利用者の意向)

- EmbeddingGemma 2 を使った VMS 向けマルチモーダル検索基盤を、**公開リポジトリ**で作る。開発中の用途、**自己責任**。
- **セキュリティ対策は不要**(認証・TLS・堅牢化は追加しない)。バックエンド専用だが、開発用 WebUI は付ける。
- ビルドは **GitHub Actions** に任せ、利用者は `docker pull` して**追加の設定なしで起動**できること。
- ドキュメントとコメントは**日本語**。**Mac は想定しない**(今後も不要)。実装は**全て Python**。
- 構成: `app`(取り込み+検索+DB+WebUI)/ `compute`(ベクトル化のみ)/ `all`(両方)。app は内蔵か外部 compute かを設定で選ぶ。取り込みと検索は同じコンテナ。
- 動作モード: **CPU / NVIDIA GPU / その他 GPU**。利用者の機材は **RTX 3060 12GB** と **RX 9060 XT 16GB**。Intel 内蔵 GPU でも動作することが望ましい(要望)。
- 「ネイティブに変換」= 初回起動時に PyTorch モデルを ONNX Runtime / OpenVINO / TensorRT 等へ変換してキャッシュし速度を出す、という意味で合意済み(**未実装・フェーズ 2**)。

## 2. 現在の状態

- リポジトリ: https://github.com/miyabiver39/embeddinggemma-poc (main に初版を push 済み)
- GitHub Actions(`.github/workflows/build.yml`): test → 5 イメージのビルドと ghcr.io への公開まで**全て成功**。
- イメージ: `ghcr.io/miyabiver39/embeddinggemma-poc:{slim,cpu,cuda,rocm,intel}`(`:latest` = cpu)。**匿名で manifest を取得できることを確認済み**(公開設定は不要だった)。
- ローカルテスト: `pytest -q` で 13 passed / 1 skipped(skip は実モデルのテスト)。`ruff check src tests` クリーン。
- **コンテナの起動確認を実施**(GPU なしのクラウド環境・4 vCPU。`dockerd` を手で起動して使えた): `:cpu` 単体、`:slim` + `:cpu`(compute)の分離構成、`:cuda` / `:intel` の CPU フォールバック。結果は `AGENTS.md` 第 6 節と `docs/operations.md`・`docs/gpu.md`。
- 構成物の一覧は `AGENTS.md` の「7. ディレクトリ構成」を参照。

## 3. 確認済み・未確認

確認済み(CPU・合成データ):

- 実モデル(CPU)で、動画取り込み → テキスト検索。「a solid blue screen」で青い動画が 1 位(スコア 0.78、次点の緑が 0.72)。
- sentence-transformers 6.1.0 で text / image / audio / video 配列 / 構造化メッセージ(順序つき混在入力)が 1 ベクトルを返す。
- ダミー埋め込みでの API 通し(取り込み・検索・絞り込み・サムネイル・Range 配信・削除・次元不一致 409)、app/compute 分離。
- ONNX 版リポジトリ `onnx-community/embeddinggemma-2-ONNX` の中身(`model.onnx` / `vision_encoder.onnx` / `audio_encoder.onnx`、fp32/fp16/q4/q4f16/quantized)。変換や精度比較は未実施。

**未確認(ここが次の作業の中心)**:

1. **GPU 3 種**(cuda / rocm / xpu)を**実機の GPU で**動かした結果。bf16 での NaN の有無、CPU 版とのベクトル一致度、RX 9060 XT が ROCm 7.2 ホイールで動くか。手順と結果記入欄は `docs/gpu.md`。
   `:cuda` と `:intel` は「GPU なしで起動して CPU で動く」ところまで確認済み。`:rocm` は展開後の容量が作業環境のディスク枠に入らず未実施。
2. 窓あたりの処理時間(GPU)、実際の監視映像での検索精度。

解決済み(2026-10-08): コンテナの起動・取り込み・検索(`:cpu` / 分離構成)、`tav` / `audio` / 画像クエリの実モデル通し、音声のトークン率(実測 25/秒)、CPU での窓あたりの処理時間。

## 3.1 2026-10-08 に追加したもの

- 取り込みの受付(`ingest_files.py`): 拡張子と ffprobe の検証、同一パスの重複防止、ファイル名からの録画開始時刻、フォルダ一括(`/api/ingest/dir`)。
  以前は任意のパスを登録でき、`/api/media` からコンテナ内の任意のファイル(`/etc/shadow` を含む)を取得できた。入力検証の不足として修正済み(認証は追加していない)。
- 監視フォルダ(`watcher.py`、`WATCH_DIRS`): 定期走査で新しい録画を自動取り込み。`:cpu` イメージ上で、起動時の取り込みと稼働中に追加したファイルの取り込みを確認済み。
- リリース時の SBOM・脆弱性検査(`.github/workflows/security.yml`、`docs/release.md`): タグ `v*` の push で、全イメージの SBOM(SPDX / CycloneDX)と grype の結果をリリースに添付。毎週の定期実行あり。
  手動実行(main の最新イメージ)で 5 イメージとも成功を確認済み(約 7 分)。初回は Critical 0、High 3(setuptools 由来)。pip / setuptools 更新後の再検査で 5 イメージとも Critical 0・High 0。Medium の大半は ffmpeg(`docs/release.md`)。
- `/api/info` に版・リビジョン、検索応答に `embed_ms`、起動時の DB 整合性の警告、pip / setuptools の更新、Dependabot。

## 4. 既知の課題・注意点(コードを読んで気づいたもの)

- (対処済み)音声クエリ(`/api/search/audio`)の `kind=auto` は `audio` の窓を探すように変更した。以前は `frames` を探し、無関係な映像が並んでいた(実測)。文字・画像クエリの `auto` は従来どおり `INCLUDE_AUDIO` で `tav` か `frames`。
- (対処済み)`:intel` イメージは Intel GPU が見えないと `torch.xpu` の判定がセグメンテーション違反で落ちていた。判定を子プロセスにし、使わないときは transformers の読み込み前に XPU を 0 台に見せる(`embedders/local.py`)。根本は Ubuntu 24.04 の古い Intel ドライバと PyTorch XPU 版の組み合わせなので、実機で問題が出たら Intel 公式のドライバ(compute-runtime)への差し替えを検討。
- 次元・モデルの整合性チェック(`ensure_compat`)は**取り込み時・検索時**に行う。起動時には行わない(remote の compute が落ちていても app が起動できるようにするため)。起動時警告が欲しければ追加。
- DB に記録する「窓の設定」の照合は、モデルID・次元が中心。AGENTS.md の第 4 節にある窓長・フレーム数などの照合が全項目実装されているかは未精査。
- `torchcodec` が読み込めない環境があったため、動画ファイルをモデルに渡さず **ffmpeg で復号した配列を渡す**設計にしている。これは意図的(変えないこと)。
- `transformers` は PyPI 版に `embedding_gemma2` が無く、Dockerfile でコミット `cb33194ad6152bd9fad6305378d92db385dd7b32` のアーカイブを入れている。PyPI に入ったら通常版へ戻せる(`docker/Dockerfile` の `TRANSFORMERS_COMMIT`)。
- 全バリアント Ubuntu 24.04 ベース(Intel の `libze-intel-gpu1` / `intel-opencl-icd` が Debian trixie に無かったため)。
- イメージの容量(圧縮/展開): slim 0.27/1.1GB、cpu 1.46/5.1GB、cuda 4.94/14.4GB、intel 3.82/14.1GB、rocm 7.89GB/未計測。
- 推論は `PriorityGate` で 1 件ずつ直列。GPU のバッチ処理は未実装(速度改善の余地)。
- `pyproject.toml` の pytest marker 説明文は「RUN_MODEL_TESTS=1 のときだけ実行」。`tests/conftest.py` がそれを実装している。

## 5. 次にやること(優先順)

1. **最初のリリース(例 `v0.2.0`)を作成**し、SBOM と脆弱性検査の結果がリリースに添付されることを確認する(タグ契機の実行は未確認。手順は `docs/release.md`)。
2. **GPU 実機確認**(利用者が RTX 3060 / RX 9060 XT を持っている)。`docs/gpu.md` の表を埋め、不具合があれば Dockerfile(ホイールの版、`HSA_OVERRIDE_GFX_VERSION` など)を直す。
3. GPU での窓あたりの処理時間・取り込み速度。結果を `docs/operations.md` と `AGENTS.md` の「検証状況」に反映。
4. 速度改善: 推論が 1 件ずつ直列(CPU で約 8.5 秒/窓 = 実時間の約 0.24 倍速)。GPU でのバッチ処理を検討。
5. フェーズ 2: ネイティブ化(ONNX Runtime → OpenVINO / TensorRT)。受け入れ条件は「PyTorch 版との出力コサイン類似度」と速度。設計は `docs/design.md` の第 7 節。
6. (任意)規模が増えたときの検索のスケール対策(`Store.search` を専用ベクトル DB に差し替えられるよう、すでに `Store` に閉じてある)。

## 6. 作業環境メモ

- テスト: `pip install -e ".[dev]" && pytest -q`。実モデル: `pip install -e ".[model]"`(transformers は固定コミット、torchvision 必須)のうえ `RUN_MODEL_TESTS=1 pytest -m model`(CPU で約 90 秒)。
- 手元起動(モデルなし): `EMBEDDING_BACKEND=dummy DATA_DIR=./data uvicorn --factory vmsembed.main:create_app --reload`
- ffmpeg / ffprobe が必須(テストの動画は ffmpeg の `lavfi` で合成。実在の映像は使わない)。
- PyTorch は 2.14.1 / torchvision 0.29.1 に固定。入手元: cpu / cu126 / rocm7.2 / xpu(cp312 ホイールの存在を確認済み)。
- クラウドの作業環境では Docker デーモンが止まっていることがある。`dockerd &` で起動できた(root)。ディスク枠は約 40GB なので、GPU イメージは 1 つずつ pull して確認後に消す。
- コミットの末尾には、実行環境が指定する `Co-Authored-By` などの行を付ける(`AGENTS.md` 第 10 節)。

## 7. 利用者への報告で伝えるべきこと

- 「GPU 版は実機の GPU で未確認」であることを、できた/できないに関わらず明記する(推測で「動く」と書かない)。コンテナの起動は CPU で確認済み。
- README の自己責任・個人情報の注意書きは消さない。セキュリティ機能は勝手に足さない。
