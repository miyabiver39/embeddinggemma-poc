"""推論の高速化(結果を変えない、または精度を上げる方向の最適化だけを行う)。

RTX 3060 12GB での実測(1080p・2 秒窓、合成の場面 32 件。docs/gpu.md の「推論の高速化」):

| 対象 | 最適化なし | 最適化あり |
|---|---|---|
| 1 窓(画像 2 枚) | 286 ms | 169 ms |
| 画像 1 枚 | 174 ms | 97 ms |
| 文章 1 件 | 48.5 ms | 15.2 ms |

品質は、float32 で計算した結果との類似度で確かめています。最適化ありのほうが float32 に近く
(窓の最小 0.999876 → 0.999925)、文章から窓・画像を探したときの 1 位も変わりません。

1. 画像の詰め物(パディング)を外す(trim_image_padding)
   前処理は、画像を最大 2520 パッチの枠に詰め、余りを詰め物で埋めます
   (16:9 の画像では 2376 パッチが有効で、144 が詰め物)。詰め物があると注意機構にマスクが必要になり、
   遅い計算方式(Memory Efficient)が選ばれます。同じ大きさの画像だけのときは、詰め物を外してから計算します。
   詰め物は計算の中で無視されるため数学的には同じ結果になり、高速な Flash 方式で計算できます。
2. 層ごとの torch.compile(compile_layers)
   正規化や型の変換などの細かい計算を 1 つにまとめ、GPU の呼び出し回数を減らします。
   まとめた計算では途中の値を float32 のまま扱えるため、float32 の結果にむしろ近づきます。
   モデル全体ではなく層ごとに compile するのは、マスクを作る処理を compile の外に置き、1 の Flash 方式を
   使えるようにするためです(compile の中では、transformers がマスクを省く判定を行わない)。
   準備(compile)に時間がかかるため、結果を DATA_DIR/cache/torch に保存し、2 回目以降の起動を速くします。
"""

from __future__ import annotations

import logging
import os
import shutil
import types
from pathlib import Path

log = logging.getLogger(__name__)


def compiler_available() -> bool:
    """torch.compile(Triton)が使う C コンパイラがあるか。cuda イメージには gcc を入れている。"""
    cc = os.environ.get("CC")
    return bool(cc and shutil.which(cc)) or shutil.which("gcc") is not None or shutil.which("cc") is not None


def setup_compile_cache(data_dir: Path) -> Path | None:
    """compile の結果の保存先を DATA_DIR/cache/torch にします(既に環境変数で指定されていれば、それを使う)。"""
    root = Path(data_dir) / "cache" / "torch"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # 書けない場所なら、保存せずに続ける(毎回の起動で compile する)
        log.warning("compile の結果の保存先を作れませんでした(%s)。起動のたびに準備を行います", exc)
        return None
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(root / "triton"))
    return root


def trim_image_padding(model) -> None:
    """画像の詰め物を、同じ大きさの画像だけのときに外してから画像側のモデルに渡します。

    model は transformers の EmbeddingGemma2Model。get_image_features を、このインスタンスだけ置き換えます。
    詰め物の位置は (-1, -1) で示され、末尾にまとまっています。有効なパッチの数が全画像で同じで、
    プーリング(3x3)で割り切れるときだけ外します。それ以外は元のまま(マスクつき)で計算します。
    """
    original = model.get_image_features
    kernel = model.config.vision_config.pooling_kernel_size ** 2

    def get_image_features(self, pixel_values, image_position_ids=None, **kwargs):
        if image_position_ids is not None:
            valid = (image_position_ids != -1).all(dim=-1)
            counts = valid.sum(dim=-1)
            n = int(counts[0])
            if (
                n < valid.shape[1]
                and n % kernel == 0
                and bool((counts == n).all())
                and bool(valid[:, :n].all())  # 詰め物が末尾にまとまっている
            ):
                pixel_values = pixel_values[:, :n]
                image_position_ids = image_position_ids[:, :n]
        return original(pixel_values=pixel_values, image_position_ids=image_position_ids, **kwargs)

    model.get_image_features = types.MethodType(get_image_features, model)


def _layers(model) -> list:
    return (
        list(model.vision_tower.encoder.layers)
        + list(model.language_model.layers)
        + list(model.audio_tower.layers)
    )


def compile_layers(model) -> None:
    """画像側・文章側・音声側の各層を torch.compile します(実際の compile は最初の推論のとき)。

    文章と音声の長さは毎回変わるため、文章側と音声側は長さを可変(dynamic)にして作り直しを防ぎます。
    画像側は、まとめる件数や画像の大きさが変わったときだけ作り直します(大きさは動画ごとにほぼ一定)。
    音声側は 10 秒の音声 1 件が 97 ms → 76 ms(float32 の結果との類似度は 0.99979 → 0.99982。RTX 3060 で実測)。
    """
    import torch

    # 層は同じコードを共有するため、作り直しの回数は全層の合計で数えられる。文章側は、層の種類(注意機構の
    # 大きさ 256 / 512)、マスクの有無、窓かどうかで別の形になり、既定の上限(8)を超えると compile しない
    # 通常の計算に戻ってしまう(実測。4 件まとめた文章が 14 ms → 48 ms になった)。必要な数より余裕を持たせる
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    torch._dynamo.config.accumulated_recompile_limit = max(torch._dynamo.config.accumulated_recompile_limit, 512)
    for layer in model.vision_tower.encoder.layers:
        layer.compile()
    for layer in model.language_model.layers:
        layer.compile(dynamic=True)
    for layer in model.audio_tower.layers:
        layer.compile(dynamic=True)


def uncompile_layers(model) -> None:
    """compile をやめ、通常の計算に戻します(compile に失敗したときに使う)。"""
    import torch

    for layer in _layers(model):
        layer._compiled_call_impl = None
    torch._dynamo.reset()


class TextGraphs:
    """文章だけの入力の推論を、件数と長さの段階ごとに CUDA Graph に記録しておき、再生します。

    文章の推論は計算が小さく、GPU の計算は 1 件 約 3 ms ですが、GPU に命令を出す CPU 側の処理に
    約 11 ms かかっていました(RTX 3060 で実測。層ごとに compile した状態)。CUDA Graph は命令の並びを
    記録して丸ごと再生するため、この処理を省けます(1 件 15 ms → 4 ms、4 件 18 ms → 7 ms)。

    - 記録は件数 TEXT_GRAPH_BATCHES × 長さ TEXT_GRAPH_LENGTHS の段階ごとに 1 つ。入力は段階の大きさまで詰め物をして渡す
      (詰め物はマスクで無視されるため、検索の文章をまとめたときと同じく、float32 の結果との類似度は変わらない)。
    - 段階に収まらない入力(長い文章、画像・音声を含む窓)は、記録を使わずに通常どおり計算する。
    - マスクは記録の外で毎回作り、記録したマスクの領域に書き写す
      (マスクを作る処理は CPU で値を確かめるため、記録できない)。
    - 再生は推論の順番待ち(PriorityGate)の中だけで行うため、記録した入出力の領域を同時に使うことはない。
    """

    def __init__(self, language_model, batches: tuple[int, ...], lengths: tuple[int, ...]) -> None:
        self._lm = language_model
        self._orig = language_model.forward
        self._batches = tuple(sorted(batches))
        self._lengths = tuple(sorted(lengths))
        self._graphs: dict[tuple[int, int], tuple] = {}
        self._pool = None
        self.replays = 0  # 記録を使った回数(確認用)

    def _masks(self, embeds, mask2d) -> dict:
        from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask

        kw = {"config": self._lm.config, "inputs_embeds": embeds, "attention_mask": mask2d}
        # 記録した形と合わせるため、詰め物が無いときもマスクを省かない(allow_is_bidirectional_skip=False)
        return {
            "full_attention": create_bidirectional_mask(**kw, allow_is_bidirectional_skip=False),
            "sliding_attention": create_bidirectional_sliding_window_mask(**kw, allow_is_bidirectional_skip=False),
        }

    def _capture(self, batch: int, length: int) -> None:
        import torch

        param = next(self._lm.parameters())
        embeds = torch.zeros(batch, length, self._lm.config.hidden_size, device=param.device, dtype=param.dtype)
        mask2d = torch.ones(batch, length, device=param.device, dtype=torch.long)
        mask2d[:, length // 2 :] = 0  # 詰め物がある形で記録する(マスクは必ずテンソルになる)
        masks = self._masks(embeds, mask2d)
        if any(m is None for m in masks.values()):
            raise RuntimeError("マスクを作れませんでした")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # 記録の前に、同じ形で一度計算する(compile をここで済ませる)
            self._orig(inputs_embeds=embeds, attention_mask=masks)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            out = self._orig(inputs_embeds=embeds, attention_mask=masks).last_hidden_state
        self._pool = graph.pool()  # 記録どうしで GPU メモリを共有する
        self._graphs[(batch, length)] = (graph, embeds, mask2d, masks, out)

    def install(self) -> None:
        """全段階を記録し、language_model の forward を置き換えます(推論の順番待ちの外で、起動時に呼ぶ)。"""
        import types

        import torch

        with torch.inference_mode():
            for batch in self._batches:
                for length in self._lengths:
                    self._capture(batch, length)
        self._lm.forward = types.MethodType(lambda _lm, *args, **kwargs: self._forward(*args, **kwargs), self._lm)

    def uninstall(self) -> None:
        self._lm.forward = self._orig
        self._graphs.clear()

    # sentence-transformers が受け渡しのために付ける引数。文章側のモデルの計算には使われない
    _IGNORED = frozenset({"return_dict", "sentence_embedding", "token_embeddings"})

    def _forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, **kwargs):
        import torch
        from transformers.modeling_outputs import BaseModelOutput

        key = None
        if (
            inputs_embeds is not None
            and input_ids is None
            and position_ids is None
            and set(kwargs) <= self._IGNORED
            and (attention_mask is None or (torch.is_tensor(attention_mask) and attention_mask.dim() == 2))
        ):
            batch, length = inputs_embeds.shape[:2]
            key = next(((b, n) for b in self._batches for n in self._lengths if b >= batch and n >= length), None)
        if key is None:
            return self._orig(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        graph, embeds, mask2d, masks, out = self._graphs[key]
        embeds.zero_()
        embeds[:batch, :length] = inputs_embeds
        mask2d.zero_()
        mask2d[:, 0] = 1  # 使わない行も先頭だけ有効にする(すべて無効だと、その行の注意機構が NaN になるため)
        mask2d[:batch, :length] = attention_mask if attention_mask is not None else 1
        for name, mask in self._masks(embeds, mask2d).items():
            masks[name].copy_(mask)
        graph.replay()
        self.replays += 1
        return BaseModelOutput(last_hidden_state=out[:batch, :length].clone())
