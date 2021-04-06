
from typing import Any, Dict, List

from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types

from expats.profiler.base import TextClassifier


class LITModelForTextClassifier(lit_model.Model):
    def __init__(self, profier: TextClassifier, labels: List[str]):
        self._profiler = profier
        self._labels = labels

    def max_minibatch_size(self):
        return 32

    def predict_minibatch(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [input_["sentence"] for input_ in inputs]
        ys = [input_["label"] for input_ in inputs]
        _output = self._profiler.interprete_via_prediction(texts, ys)
        try:
            output = [{
                "tokens": _output_per_inst["tokens"],
                "probas": _output_per_inst["probas"],
                "cls_emb": _output_per_inst["cls_emb"],
                "token_grad_sentence": _output_per_inst["token_grad_sentence"]
            } for _output_per_inst in _output]
            return output
        except KeyError as e:
            raise KeyError(f"Output spec of interprete_via_prediction seems to be not fit. error={e}")

    def input_spec(self) -> lit_types.Spec:
        return {
            "sentence": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(vocab=self._labels, required=False)
        }

    def output_spec(self) -> lit_types.Spec:
        return {
            "tokens": lit_types.Tokens(),
            "probas": lit_types.MulticlassPreds(parent="label", vocab=self._labels),
            "cls_emb": lit_types.Embeddings(),
            "token_grad_sentence": lit_types.TokenGradients(align="tokens")
        }


class LITModelForTextRegressor(lit_model.Model):
    def __init__(self, profier: TextClassifier):
        self._profiler = profier

    def max_minibatch_size(self):
        return 32

    def predict_minibatch(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [input_["sentence"] for input_ in inputs]
        ys = [input_["label"] for input_ in inputs]
        _output = self._profiler.interprete_via_prediction(texts, ys)
        try:
            output = [{
                "tokens": _output_per_inst["tokens"],
                "cls_emb": _output_per_inst["cls_emb"],
                "logits": _output_per_inst["logits"],
                "token_grad_sentence": _output_per_inst["token_grad_sentence"]
            } for _output_per_inst in _output]
            return output
        except KeyError as e:
            raise KeyError(f"Output spec of interprete_via_prediction seems to be not fit. error={e}")

    def input_spec(self) -> lit_types.Spec:
        return {
            "sentence": lit_types.TextSegment(),
            "label": lit_types.RegressionScore(required=False)
        }

    def output_spec(self) -> lit_types.Spec:
        return {
            "tokens": lit_types.Tokens(),
            "logits": lit_types.RegressionScore(),
            "cls_emb": lit_types.Embeddings(),
            "token_grad_sentence": lit_types.TokenGradients(align="tokens")
        }
