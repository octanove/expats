
from copy import deepcopy
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer

from expats.common.instantiate import BaseConfig
from expats.common.tensor import unbatch_preds
from expats.common.type import ClassificationOutput, RegressionOutput
from expats.data.helper import dict_to_hfdataset
from expats.ml.nn.transformer import (
    TransformerClassifierNetConfig, TransformerClassifierNet, TransformerRegressorNetConfig, TransformerRegressorNet
)
from expats.ml.nn.trainer import PLTrainerConfig, create_pltrainer
from expats.profiler.base import TextProfiler, TextClassifier, TextRegressor
from expats.settings import SETTINGS


T = TypeVar("T", bound=Union[ClassificationOutput, RegressionOutput])

NET_FILENAME = "model.pt"
LABEL_ENCORDER_FILENAME = "label_encorder.pkl"
MAP_BATCHSIZE = 1000  # NOTE: it's default value in datasets https://github.com/huggingface/datasets/blob/441e2040ead59dc4b0c9f2f7998f57d4fa2fd99f/src/datasets/arrow_dataset.py#L1332  # noqa


class DataLoaderConfig(BaseConfig):
    batch_size: int = 32
    num_workers: int = 0


class _ClassifierConfig(BaseConfig):
    trainer: PLTrainerConfig
    network: TransformerClassifierNetConfig
    data_loader: DataLoaderConfig
    val_ratio: float = 0.0
    max_length: Optional[int] = None


class _RegressorConfig(BaseConfig):
    trainer: PLTrainerConfig
    network: TransformerRegressorNetConfig
    data_loader: DataLoaderConfig
    val_ratio: float = 0.0
    max_length: Optional[int] = None


class TransformerBase(TextProfiler[T]):
    def predict_batch(self, inputs: List[str]) -> List[T]:
        batchsize = 10
        results = []
        for i in range(0, len(inputs), batchsize):
            _batch = inputs[i:i + batchsize]
            results.extend(self._predict_batch_each(_batch))
        return results

    @classmethod
    def create(cls, params: BaseConfig):
        raise NotImplementedError

    def _load_internal(cls, artifact_path: str, params: BaseConfig) -> "TransformerBase":
        raise NotImplementedError

    def _save_internal(self, artifact_path: str):
        raise NotImplementedError


@TextClassifier.register
class TransformerClassifier(TransformerBase[ClassificationOutput]):
    config_class = _ClassifierConfig

    def __init__(
        self,
        params: _ClassifierConfig,
        net: TransformerClassifierNet,
        tokenizer: AutoTokenizer,
        label_encorder: LabelEncoder,
    ):
        super().__init__(params)
        self._net = net
        self._tokenizer = tokenizer
        self._label_encorder = label_encorder
        self._model_input_names = tokenizer.model_input_names

    def _predict_batch_each(self, inputs: List[str]) -> List[str]:
        # NOTE: copy tokenizer to avoid https://github.com/huggingface/transformers/issues/8453
        _dataset = _convert_to_torch_format_dataset(
            {"text": inputs}, "text", deepcopy(self._tokenizer), self._params.max_length, self._model_input_names
        )
        logits = self._net.forward(**{
            key: _dataset[key] for key in self._model_input_names
        })
        # decode
        idxs = self._label_encorder.inverse_transform(logits.argmax(-1).tolist())
        return [idx for idx in idxs.tolist()]

    def fit(self, inputs: List[str], ys: List[str]):
        _ys = self._label_encorder.fit_transform(ys)
        train_inputs, val_inputs, train_ys, val_ys = _train_test_split(
            inputs, _ys, self._params.val_ratio
        )
        train_dataset = _convert_to_torch_format_dataset(
            {"text": train_inputs, "label": train_ys}, "text", deepcopy(self._tokenizer),
            self._params.max_length,
            self._model_input_names + ["label"]
        )
        val_dataset = _convert_to_torch_format_dataset(
            {"text": val_inputs, "label": val_ys}, "text", deepcopy(self._tokenizer),
            self._params.max_length,
            self._model_input_names + ["label"]
        )
        trainer = create_pltrainer(self._params.trainer)
        trainer.fit(
            self._net,
            torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self._params.data_loader.batch_size,
                num_workers=self._params.data_loader.num_workers,
                shuffle=True
            ),
            torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self._params.data_loader.batch_size,
                num_workers=self._params.data_loader.num_workers,
                shuffle=False
            )
        )

    def interprete_via_prediction(self, inputs: List[str], ys: List[str]) -> List[Dict[str, Any]]:
        _ys = self._label_encorder.transform(ys)
        _dataset = _convert_to_torch_format_dataset(
            {"text": inputs, "label": _ys}, "text", deepcopy(self._tokenizer),
            self._params.max_length,
            self._model_input_names + ["label"]
        )
        results = self._net.forward_for_interpretation(**{
            key: _dataset[key] for key in self._model_input_names
        })
        # to ndarray
        results = {k: v.detach().numpy() for k, v in results.items()}
        unbatched_results = unbatch_preds(results)
        unbatched_results = [
            {
                "tokens": self._tokenizer.convert_ids_to_tokens(res["input_ids"][:res["ntok"]]),
                "token_grad_sentence": res["input_emb_grad"][:res["ntok"]],
                **res
            }
            for res in unbatched_results
        ]
        return unbatched_results

    @classmethod
    def create(cls, params: _ClassifierConfig):
        net = TransformerClassifierNet.create(params.network.to_dict())
        tokenizer = AutoTokenizer.from_pretrained(params.network.pretrained_model_name_or_path)
        label_encorder = LabelEncoder()
        return cls(params, net, tokenizer, label_encorder)

    @classmethod
    def _load_internal(cls, artifact_path: str, params: _ClassifierConfig) -> "TransformerClassifier":
        net = TransformerClassifierNet.load(os.path.join(artifact_path, NET_FILENAME))
        with open(os.path.join(artifact_path, LABEL_ENCORDER_FILENAME), "rb") as f:
            label_encoder = pickle.load(f)
        tokenizer = AutoTokenizer.from_pretrained(params.network.pretrained_model_name_or_path)
        return cls(params, net, tokenizer, label_encoder)

    def _save_internal(self, artifact_path: str):
        self._net.save(os.path.join(artifact_path, NET_FILENAME))
        with open(os.path.join(artifact_path, LABEL_ENCORDER_FILENAME), "wb") as fw:
            pickle.dump(self._label_encorder, fw)


@TextRegressor.register
class TransformerRegressor(TransformerBase[RegressionOutput]):
    config_class = _RegressorConfig

    def __init__(
        self,
        params: _RegressorConfig,
        net: TransformerRegressorNet,
        tokenizer: AutoTokenizer,
    ):
        super().__init__(params)
        self._net = net
        self._tokenizer = tokenizer
        self._model_input_names = tokenizer.model_input_names

    def _predict_batch_each(self, inputs: List[str]) -> List[float]:
        _dataset = _convert_to_torch_format_dataset(
            {"text": inputs}, "text", deepcopy(self._tokenizer), self._params.max_length, self._model_input_names
        )
        logits = self._net.forward(**{
            key: _dataset[key] for key in self._model_input_names
        })
        return logits.squeeze().tolist()

    def fit(self, inputs: List[str], ys: List[float]):
        train_inputs, val_inputs, train_ys, val_ys = _train_test_split(
            inputs, ys, self._params.val_ratio
        )
        train_dataset = _convert_to_torch_format_dataset(
            {"text": train_inputs, "label": train_ys}, "text", deepcopy(self._tokenizer),
            self._params.max_length,
            self._model_input_names + ["label"]
        )
        val_dataset = _convert_to_torch_format_dataset(
            {"text": val_inputs, "label": val_ys}, "text", deepcopy(self._tokenizer),
            self._params.max_length,
            self._model_input_names + ["label"]
        )
        trainer = create_pltrainer(self._params.trainer)
        trainer.fit(
            self._net,
            torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self._params.data_loader.batch_size,
                num_workers=self._params.data_loader.num_workers,
                shuffle=True
            ),
            torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self._params.data_loader.batch_size,
                num_workers=self._params.data_loader.num_workers,
                shuffle=False
            )
        )

    def interprete_via_prediction(self, inputs: List[str], ys: List[float]) -> List[Dict[str, Any]]:
        _dataset = _convert_to_torch_format_dataset(
            {"text": inputs, "label": ys}, "text", deepcopy(self._tokenizer),
            self._params.max_length,
            self._model_input_names + ["label"]
        )
        results = self._net.forward_for_interpretation(**{
            key: _dataset[key] for key in self._model_input_names
        })
        # to ndarray
        results = {k: v.detach().numpy() for k, v in results.items()}
        unbatched_results = unbatch_preds(results)
        unbatched_results = [
            {
                "tokens": self._tokenizer.convert_ids_to_tokens(res["input_ids"][:res["ntok"]]),
                "token_grad_sentence": res["input_emb_grad"][:res["ntok"]],
                **res
            }
            for res in unbatched_results
        ]
        return unbatched_results

    @classmethod
    def create(cls, params: _RegressorConfig):
        net = TransformerRegressorNet.create(params.network.to_dict())
        tokenizer = AutoTokenizer.from_pretrained(params.network.pretrained_model_name_or_path)
        return cls(params, net, tokenizer)

    @classmethod
    def _load_internal(cls, artifact_path: str, params: _RegressorConfig) -> "TransformerRegressor":
        net = TransformerRegressorNet.load(os.path.join(artifact_path, NET_FILENAME))
        tokenizer = AutoTokenizer.from_pretrained(params.network.pretrained_model_name_or_path)
        return cls(params, net, tokenizer)

    def _save_internal(self, artifact_path: str):
        self._net.save(os.path.join(artifact_path, NET_FILENAME))


def _convert_to_torch_format_dataset(
    input_dict: Dict[str, Any],
    text_key: str,
    tokenizer: AutoTokenizer,
    max_length: Optional[int],
    columns: List[str]
) -> HFDataset:
    if text_key not in input_dict:
        raise ValueError(f"Specified key({text_key}) must be contained in input_dict.")
    dataset = dict_to_hfdataset(input_dict)
    dataset = dataset.map(
        # NOTE: Padding to specific length.
        # https://huggingface.co/transformers/preprocessing.html#everything-you-always-wanted-to-know-about-padding-and-truncation  # noqa
        lambda batch: tokenizer(batch[text_key], padding="max_length", truncation=True, max_length=max_length),
        batched=True, batch_size=MAP_BATCHSIZE
    )
    dataset.set_format('torch', columns=columns)
    return dataset


def _train_test_split(
    arr1: List[Any], arr2: List[Any], test_ratio: float = 0.0
) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    if len(arr1) != len(arr2):
        raise ValueError(f"Inconsistent length: {len(arr1)} != {len(arr2)}")
    if (test_ratio > 1.0) or (test_ratio < 0.0):
        raise ValueError(f"Invalid ratio: {test_ratio}")
    if test_ratio == 0:
        return arr1, [], arr2, []
    else:
        train_arr1, test_arr1, train_arr2, test_arr2 = train_test_split(
            arr1, arr2, test_size=test_ratio, random_state=SETTINGS.random_seed)
        return train_arr1, test_arr1, train_arr2, test_arr2
