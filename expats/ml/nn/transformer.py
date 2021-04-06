
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel
from transformers.file_utils import ModelOutput
from transformers.optimization import AdamW

from expats.common.instantiate import BaseConfig
from expats.ml.nn.base import NNModuleBase


class TransformerClassifierNetConfig(BaseConfig):
    num_class: int
    pretrained_model_name_or_path: str
    lr: float


class TransformerRegressorNetConfig(BaseConfig):
    pretrained_model_name_or_path: str
    lr: float
    output_normalized: bool


class TransformerNetBase(NNModuleBase):
    def __init__(
        self,
        params: BaseConfig,
        transformer: PreTrainedModel,
        linear: torch.nn.Linear,
    ):
        super().__init__(params=params)
        self._transformer = transformer
        self._linear = linear

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None
    ):
        output = self._forward_transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False
        )
        return self._transformer_output2logit(output)

    def _forward_transformer(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False
    ) -> ModelOutput:
        if token_type_ids is None:
            output = self._transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states
            )
        else:
            output = self._transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=output_hidden_states
            )
        return output

    def _transformer_output2logit(self, output: ModelOutput):
        raise NotImplementedError

    def _calculate_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_nb):
        y_hat = self(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids", None))
        loss = self._calculate_loss(y_hat, batch["label"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_nb):
        y_hat = self(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids", None))
        loss = self._calculate_loss(y_hat, batch["label"])
        self.log("val_loss", loss)

    def configure_optimizers(self):
        # FIXME: support to freeze BERT params
        optimizer = AdamW(self.parameters(), lr=self._params.lr)
        return optimizer


class TransformerClassifierNet(TransformerNetBase):
    """BERT-based classifier
    """
    config_class = TransformerClassifierNetConfig

    def _transformer_output2logit(self, output):
        # (batchsize, num_token, hidden_size)
        h = output["last_hidden_state"]
        # (batchsize, hidden_size)
        h_cls = h[:, 0]
        logits = self._linear(h_cls)
        return logits

    def forward_for_interpretation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None
    ):
        with torch.set_grad_enabled(True):
            output = self._forward_transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True
            )
            logits = self._transformer_output2logit(output)
        probas = torch.nn.functional.softmax(logits, dim=-1)
        # NOTE: https://github.com/PAIR-code/lit/blob/a88c58005c3b15694125e15e6165ee5fba7407d0/lit_nlp/examples/sst_pytorch_demo.py#L129
        # (batchsize, )
        scalar_pred_for_gradients = torch.max(probas, dim=1, keepdim=False)[0]
        # NOTE: gradient with respect to hidden states in first layer
        # (batchsize, num_token, hidden_size)
        input_emb_grad = torch.autograd.grad(
            scalar_pred_for_gradients, output["hidden_states"][0],
            grad_outputs=torch.ones_like(scalar_pred_for_gradients),
        )[0]
        return {
            "probas": probas,
            "input_ids": input_ids,
            "ntok": torch.sum(attention_mask, dim=1),
            "cls_emb": output["last_hidden_state"][:, 0],
            "input_emb_grad": input_emb_grad
        }

    def _calculate_loss(self, logits, target):
        return F.cross_entropy(logits, target)

    @classmethod
    def create(cls, params: Dict) -> "TransformerClassifierNet":
        params_ = TransformerClassifierNetConfig.from_dict(params)
        transformer = AutoModel.from_pretrained(params_.pretrained_model_name_or_path)
        linear = torch.nn.Linear(transformer.config.hidden_size, params_.num_class)
        return cls(
            params_,
            transformer,
            linear,
        )


class TransformerRegressorNet(TransformerNetBase):
    config_class = TransformerRegressorNetConfig

    def _transformer_output2logit(self, output):
        # (batchsize, num_token, hidden_size)
        h = output["last_hidden_state"]
        # (batchsize, hidden_size)
        h_cls = h[:, 0]
        logits = self._linear(h_cls)
        if self._params.output_normalized:
            logits = torch.sigmoid(logits)
        return logits

    def _calculate_loss(self, logits, target):
        return F.mse_loss(logits, torch.unsqueeze(target.float(), 1))

    def forward_for_interpretation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None
    ):
        with torch.set_grad_enabled(True):
            output = self._forward_transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True
            )
            logits = self._transformer_output2logit(output)
        scalar_pred_for_gradients = torch.squeeze(logits)
        # NOTE: gradient with respect to hidden states in first layer
        # (batchsize, num_token, hidden_size)
        input_emb_grad = torch.autograd.grad(
            scalar_pred_for_gradients, output["hidden_states"][0],
            grad_outputs=torch.ones_like(scalar_pred_for_gradients),
        )[0]
        return {
            "logits": torch.squeeze(logits),
            "input_ids": input_ids,
            "ntok": torch.sum(attention_mask, dim=1),
            "cls_emb": output["last_hidden_state"][:, 0],
            "input_emb_grad": input_emb_grad
        }

    @classmethod
    def create(cls, params: Dict) -> "TransformerRegressorNet":
        params_ = TransformerRegressorNetConfig.from_dict(params)
        transformer = AutoModel.from_pretrained(params_.pretrained_model_name_or_path)
        linear = torch.nn.Linear(transformer.config.hidden_size, 1)
        return cls(
            params_,
            transformer,
            linear,
        )
