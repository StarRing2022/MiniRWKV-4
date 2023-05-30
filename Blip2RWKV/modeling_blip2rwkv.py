import copy
import os
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
import warnings
from torch import Tensor, nn

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Blip2VisionModel,
    Blip2QFormerModel,
    Blip2Model,
    Blip2PreTrainedModel,
    Blip2ForConditionalGeneration,
    GenerationConfig,
)
from transformers.models.blip_2.modeling_blip_2 import (
    Blip2ForConditionalGenerationModelOutput,
)
from transformers.utils import logging
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList

from modeling_rwkv import (
    RwkvForCausalLM
)
from configuration_blip2rwkv import Blip2RWKVConfig


logger = logging.get_logger(__name__)


class Blip2RWKVConditionalGeneration(Blip2ForConditionalGeneration):
    config_class = Blip2RWKVConfig

    def __init__(self, config: Blip2RWKVConfig):
        Blip2PreTrainedModel.__init__(self, config)
        # NOTE: we only initialize Blip2PreTrainedModel
        # directly call super().__init__() will cause error since ChatGLM cannot be found by AutoModel

        self.vision_model = Blip2VisionModel(config.vision_config).to("cuda")

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size).to("cuda")
        )
        self.qformer = Blip2QFormerModel(config.qformer_config).to("cuda")

        self.language_projection = nn.Linear(
            config.qformer_config.hidden_size, config.text_config.hidden_size
        ).to("cuda")
        #self.language_model = RwkvForCausalLM(config.text_config)
        self.language_model = RwkvForCausalLM.from_pretrained("RWKV-4-Raven-3B-v11-zh",device_map='auto').to("cuda")
        #print(self.language_model )

        # Initialize weights and apply final processing
        # self.post_init()

    def setup_dtype(self, vision_encoder_dtype: str = "fp32", lm_dtype: str = "fp16"):
        if vision_encoder_dtype == "fp32":
            self.vision_model = self.vision_model.float().cuda()
        elif vision_encoder_dtype == "fp16":
            self.vision_model = self.vision_model.half().cuda()
        else:
            raise NotImplementedError(
                f"Unsupported vision_encoder_dtype: {vision_encoder_dtype}"
            )

        if lm_dtype == "fp32":
            self.language_model = self.language_model.float()
        elif lm_dtype == "fp16":
            self.language_model = self.language_model.half()
        elif lm_dtype == "int4":
            self.language_model = self.language_model.half().quantize(4)
        elif lm_dtype == "int8":
            self.language_model = self.language_model.half().quantize(8)
        else:
            raise NotImplementedError(f"Unsupported lm_dtype: {lm_dtype}")

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        image_slot_offset: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        """_summary_

        Args:
            pixel_values (torch.FloatTensor): _description_
            input_ids (torch.FloatTensor): input_ids[:, :num_query_tokens] should be filled with tokenizer.unk_token_id
            image_slot_offset (Optional[torch.LongTensor], optional): if not set, all vtokens are placed as prefix (image_slot_offset = torch.zeros(bsz)). Defaults to None.
            attention_mask (Optional[torch.LongTensor], optional): _description_. Defaults to None.
            output_attentions (Optional[bool], optional): _description_. Defaults to None.
            output_hidden_states (Optional[bool], optional): _description_. Defaults to None.
            labels (Optional[torch.LongTensor], optional): _description_. Defaults to None.
            return_dict (Optional[bool], optional): _description_. Defaults to None.

        Returns:
            Union[Tuple, Blip2ForConditionalGenerationModelOutput]: _description_
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        # 关键步骤，将图片进行embedding编码，然后送入LM
        language_model_inputs = self.language_projection(query_output)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        #print(inputs_embeds.shape) #[1,27,2560]
        if image_slot_offset is None:
            # image as prefix
            # update data to avoid inplace operation of leaf Variable
            inputs_embeds.data[
                :, : self.config.num_query_tokens, : # num_query_tokens = 27
            ] = language_model_inputs
        else:
            for i, offset in enumerate(image_slot_offset):
                inputs_embeds.data[
                    i, offset : offset + self.config.num_query_tokens, :
                ] = language_model_inputs[i]

        outputs = self.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]
        loss = None
        # we compute the loss here since we need to take into account the sequence length of the query embeds
        if labels is not None:
            logits = logits[:, -labels.size(1) :, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="mean")

            loss = loss_fct(
                shift_logits.view(-1, self.config.text_config.vocab_size),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

if __name__ == "__main__":
    #Blip2RWKV测试
    blip2rwkvconfig = Blip2RWKVConfig()
    blip2RWKVConditionalGeneration = Blip2RWKVConditionalGeneration(config=blip2rwkvconfig)
    blip2RWKVConditionalGeneration.setup_dtype()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    
    from PIL import Image
    from transformers import BlipProcessor
    from lavis.models import load_model_and_preprocess
    
    raw_image = Image.open('gen.png').convert('RGB')
    caption = "一个男孩抱着一只猫，猫咪看起来很享受。"

    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
    model = model.to(device)

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    text_input = txt_processors["eval"](caption)
    sample = {"image": image, "text_input": [text_input]}

    #print(image.shape)
    #print(text_input)

    from transformers import GPTNeoXTokenizerFast
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("RWKV-4-Raven-3B-v11-zh")
    text_input = tokenizer.encode(text_input, return_tensors='pt')
    blip2rwkvoutput = blip2RWKVConditionalGeneration.forward(pixel_values=image.to(device),input_ids=text_input.to(device),labels=text_input.to(device))
    #print(blip2rwkvoutput)