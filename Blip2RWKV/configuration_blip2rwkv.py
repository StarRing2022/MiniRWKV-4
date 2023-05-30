from transformers import (
    PretrainedConfig,
    Blip2VisionConfig, Blip2QFormerConfig
)
from configuration_rwkv import RwkvConfig

import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)



class Blip2RWKVConfig(PretrainedConfig):
    """Mainly based on Blip2Config

    Args:
        PretrainedConfig (_type_): _description_
    """
    is_composition = True

    def __init__(self, vision_config=None, qformer_config=None, text_config=None, num_query_tokens=27, **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the Blip2VisionConfig with default values.")

        if qformer_config is None:
            qformer_config = {}
            logger.info("qformer_config is None. Initializing the Blip2QFormerConfig with default values.")

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the text config with default values (`OPTConfig`).")

        self.vision_config = Blip2VisionConfig(**vision_config)
        self.qformer_config = Blip2QFormerConfig(**qformer_config)
        # text_model_type = text_config["model_type"] if "model_type" in text_config else "opt"
        # self.text_config = CONFIG_MAPPING[text_model_type](**text_config)
        self.text_config = RwkvConfig(**text_config)

        # self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.tie_word_embeddings = False                # I don't know what this is
        # self.is_encoder_decoder = self.text_config.is_encoder_decoder
        self.is_encoder_decoder = True                  # chatglm is an encoder-decoder model

        self.num_query_tokens = num_query_tokens
        self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size
        # self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        self.use_decoder_only_language_model = True             # chatglm has no encoder
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    @classmethod
    def from_vision_qformer_text_configs(
        cls,
        vision_config: Blip2VisionConfig,
        qformer_config: Blip2QFormerConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`Blip2Config`] (or a derived class) from a BLIP-2 vision model, Q-Former and language model
        configurations.

        Returns:
            [`Blip2Config`]: An instance of a configuration object
        """

        return cls(
            vision_config=vision_config.to_dict(),
            qformer_config=qformer_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["qformer_config"] = self.qformer_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output

if __name__ == "__main__":
    blip2rwkvconfig = Blip2RWKVConfig()
    print(blip2rwkvconfig)
