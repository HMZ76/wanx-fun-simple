import importlib.util

from diffusers import AutoencoderKL
from transformers import (AutoProcessor, AutoTokenizer, CLIPImageProcessor,
                          CLIPTextModel, CLIPTokenizer,
                          CLIPVisionModelWithProjection, LlamaModel,
                          LlamaTokenizerFast, LlavaForConditionalGeneration,
                          Mistral3ForConditionalGeneration, PixtralProcessor,
                          Qwen3ForCausalLM, T5EncoderModel, T5Tokenizer,
                          T5TokenizerFast)

try:
    from transformers import (Qwen2_5_VLConfig,
                              Qwen2_5_VLForConditionalGeneration,
                              Qwen2Tokenizer, Qwen2VLProcessor)
except:
    Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer = None, None
    Qwen2VLProcessor, Qwen2_5_VLConfig = None, None
    print("Your transformers version is too old to load Qwen2_5_VLForConditionalGeneration and Qwen2Tokenizer. If you wish to use QwenImage, please upgrade your transformers package to the latest version.")


from .wan_image_encoder import CLIPModel
from .wan_text_encoder import WanT5EncoderModel
from .wan_transformer3d import (Wan2_2Transformer3DModel, WanRMSNorm,
                                WanSelfAttention, WanTransformer3DModel)

from .wan_vae import AutoencoderKLWan, AutoencoderKLWan_


