# MiniRWKV-4
1.工程介绍：<br>
为使RWKV模型能够具有图文描述，对话，推理等多模态图文能力，主要使用了RWKV作为LLM模型，再配合CLIP，VIT等预训练模型，和Two Stage二阶段思维连提示工程技巧，完成工作。 <br>

新添加的blip2rwkv工程，则是实现了使用预训练的RWKV Raven（RWKV World模型同理，只是词表和tokenizer不同，而Dlip-RWKV则基于了RWKV World模型）预训练模型，对图片进行编码。  <br>

要注意的是，blip2rwkv使用的RWKV Raven模型为HF格式，而非原生Pth，见https://huggingface.co/StarRing2022/RWKV-4-Raven-3B-v11-zh <br>

2.主要聚合模型：<br>
config/minirwkv4.yaml 文件中有详细配置<br>
RWKV-4-Raven-3B、RWKV-4-Raven-7B（原生pth，推荐V11或V12的Eng49%-Chn49%版本）<br>
blip-image-captioning-large、vit-gpt2-image-captioning、blip-vqa-capfilt-large、vilt-b32-finetuned-vqa、vilt-b32-finetuned-vqa（图片-文本链接模型）<br>
EasyNMT（中英文翻译模型）

3.使用：<br>
环境：WIN10+Torch1.31+Cuda11.6<br>
python app.py<br>
一些测试结果在assets文件夹
