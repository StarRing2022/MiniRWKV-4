import gradio as gr
import os, gc, copy, torch
os.environ["RWKV_CUDA_ON"] = '1' # if '1' then use CUDA kern0el for seq mode (much faster)
from datetime import datetime
import yaml
import time
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from easynmt import EasyNMT
#from minirwkv4 import blipcaption
from minirwkv4 import vitgptcaption
#from minirwkv4 import vitvqa
from minirwkv4 import blipvqa

translatemodel = EasyNMT('opus-mt')

ctx_limit = 2048 #3B模型最大值为4096，7B为8192
title = "MiniRWKV-4：基于RWKV-4 + BLIP/VIT-GPT的多模态图文对话大模型"
description = """<h3>MiniRWKV-4的例子，上传你的图片并开始聊天!</h3>"""
article = """<p>开源地址：<a href='https://github.com/Vision-CAIR/MiniGPT-4'>StarRing2022/MiniRWKV-4</a></p>"""

def readcog(path):
    with open(path, 'r',encoding='UTF-8') as file:
        data = file.read()
        result = yaml.safe_load(data)
        return result

LMyamlres = readcog("./config/minirwkv4.yaml")

#model_path = LMyamlres['model-language']['3Bpath']
model_path = LMyamlres['model-language']['7Bpath']
model = RWKV(model=model_path, strategy='cuda fp16i8 *8 -> cuda fp16') #加载模型
tokenizer_path = LMyamlres['model-language']['tokenizer']
pipeline = PIPELINE(model, tokenizer_path) #加载tokenizer




def upload_file(chatbot, upload_btn):
    chat_history = chatbot
    file = upload_btn

    lipres=""
    #vcaption = blipcaption.get_blipcap(file.name)
    vcaption = vitgptcaption.get_vitgptcap(file.name)

    lipres = translatemodel.translate(vcaption, target_lang='zh')

    lipres = str(lipres)

    time.sleep(1)

    
    rwkvres = get_answer(botmode = 0,message = lipres)
    #print(rwkvres)

    chatres = str(lipres+"。"+rwkvres)

    #chat_history = chat_history + [((file.name,), lipres)]

    chat_history = chat_history + [((file.name,), chatres)]

    return chat_history

def reset_chat(input_txt,chatbot):
    return None, None

def dispic(upload_btn):
    try:
        if not upload_btn:
            return upload_btn 
        else:
            #print(upload_btn.name)
            upload_btn.name=""
            upload_btn=None
    except:
        pass

    return upload_btn 



def generate_prompt(prompt,cardiogenic_prompt=None,operability_prompt=None,exogenous_prompt=None):
    promptalter = ""
    if cardiogenic_prompt:
        promptalter = promptalter + cardiogenic_prompt
    if operability_prompt:
        promptalter = promptalter + operability_prompt
    if exogenous_prompt:
        promptalter = promptalter + exogenous_prompt
    promptalter = promptalter + prompt
    #print(promptalter)
    return f"Human: {promptalter} \nAssistant:"

def get_answer(botmode,message,token_count=500,temperature=0.8,top_p=0.7,presencePenalty=0.1,countPenalty=0.1):
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = float(presencePenalty),
                     alpha_presence = float(countPenalty),
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0]) # stop generation whenever you see any token here
    message = message.strip().replace('\r\n','\n')

   
    #prompt种类：cardiogenic,operability,exogenous
    CPyamlres = readcog("./prompts/cardiogenic.yaml")
    cardiogenic_prompt=CPyamlres['promptwords']['nature']
    #print(cardiogenic_prompt) #心源性
    OPyamlres = readcog("./prompts/operability.yaml")
    operability_prompt=OPyamlres['promptwords']['task']
    #print(operability_prompt) #操作性
    EXyamlres = readcog("./prompts/exogenous.yaml")
    exogenous_prompt=EXyamlres['promptwords']['instruction'] #外因性
    #print(exogenous_prompt)

    # 判断提示模式
    if(botmode==1):
        # 提示模式1
        ctx = generate_prompt(message,cardiogenic_prompt=cardiogenic_prompt).strip()
        #print(ctx)
    elif(botmode==2):
        # 提示模式2
        ctx = generate_prompt(message,cardiogenic_prompt=cardiogenic_prompt,operability_prompt=operability_prompt).strip()
        #print(ctx)
    elif(botmode==3):
        # 提示模式3
        ctx = generate_prompt(message,cardiogenic_prompt=cardiogenic_prompt,operability_prompt=operability_prompt,exogenous_prompt=exogenous_prompt).strip()
        #print(ctx)
    elif(botmode==0):
        # 不使用提示
        ctx = generate_prompt(message).strip()
        #print(ctx)

    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    for i in range(int(token_count)):
        out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state)
       
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        
        tmp = pipeline.decode(all_tokens[out_last:])

        if '\ufffd' not in tmp:
            out_str += tmp
            out_last = i + 1

        
    del out
    del state
    gc.collect()
    torch.cuda.empty_cache()
    answer = out_str.strip()
    
    return answer


def gen_response(
    input_txt,
    chatbot,
    upload_btn,
    temperature=0.9,
    top_p=0.7,
    presencePenalty = 0.1,
    countPenalty = 0.1,
):  
    usrmsg = input_txt
    chat_history = chatbot


    response = ""
    #判断是否结合图片进行对话
    BotMode = 1 # 1为只加载心源性提示；2为加载心源性提示和操作性提示；3为三种提示都加载
    try:
        if not upload_btn:
            BotMode = 1
            response = get_answer(botmode = BotMode,message=usrmsg,token_count=1024,temperature=temperature,top_p=top_p,presencePenalty=presencePenalty,countPenalty=countPenalty) 
        else:
            BotMode = 3

            #print(upload_btn.name)
            file = upload_btn
            imgquery = translatemodel.translate(input_txt, target_lang='en')
            #print(imgquery)

            #vqares = vitvqa.get_vqares(file.name,imgquery)
            vqares = blipvqa.get_bqares(file.name,imgquery)
            #print(vqares)

            if vqares.isdigit():
                pass
            else:
                vqares = translatemodel.translate(vqares, target_lang='zh')
            
            #print(vqares)

            msgvqa = f"已知问答题，对于问题：{usrmsg}，问题的答案是：{vqares}。请再次回答：{usrmsg}"
            
            #二阶段推理
            response_step1 = get_answer(botmode = 0,message=msgvqa,token_count=1024,temperature=temperature,top_p=top_p,presencePenalty=presencePenalty,countPenalty=countPenalty)

            response_step2 = get_answer(botmode = 3,message=response_step1,token_count=1024,temperature=temperature,top_p=top_p,presencePenalty=presencePenalty,countPenalty=countPenalty)

            response = response_step1+"\n"+response_step2

    except:
        BotMode = 2
        response = get_answer(botmode = BotMode,message=usrmsg,token_count=1024,temperature=temperature,top_p=top_p,presencePenalty=presencePenalty,countPenalty=countPenalty)
    
    #print(response)
    chat_history.append((usrmsg, response))
    
    return "",chat_history



with gr.Blocks(title = "MiniRWKV-4 Demo") as demo:

    gr.HTML(f"<div style=\"text-align: center;\">\n<h1>🐦{title}</h1>\n</div>")
    gr.Markdown(description)
    gr.Markdown(article)

    with gr.Row():
        chatbot = gr.Chatbot(value=[], label = "MiniRWKV-4",elem_id="chatbot").style(height=500)
    
    with gr.Row():
        with gr.Column(scale=0.85):
            input_txt = gr.Textbox(show_label=False,placeholder="输入内容，或上传一张图片")
        with gr.Column(scale=0.15, min_width=0):
            upload_btn = gr.UploadButton("📁", file_types=["image"])
            disload_btn = gr.Button("清除图片")
        
    with gr.Row():
        temperature = gr.Slider(0.2, 2.0, label="创造力", step=0.1, value=1.2)
        top_p = gr.Slider(0.0, 1.0, label="注意力参数", step=0.05, value=0.5)
        presence_penalty = gr.Slider(0.0, 1.0, label="在场惩罚参数", step=0.1, value=0.4)
        count_penalty = gr.Slider(0.0, 1.0, label="计数惩罚参数", step=0.1, value=0.4)
    
    submit_btn = gr.Button("提交", variant="primary")
    clear_btn = gr.Button("清空", variant="secondary")
            
    input_txt.submit(gen_response, [input_txt, chatbot, upload_btn, temperature, top_p, presence_penalty, count_penalty], [input_txt, chatbot])
    submit_btn.click(gen_response, [input_txt, chatbot, upload_btn, temperature, top_p, presence_penalty, count_penalty], [input_txt, chatbot])
    clear_btn.click(reset_chat, [input_txt,chatbot], [input_txt,chatbot])

    upload_btn.upload(upload_file, [chatbot, upload_btn], [chatbot])
    disload_btn.click(dispic,[upload_btn],[upload_btn])
    
demo.queue(concurrency_count=1, max_size=10)
demo.launch(share=False)

# if __name__ == "__main__":
#     token_count = 500
#     args = PIPELINE_ARGS(temperature = max(0.2, float(0.8)), top_p = float(0.7),
#                      alpha_frequency = 0.1,
#                      alpha_presence = 0.1,
#                      token_ban = [], # ban the generation of some tokens
#                      token_stop = [0]) # stop generation whenever you see any token here
#     message = "你好"
#     message = message.strip().replace('\r\n','\n')
#     ctx = generate_prompt(message).strip()
#     #print(ctx)

#     all_tokens = []
#     out_last = 0
#     out_str = ''
#     occurrence = {}
#     state = None
#     for i in range(int(token_count)):
#         out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state)
       
#         for n in occurrence:
#             out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

#         token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
#         if token in args.token_stop:
#             break
#         all_tokens += [token]
#         if token not in occurrence:
#             occurrence[token] = 1
#         else:
#             occurrence[token] += 1
        
#         tmp = pipeline.decode(all_tokens[out_last:])

#         if '\ufffd' not in tmp:
#             out_str += tmp
#             out_last = i + 1

        
#     del out
#     del state
#     gc.collect()
#     torch.cuda.empty_cache()
#     answer = out_str.strip()

#     print(answer)


    