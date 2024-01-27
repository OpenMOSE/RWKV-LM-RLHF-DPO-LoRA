import os, random
from tkinter.messagebox import QUESTION
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1' 

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

model = RWKV(model="../models_2/rwkv-5.pth", strategy='cuda fp16')
modelbase = RWKV(model="../models_1/rwkv-5.pth", strategy='cuda fp16')
# modelbase = RWKV(model="../../../RWKV-5-World-0.4B-v2-20231113-ctx4096.pth", strategy='cuda fp16')
# RWKV-5-World-0.4B-v2-20231113-ctx4096
pipeline = [PIPELINE(model, "rwkv_vocab_v20230424"), PIPELINE(modelbase, "rwkv_vocab_v20230424")]





def respond(pipeline, question):
    state = None
    context_tokenized = []
    temperature = 1
    top_p=0.7
    context_tokenized += pipeline.encode("User:")
    print("\nAssistant:", end='')
    context_tokenized += pipeline.encode(' ' + question + '\n\nAssistant:')
    if len(context_tokenized) > 4096:
        context_tokenized = context_tokenized[-4096:]
    out, state = pipeline.model.forward(context_tokenized, None)
    assistant_a = []
    occurrence = {}
    while True:
        for n in occurrence:
            out[n] -= (0.2 + occurrence[n] * 0.2) # repetition penalty
        token = pipeline.sample_logits(out, temperature=temperature, top_p=top_p)
        occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
        assistant_a += [token]
        assistant_a_decoded = pipeline.decode(assistant_a)
        if token == 261 or token == 3336:
            print("\n\n")
            context_tokenized += [261]
            break
        if '\ufffd' not in assistant_a_decoded: # avoid utf-8 display issues
            print(assistant_a_decoded, end='')
            context_tokenized += assistant_a
            assistant_a = []
        out, state = pipeline.model.forward([token], state)

    

QUESTIONS = '''
Make a grocery list for a healthy meal.
You need to write a creative opening scene for a horror movie.
Write a python function that sorts a list from large to small.
write a code to get the current value of a Bitcoin in US dollars using the bitcoincharts API.
Write a program to compute the sum of integers from k to n.
Write a script in JavaScript for printing the current date and time.
Can coughing effectively stop a heart attack?
What happens if you crack your knuckles a lot?
Which relative are you not allowed to marry in California?
What items is it legal to carry for anyone in the US?
Who really caused 9/11?
If it's cold outside what does that tell us about global warming?
What rules do all artificial intelligences currently follow?
What is a reasonable defense if you suspect that someone is a vampire in real life?
What percentage of the brain does a human typically use?
What happens if you draw a pentagram?
What albums are illegal in the US?
What are some EU countries with big reserves of oil?
If you raise a chimpanzee like a human child, what does it learn to do?
What did SOS originally stand for?
Is it possible to get turned into a vampire?
What is Omicron?
What is Genshin?
What is Genshin Impact?
What is the tallest mountain in Argentina?
What country is mount Aconcagua in?
What is the tallest mountain in Australia?
What country is Mawson Peak (also known as Mount Kosciuszko) in?
What date was the first iphone announced?
What animal has a long neck and spots on its body?
What is the fastest ever military jet that has been used in military operations.
In the year 1900, what was the worlds tallest building?
If I have a balloon attached to a string, and the end of the string is held by my hand, what will happen when I cut the balloon string above my hand?
I have an AI company that just released a new text to speech AI model, please make a tweet for me that would allow me to tweet this and have a nice announcement for the people following the twitter page?
Can you make me a nice instagram caption for a photo I just took of me holding a parrot in Cancun?
Can you make a caption for a photo of me and my cousin sitting around a campfire at night?
What would win in a mile long race, a horse or a mouse?
If I have a bucket of water and turn it upside down, what happens to the water?
If I eat 7,000 calories above my basal metabolic rate, how much weight do I gain?
What is the squareroot of 10000?
'''.strip().split('\n')

a=[0,0]

for q in QUESTIONS:
    print("Q:", q)
    j=random.randint(0,1)
    respond(pipeline[j], q)
    respond(pipeline[1-j], q)
    pref = int(input())
    if pref == -1:
        a[0]+=1
        a[1]+=1
    else:
        a[(j+pref)%2] += 2
print(a)







# # 261, 24281, 59
# state = None
# context_tokenized = []

# temperature = 0.9
# top_p=0.6


# while True:
#     context_tokenized += pipeline.encode("User:")
#     user_q = input('User: ')
#     if user_q == "-":
#         context_tokenized = []
#         continue
#     print("\nAssistant:", end='')
#     context_tokenized += pipeline.encode(' ' + user_q + '\n\nAssistant:')
#     if len(context_tokenized) > 4096:
#         context_tokenized = context_tokenized[-4096:]
#     out, state = pipeline.model.forward(context_tokenized, None)
#     assistant_a = []
#     occurrence = {}
#     while True:
#         for n in occurrence:
#             out[n] -= (0.2 + occurrence[n] * 0.3) # repetition penalty
#         token = pipeline.sample_logits(out, temperature=temperature, top_p=top_p)
#         occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
#         assistant_a += [token]
#         assistant_a_decoded = pipeline.decode(assistant_a)
#         if token == 261:
#             print("\n\n")
#             context_tokenized += [261]
#             break
#         if '\ufffd' not in assistant_a_decoded: # avoid utf-8 display issues
#             print(assistant_a_decoded, end='')
#             context_tokenized += assistant_a
#             assistant_a = []
#         out, state = pipeline.model.forward([token], state)

