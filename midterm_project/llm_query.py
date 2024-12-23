"""
Script for querying LLMs to make the dataset

Author: Kade E. Carlson
Date: 09/25/24

LLMs used:
    - GPT-4o mini (OpenAI)
    - Claude 2.1 (Anthropic)
    - Command (Cohere)
    - Jamba (AI21)
    - LLaMa 2 (Meta)
    - Mistral 7B
    - GPT-Neo 1.3B
"""
import json
import os
import random
from ai21 import AI21Client
from ai21.models.chat import UserMessage
from anthropic import Anthropic
from cohere import ClientV2
from mistralai import Mistral
from openai import OpenAI
from pathlib import Path
from llama_models.models.llama3.reference_impl.generation import Llama
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
# OpenAI GPT-4o mini
client_openai = OpenAI()
client_anthropic = Anthropic()
client_co = ClientV2()
client_ai21 = AI21Client()
THIS_DIR = Path(__file__).parent.resolve()
print(THIS_DIR)
ckpt_dir = model_path = Path.home() / ".llama" / "checkpoints" / "Llama3.2-1B"
tokenizer_path = str(THIS_DIR / "llama_models/models/llama3/api/tokenizer.model")
generator = Llama.build(
                    ckpt_dir=ckpt_dir,
                    tokenizer_path=tokenizer_path,
                    max_seq_len=60,
                    max_batch_size=4
                )
api_key = os.environ["MISTRAL_API_KEY"]
client_mis = Mistral(api_key=api_key)
client_neo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
neo_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

prompts = [
    "Today I went",
    "Yesterday I was feeling",
    "I had an idea about",
    "I played a game",
    "I heard this fact the other day: ",
    "The best meal I ever had was",
    "A strange dream I had involved",
    "My inspritation comes from",
    "Technology has shaped us in these ways",
    "If I could meet any historical figure it would be",
    "Great hobbies include",
    "My favorite song is",
    "My favorite food is",
    "My daily routine involves",
    "I like this t-shirt because",
    "My biggest fear is",
    "I find fault in",
    "I like math because",
    "12 times 12 is 144. I think about it this way: ",
    "Listen to this one-liner joke: "
]

few_shot_prompts = [
    "Today I went [Example: Today I went shopping for new boots.]",
    "Yesterday I was feeling [Example: Yesterday I was feeling sick, but now I'm better.]",
    "I had an idea about [Example: I had an idea about a new game.]",
    "I played a game [Example: I played a game with friends and had fun.]",
    "I heard this fact the other day: [Example: I heard this fact the other day: Fish can only breathe underwater.]",
    "The best meal I ever had was [Example: The best meal I ever had was steak and eggs.]",
    "A strange dream I had involved [Example: A strange dream I had involved a skinned polar bear.]",
    "My inspritation comes from [Example: My inspiration comes from famous mathematicians before me.]",
    "Technology has shaped us in these ways: [Example: Technology has shaped us in these ways: less attention, more information.]",
    "If I could meet any historical figure it would be [Example: If I could meet any historical figure it would be Marcus Aurelius.]",
    "Great hobbies include [Example: Great hobbies include reading, watchmaking, and playing card games.]",
    "My favorite song is [Example: My favorite song is One by Metallica.]",
    "My favorite food is [Examle: My favorite food is mac and cheese.]",
    "My daily routine involves [Example: My daily routine involves brushing my teeth and taking a shower]",
    "I like this t-shirt because [Example: I like this t-shirt because the graphic is cool.]",
    "My biggest fear is [Example: My biggest fear is flying.]",
    "I find fault in [Example: I find fault in people that interrupt me.]",
    "I like math because [Example: I like math because it challenges me.]",
    "12 times 12 is 144. I think about it this way: [Example: 12 times 12 is 144. I think about it this way: 12 times 10 is 120 + 12 times 2 is 24 makes 144]",
    "Listen to this one-liner joke: [Example: Listen to this one-liner joke: I have an inferiority complex but it's not a very good one.]"
]

# few_shot_prompts_small = [
#     "My favorite song is [Example: My favorite song is One by Metallica.]",
#     "My favorite food is [Examle: My favorite food is mac and cheese.]",
#     "My daily routine involves [Example: My daily routine involves brushing my teeth and taking a shower]",
#     "I like this t-shirt because [Example: I like this t-shirt because the graphic is cool.]",
#     "My biggest fear is [Example: My biggest fear is flying.]",
#     "I find fault in [Example: I find fault in people that interrupt me.]",
#     "I like math because [Example: I like math because it challenges me.]",
#     "12 times 12 is 144. I think about it this way: [Example: 12 times 12 is 144. I think about it this way: 12 times 10 is 120 + 12 times 2 is 24 makes 144]",
#     "Listen to this one-liner joke: [Example: Listen to this one-liner joke: I have an inferiority complex but it's not a very good one.]"
# ]

test_prompts = [
    "Someone hurt me",
    "I experienced a miracle",
    "I wasted my life doing",
    "This new trend",
    "I can't believe I am saying this but"
]

models = ['gpt-4o-mini', 'claude-2.1', 'command-r-plus', 'jamba-1.5-large', 'llama-3.2-1B', 'mistral-large-latest', 'gpt-neo-1.3B']
filename = "llm_train_zero_shot_long.json"
if not os.path.exists(filename):
    with open(filename, 'w') as f:
        json.dump([], f)

for model in models:
    for j, prompt in enumerate(few_shot_prompts):
        temperature = random.uniform(0.3, 1)
        max_tokens = random.randint(30, 100)
        for i in range(100):
            if model == 'gpt-4o-mini':
                completion = client_openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant helping me finish prompts that I am going to feed you. You should finish theses prompts so that the final results is at minimum one sentence long and maximum three sentences long. Try to vary the responses considerably."},
                        {"role": "user", "content": f"Finish this sentence: {prompt}"}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                comp_text = completion.choices[0].message.content
                new_data = {
                    "input_prompt": prompt,
                    "completion": comp_text,
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            elif model == 'claude-2.1':
                completion = client_anthropic.messages.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": f"Finish this sentence: {prompt}"}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                comp_text = completion.content[0].text
                new_data = {
                    "input_prompt": prompt,
                    "completion": comp_text,
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            elif model == 'command-r-plus':
                completion = client_co.chat(
                    model=model,
                    messages=[
                        {"role": "user", "content": f"Finish this sentence: {prompt}"}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                comp_text = completion.message.content[0].text
                new_data = {
                    "input_prompt": prompt,
                    "completion": comp_text,
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            elif model == 'jamba-1.5-large':
                completion = client_ai21.chat.completions.create(
                    model=model,
                    messages=[
                        UserMessage(
                            content=f"Finish this sentence: {prompt}"
                        )
                    ]
                )

                comp_text = completion.choices[0].message.content
                new_data = {
                    "input_prompt": prompt,
                    "completion": comp_text,
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            elif model == 'llama-3.2-1B':
                result = generator.text_completion(
                    prompt,
                    temperature=temperature,
                    top_p=0.9,
                    max_gen_len=max_tokens,
                    logprobs=False,
                )

                comp_text = result.generation
                new_data = {
                    "input_prompt": prompt,
                    "completion": comp_text,
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            elif model == 'mistral-large-latest':
                completion = client_mis.chat.complete(
                    model=model,
                    messages=[
                        {"role": "user", "content": f"finish this sentence: {prompt}"}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                comp_text = completion.choices[0].message.content
                new_data = {
                    "input_prompt": prompt,
                    "completion": comp_text,
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            elif model == 'gpt-neo-1.3B':
                input_ids = neo_tokenizer(prompt, return_tensors="pt").input_ids
                result = client_neo.generate(
                    input_ids,
                    do_sample=True,
                    temperature=temperature,
                    max_length=max_tokens,
                )

                comp_text = neo_tokenizer.batch_decode(result)[0]
                new_data = {
                    "input_prompt": prompt,
                    "completion": comp_text,
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }

            with open(filename, 'r') as f:
                data = json.load(f)
            data.append(new_data)
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Saved prompt, model {model}, iter {i}, prompt {j}")
