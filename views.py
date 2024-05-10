from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.db import connection
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv
import os
import ast
import random
import json


def home(request):
    return render(request, 'home.html')
def generate_lyrics(request):
    lyrics_list=[]
    if request.method == 'POST' :
        print(request.POST)
        artist = request.POST.get('artist')
        genre = request.POST.get('genre')
        prompt = request.POST.get('prompt')
        prompt1 = prompt
        prompt = " ".join([prompt_explore(artist, genre), prompt])
        print(prompt)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # model = GPT2LMHeadModel.from_pretrained('gpt2')
        current_directory = os.path.dirname(os.path.abspath(__file__))
        model_file_path = os.path.join(current_directory, 'lyrics_generator_fine_tuned_model.pt')
        model = torch.load(model_file_path, map_location=torch.device('cpu'))
        lyrics = generate(model, tokenizer, prompt)
        print("*******************aFTER THE fUNCTION CALL DONE****************")
        print(lyrics)
        lyrics_list = print_lyrics(lyrics)
        print(lyrics_list)
        context = {'lyrics_list': lyrics_list,
                   'genre': genre,
                   'artist':artist,
                   'prompt': prompt1}  # Pass lyrics_list to context
        return render(request, 'home.html', context=context)  # Pass context to render
def prompt_explore(artist,genre):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(BASE_DIR, 'static', 'images', 'Lyrics_Prompt.txt')
    text=''
    with open(file_path, 'r') as file:
        file_content = file.read()
        data_dict = ast.literal_eval(file_content)
        text = random.choice(data_dict[artist])
    return text



def print_lyrics(lyrics):
    lyrics_list=[]
    for curse in lyrics:
        for verse in curse:
            for lines in verse:
                lyrics_list.append(lines)
                print(lines)
            print()
            lyrics_list.append('\n')
    return lyrics_list

def generate(
    model,
    tokenizer,
    prompt,
    entry_count=1,
    entry_length=150,  # maximum number of words
    top_p=0.8,
    temperature=1.,
):
    model.eval()

    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                # Break the loop if the generated token corresponds to the end-of-text token
                if next_token == tokenizer.eos_token_id:
                    break

            output_list = list(generated.squeeze().numpy())
            output_text = tokenizer.decode(output_list)
            print("*******************Before the SPACES getting added****************")
            print(output_text)
            # Split the generated text into words
            words = output_text.split(' ')

            # Divide the words into lines with 5-8 words per line
            lines = []
            line = []
            word_count = 0
            for word in words:
                if word_count + len(word.split()) <= 8:
                    line.append(word)
                    word_count += len(word.split())
                else:
                    lines.append(' '.join(line))
                    line = [word]
                    word_count = len(word.split())
            if line:
                lines.append(' '.join(line))

            print("*******************Before the Verse getting added****************")
            print(output_text)
            # Divide the lines into verses with varying number of lines
            verses = []
            verse = []
            for line in lines:
                if len(verse) < 5:  # Targeting 5-8 lines per verse
                    verse.append(line)
                else:
                    verses.append(verse)
                    verse = [line]
            if verse:
                if len(verse) > 1:  # Avoid separate verses for just a line or two
                    verses.append(verse)

            # Mix the last verse with the rest of the lines
            if len(verses) > 1:
                last_verse = verses.pop()
                verses[-1].extend(last_verse)

            generated_list.append(verses)

    return generated_list
        







