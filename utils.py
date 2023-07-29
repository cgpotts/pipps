from contextlib import nullcontext
import scikits.bootstrap as boot
from nltk.tokenize import TweetTokenizer
import numpy as np
import openai
import pandas as pd
import pytest
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


main_regex = re.compile(r"""
    (\S+)
    \s+
    (?:though|as)
    \s+
    (?:\S+\s+)+
    """, re.VERBOSE | re.I)

def is_match(s):
    m = main_regex.search(s)
    return m and m.group(1).lower() not in {"as", "even", "but", "and"}


@pytest.mark.parametrize("ex, expected", [
    ("Happy though we were", True),
    ("Happy though we felt", True),
    ("Even though we were there", False),
    ("Happy as we were", True),
    ("HAPPY AS WE WERE", True)
])
def test_regex(ex, expected):
    result = is_match(ex)
    assert result == expected


tokenizer = TweetTokenizer()

def item(ex, embedding="", preposition="though"):
    assert "*" in ex
    assert "GAP" in ex

    if embedding:
        ex = ex.replace("though", f"though {embedding}")

    if preposition == "asas":
        func = lambda x: f"as {x.group(1).lower()} as"
        ex = re.sub(r"(\w+\*)\s+though", func, ex, re.I)
    else:
        ex = ex.replace("though", preposition)

    toks = tokenizer.tokenize(ex)

    # Get target predicate:
    ai = toks.index("*")
    pred = toks[ai - 1]
    ex = ex.replace("*", "")

    # Get the word after the GAP:
    gi = toks.index("GAP")
    t = toks[gi + 1]

    # The PiPP:
    c1 = ex.replace("GAP", "").strip()

    # Regular Adverbial clause:
    c2 = ex.replace(pred, "").replace("GAP", pred.lower()).strip()

    # Fronting with no gap:
    c3 = ex.replace("GAP", pred.lower()).strip()

    # No fronting but with a gap:
    c4 = ex.replace(pred, "").replace("GAP", "").strip()

    t2 = pred.lower()

    return {
        "PiPP (Filler/Gap)": (cleanup(c1), t),
        "PP (No Filler/No Gap)": (cleanup(c2), t2),
        "Filler/No Gap": (cleanup(c3), t2),
        "No Filler/Gap": (cleanup(c4), t)}


def cleanup(s):
    s = s.replace(" ,", ",").replace(" .", ".").replace("  ", " ")
    s = s.replace("as as", "as")
    s = s[0].upper() + s[1: ]
    return s


@pytest.mark.parametrize("ex, embedding, preposition, expected", [
    (
        "Happy* though we were GAP with the idea, we had to reject it.", "", "though",
        {'PiPP (Filler/Gap)': ('Happy though we were with the idea, we had to reject it.', 'with'),
         'PP (No Filler/No Gap)': ('Though we were happy with the idea, we had to reject it.', 'happy'),
         'Filler/No Gap': ('Happy though we were happy with the idea, we had to reject it.', 'happy'),
         'No Filler/Gap': ('Though we were with the idea, we had to reject it.', 'with')
        }
    ),
    (
        "Happy* though we were GAP with the idea, we had to reject it.", "", "asas",
        {'PiPP (Filler/Gap)': ('As happy as we were with the idea, we had to reject it.', 'with'),
         'PP (No Filler/No Gap)': ('As we were happy with the idea, we had to reject it.', 'happy'),
         'Filler/No Gap': ('As happy as we were happy with the idea, we had to reject it.', 'happy'),
         'No Filler/Gap': ('As we were with the idea, we had to reject it.', 'with')
        }
    ),
    (
        "Happy* though we were GAP with the idea, we had to reject it.", "they said that we knew that", "as",
        {'PiPP (Filler/Gap)': ('Happy as they said that we knew that we were with the idea, we had to reject it.', 'with'),
         'PP (No Filler/No Gap)': ('As they said that we knew that we were happy with the idea, we had to reject it.', 'happy'),
         'Filler/No Gap': ('Happy as they said that we knew that we were happy with the idea, we had to reject it.', 'happy'),
         'No Filler/Gap': ('As they said that we knew that we were with the idea, we had to reject it.', 'with')
        }
    ),
    (
        "It was reportedly an accident, deliberate* though it may have seemed GAP.", "they said that we knew that", "[MASK]",
        {'PiPP (Filler/Gap)': ('It was reportedly an accident, deliberate [MASK] they said that we knew that it may have seemed.', '.'),
         'PP (No Filler/No Gap)': ('It was reportedly an accident, [MASK] they said that we knew that it may have seemed deliberate.', 'deliberate'),
         'Filler/No Gap': ('It was reportedly an accident, deliberate [MASK] they said that we knew that it may have seemed deliberate.', 'deliberate'),
         'No Filler/Gap': ('It was reportedly an accident, [MASK] they said that we knew that it may have seemed.', '.')
        }
    )
])
def test_item(ex, embedding, preposition, expected):
    result = item(ex, embedding=embedding, preposition=preposition)
    assert result == expected



def run_gpt3(prompts, engine="text-davinci-003", temperature=0.1, top_p=0.95, max_tokens=1, **gpt3_kwargs):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompts,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        echo=True,
        logprobs=1,
        n=1,
        **gpt3_kwargs)

    # From here, we parse each example to get the values we need:
    data = []
    for ex, prompt in zip(response["choices"], prompts):
        tokens = ex["logprobs"]["tokens"]
        logprobs = list(ex["logprobs"]["token_logprobs"])
        logprobs[0] = 0.0 # The API puts None in this position.
        if "<|endoftext|>" in tokens:
            end_i = tokens.index("<|endoftext|>")
            tokens = tokens[ : end_i]  # This leaves off the "<|endoftext|>"
            probs = probs[ : end_i]    # token -- perhaps dubious.

        prompt_indices, gen_indices = _find_generated_answer(tokens, prompt)
        
        gen_tokens = [tokens[i] for i in gen_indices]
        gen_scores = [logprobs[i] for i in gen_indices]
        gen_text = "".join(gen_tokens)

        prompt_tokens = [tokens[i] for i in prompt_indices]
        prompt_scores = [logprobs[i] for i in prompt_indices]
        prompt_text = "".join(prompt_tokens)
        assert prompt_text == prompt, f"Prompt: {prompt}\nOurs: {prompt_text}"

        data.append({
            "fulltext": ex['text'],
            "prompt": prompt_text,
            "prompt_tokens": prompt_tokens,
            "prompt_scores": prompt_scores,
            "gen_text": gen_text,
            "gen_tokens": gen_tokens,
            "gen_scores": gen_scores
        })

    return data[0]


def _find_generated_answer(tokens, prompt):
    extended = ""
    for i, tok in enumerate(tokens):
        extended += tok
        if extended == prompt:
            return list(range(i+1)), list(range(i+1, len(tokens)))
    return None, None


def load_hugging_face_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        padding='longest',
        truncation='longest_first',
        max_length=2000)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model


def run_hugging_face(prompts, tokenizer, model, scorer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(device)

    with torch.inference_mode():
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
            model_output = model(prompt_ids)

    prompt_scores = model_output.logits.softmax(-1).log().detach().numpy()

    data = []
    iterator = zip(prompts, prompt_ids, prompt_scores)
    for prompt, prompt_id, prompt_score in iterator:
        prompt_tokens = [t.strip("Ġ") for t in tokenizer.convert_ids_to_tokens(prompt_id)]

        pred_scores = scorer(prompt_score, prompt_id)

        data.append({
            "fulltext": prompt,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "prompt_scores": pred_scores})

    return data[0]


def run_hugging_face_autoregressive(prompts, tokenizer, model):

    def scorer(prompt_score, prompt_id):
        pred_scores = [0.0] # First token gets probability 1.
        # Subsequent token logprobs are those predicted at the previous
        # timestep.
        for i, tokid in enumerate(prompt_id[1: ]):
            pred_scores.append(float(prompt_score[i][tokid]))
        return pred_scores

    return run_hugging_face(prompts, tokenizer, model, scorer)


def run_hugging_face_bidirectional(prompts, tokenizer, model):

    def scorer(prompt_score, prompt_id):
        return [float(prompt_score[i][tokid]) for i, tokid in enumerate(prompt_id)]

    return run_hugging_face(prompts, tokenizer, model, scorer)


def get_cis(s):
    vals = np.array([x for x in s.values if not pd.isnull(x)])
    mu = vals.mean()
    s1a, s1b = boot.ci(vals)
    s1a, s1b = mu -s1a, s1b-mu
    return [s1a, s1b]
