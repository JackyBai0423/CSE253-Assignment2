import torchaudio
from tqdm import trange
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--weights_path', type=str, default=None)
parser.add_argument('--model_id', type=str, default='small')
parser.add_argument('--save_path', type=str, default='test.wav')
parser.add_argument('--duration', type=float, default=30)
parser.add_argument('--sample_loops', type=int, default=4)
parser.add_argument('--use_sampling', type=bool, default=True)
parser.add_argument('--two_step_cfg', type=bool, default=False)
parser.add_argument('--top_k', type=int, default=250)
parser.add_argument('--top_p', type=float, default=0.0)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--cfg_coef', type=float, default=3.0)
args = parser.parse_args()

# Load model
model: MusicGen = MusicGen.get_pretrained(args.model_id)

# Load fine-tuned LM weights if specified
if args.weights_path is not None:
    print(f"Loading fine-tuned weights from: {args.weights_path}")
    state_dict = torch.load(args.weights_path, map_location="cuda")
    model.lm.load_state_dict(state_dict)

# Prepare conditioning
attributes, prompt_tokens = model._prepare_tokens_and_attributes([args.prompt], None)

# Generation parameters
model.generation_params = {
    'max_gen_len': int(args.duration * model.frame_rate),
    'use_sampling': args.use_sampling,
    'temp': args.temperature,
    'top_k': args.top_k,
    'top_p': args.top_p,
    'cfg_coef': args.cfg_coef,
    'two_step_cfg': args.two_step_cfg,
}

# Generation loop
prompt_len = prompt_tokens.shape[-1] if prompt_tokens is not None else 0
generated = []
for _ in trange(args.sample_loops, desc="Generating"):
    with model.autocast:
        gen_tokens = model.lm.generate(prompt_tokens, attributes, callback=None, **model.generation_params)
        generated.append(gen_tokens[..., prompt_len:])
        if prompt_tokens is not None:
            prompt_tokens = gen_tokens[..., -prompt_len // 2:]

gen_tokens = torch.cat(generated, dim=-1)

# Decode audio
with torch.no_grad():
    audio = model.compression_model.decode(gen_tokens, None)

audio = audio.cpu()
torchaudio.save(args.save_path, audio[0], model.sample_rate)
print(f"Saved to {args.save_path}")
