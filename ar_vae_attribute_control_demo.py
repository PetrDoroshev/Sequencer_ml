import torch
import numpy as np
import matplotlib.pyplot as plt
from symusic import Score
from pathlib import Path
from miditok import REMI
from tqdm import tqdm
from model.VAE import VAE 
from data.attrubute import compute_attributes_for_track

def compute_attributes_from_tokens(tokens, tokenizer, pitch_diff):
    
    score = tokenizer.decode([tokens])  
    track = score.tracks[0]
    attrs = compute_attributes_for_track(score, track, pitch_diff)
    return attrs

def generate_from_z(z, temperature=1.0):

    with torch.no_grad():
        start_token = torch.full((1,), bos_id, dtype=torch.long, device=device)
        tokens = model.generate(start_token, z=z, max_len=max_len, temperature=temperature)
   
    tokens = tokens[0].cpu().tolist()
    if tokens[0] == bos_id:
        tokens = tokens[1:]
    if eos_id in tokens:
        tokens = tokens[:tokens.index(eos_id)]
    return tokens

def traverse_dim(dim_idx, min_val=-2.0, max_val=2.0, steps=7):
    
    z_base = torch.zeros(1, model.latent_dim, device=device) 
    values = np.linspace(min_val, max_val, steps)
    all_tokens = []
    for val in values:
        z = z_base.clone()
        z[0, dim_idx] = val
        tokens = generate_from_z(z, temperature=temp)
        all_tokens.append(tokens)
    return all_tokens, values

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "ar_vae_epoch50.pt"                
    tokenizer_dir = "./tokenizer"                 
    scaler_path = "attr_scaler.pkl"   

    tokenizer = REMI.from_pretrained(tokenizer_dir, local_files_only=True)
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer._tokens_to_ids(["BOS_None"])[0]
    eos_id = tokenizer._tokens_to_ids(["EOS_None"])[0]

  
    model = VAE(vocab_size, embed_dim=256, hidden_dim=256, latent_dim=64, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    reg_dims = [0, 1, 2]           
    attr_names = ['pitch_range', 'note_density', 'contour']
    num_steps = 10          
    temp = 0.9                  
    max_len = 256
    output_dir = Path("./traversal_demo")
    output_dir.mkdir(exist_ok=True)

    for dim_idx in reg_dims:
        print(f"\n=== Traversal по измерению {dim_idx} (атрибут: {attr_names[dim_idx]}) ===")
        tokens_list, values = traverse_dim(dim_idx, min_val=-3.0, max_val=3.0, steps=num_steps)
      
        attr_vals = []
        for i, (val, tokens) in enumerate(zip(values, tokens_list)):
           
            midi_path = output_dir / f"dim{dim_idx}_val{val:.2f}.mid"
            score = tokenizer.decode([tokens])
            score.dump_midi(str(midi_path))
            
            attrs = compute_attributes_from_tokens(tokens, tokenizer, 
                                                   tokenizer.config.pitch_range[1] - tokenizer.config.pitch_range[0])
            attr_val = attrs[attr_names[dim_idx]]
            attr_vals.append(attr_val)
            print(f"  val = {val:.2f} -> {attr_names[dim_idx]} = {attr_val:.3f}")

      
        plt.figure()
        plt.plot(values, attr_vals, 'o-')
        plt.xlabel(f'Значение измерения {dim_idx}')
        plt.ylabel(attr_names[dim_idx])
        plt.title(f'Traversal по измерению {dim_idx}')
        plt.grid(True)
        plt.savefig(output_dir / f"dim{dim_idx}_plot.png")
        plt.show()

   
    print("\n=== Интерполяция между двумя случайными точками ===")
    z1 = torch.randn(1, model.latent_dim, device=device)
    z2 = torch.randn(1, model.latent_dim, device=device)

    alphas = np.linspace(0, 1, 7)
    for i, alpha in enumerate(alphas):
        z_interp = (1 - alpha) * z1 + alpha * z2
        tokens = generate_from_z(z_interp, temperature=temp)
        score = tokenizer.decode([tokens])
        midi_path = output_dir / f"interp_{alpha:.2f}.mid"
        score.dump_midi(str(midi_path))
        print(f"  alpha = {alpha:.2f} -> сохранён {midi_path.name}")

    print("\nГотово! MIDI-файлы и графики сохранены в", output_dir)

