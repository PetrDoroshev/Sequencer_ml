import torch
from torch import jit
from model.VAE import VAE
from pathlib import Path
from miditok import REMI
from data.midi_pp import directory_min_max_pitch
from data.create_ds import create_tokenizer, open_tokenizer



if __name__ == "__main__":

    midi_dir = Path("./ds/vgmusic_midi/512/chunks_dedup")

    '''
    MIN_PITCH, MAX_PITCH = directory_min_max_pitch(midi_dir)
    
    TOKENIZER_PARAMS = {

        "pitch_range": (MIN_PITCH, MAX_PITCH),
        "beat_res": {(0, 4): 8, (4, 12): 4},
        "special_tokens": ["PAD", "BOS", "EOS"],
        "use_velocities": False,
        "use_chords": False,
        "use_rests": True,
        "use_tempos": True,
        "use_time_signatures": False,
        "use_programs": False,
        "use_pitch_intervals": True,
        "num_tempos": 32,  # number of tempo bins
        "tempo_range": (40, 250),  # (min, max)
        "remove_duplicated_notes": True
    } 
    ''' 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_dir = Path("./tokenizer")
    model_path = Path("best_ar_vae.pt")

    tokenizer = open_tokenizer(tok_dir)
    vocab_size = tokenizer.vocab_size

    model = VAE(vocab_size, embed_dim=256, hidden_dim=256, latent_dim=64, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    dummy_src = torch.randint(0, vocab_size, (1, 32), device=device)
    dummy_tgt = torch.randint(0, vocab_size, (1, 32), device=device)
    scripted_module = torch.jit.script(model, (dummy_src, dummy_tgt))
    scripted_module.save("ar_vae.pt")

    