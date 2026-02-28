import torch
import argparse
from pathlib import Path
from os.path import exists
from data.create_ds import create_dataloader, create_tokenizer, open_tokenizer
from model.VAE import VAE
from model.train import train_epoch, train_epoch_ar, validate_epoch

TOKENIZER_PARAMS = {

    "pitch_range": (21, 109),
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

parser = argparse.ArgumentParser()
parser.add_argument("--TOK_DIR", type=str, help="tokenizer directory")
parser.add_argument("--MIDI_DIR", type=str, help="midi directory")
parser.add_argument("--use_ar_vae", action="store_true", help="enable AR-VAE training loss")
parser.add_argument("--reg_dims", type=int, nargs="+", default=[0], help="latent dims regularized by AR-VAE")
parser.add_argument("--gamma", type=float, default=1.0, help="AR regularization strength")
parser.add_argument("--delta", type=float, default=10.0, help="AR distance scaling")

if __name__ == "__main__":

    args = parser.parse_args()

    TOK_DIR = args.TOK_DIR
    MIDI_DIR = args.MIDI_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not exists(Path(TOK_DIR, "tokenizer.json").resolve()):
        tokenizer = create_tokenizer(MIDI_DIR, TOK_DIR, TOKENIZER_PARAMS)
    else:
        tokenizer = open_tokenizer(TOK_DIR)

    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer._tokens_to_ids(["BOS_None"])[0]
    eos_id = tokenizer._tokens_to_ids(["EOS_None"])[0]

    train_loader, val_loader = create_dataloader(MIDI_DIR, tokenizer)

    model = VAE(vocab_size, embed_dim=256, hidden_dim=512, latent_dim=64, num_layers=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        if args.use_ar_vae:
            train_loss, train_recon, train_kl, train_reg = train_epoch_ar(
                model,
                train_loader,
                optimizer,
                bos_id,
                eos_id,
                pad_id,
                device,
                reg_dims=args.reg_dims,
                beta=0.1,
                gamma=args.gamma,
                delta=args.delta,
            )

            val_loss, val_recon, val_kl, val_reg = validate_epoch(
                model,
                val_loader,
                bos_id,
                eos_id,
                pad_id,
                device,
                beta=0.1,
                use_ar_vae=True,
                reg_dims=args.reg_dims,
                gamma=args.gamma,
                delta=args.delta,
            )

            print(
                f"Epoch {epoch+1:2d} | "
                f"Train Loss: {train_loss:.3f} (Recon: {train_recon:.3f} KL: {train_kl:.3f} Reg: {train_reg:.3f}) | "
                f"Val Loss: {val_loss:.3f} (Recon: {val_recon:.3f} KL: {val_kl:.3f} Reg: {val_reg:.3f})"
            )
        else:
            train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, bos_id, eos_id, pad_id, device, beta=0.1)
            
            val_loss, val_recon, val_kl = validate_epoch(
                model, val_loader, bos_id, eos_id, pad_id, device, beta=0.1)
        
            print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.3f} (Recon: {train_recon:.3f} KL: {train_kl:.3f}) | "
              f"Val Loss: {val_loss:.3f} (Recon: {val_recon:.3f} KL: {val_kl:.3f})")
        torch.save(model.state_dict(), f"vae_epoch{epoch+1}.pt")
