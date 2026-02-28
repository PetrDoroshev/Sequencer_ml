import torch
from model.VAE import VAE, vae_loss, ar_vae_loss
from pathlib import Path
from data.create_ds import open_tokenizer, create_dataloader, create_tokenizer
from os.path import exists

def train_epoch(model, dataloader, optimizer, bos_id, eos_id, pad_id, device, beta=1.0):

    model.train()
    
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device)

        batch_size, seq_len = input_ids.shape

        src = input_ids  # (B, T)
        bos = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        decoder_input = torch.cat([bos, input_ids[:, :-1]], dim=1)  # (B, T)
        
        target = input_ids  # (B, T)

        logits, mu, logvar = model(src, decoder_input)
        
        loss, recon, kl = vae_loss(logits, target, mu, logvar, pad_id, beta)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()


    n = len(dataloader)
    return total_loss / n, total_recon / n, total_kl / n


def train_epoch_ar(
    model,
    dataloader,
    optimizer,
    bos_id,
    eos_id,
    pad_id,
    device,
    reg_dims,
    beta=1.0,
    gamma=1.0,
    delta=10.0,
):

    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_reg = 0.0

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device)
        batch_size, _ = input_ids.shape

        src = input_ids
        bos = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        decoder_input = torch.cat([bos, input_ids[:, :-1]], dim=1)
        target = input_ids

        logits, mu, logvar = model(src, decoder_input)

        if "attributes" in batch:
            attributes = batch["attributes"].to(device).float()
        else:
            # Заглушка: упорядоченные псевдо-атрибуты для AR-регуляризации.
            attributes = torch.linspace(-1.0, 1.0, steps=batch_size, device=device).unsqueeze(1)
            if len(reg_dims) > 1:
                attributes = attributes.repeat(1, len(reg_dims))

        loss, recon, kl, reg = ar_vae_loss(
            logits=logits,
            targets=target,
            mu=mu,
            logvar=logvar,
            pad_id=pad_id,
            attributes=attributes,
            reg_dims=reg_dims,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        total_reg += reg.item()

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_kl / n, total_reg / n

def validate_epoch(
    model,
    dataloader,
    bos_id,
    eos_id,
    pad_id,
    device,
    beta=1.0,
    use_ar_vae=False,
    reg_dims=None,
    gamma=1.0,
    delta=10.0,
):
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    total_reg = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            batch_size, seq_len = input_ids.shape
            
            '''
            bos = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
            eos = torch.full((batch_size, 1), eos_id, dtype=torch.long, device=device)

            src = torch.cat([bos, input_ids], dim=1)
            tgt = torch.cat([bos, input_ids], dim=1)
            target = torch.cat([input_ids, eos], dim=1)
            '''
            
            src = input_ids  # (B, T)
            bos = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
            decoder_input = torch.cat([bos, input_ids[:, :-1]], dim=1)  # (B, T)

            target = input_ids  # (B, T)
            
            logits, mu, logvar = model(src, decoder_input)

            if use_ar_vae:
                if reg_dims is None or len(reg_dims) == 0:
                    raise ValueError("reg_dims must be provided when use_ar_vae=True")

                if "attributes" in batch:
                    attributes = batch["attributes"].to(device).float()
                else:
                    # Заглушка: упорядоченные псевдо-атрибуты для AR-регуляризации.
                    attributes = torch.linspace(-1.0, 1.0, steps=batch_size, device=device).unsqueeze(1)
                    if len(reg_dims) > 1:
                        attributes = attributes.repeat(1, len(reg_dims))

                loss, recon, kl, reg = ar_vae_loss(
                    logits=logits,
                    targets=target,
                    mu=mu,
                    logvar=logvar,
                    pad_id=pad_id,
                    attributes=attributes,
                    reg_dims=reg_dims,
                    beta=beta,
                    gamma=gamma,
                    delta=delta,
                )
                total_reg += reg.item()
            else:
                loss, recon, kl = vae_loss(logits, target, mu, logvar, pad_id, beta)

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

    n = len(dataloader)
    if use_ar_vae:
        return total_loss / n, total_recon / n, total_kl / n, total_reg / n
    return total_loss / n, total_recon / n, total_kl / n
