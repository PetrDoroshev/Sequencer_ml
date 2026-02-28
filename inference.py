import torch
from pathlib import Path
from model.VAE import VAE  
from miditok import REMI
from sklearn.neighbors import NearestNeighbors
import numpy as np
from data.create_ds import create_dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt


def generate_melody(model, tokenizer, bos_id, eos_id, max_len=512, temperature=1.0, z=None, device='cpu'):
    """
    Генерирует мелодию и возвращает список токенов (без BOS, обрезанный по EOS).
    """
    model.eval()
    with torch.no_grad():
        start_token = torch.tensor([bos_id], device=device)
        if z is None:
            z = torch.randn(1, model.latent_dim, device=device)
        else:
            z = z.unsqueeze(0).to(device) if z.dim() == 1 else z.to(device)
        print(start_token)
        # Генерация последовательности токенов (включая BOS)
        generated = model.generate(start_token, z=z, max_len=max_len, temperature=temperature)
        # generated: (1, seq_len)

    tokens = generated[0].cpu().tolist()  # список токенов

    # Убираем BOS (первый токен) и всё после EOS (если встретится)
    if tokens[0] == bos_id:
        tokens = tokens[1:]
    if eos_id in tokens:
        tokens = tokens[:tokens.index(eos_id)]

    return tokens

def collect_latent_vectors(model, dataloader, device):
    """Собирает μ (средние) из энкодера для всех примеров в даталоадере."""
    model.eval()
    z_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting latent vectors"):
            input_ids = batch["input_ids"].to(device)
            batch_size, seq_len = input_ids.shape
            # Добавляем BOS для энкодера (как при обучении)
            bos_tensor = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
            src = torch.cat([bos_tensor, input_ids], dim=1)
            mu, _ = model.encoder(src)
            z_list.append(mu.cpu().numpy())

    return np.concatenate(z_list, axis=0)

def generate_samples_and_encode(model, tokenizer, num_samples, max_len, temperature, device):
    """Генерирует num_samples мелодий из случайных z и кодирует их обратно в латентное пространство."""
    model.eval()
    z_generated = []
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Generating samples"):
            # Случайный z из N(0,I)
            z = torch.randn(1, model.latent_dim, device=device)
            # Генерируем мелодию
            tokens = generate_melody(model, tokenizer, bos_id, eos_id, max_len=max_len,
                                     temperature=temperature, z=z, device=device)
            if len(tokens) == 0:
                continue  # на всякий случай
            # Кодируем сгенерированную мелодию обратно в латентный вектор
            # Для этого нужно преобразовать токены в тензор с BOS
            input_tensor = torch.tensor([bos_id] + tokens, device=device).unsqueeze(0)
            mu, _ = model.encoder(input_tensor)
            z_generated.append(mu.cpu().numpy())
    return np.concatenate(z_generated, axis=0) if z_generated else np.array([])

def nearest_neighbor_analysis(z_train, z_gen):
    """Для каждого сгенерированного вектора находит расстояние до ближайшего обучающего."""
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(z_train)
    distances, _ = nn.kneighbors(z_gen)
    return distances.flatten()

def plot_distance_distribution(dist_train, dist_gen, title="Distance to nearest neighbor"):
    """Строит гистограммы распределений расстояний."""
    plt.figure(figsize=(10, 5))
    plt.hist(dist_train, bins=50, alpha=0.5, label='Train vs Train', density=True)
    plt.hist(dist_gen, bins=50, alpha=0.5, label='Generated vs Train', density=True)
    plt.xlabel('Euclidean distance')
    plt.ylabel('Density')
    plt.legend()
    plt.title(title)
    plt.show()

def val_latent():
    midi_dir = Path("./ds/lakhmidi/preprocessed")  # замените на ваш путь
    train_loader, _ = create_dataloader(midi_dir, tokenizer)

    # 1. Собираем латентные векторы для обучающей выборки
    print("Сбор латентных векторов для обучающей выборки...")
    z_train = collect_latent_vectors(model, train_loader, device)
    print(f"Собрано {z_train.shape[0]} векторов размерности {z_train.shape[1]}")

    # 2. Вычисляем расстояния между обучающими примерами (для сравнения)
    # Выберем случайное подмножество для ускорения (например, 5000 пар)
    np.random.seed(42)
    idx = np.random.choice(z_train.shape[0], size=min(5000, z_train.shape[0]), replace=False)
    z_train_subset = z_train[idx]
    nn_train = NearestNeighbors(n_neighbors=2, metric='euclidean')  # ищем ближайшего кроме себя
    nn_train.fit(z_train_subset)
    dist_train_to_train, _ = nn_train.kneighbors(z_train_subset, n_neighbors=2)
    dist_train_to_train = dist_train_to_train[:, 1]  # расстояние до самого себя = 0, берём второе

    # 3. Генерируем новые примеры и получаем их латентные векторы
    num_gen = 200  # количество генерируемых примеров
    max_len = 512
    temperature = 0.9
    print(f"Генерация {num_gen} примеров...")
    z_gen = generate_samples_and_encode(model, tokenizer, num_gen, max_len, temperature, device)
    if z_gen.size == 0:
        print("Не удалось сгенерировать ни одного примера.")
        exit()
    print(f"Сгенерировано и закодировано {z_gen.shape[0]} примеров")

    # 4. Расстояния от сгенерированных до обучающих
    dist_gen_to_train = nearest_neighbor_analysis(z_train, z_gen)

    # 5. Статистика
    print("\n--- Статистика расстояний до ближайшего соседа ---")
    print(f"Среднее расстояние (Train-Train): {np.mean(dist_train_to_train):.4f}")
    print(f"Медиана (Train-Train): {np.median(dist_train_to_train):.4f}")
    print(f"Среднее расстояние (Generated-Train): {np.mean(dist_gen_to_train):.4f}")
    print(f"Медиана (Generated-Train): {np.median(dist_gen_to_train):.4f}")

    # 6. Визуализация
    plot_distance_distribution(dist_train_to_train, dist_gen_to_train,
                               title="Распределение расстояний до ближайшего соседа")

    # 7. Дополнительно: t-SNE визуализация (если установлен sklearn)
    try:
        from sklearn.manifold import TSNE
        # Возьмём подвыборку для t-SNE
        sample_size = min(1000, z_train.shape[0])
        idx_tsne = np.random.choice(z_train.shape[0], size=sample_size, replace=False)
        z_train_tsne = z_train[idx_tsne]
        z_gen_tsne = z_gen[np.random.choice(z_gen.shape[0], size=min(200, z_gen.shape[0]), replace=False)]
        combined = np.vstack([z_train_tsne, z_gen_tsne])
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        combined_2d = tsne.fit_transform(combined)
        train_2d = combined_2d[:len(z_train_tsne)]
        gen_2d = combined_2d[len(z_train_tsne):]

        plt.figure(figsize=(10, 8))
        plt.scatter(train_2d[:, 0], train_2d[:, 1], c='blue', alpha=0.5, label='Train')
        plt.scatter(gen_2d[:, 0], gen_2d[:, 1], c='red', alpha=0.7, label='Generated')
        plt.legend()
        plt.title('t-SNE проекция латентного пространства')
        plt.show()
    except ImportError:
        print("sklearn не установлен, пропускаем t-SNE")


if __name__ == "__main__":

    midi_dir = "./ds/lakhmidi/preprocessed"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_dir = Path("./tokenizer")
    model_path = "vae_epoch50 (1).pt"

    tokenizer = REMI.from_pretrained(tok_dir, local_files_only=True)
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer._tokens_to_ids(["BOS_None"])[0]   
    eos_id = tokenizer._tokens_to_ids(["EOS_None"])[0]

    model = VAE(vocab_size, embed_dim=256, hidden_dim=256, latent_dim=64, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    model.eval() 

    #train_loader = create_dataloader(Path("./ds/lakhmidi/preprocessed"), tokenizer)

    tokens1 = generate_melody(model, tokenizer, bos_id, eos_id, max_len=32, temperature=1.0, device=device)
    score1 = tokenizer.decode([tokens1])
    score1.dump_midi("generated_cold.mid")
    
    #val_latent()
    '''
    train_loader, _ = create_dataloader(midi_dir, tokenizer)

    # Преобразуем обучающие последовательности в строки для быстрого сравнения
    train_sequences = []
    for batch in train_loader:
        for seq in batch["input_ids"]:
            # Убираем паддинги и превращаем в кортеж
            seq = seq[seq != pad_id].tolist()
            train_sequences.append(tuple(seq))
    train_set = set(train_sequences)

    # Генерируем новые последовательности
    generated_sequences = []
    for _ in range(200):
        tokens = generate_melody(model, tokenizer, bos_id, eos_id, max_len=512, temperature=0.9, device=device)
        generated_sequences.append(tuple(tokens))

    # Проверяем пересечение
    common = set(generated_sequences) & train_set
    print(f"Найдено точных совпадений: {len(common)} из {len(generated_sequences)}")
    '''