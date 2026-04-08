import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time
from pathlib import Path
import logging


BASE_URL = "https://www.vgmusic.com/"

DOWNLOAD_DIR = "./ds/vgmusic_midi/full"

REQUEST_DELAY = 0.1

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_session():
    
    session = requests.Session()
    session.headers.update(HEADERS)
    return session



def download_file(session, url, filepath):

    try:
        # Проверяем, существует ли уже файл
        if os.path.exists(filepath):
            logger.info(f"Файл уже существует, пропускаем: {filepath}")
            return True

        response = session.get(url, stream=True, timeout=15)
        response.raise_for_status()

        # Проверка Content-Type для надёжности (не все серверы присылают правильный тип)
        content_type = response.headers.get('Content-Type', '').lower()
        if 'audio/midi' not in content_type and not url.lower().endswith(('.mid', '.midi')):
            logger.warning(f"Ссылка {url} не похожа на MIDI (Content-Type: {content_type}), пропускаем.")
            return False

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Скачано: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при скачивании {url}: {e}")
        return False
    

def get_platform_links_from_main(session, main_url):
    """
    Извлекает ссылки на страницы платформ из левого меню главной страницы.
    Возвращает список словарей: {'name': 'snes', 'url': 'полный_url'}.
    """
    logger.info(f"Загружаем главную страницу: {main_url}")
    try:
        response = session.get(main_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        platform_links = []
        # Левое меню находится в div с id="sidebar"
        sidebar = soup.find('div', id='sidebar')
        if not sidebar:
            logger.error("Не удалось найти левое меню (div#sidebar) на главной странице.")
            return []

        # Ищем все ссылки внутри левого меню, ведущие на страницы музыки
        # Обычно это ссылки вида /music/console/...
        for link in sidebar.find_all('a', href=True):
            href = link['href']
            # Нас интересуют только ссылки, начинающиеся с /music/ (или полные url)
            if href.startswith('/music/') or 'vgmusic.com/music/' in href:
                full_url = urljoin(main_url, href)
                # Извлекаем имя платформы из последней части пути для названия папки
                parsed = urlparse(full_url)
                path_parts = [p for p in parsed.path.split('/') if p]
                # Имя папки: последний элемент пути (например, 'snes', 'nes')
                platform_name = path_parts[-1] if path_parts else 'unknown'
                platform_links.append({'name': platform_name, 'url': full_url})

        # Убираем дубликаты (одни и те же ссылки могут встречаться несколько раз)
        unique_links = {link['url']: link for link in platform_links}.values()
        result = list(unique_links)
        logger.info(f"Найдено ссылок на платформы: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"Ошибка при загрузке главной страницы: {e}")
        return []
    

def extract_midi_links_from_platform_page(session, page_url):
    """
    Находит все прямые ссылки на MIDI-файлы на странице платформы.
    Возвращает список абсолютных URL.
    """
    logger.info(f"Анализируем страницу платформы: {page_url}")
    try:
        response = session.get(page_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        midi_links = []
        # Ищем все ссылки, у которых href заканчивается на .mid или .midi
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            if href.endswith(('.mid', '.midi')):
                full_url = urljoin(page_url, link['href'])
                midi_links.append(full_url)

        # Убираем дубликаты (если вдруг ссылка встречается дважды)
        unique_midi_links = list(set(midi_links))
        logger.info(f"Найдено MIDI-файлов: {len(unique_midi_links)}")
        return unique_midi_links
    except Exception as e:
        logger.error(f"Ошибка при обработке страницы {page_url}: {e}")
        return []
    
def main():
    session = create_session()

    # Шаг 1: Получить ссылки на все платформы с главной
    platforms = get_platform_links_from_main(session, BASE_URL)
    if not platforms:
        logger.error("Не удалось получить список платформ. Завершение.")
        return

    total_midi_found = 0
    total_downloaded = 0

    # Шаг 2: Для каждой платформы собрать и скачать MIDI
    for platform in platforms:
        platform_name = platform['name']
        platform_url = platform['url']

        logger.info(f"--- Обработка платформы: {platform_name} ---")

        # Создаем папку для платформы
        platform_dir = os.path.join(DOWNLOAD_DIR, platform_name)
        Path(platform_dir).mkdir(parents=True, exist_ok=True)

        # Извлекаем все ссылки на MIDI со страницы платформы
        midi_urls = extract_midi_links_from_platform_page(session, platform_url)
        total_midi_found += len(midi_urls)

        # Скачиваем каждый файл
        for midi_url in midi_urls:
            # Формируем имя файла (берём последнюю часть URL)
            parsed_url = urlparse(midi_url)
            file_name = os.path.basename(parsed_url.path)
            # Очистка имени от недопустимых символов (на всякий случай)
            safe_filename = "".join(c for c in file_name if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
            if not safe_filename:
                safe_filename = f"midi_{hash(midi_url)}.mid"

            file_path = os.path.join(platform_dir, safe_filename)

            success = download_file(session, midi_url, file_path)
            if success:
                total_downloaded += 1

            # Задержка между скачиваниями
            time.sleep(REQUEST_DELAY)

        # Небольшая задержка между обработкой разных платформ
        time.sleep(REQUEST_DELAY)

    logger.info(f"=== Сбор завершен ===")
    logger.info(f"Всего найдено MIDI-файлов: {total_midi_found}")
    logger.info(f"Успешно скачано: {total_downloaded}")


if __name__ == "__main__":
    main()