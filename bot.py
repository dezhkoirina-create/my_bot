import os
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
import openai
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
from dotenv import load_dotenv

# 1. Загружаем секретные ключи из файла .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")

# 2. Настраиваем OpenAI
openai.api_key = OPENAI_API_KEY

# 3. Настраиваем нашу базу данных (Chroma)
# Используем локальную модель для эмбеддингов (без интернета и OpenAI)
from chromadb.utils import embedding_functions

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # Лёгкая и быстрая локальная модель
)

# Создаем или подключаемся к базе данных, которая будет сохраняться в папке "chroma_db"
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Создаем "коллекцию" (как папку) для наших документов
collection = chroma_client.get_or_create_collection(
    name="my_knowledge",
    embedding_function=embed_fn
)
# 4. Настраиваем Телеграм-бота
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# 5. Обработчик команды /start
@dp.message(Command("start"))
async def start_command(message: Message):
    await message.answer("Привет! Я твой второй мозг. Отправь мне PDF-файл, и я запомню его. А потом сможешь задавать вопросы по его содержанию.")

# 6. Обработчик получения PDF-файла
@dp.message(lambda message: message.document is not None)
async def handle_docs(message: Message):
    # Проверяем, что это PDF
    if not message.document.file_name.endswith('.pdf'):
        await message.answer("Пожалуйста, отправь PDF-файл.")
        return

    await message.answer("Принимаю файл, подожди немного...")

    # Скачиваем файл от пользователя
    file = await bot.get_file(message.document.file_id)
    file_path = f"./{message.document.file_name}"
    await bot.download_file(file.file_path, file_path)

    # Читаем текст из PDF
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        await message.answer(f"Ошибка при чтении PDF: {e}")
        return

    if not text.strip():
        await message.answer("Не удалось извлечь текст из PDF. Возможно, он защищен или это сканированная копия.")
        return

    # Разбиваем текст на кусочки (чанки) по 1000 символов
    chunk_size = 1000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Добавляем кусочки в нашу базу данных
    ids = [f"{message.document.file_name}_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=[{"source": message.document.file_name}] * len(chunks)
    )

    await message.answer(f"Готово! Файл '{message.document.file_name}' запомнен. Теперь ты можешь задавать вопросы по нему.")

# 7. Обработчик текстовых вопросов
@dp.message()
async def answer_question(message: Message):
    user_query = message.text

    # Ищем в базе данных 3 самых похожих по смыслу кусочка
    results = collection.query(query_texts=[user_query], n_results=3)

    # Если ничего не нашли
    if not results['documents'] or len(results['documents'][0]) == 0:
        await message.answer("Я не нашел информации по этому вопросу в загруженных файлах.")
        return

    # Собираем найденные кусочки в один контекст
    context = "\n---\n".join(results['documents'][0])

    # Отправляем запрос в OpenAI, чтобы он сформулировал ответ на основе контекста
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты — полезный ассистент. Отвечай на вопрос пользователя, используя только информацию из контекста. Если ответа нет в контексте, скажи 'Я не знаю'."},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {user_query}"}
            ],
            temperature=0.3
        )
        answer = response.choices[0].message.content
        await message.answer(answer)
    except Exception as e:
        await message.answer(f"Произошла ошибка при обращении к ИИ: {e}")

# 8. Запускаем бота
if __name__ == "__main__":
    print("Бот запущен!")
    dp.run_polling(bot)