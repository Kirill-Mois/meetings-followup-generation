
# Meetings Follow-up Generation

Этот скрипт предназначен для генерации follow-up (bullet-point summary) на основе транскриптов встреч в формате Markdown (md). В follow-up содержится самая важная информация (итоги, договоренности, ключевые решения), сгруппированная по темам.

## Установка

Склонируйте репозиторий:

```bash
git clone <URL вашего репозитория>
cd <название репозитория>
```

Настройте виртуальное окружение:

```bash
python -m venv .venv
source .venv/bin/activate   # для Windows используйте .venv\Scripts\activate
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

Проверьте корректность установки зависимостей:

```bash
python -m pip check
```

## Настройка

Перед запуском скрипта необходимо настроить файл .env для использования OpenAI API.

1. Создайте файл .env в корневой директории проекта.
2. Добавьте в этот файл ваш API-ключ OpenAI:

```bash
OPENAI_API_KEY="sk-..."
```

## Запуск

Для запуска скрипта используйте следующую команду:

```bash
PYTHONPATH=$PWD python src/run.py --config_path "config/<config>.yaml" --markdown_path "data/<markdown>.md"
```

Замените `<config>` на имя файла конфигурации и `<markdown>` на имя вашего файла с транскриптом встречи.

**Пример:**
```bash
PYTHONPATH=$PWD python src/run.py --config_path "config/map_reduce.yaml" --markdown_path "data/interview-durov-markdown.md"
```

## Примечания

- Убедитесь, что файл конфигурации и файл транскрипта находятся в соответствующих директориях (config и data).
- Убедитесь, что файл .env правильно настроен и содержит действующий API-ключ OpenAI.
