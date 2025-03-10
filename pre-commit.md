# Pre-commit

Начинаем помаленьку приближаться к полноценному CI/СD, но сделаем это мягко и начнем с пре-коммитов.

**ChatGPT**  
Pre-commit – это фреймворк для управления и поддержания pre-commit хуков. Pre-commit хуки – это скрипты, которые автоматически запускаются перед выполнением коммита в git. Они могут использоваться для выполнения различных задач, таких как проверка стиля кодирования, форматирование кода, запуск линтеров и выполнение тестов, что помогает обеспечить соблюдение определенных критериев качества кода до того, как изменения будут зафиксированы в репозитории.

# Установка
pip install pre-commit

# Использование  
Итак, смысл в том, чтобы не давать самому себе сделать коммит, если не прогнался линтер и тест. Да, теперь еще и тесты, но сами их мы писать, конечно же, не будем, для этого у нас скоро появится специальный человек, поэтому напишите хотя бы один, самый глупый тест на Pytest. 

**Важно, что на Pytest, пока будем использовать именно его:**
```bash
pip install pytest
pytest
```
  
Тогда ваша репа в корне будет выглядеть, например, вот так:
```
project_name/
│
├── src/
│   ├── main.py
│   └── DataModule.py
│   └── ...
├── notebooks/
│   ├── EDA.ipynb
│   └── ...
├── tests/
│   └── test_DataModule.py
├── Makefile
├── requirements.txt
└── setup.cfg
```

И чтобы нам ручками не делать вот это:
```
flake8 src
pytest
```

Мы создаем файл ```.pre-commit-config.yaml``` и кладем его в корень репы:
```yaml
repos:
  - repo: https://github.com/pycqa/flake8
    rev: '7.0.0'
    hooks:
      - id: flake8
        additional_dependencies: [flake8]
        files: ^src/  # Да, имеенно с апострофом, чтобы прогонялся весь код в src

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        always_run: true
        pass_filenames: false

```
# Успех
```bash
git add .
```
И далее, если вы увидели примерно следующую картину, то пре-коммит прогнался и вы победили, можно пушить  
```bash
(venv) C:\Users\usenk\PycharmProjects\SatelliteTo3D_Raduga>git commit -m "test pre-commit"
[INFO] Initializing environment for https://github.com/pycqa/flake8.
[INFO] Initializing environment for https://github.com/pycqa/flake8:flake8.
[INFO] Installing environment for https://github.com/pycqa/flake8.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
flake8...................................................................Passed
pytest...................................................................Passed
[main f8fc534] test pre-commit
 3 files changed, 3 insertions(+), 6 deletions(-)
```
# Провал

**Под провалом имеется в виду ситуация, не когда вас не пустил пре-коммит, а когда он у вас тупо не работает**

Провал в том случае, если вы не увидели того, что было в успехе - значит пре-коммит не прогнался, тогда попробуйте:  
### Вызвать пре-коммит вручную
```bash
pre-commit run --all-files 
```
### Реиницаилизировать
```bash
pre-commit install 
pre-commit autoupdate 
```
И снова попробовать сделать коммит