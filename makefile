# Makefile para instalação dentro do WSL

.PHONY: all install ollama deps run

all: ollama deps run

install: ollama deps

ollama:
	@echo "Configurando Ollama..."
	curl -fsSL https://ollama.com/install.sh | sh
	ollama pull nomic-embed-text
	ollama pull llama3:8b

deps:
	@echo "Instalando dependências..."
	sudo apt update
	sudo apt install -y python3-pip python3-venv
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

run:
	@echo "Iniciando aplicação..."
	. venv/bin/activate && streamlit run app.py