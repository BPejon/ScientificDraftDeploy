# Makefile para instalação dentro do WSL

.PHONY: all install ollama deps run fix-dpk update

all: ollama deps run

install: ollama deps

fix-dpk:
	@echo "Reparando possíveis problemas do dpkg..."
	sudo dpkg --configure -a
	sudo apt --fix-broken install -y

ollama:
	@echo "Configurando Ollama..."
	curl -fsSL https://ollama.com/install.sh | sh
	sudo systemctl start ollama
	sleep 10
	ollama pull nomic-embed-text
	ollama pull llama3:8b

deps: fix-dpk
	@echo "Instalando dependências..."
	sudo apt update -y
	sudo apt install -y python3-pip python3-venv
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

update:
	@echo "Atualizando código..."
	git pull
	@echo "Código atualizado!"

run:
	@echo "Iniciando aplicação..."
	. venv/bin/activate && streamlit run app.py
