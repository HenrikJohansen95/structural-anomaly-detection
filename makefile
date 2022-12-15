JENA_PATH = "./jena"
LUBM_PATH = "./lubm"
NR_UNIVERSITY = 16

.PHONY = lubm_generate start_jena jena_uninstall lubm_clean lubm_uninstall clean

all: jena lubm

jena:
	wget -P $(JENA_PATH) https://archive.apache.org/dist/jena/binaries/apache-jena-fuseki-4.4.0.zip && \
	cd $(JENA_PATH) && \
	unzip apache-jena-fuseki-4.4.0.zip && \
	rm apache-jena-fuseki-4.4.0.zip && \
	mv apache-jena-fuseki-4.4.0 fuseki

start_jena: jena
	cd $(JENA_PATH) && \
	fuseki/fuseki-server -tdb1 --memTDB --timeout 300000 /gnn

jena_uninstall:
	rm -rf $(JENA_PATH)

lubm:
	wget -P $(LUBM_PATH) http://swat.cse.lehigh.edu/projects/lubm/uba1.7.zip && \
	cd $(LUBM_PATH) && \
	unzip uba1.7.zip -d uba && \
	rm uba1.7.zip



lubm_generate: lubm
	cd $(LUBM_PATH)/uba && \
	java -cp classes edu.lehigh.swat.bench.uba.Generator -univ $(NR_UNIVERSITY) -onto http://example.org/

lubm_clean: lubm
	rm lubm/*log.txt lubm/*.owl

lubm_uninstall:
	rm -rf $(LUBM_PATH)

venv:
	python -m venv venv && \
	. venv/bin/activate && \
	pip install wheel && \
	pip install torch --extra-index-url https://download.pytorch.org/whl/cu117 && \
	pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
	pip install -r requirements.txt

conda:
	conda env create -f environment.yml -p ./venv

clean:
	rm -r lightning_logs
	rm -r plots
	rm -r wandb