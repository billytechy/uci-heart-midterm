# Optional: runs the whole pipeline in a Linux container.
#   docker build -t uci-heart-midterm .
#   docker run --rm -v "$PWD":/app uci-heart-midterm
FROM condaforge/miniforge3
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy
CMD ["conda", "run", "--no-capture-output", "-n", "dpbs-midterm", "python", "run_all.py"]
