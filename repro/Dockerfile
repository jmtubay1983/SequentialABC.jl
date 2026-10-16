# Minimal reproducibility image for SequentialABC.jl.
#
# Build from the repository root:
#   docker build -f repro/Dockerfile -t sequentialabc-repro .
# Run (executes repro/run_all.sh):
#   docker run --rm sequentialabc-repro
FROM julia:1.13.2

WORKDIR /app
COPY . /app

CMD ["bash", "repro/run_all.sh"]
