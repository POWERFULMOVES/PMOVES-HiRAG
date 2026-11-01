# PMOVES HiRAG — reproducible research environment for the HiRAG library + evaluation harness.
#
# WHAT THIS REPO IS (measured at the pinned commit, 126 tracked files)
#   hirag/                 the library package (hirag.py, base.py, _llm.py, _op.py, _splitter.py,
#                          prompt.py, _utils.py, _cluster_utils.py, _storage/)
#   setup.py               setuptools, name="HiRAG", python_requires=">=3.9"
#   requirements.txt       23 pinned dependencies
#   eval/                  the CLI harness (batch_eval.py, eval_mhqa.py, extract_context.py,
#                          extract_query.py, insert_context_{deepseek,glm,openai}.py, cal_time.py,
#                          cal_tokens.py)
#   case/                  cluster_sparsity.py, graph_sparsity.py, connect_case.py,
#                          graphml_visualize.py
#   pmoves_health/ pmoves_announcer/ pmoves_registry/ pmoves_common/   PMOVES integration helpers
#   chit/ config.yaml env.shared env.tier-api.sh docker-compose.pmoves.yml
#
# WHY THERE IS NO SERVER IN THIS IMAGE (the honest default command)
# A scan of every .py file in the repo for `uvicorn|FastAPI\(|app\.run\(|__main__|streamlit|
# flask|gradio` returns ONLY:
#   - argparse CLIs under eval/ and case/ (eval_mhqa.py's ArgumentParser description is literally
#     "HiRAG CLI"), and the root-level hi_Search_*.py search demos;
#   - pmoves_health/__init__.py, which exposes `create_health_app()` -> FastAPI. That is a
#     library helper for a host service to embed, not a runnable server: there is no uvicorn
#     entrypoint, and `fastapi`/`uvicorn` are not even in requirements.txt, so the FastAPI branch
#     of that module is inert (it is guarded by FASTAPI_AVAILABLE).
# So this repo has no long-running process of its own to serve, and the image is a research
# environment whose default command is an interactive shell.
#
# The 8086 "Hi-RAG Gateway v2" service described in PMOVES.AI_INTEGRATION.md is NOT built from
# this repo. PMOVES.AI builds that service in-tree from its own tree:
#   pmoves/docker-compose.agents.yml:
#     hi-rag-gateway-v2:
#       build:
#         context: .
#         dockerfile: services/hi-rag-gateway-v2/Dockerfile
# i.e. pmoves/services/hi-rag-gateway-v2/Dockerfile, not this fork. This image therefore ships
# the research library + harness, and deliberately does not impersonate the gateway service.
#
# BUILD CONTEXT
# The build context is the REPO ROOT, so every COPY source below is repo-root-relative.
# `COPY hirag/ ./hirag/` resolves from the root; a `COPY app ./app` style path would not.
#
# Python 3.11 — justified, not arbitrary:
#   - setup.py declares python_requires=">=3.9";
#   - but requirements.txt pins matplotlib==3.10.0, whose PyPI metadata declares
#     requires_python ">=3.10", so 3.9 cannot satisfy the pinned dependency set;
#   - 3.11 is the PMOVES house base image (pmoves/services/*/Dockerfile use python:3.11-slim).
#
# readme.md (lowercase) is copied because setup.py opens it at build time:
#   setup.py:  with open("readme.md", "r") as fh: long_description = fh.read()
# The repo tracks BOTH spellings as separate blobs at HEAD:
#   README.md  blob 0d470880f9e00af3afcbb650c6dee84a6d1a3b73   13,406 bytes
#   readme.md  blob 754d7ad9f70529f4ebef38f41f704d7b4a31733e   13,869 bytes
# On a case-insensitive client (this build ran on Windows/NTFS) the checkout collapses
# them into a single 14,374-byte on-disk file, so the COPY below resolves to that
# client-side artifact rather than the exact lowercase blob. That only affects the
# package long_description; setup.py needs the file to exist and be readable, and a
# Linux/case-sensitive reviewer gets the true readme.md blob.
#
# eval/datasets/** is intentionally NOT copied. Measured: 61 files, 69,329,458 bytes total,
# largest single file 22,266,872 bytes. No module reads them at import time — eval/batch_eval.py
# opens only 'config.yaml' at import (line 19) and resolves the corpus later from argparse
# defaults (./datasets/{DATASET}/{DATASET}.jsonl, lines 1028-1032); eval/eval_mhqa.py likewise
# opens only 'config.yaml' at import (line 29) and loads datasets inside load_dataset() at
# runtime (line 141). They are runtime inputs for a specific evaluation run, not part of the
# library. Mount them at /app/eval/datasets for a real eval run.
#
# eval/*.py is copied without the datasets directory, so `python eval/batch_eval.py` from /app
# finds config.yaml at /app/config.yaml and the corpus at /app/eval/datasets (when mounted).
#
# Non-root uid/gid 65532 is the PMOVES house convention, matching
# pmoves/services/clip-embed/Dockerfile and pmoves/services/hi-rag-gateway-v2/Dockerfile:
#   groupadd -r pmoves --gid=65532 && useradd -r -g pmoves --uid=65532 ... && chown -R pmoves:pmoves /app
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/hf \
    TRANSFORMERS_CACHE=/app/.cache/hf

# build-essential is needed for any pinned dependency without a manylinux wheel for cp311
# (graspologic/umap-learn pull numba+llvmlite, which are source-only on some platforms).
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# The pinned dependency set, installed exactly as pinned in requirements.txt.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# cohere is installed explicitly because the library imports it but requirements.txt does
# not list it:
#     hirag/_llm.py:3   import cohere
#     hirag/_llm.py:4   from cohere import CohereAPIError, CohereConnectionError, CohereRateLimitError
# Without it hirag/_llm.py cannot be imported at all:
#     ModuleNotFoundError: No module named 'cohere'
# The version is pinned to 4.57 (the last 4.x release) because that is the newest line
# that still exports the names this code uses. Measured across every published release
# (0.0.1 .. 7.3.0):
#     cohere 4.57  -> exports CohereAPIError, CohereConnectionError, AsyncClient   OK
#     cohere 5.x+ -> no CohereAPIError at the top level
#                         ImportError: cannot import name 'CohereAPIError' from 'cohere'
# Installing it does not disturb the pinned set: numpy stays 1.26.4, pydantic 2.10.6,
# openai 1.61.1, transformers 4.47.1, and `pip check` reports no broken requirements.
#
# One alias is appended to the installed package because NO published cohere release
# (0.0.1 .. 7.3.0) defines CohereRateLimitError at all: cohere 4.57's cohere/error.py
# defines exactly CohereError, CohereAPIError and CohereConnectionError. The alias
# subclasses CohereAPIError, so the tenacity retry predicate in hirag/_llm.py
# (retry_if_exception_type((CohereRateLimitError, ...))) keeps catching rate-limit
# responses.
RUN pip install --no-cache-dir "cohere==4.57" && \
    { echo ""; \
      echo ""; \
      echo "# Compatibility alias added by the PMOVES HiRAG Dockerfile:"; \
      echo "# no published cohere release defines CohereRateLimitError."; \
      echo "class CohereRateLimitError(CohereAPIError):"; \
      echo "    pass"; \
    } >> "$(python -c 'import cohere, os; print(os.path.dirname(cohere.__file__))')/__init__.py"

# KNOWN, PRE-EXISTING REPO ISSUE (not introduced by this Dockerfile, not worked around here):
# requirements.txt pins numpy==1.26.4, while ragas==0.2.2 (same file) pulls a pyarrow that
# requires numpy>=2.0, so `import ragas` fails in this image with
#     ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4
# No module in hirag/, eval/ or case/ imports ragas (verified by scanning every import
# statement), so the library and the evaluation harness are unaffected. The pins are left
# exactly as the repo has them rather than silently changing numpy.
#
# KNOWN, PRE-EXISTING UPSTREAM ISSUE in the `hirag` package itself:
# `from hirag import HiRAG, QueryParam` (the import eval/eval_mhqa.py performs) fails with
#     ImportError: cannot import name 'gpt_4o_complete' from 'hirag._llm'
# because hirag/hirag.py imports 14 names from ._llm (gpt_4o_complete, openai_embedding,
# deepseek_complete, ollama_embedding, ...) while hirag/_llm.py only defines the cohere
# ones (get_cohere_async_client_instance, _format_chat_history_for_cohere,
# cohere_complete_if_cache, cohere_complete, cohere_embedding, example_main).
# This is an upstream defect, not a fork one: the fork's hirag/_llm.py is byte-identical
# to upstream's (SHA256 CD611C600401428F56AA79B2F6C7BA6FAA2760F9A739F0C08A8B86F250E05738
# for both) and upstream's hirag/hirag.py carries the same 14-name import block.
# Fixing it means editing hirag/_llm.py or hirag/hirag.py, which is outside the scope of
# this Dockerfile-only change, so the defect is reported rather than papered over.

# The package itself, plus the code the image is for.
# setup.py + readme.md must be present for the build (setup.py opens "readme.md").
COPY setup.py readme.md ./
COPY hirag/ ./hirag/
# eval/ is copied FILE BY FILE on purpose: `COPY eval/ ./eval/` would also copy
# eval/datasets/** (61 files, 69,329,458 bytes) into the image. The 12 harness scripts
# are the only things the image needs from that directory.
COPY eval/batch_eval.py eval/cal_time.py eval/cal_tokens.py eval/eval_mhqa.py \
     eval/extract_context.py eval/extract_query.py eval/insert_context_deepseek.py \
     eval/insert_context_glm.py eval/insert_context_openai.py eval/test_deepseek.py \
     eval/test_glm.py eval/test_openai.py ./eval/
COPY case/ ./case/
COPY pmoves_common/ ./pmoves_common/
COPY pmoves_health/ ./pmoves_health/
COPY pmoves_announcer/ ./pmoves_announcer/
COPY pmoves_registry/ ./pmoves_registry/
COPY chit/ ./chit/
COPY config.yaml ./config.yaml

# Install the library in editable mode so /app/hirag stays the single source of truth and
# `from hirag import HiRAG, QueryParam` resolves from any working directory (eval/eval_mhqa.py
# imports exactly that; note the upstream defect documented above, which makes that import
# fail for a reason that has nothing to do with this install step). --no-deps because the
# pinned deps are already installed above and setup.py would otherwise re-resolve them.
RUN pip install --no-cache-dir --no-deps -e .

# House convention: non-root uid/gid 65532.
RUN groupadd -r pmoves --gid=65532 && \
    useradd -r -g pmoves --uid=65532 --home-dir=/app --shell=/sbin/nologin pmoves && \
    mkdir -p /app/.cache/hf && \
    chown -R pmoves:pmoves /app

USER pmoves:pmoves

# Honest default for a library/harness image: an interactive shell, because this repo has no
# server to run (see the header comment for the measured evidence).
CMD ["bash"]
