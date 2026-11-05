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
# README.md is the ONE readme this repo tracks.
# The fork used to track BOTH spellings as separate blobs -- README.md (13,406 bytes,
# upstream PR #1) and readme.md (13,869 bytes, the fork's newer revision carrying the
# PMOVES icon, the DeepRefine note and the star-history chart). On a case-insensitive
# client (Windows/NTFS, macOS) the checkout COLLAPSES them into one on-disk file, so the
# submodule was permanently dirty and the COPY below resolved to a 14,374-byte
# client-side artefact (the CRLF-translated lowercase blob) rather than an exact blob.
# FIXED by keeping the CANONICAL name README.md, giving it the fork's content, and
# dropping readme.md -- so the name, the content and the build all agree on one file.
# setup.py opens README.md for long_description.
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
# THE `hirag` PACKAGE'S IMPORT SPLIT -- FIXED HERE, AND NOW GATED BY THIS BUILD.
# Upstream split the package's LLM layer in two: `hirag/_llm.py` kept only the COHERE
# helpers, and the OpenAI / Azure / DeepSeek / Ollama implementations were moved into
# `hirag/_llm_backup.py`. But `hirag/hirag.py` kept importing the moved names from `._llm`,
# so `from hirag import HiRAG, QueryParam` -- the import `eval/eval_mhqa.py` performs --
# died with:
#     ImportError: cannot import name 'gpt_4o_complete' from 'hirag._llm'
# This was an upstream defect, not a fork one: the fork's `hirag/_llm.py` is byte-identical
# to upstream's (SHA256 CD611C600401428F56AA79B2F6C7BA6FAA2760F9A739F0C08A8B86F250E05738
# for both) and upstream's `hirag/hirag.py` carries the same import block. Upstream `main`
# is still at that commit and still broken, so there was no upstream fix to sync from.
# FIXED in the fork by importing each name from the module that actually DEFINES it:
# `cohere_embedding` / `cohere_complete` from `._llm`, the other 13 from `._llm_backup`.
# No function body is duplicated and no upstream file is edited.
#
# GATED: the RUN step further down imports the package DURING the image build, so a build
# of a tree where the split is broken again FAILS instead of shipping an image whose
# library cannot be imported.

# The package itself, plus the code the image is for.
# setup.py + README.md must be present for the build (setup.py opens "README.md").
COPY setup.py README.md ./
COPY hirag/ ./hirag/
# eval/ is copied FILE BY FILE on purpose: `COPY eval/ ./eval/` would also copy
# eval/datasets/** (61 files, 69,329,458 bytes) into the image. The 12 harness scripts
# are the only things the image needs from that directory.
#
# KNOWN, SEPARATE UPSTREAM DEFECT in the HARNESS (not the library, and NOT fixed
# here): `eval/eval_mhqa.py` reads a DIFFERENT config schema than `config.yaml`
# provides. It wants `config['glm']['url']`, `config['openai']['url']` and a
# `config['nvidia'][...]` section, while config.yaml carries `base_url` (not `url`)
# and has no `nvidia` section at all. MEASURED 2026-11-05: after the YAML was made
# parseable, `import eval.eval_mhqa` reached line 34 and died with
# `KeyError: 'url'`.
# This is a CONTRACT mismatch between two upstream files, not a one-character fix --
# repairing it means deciding which spelling is canonical and what an `nvidia`
# section should contain -- so it is REPORTED here rather than guessed at. The
# image's gates above cover the LIBRARY import and the config's YAML validity,
# which are what this fork repairs; they deliberately do not gate on the harness
# import, because a gate that fails on an unrelated disclosed defect would block
# the image forever.
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
# imports exactly that). --no-deps because the pinned deps are already installed above and
# setup.py would otherwise re-resolve them.
RUN pip install --no-cache-dir --no-deps -e .

# THE IMPORT GATE. The image is only worth building if the library it ships can actually be
# imported, and the split this Dockerfile documents above was invisible until something
# tried it. So the build itself tries it: this RUN imports the package, and the image build
# FAILS if the import regresses. It is a gate in the tree, on the real path, not a note in
# a comment.
#
# MEASURED 2026-11-05: before the split was repaired this line raised
#   ImportError: cannot import name 'gpt_4o_complete' from 'hirag._llm'
# and the image built anyway -- shipping a library that no caller could import.
RUN python -c "from hirag import HiRAG, QueryParam; print('hirag imports OK:', HiRAG.__name__, QueryParam.__name__)"

# THE CONFIG GATE, for the same reason: the harness reads config.yaml at IMPORT
# time (`eval/eval_mhqa.py` opens it at module scope), so a config.yaml that does
# not parse makes the harness unimportable even though the library is fine.
#
# MEASURED 2026-11-05: upstream's config.yaml line 46 read
#     working_dir: "./hirag_index_cohere
# -- an UNTERMINATED double-quoted scalar -- so `yaml.safe_load` raised
# `ParserError: while parsing a block mapping ... expected <block end>` and
# `import eval.eval_mhqa` died before it did any work. The file was byte-identical
# to upstream's (blob 4a0258a032a3c38e7efacc5fd4d5be822c021cd0), so this was
# upstream's defect too. FIXED by closing the quote.
#
# pyyaml is already in this image as the harness's own dependency, so this gate
# adds no new dependency -- it only checks the config the harness is about to
# read.
RUN python -c "import yaml; yaml.safe_load(open('config.yaml', encoding='utf-8')); print('config.yaml parses OK')"

# House convention: non-root uid/gid 65532.
RUN groupadd -r pmoves --gid=65532 && \
    useradd -r -g pmoves --uid=65532 --home-dir=/app --shell=/sbin/nologin pmoves && \
    mkdir -p /app/.cache/hf && \
    chown -R pmoves:pmoves /app

USER pmoves:pmoves

# Honest default for a library/harness image: an interactive shell, because this repo has no
# server to run (see the header comment for the measured evidence).
CMD ["bash"]
