1. Install Python +3.11
2. Run `pip install -r requirements.txt`

---

- The `src/pandas_impl.py` is based on Flowise's CSV Agent approach: https://github.com/FlowiseAI/Flowise/blob/main/packages/components/nodes/agents/CSVAgent/CSVAgent.ts

- The `src/vector_store_impl.py` is a simple RAG approach that stores CSV data and do a similarity search on it.
