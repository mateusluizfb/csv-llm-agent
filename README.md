1. Install Python +3.11
2. Run `pip install -r requirements.txt`

---

- The `src/pandas_impl.py` is based on Flowise's CSV Agent approach: https://github.com/FlowiseAI/Flowise/blob/main/packages/components/nodes/agents/CSVAgent/CSVAgent.ts

- The `src/vector_store_impl.py` is follows a simple RAG approach to store CSV and do a similarity search on the data.
