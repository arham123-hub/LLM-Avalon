# LLM-Game-Agent (Reproducibility Edition)

This is our reproduction of the EMNLP 2024 paper:  
**“LLM-Based Agent Society Investigation: Collaboration and Confrontation in Avalon Gameplay”**  
🔗 [Original Paper on arXiv](https://arxiv.org/abs/2310.14985)

We recreated the full Avalon game simulation with LLM-based agents using **Groq's LLaMA-3 API**, and optionally a local **TinyLlama** model.

---

## 🔧 Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure you are using **Python 3.9+** and have the following additional packages installed (not in the original `requirements.txt`):

```bash
pip install transformers accelerate sentence-transformers
```

---

## 🔑 API Configuration

To run with **Groq API** (LLaMA-3 model):

1. Get a free API key from [https://console.groq.com](https://console.groq.com)
2. Add this to your environment:
   ```python
   import os
   os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
   ```

Alternatively, you can set it in the terminal before running:
```bash
export GROQ_API_KEY=your_groq_api_key_here
```

---

## 🕹️ Run Avalon Simulation

### Good Side
```bash
python run_avalon_battle.py --exp_name battle --camp good --game_count 3 --start_game_idx 0
```

### Evil Side
```bash
python run_avalon_battle.py --exp_name battle --camp evil --game_count 3 --start_game_idx 0
```

---

## 🧪 Model Notes

- The **Groq API** runs fast and supports long context via `llama3-8b-8192`.
- The **TinyLlama** model can be used locally for cost-free testing (but may break due to memory limits or weak responses).
- You can configure which model is used in `chatgpt_agent.py` by switching between GroqAgent or a local LLaMA agent.

---

## 📦 Project Structure

```
LLM-Game-Agent/
│
├── run_avalon_battle.py      # Main entry point
├── src/
│   ├── agents/               # LLM Agent modules
│   ├── games/avalon/         # Game mechanics & rules
│   ├── apis/groq_api.py      # Groq API wrapper (new)
│   └── ...
├── requirements.txt
├── README.md
```

---

## 💡 Acknowledgement

Original code: [lanluxuan/LLM-Game-Agent](https://github.com/lanluxuan/LLM-Game-Agent)