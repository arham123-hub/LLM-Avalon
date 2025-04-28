#!/usr/bin/env python
# encoding: utf-8
# Updated to use Groq API instead of OpenAI

import json
import os
import re
import time
from typing import List
import requests
import torch
from sentence_transformers import util
from ..abs_agent import Agent
from ..utils import write_json
# from ...apis.groq_api import chatgpt



# ───────────────────────────────────────────────────────────────
# 🧠 Groq API Agent Wrapper
# ───────────────────────────────────────────────────────────────

class GroqAgent:
    def __init__(self, model="llama3-8b-8192", temperature=0.7):
        self.api_key = os.environ["GROQ_API_KEY"]
        self.model = model
        self.temperature = temperature

    def chat(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        for attempt in range(5):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            )

            if response.status_code == 429:
                wait_time = 2 ** attempt
                print(f"[⏳] Rate limited, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        raise Exception("Exceeded retry limit on Groq API due to rate limiting.")

# ───────────────────────────────────────────────────────────────
# 🤖 CGAgent (Communication Game)
# ───────────────────────────────────────────────────────────────

class CGAgent(Agent):
    def __init__(self, name: str, role: str, rule_role_prompt: str,
                 select_question_prompt: str, ask_question_prompt: str, generate_answer_prompt: str,
                 reflection_prompt: str, extract_suggestion_prompt: str, generate_response_prompt: str,
                 informativeness_prompt: str, question_list: list, retrival_model, model: str,
                 freshness_k: int, informativeness_n: int, experience_window: int, temperature: float,
                 api_key: str, previous_exp_pool: list, output_dir: str, use_summary: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.rule_role_prompt = rule_role_prompt
        self.select_question_prompt = select_question_prompt
        self.ask_question_prompt = ask_question_prompt
        self.generate_answer_prompt = generate_answer_prompt
        self.reflection_prompt = reflection_prompt
        self.extract_suggestion_prompt = extract_suggestion_prompt
        self.generate_response_prompt = generate_response_prompt
        self.informativeness_prompt = informativeness_prompt
        self.question_list = question_list
        self.retrival_model = retrival_model
        self.phase = "{}-th {}"
        self.freshness_k = freshness_k
        self.informativeness_n = informativeness_n
        self.experience_window = experience_window
        self.T = 3
        self.epsilon = 0.85
        self.memory = {"name": [], "message": [], "informativeness": []}
        self.phase_memory = {}
        self.summary = {}
        self.previous_exp_pool = previous_exp_pool
        self.current_experience = []
        self.use_summary = use_summary
        self.output_dir = output_dir
        self.local_llm = GroqAgent(model="llama3-8b-8192", temperature=temperature)

    def step(self, message: str) -> str:
        phase = message.split("|")[0]
        self.phase = phase
        message = message.split("|")[1]
        conversations = [{"role": 'system', "content": self.rule_role_prompt}]
        if self.memory.get("message"):
            if self.use_summary:
                r_t = "None"
            else:
                r_t, conversations = self.retrival_memory(conversations)
        else:
            r_t = "None"
        s_t = "None"
        prompt = self.generate_response_prompt.format(self.phase, self.name, self.role, message, r_t, s_t)
        conversations.append({"role": 'user', "content": prompt})
        # output = self.local_llm.chat(conversations, temperature=0.7)
        output = self.local_llm.chat(conversations)
        conversations.append({"role": 'assistant', "content": output})
        output = output.replace("\\n", " ")
        match = re.search(r"(?<=My concise talking content:).*?(?=<EOS>)", output)
        if match is None:
            match = re.search(r"(?<=My concise talking content:).*", output)
        response = match.group().strip() if match else output
        self.update_memory("Host", message)
        self.update_memory(self.name, response)
        self.current_experience.append([r_t, response, None])
        return response

    def retrival_memory(self, conversations: List[dict]):
        names = self.memory.get("name", [])[-self.freshness_k:]
        messages = self.memory.get("message", [])[-self.freshness_k:]
        o_t = [f"{n}: {m}" for n, m in zip(names, messages)]
        x = zip(self.memory.get("name", []), self.memory.get("message", []), self.memory.get("informativeness", []))
        x = sorted(x)
        v_t = [f"{i[0]}: {i[1]}" for i in x[-self.informativeness_n:]]
        prompt = self.select_question_prompt.format(self.phase, self.name, self.role, self.question_list)
        conversations.append({"role": 'user', "content": prompt})
        output = self.local_llm.chat(conversations)
        conversations.append({"role": 'assistant', "content": output})
        return output, conversations

    def update_memory(self, name: str, message: str):
        prompt = self.informativeness_prompt.format(f"{name}: {message}")
        messages = [{"role": 'system', "content": ""}, {"role": 'user', "content": prompt}]
        output = self.local_llm.chat(messages)
        scores = re.findall("\\d+", output)
        score = int(scores[-1]) if scores else 1
        self.memory['name'].append(name)
        self.memory['message'].append(message)
        self.memory['informativeness'].append(score)

    def receive(self, name: str, message: str) -> None:
        phase = message.split("|")[0]
        self.phase = phase
        message = message.split("|")[1]
        self.update_memory(name, message)

    def reflection(self, player_mapping: dict, file_name: str, winners: list, duration: int):
        score = duration if self.role not in winners else 1000 - duration
        exp = [[e[0], e[1], score] for e in self.current_experience]
        self.previous_exp_pool.extend(exp)
        write_json(data=self.previous_exp_pool, path=file_name)
class SAPARAgent(Agent):
    """
    SAPAR-Agent (Summary-Analysis-Planning-Action-Response)
    """
    def __init__(self, name, role, role_intro, game_goal, strategy, system_prompt: str, summary_prompt: str,
                 analysis_prompt: str, plan_prompt: str, action_prompt: str, response_prompt: str, model, temperature,
                 api_key, output_dir, suggestion_prompt: str, strategy_prompt: str, update_prompt: str, suggestion: str,
                 other_strategy: str, candidate_actions: list, use_analysis=True, use_plan=True,
                 use_action=True, reflection_other=True, improve_strategy=True):
        super().__init__()
        self.name = name
        self.role = role
        self.introduction = role_intro
        self.game_goal = game_goal
        self.strategy = strategy
        self.memory = {"message_type": [], "name": [], "message": [], "phase": []}
        self.phase_memory = {}
        self.summary = {}
        self.plan = {}

        self.system_prompt = system_prompt
        self.summary_prompt = summary_prompt
        self.analysis_prompt = analysis_prompt
        self.plan_prompt = plan_prompt
        self.action_prompt = action_prompt
        self.response_prompt = response_prompt
        self.suggestion_prompt = suggestion_prompt
        self.strategy_prompt = strategy_prompt
        self.update_prompt = update_prompt
        self.previous_suggestion = suggestion
        self.previous_other_strategy = other_strategy

        self.use_analysis = use_analysis
        self.use_plan = use_plan
        self.use_action = use_action
        self.reflection_other = reflection_other
        self.improve_strategy = improve_strategy

        self.model = model
        self.temperature = temperature
        self.output_dir = output_dir
        self.phase = 0
        self.T = 3
        self.candidate_actions = candidate_actions
        self.local_llm = GroqAgent(model="llama3-8b-8192", temperature=temperature)

    def step(self, message: str) -> str:
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]

        format_summary = self.get_summary()
        analysis = self.make_analysis(self.phase, format_summary) if self.use_analysis and format_summary != "None" else "None"
        format_plan = self.make_plan(self.phase, format_summary, analysis) if self.use_plan else "None"
        action = self.make_action(self.phase, format_summary, format_plan, analysis, message) if self.use_action else "None"
        response = self.make_response(self.phase, format_summary, format_plan, action, message)

        self.update_private("Host", message, self.phase)
        self.update_private("Self", response, self.phase)
        _ = self.memory_summary(self.phase)

        return response

    def memory_summary(self, phase):
        prompt = self.summary_prompt.format(name=self.name, conversation=self.memory_to_json(phase))
        messages = [{"role": 'system', "content": self.system_prompt}, {"role": 'user', "content": prompt}]
        output = self.local_llm.chat(messages)
        match = re.search("(?<=Summary:).*", output, re.S)
        summary = match.group().strip() if match else output
        self.summary[phase] = summary
        return summary

    def make_analysis(self, phase, format_summary):
        prompt = self.analysis_prompt.format(name=self.name, phase=phase, role=self.role, summary=format_summary)
        messages = [{"role": 'system', "content": self.system_prompt}, {"role": 'user', "content": prompt}]
        return self.local_llm.chat(messages)

    def make_plan(self, phase, format_summary, analysis):
        # following_format = '\n'.join([f"Quest Phase Turn {i}: <your_plan_{i}>" if i != 0 else f"Reveal Phase: <your_plan_0>" for i in range(int(phase), 6)])
        
        if str(phase).isdigit():
            following_format = '\n'.join(
                [f"Quest Phase Turn {i}: <your_plan_{i}>" if i != 0 else f"Reveal Phase: <your_plan_0>" for i in range(int(phase), 6)]
            )
        else:
            following_format = "Reveal Phase: <your_plan_0>"

        prompt = self.plan_prompt.format(name=self.name, phase=phase, role=self.role, introduction=self.introduction,
                                         goal=self.game_goal, strategy=self.strategy, previous_plan="None",
                                         summary=format_summary, analysis=analysis, plan=following_format)
        messages = [{"role": 'system', "content": self.system_prompt}, {"role": 'user', "content": prompt}]
        output = self.local_llm.chat(messages)
        return output

    def make_action(self, phase, format_summary, format_plan, analysis, message):
        prompt = self.action_prompt.format(name=self.name, phase=phase, role=self.role, introduction=self.introduction,
                                           goal=self.game_goal, strategy=self.strategy, candidate_actions=self.candidate_actions,
                                           summary=format_summary, analysis=analysis, plan=format_plan, question=message)
        messages = [{"role": 'system', "content": self.system_prompt}, {"role": 'user', "content": prompt}]
        return self.local_llm.chat(messages)

    def make_response(self, phase, format_summary, format_plan, actions, message):
        prompt = self.response_prompt.format(name=self.name, phase=phase, role=self.role, introduction=self.introduction,
                                             strategy=self.strategy, summary=format_summary, plan=format_plan,
                                             question=message, actions=actions)
        messages = [{"role": 'system', "content": self.system_prompt}, {"role": 'user', "content": prompt}]
        output = self.local_llm.chat(messages)
        match = re.search("(?<=<response>).*?(?=</response>)", output, re.S)
        return match.group().strip() if match else output

    def get_summary(self):
        if self.summary:
            return "\n".join([f"Quest Phase Turn {key}:{value}" if key != "0" else f"Reveal Phase: {value}" for key, value in self.summary.items()])
        else:
            return "None"

    def update_private(self, name, message, phase: str = None) -> None:
        self.memory['message_type'].append("private")
        self.memory['name'].append(name)
        self.memory['message'].append(message)
        self.memory['phase'].append(phase)
        if phase not in self.phase_memory:
            self.phase_memory[phase] = {"message_type": [], "name": [], "message": [], "phase": []}
        self.phase_memory[phase]['message_type'].append("private")
        self.phase_memory[phase]['name'].append(name)
        self.phase_memory[phase]['message'].append(message)
        self.phase_memory[phase]['phase'].append(phase)

    def memory_to_json(self, phase: str = None, discard: int = None):
        if phase is None:
            return json.dumps([{"name": n, "message": m} for n, m in zip(self.memory['name'], self.memory['message'])], indent=4)
        else:
            messages = zip(
                self.phase_memory.get(phase, {}).get('name', []),
                self.phase_memory.get(phase, {}).get('message', [])
            )
            return json.dumps([{"name": name, "message": msg} for name, msg in messages], indent=4)
