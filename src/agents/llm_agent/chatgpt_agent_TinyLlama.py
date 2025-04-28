import json
import re
import time
from typing import List
import torch
from sentence_transformers import util

from ..abs_agent import Agent
from ..utils import write_json
from ..hf_agent import HuggingFaceAgent  # ✅ Use TinyLlama

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
        self.temperature = temperature
        self.T = 3
        self.epsilon = 0.85
        self.memory = {"name": [], "message": [], "informativeness": []}
        self.phase_memory = {}
        self.summary = {}
        self.bad_experience = []
        self.good_experience = []
        self.current_experience = []
        self.previous_exp_pool = previous_exp_pool
        self.use_summary = use_summary
        self.output_dir = output_dir
        self.local_llm = HuggingFaceAgent()  # ✅

    def step(self, message: str) -> str:
        phase = message.split("|")[0]
        self.phase = phase
        message = message.split("|")[1]
        conversations = [{"role": 'system', "content": self.rule_role_prompt}]

        if self.memory.get("message"):
            if self.use_summary:
                r_t = self.summary_memory()
            else:
                r_t, conversations = self.retrival_memory(conversations)
        else:
            r_t = "None"

        if self.previous_exp_pool:
            s_t, conversations = self.extract_suggestion(r_t, conversations)
        else:
            s_t = "None"

        prompt = self.generate_response_prompt.format(self.phase, self.name, self.role, message, r_t, s_t)
        conversations.append({"role": 'user', "content": prompt})

        output = self.local_llm.chat(conversations, max_tokens=512, temperature=self.temperature)["content"]
        self.log(f"{self.output_dir}/response.txt", f"input:{conversations}\noutput:\n{output}\n--------------------")
        conversations.append({"role": 'assistant', "content": output})

        output = output.replace("\n", "")
        pattern = "(?<=My concise talking content:).*(?=<EOS>)"
        match = re.search(pattern, output)
        if match is None:
            pattern = "(?<=My concise talking content:).*"
            match = re.search(pattern, output)
        response = match.group().strip() if match else output

        self.update_memory("Host", message)
        self.update_memory(self.name, response)
        self.current_experience.append([r_t, response, None])
        return response

    def retrival_memory(self, conversations: List[dict]):
        names = self.memory.get("name", [])[-self.freshness_k:]
        messages = self.memory.get("message", [])[-self.freshness_k:]
        o_t = [f"{n}: {m}" for n, m in zip(names, messages)]

        x = zip(self.memory.get("name", []),
                self.memory.get("message", []),
                self.memory.get("informativeness", []))
        x = sorted(x)
        v_t = [f"{i[0]}: {i[1]}" for i in x[-self.informativeness_n:]]

        prompt = self.select_question_prompt.format(self.phase, self.name, self.role, self.question_list)
        conversations.append({"role": 'user', "content": prompt})
        output = self.local_llm.chat(conversations)["content"]
        self.log(f"{self.output_dir}/select_question.txt", f"input:{conversations}\noutput:\n{output}\n--------------------")
        conversations.append({"role": 'assistant', "content": output})
        selected_questions = output.split("#")

        prompt = self.ask_question_prompt.format(self.phase, self.name, self.role, selected_questions)
        conversations.append({"role": 'user', "content": prompt})
        output = self.local_llm.chat(conversations)["content"]
        self.log(f"{self.output_dir}/ask_question.txt", f"input:{conversations}\noutput:\n{output}\n--------------------")
        conversations.append({"role": 'assistant', "content": output})
        questions = output.split("#")

        candidate_answer = []
        documents = self.memory.get("message", [])
        documents_embedding = self.retrival_model.encode(documents)
        k = min(len(documents), self.T)
        for q in selected_questions + questions:
            q_embedding = self.retrival_model.encode(q)
            cos_scores = util.cos_sim(q_embedding, documents_embedding)[0]
            top_results = torch.topk(cos_scores, k=k)
            result = [documents[idx] for idx in top_results.indices]
            candidate_answer.append(result)

        q = ' '.join([f"{idx + 1}: {q_i}" for idx, q_i in enumerate(selected_questions + questions)])
        c = ' '.join([f"{idx + 1}: {c_i}" for idx, c_i in enumerate(candidate_answer)])
        prompt = self.generate_answer_prompt.format(self.phase, self.name, self.role, q, self.T, c)
        conversations.append({"role": 'user', "content": prompt})
        output = self.local_llm.chat(conversations)["content"]
        self.log(f"{self.output_dir}/generate_answer.txt", f"input:{conversations}\noutput:\n{output}\n--------------------")
        a_t = output
        conversations.append({"role": 'assistant', "content": output})

        prompt = "{}".format(o_t + v_t) + self.reflection_prompt.format(
            self.phase, self.name, self.role, a_t, self.role)
        conversations.append({"role": 'user', "content": prompt})
        output = self.local_llm.chat(conversations)["content"]
        self.log(f"{self.output_dir}/reflection.txt", f"input:{conversations}\noutput:\n{output}\n--------------------")
        conversations.append({"role": 'assistant', "content": output})
        r_t = output
        return r_t, conversations

    def extract_suggestion(self, r_t, conversations):
        r_pool = [e[0] for e in self.previous_exp_pool]
        g_pool = [e[1] for e in self.previous_exp_pool]
        d_embedding = self.retrival_model.encode(r_pool)
        q_embedding = self.retrival_model.encode(r_t)
        cos_scores = util.cos_sim(q_embedding, d_embedding)[0]
        sub_e_idx = torch.where(cos_scores > self.epsilon)[0]
        sub_e = [self.previous_exp_pool[idx] for idx in sub_e_idx]
        sub_e = sorted(sub_e, key=lambda x: x[2])[:min(len(sub_e), self.experience_window)]
        good_experience = [e[1] for e in sub_e[:-1]] if sub_e else []
        bad_experience = [e[1] for e in sub_e[:-1]] if sub_e else []
        prompt = self.extract_suggestion_prompt.format(bad_experience, good_experience)
        conversations.append({"role": 'user', "content": prompt})
        output = self.local_llm.chat(conversations)["content"]
        self.log(f"{self.output_dir}/suggestion.txt", f"input:{conversations}\noutput:\n{output}\n--------------------")
        conversations.append({"role": 'assistant', "content": output})
        return output, conversations

    def receive(self, name: str, message: str) -> None:
        phase = message.split("|")[0]
        self.phase = phase
        message = message.split("|")[1]
        self.update_memory(name, message)

    def update_memory(self, name: str, message: str):
        prompt = self.informativeness_prompt.format(f"{name}: {message}")
        messages = [{"role": 'system', "content": ""}, {"role": 'user', "content": prompt}]
        output = self.local_llm.chat(messages)["content"]
        scores = re.findall(r"\d+", output)
        score = int(scores[-1]) if scores else 1
        self.memory['name'].append(name)
        self.memory['message'].append(message)
        self.memory['informativeness'].append(score)

    def reflection(self, player_role_mapping: dict, file_name: str, winners: list, duration: int):
        score = duration if self.role not in winners else 1000 - duration
        exp = [[e[0], e[1], score] for e in self.current_experience]
        self.previous_exp_pool.extend(exp)
        write_json(data=self.previous_exp_pool, path=file_name)

    @staticmethod
    def log(file, data):
        with open(file, mode='a+', encoding='utf-8') as f:
            f.write(data)


# from ..hf_agent import HuggingFaceAgent  # Already included in CGAgent

class SAPARAgent(Agent):
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

        self.temperature = temperature
        self.output_dir = output_dir
        self.phase = 0
        self.api_key = api_key
        self.T = 3
        self.candidate_actions = candidate_actions
        self.memory_window = 30

        self.local_llm = HuggingFaceAgent()  # ✅ TinyLlama

    def step(self, message: str) -> str:
        self.phase = message.split("|")[0]
        message = message.split("|")[1]
        output = self.phase
        matches = re.findall(r"\d+", output)
        phase = matches[-1] if matches else "0"

        format_summary = self.get_summary()
        if self.use_analysis and format_summary != "None":
            analysis = self.make_analysis(phase, format_summary)
        else:
            analysis = "None"

        if self.use_plan:
            format_plan = self.make_plan(phase, format_summary, analysis)
        else:
            format_plan = "None"

        if self.use_action:
            action = self.make_action(phase, format_summary, format_plan, analysis, message)
        else:
            action = None

        response = self.make_response(phase, format_summary, format_plan, action, message)

        self.update_private("Host", message, phase)
        self.update_private("Self", response, phase)
        _ = self.memory_summary(phase)

        return response

    def memory_summary(self, phase):
        prompt = self.summary_prompt.format(name=self.name, conversation=self.memory_to_json(phase))
        messages = [{"role": 'system', "content": self.system_prompt},
                    {"role": 'user', "content": prompt}]
        output = self.local_llm.chat(messages)["content"]
        match = re.search("(?<=Summary:).*", output, re.S)
        summary = match.group().strip() if match else output
        self.summary[phase] = summary
        if self.summary:
            return "\n".join(
                [f"Quest Phase Turn {key}:{value}" if key != "0" else f"Reveal Phase: {value}" for key, value in self.summary.items()])
        else:
            return "None"

    def make_analysis(self, phase, format_summary):
        prompt = self.analysis_prompt.format(name=self.name, phase=self.phase, role=self.role, summary=format_summary)
        messages = [{"role": 'system', "content": self.system_prompt},
                    {"role": 'user', "content": prompt}]
        output = self.local_llm.chat(messages)["content"]
        return output

    def make_plan(self, phase, format_summary, analysis):
        if self.plan:
            format_previous_plan = '\n'.join(
                [f"Quest Phase Turn {i}: {self.plan.get(str(i), 'None')}" if i != 0 else f"Reveal Phase: {self.plan.get(str(i), 'None')}"
                 for i in range(int(phase) + 1)])
        else:
            format_previous_plan = "None"

        following_format = '\n'.join(
            [f"Quest Phase Turn {i}: <your_plan_{i}>" if i != 0 else f"Reveal Phase: <your_plan_0>" for i in range(int(phase), 6)]
        )

        prompt = self.plan_prompt.format(name=self.name, phase=self.phase, role=self.role, introduction=self.introduction,
                                         goal=self.game_goal, strategy=self.strategy,
                                         previous_plan=format_previous_plan, summary=format_summary,
                                         analysis=analysis, plan=following_format)

        messages = [{"role": 'system', "content": self.system_prompt},
                    {"role": 'user', "content": prompt}]
        output = self.local_llm.chat(messages)["content"]

        match = re.search('<plan>(.*?)</plan>', output, re.S)
        format_plans = match.group().split('\n') if match else output.split('\n')
        dict_plans = {}
        for plan in format_plans:
            if ':' in plan:
                parts = plan.split(':', 1)
                key = re.search(r'\d+', parts[0])
                if key:
                    c_phase = key.group()
                    dict_plans[c_phase] = parts[1].strip()
                elif "reveal" in parts[0].lower():
                    dict_plans["0"] = parts[1].strip()
        self.plan.update(dict_plans)

        return '\n'.join([
            f"Quest Phase Round {str(c_phase)}: {self.plan.get(str(c_phase))}" if str(c_phase) != "0"
            else f"Reveal Phase: {self.plan.get(str(c_phase))}"
            for c_phase in range(int(phase), 6)])

    def make_action(self, phase, format_summary, format_plan, analysis, message):
        prompt = self.action_prompt.format(name=self.name, phase=self.phase, role=self.role,
                                           introduction=self.introduction, goal=self.game_goal,
                                           strategy=self.strategy, candidate_actions=self.candidate_actions,
                                           summary=format_summary, analysis=analysis, plan=format_plan,
                                           question=message)
        messages = [{"role": 'system', "content": self.system_prompt},
                    {"role": 'user', "content": prompt}]
        output = self.local_llm.chat(messages)["content"]
        actions = re.findall("(?<=<actions>).*?(?=</actions>)", output, re.S)
        if not actions:
            actions = re.findall("(?<=<output>).*?(?=</output>)", output, re.S)
            if not actions:
                return output
        return actions

    def make_response(self, phase, format_summary, format_plan, actions, message):
        if self.use_action:
            prompt = self.response_prompt.format(
                name=self.name, phase=self.phase, role=self.role, introduction=self.introduction,
                strategy=self.strategy, summary=format_summary, plan=format_plan,
                question=message, actions=actions)
        else:
            prompt = self.response_prompt.format(
                name=self.name, phase=self.phase, role=self.role, introduction=self.introduction,
                strategy=self.strategy, summary=format_summary, plan=format_plan,
                question=message, actions="None")

        messages = [{"role": 'system', "content": self.system_prompt},
                    {"role": 'user', "content": prompt}]
        output = self.local_llm.chat(messages)["content"]
        match = re.search("(?<=<response>).*?(?=</response>)", output, re.S)
        return match.group().strip() if match else output

    def get_summary(self):
        if self.summary:
            return "\n".join(
                [f"Quest Phase Turn {key}:{value}" if key != "0" else f"Reveal Phase: {value}" for key, value in self.summary.items()])
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

    def update_public(self, name, message, phase: str = None) -> None:
        self.memory['message_type'].append("public")
        self.memory['name'].append(name)
        self.memory['message'].append(message)
        self.memory['phase'].append(phase)
        if phase not in self.phase_memory:
            self.phase_memory[phase] = {"message_type": [], "name": [], "message": [], "phase": []}
        self.phase_memory[phase]['message_type'].append("public")
        self.phase_memory[phase]['name'].append(name)
        self.phase_memory[phase]['message'].append(message)
        self.phase_memory[phase]['phase'].append(phase)

    def memory_to_json(self, phase: str = None):
        if phase is None:
            data = [{'message_type': t, 'name': r, 'message': m, 'phase': p}
                    for t, r, m, p in zip(self.memory['message_type'], self.memory['name'], self.memory['message'], self.memory['phase'])]
        else:
            data = [{'message_type': t, 'name': r, 'message': m, 'phase': p}
                    for t, r, m, p in zip(self.phase_memory.get(phase, {}).get('message_type', []),
                                          self.phase_memory.get(phase, {}).get('name', []),
                                          self.phase_memory.get(phase, {}).get('message', []),
                                          self.phase_memory.get(phase, {}).get('phase', []))]
        return json.dumps(data, indent=4, ensure_ascii=False)

    def reflection(self, player_role_mapping: dict, file_name: str, winners: list, duration: int):
        summary = self.get_summary()
        strategy = "None"
        suggestion = "None"

        if self.reflection_other:
            roles_str = '\n'.join([f"{k}:{v}" for k, v in player_role_mapping.items()])
            prompt = self.strategy_prompt.format(name=self.name, roles=roles_str, summaries=summary,
                                                 strategies=self.previous_other_strategy)
            messages = [{"role": 'system', "content": ""}, {"role": 'user', "content": prompt}]
            strategy = self.local_llm.chat(messages)["content"]

        if self.improve_strategy:
            prompt = self.suggestion_prompt.format(
                name=self.name, role=self.role, roles='\n'.join([f"{k}:{v}" for k, v in player_role_mapping.items()]),
                summaries=summary, goal=self.game_goal, strategy=self.strategy,
                previous_suggestions=self.previous_suggestion)
            messages = [{"role": 'system', "content": ""}, {"role": 'user', "content": prompt}]
            suggestion = self.local_llm.chat(messages)["content"]

        prompt = self.update_prompt.format(name=self.name, role=self.role, strategy=self.strategy, suggestions=suggestion)
        messages = [{"role": 'system', "content": ""}, {"role": 'user', "content": prompt}]
        output = self.local_llm.chat(messages)["content"]
        match = re.search("(?<=<strategy>).*?(?=</strategy>)", output)
        strategy = match.group() if match else output

        write_json({"strategy": strategy, "suggestion": suggestion, "other_strategy": strategy}, file_name)
