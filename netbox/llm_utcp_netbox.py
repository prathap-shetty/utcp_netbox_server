#
# LLM Generated Code
# Using utcp verion 1.x  
# TEMP chage : I am using a self-signed certificate, so I had to disable SSL verification in venv/lib/python3.10/site-packages/aiohttp/connector.py 
# I tried setting ssl_verify to false in the providers.json file, but that didn't seem to work. -->  https://www.utcp.io/security
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
import ssl
import warnings
import urllib3
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)


# Ensure HTTP plugin types (call templates) are registered
import utcp_http  # <-- critical for "http" call_template_type
import utcp_text    # <-- important: registers the Text protocol
import utcp_file   # <-- registers the 'file' call template
from utcp.utcp_client import UtcpClient
from utcp.data.tool import Tool
import openai

# -------------------------
# Config / Init
# -------------------------
ROOT = Path(__file__).resolve().parent
PROVIDERS = str(ROOT / "providers.json")
ENV_FILE = str(ROOT / ".env")

load_dotenv(ENV_FILE)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Put it in .env or export it.")

oai = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

_utcp_client: Optional[UtcpClient] = None

# -------------------------
# UTCP helpers
# -------------------------
async def init_utcp() -> UtcpClient:
    global _utcp_client
    if _utcp_client:
        return _utcp_client
    _utcp_client = await UtcpClient.create(config=PROVIDERS)
    return _utcp_client

async def discover_tools(client: UtcpClient, query: str = "dcim devices", limit: int = 100) -> List[Tool]:
    return await client.search_tools(query, limit=limit)

def tools_to_json_for_prompt(tools: List[Tool]) -> str:
    return json.dumps([t.model_dump() for t in tools], indent=2)

# -------------------------
# LLM prompts
# -------------------------
TOOL_CALL_SYSTEM = (
    "You are a careful NetBox automation assistant.\n"
    "You have access to NetBox tools via UTCP (auto-discovered from OpenAPI).\n"
    "When a tool is needed, reply with ONLY a JSON object with keys 'tool_name' and 'arguments'.\n"
    "For checking if a device has a primary IPv4, call the NetBox device list endpoint with 'name' and perhaps 'li
mit': 1, then inspect 'primary_ip4' in the response.\n"
    "Example: {\"tool_name\": \"netbox.dcim_devices_list\", \"arguments\": {\"name\":\"lond-switch-01\", \"limit\"
: 1 }}\n\n"
    "Here are the available tools (names, descriptions, and JSON argument schemas):\n"
)

FINAL_ANSWER_SYSTEM = (
    "You are a helpful assistant. Use the tool output to answer the user's request clearly.\n"
    "If the tool failed, explain the error and suggest the next troubleshooting step.\n"
)

def build_tool_call_messages(history, tools_json: str):
    content = f"{TOOL_CALL_SYSTEM}{tools_json}\n"
    msgs = [{"role": "system", "content": content}]
    for role, content in history:
        msgs.append({"role": role, "content": content})
    return msgs

def build_final_messages(history: List[Tuple[str, str]], tool_output: str) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": FINAL_ANSWER_SYSTEM}]
    for role, content in history:
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": f"Tool output:\n{tool_output}\n\nPlease answer the original request us
ing this output."})
    return msgs

async def chat_complete(messages: List[Dict[str, str]]) -> str:
    resp = await oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""

TOOL_JSON_RE = re.compile(r"```json\s*({.*?})\s*```", re.DOTALL)
def extract_tool_json(s: str) -> Optional[Dict[str, Any]]:
    m = TOOL_JSON_RE.search(s)
    if not m:
        m = re.search(r"(\{[\s\S]*\})", s)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

# -------------------------
# Orchestration
# -------------------------
async def handle_user_query(user_text: str, state_history: List[Tuple[str, str]]) -> Tuple[str, str, str, List[Tup
le[str, str]]]:
    client = await init_utcp()
    tools = await discover_tools(client, "dcim devices", limit=100)
    tools_json = tools_to_json_for_prompt(tools)

    history = state_history.copy()
    history.append(("user", user_text))
    msgs_tool = build_tool_call_messages(history, tools_json)
    assistant_raw = await chat_complete(msgs_tool)
    tool_obj = extract_tool_json(assistant_raw)

    if not tool_obj or "tool_name" not in tool_obj or "arguments" not in tool_obj:
        history.append(("assistant", assistant_raw))
        return (assistant_raw, "(no tool called)", assistant_raw, history)

    tool_name = tool_obj["tool_name"]
    arguments = tool_obj["arguments"]
    try:
        tool_result = await client.call_tool(tool_name, arguments)
        tool_result_text = json.dumps(tool_result, indent=2)
    except Exception as e:
        tool_result_text = f"Tool call error for {tool_name} with args {arguments}:\n{str(e)}"

    history.append(("assistant", json.dumps(tool_obj, indent=2)))

    msgs_final = build_final_messages(history, tool_result_text)
    final_answer = await chat_complete(msgs_final)
    history.append(("assistant", final_answer))

    return (json.dumps(tool_obj, indent=2), tool_result_text, final_answer, history)

# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks(title="UTCP + OpenAI + NetBox") as demo:
    gr.Markdown("## UTCP + OpenAI + NetBox (query devices, interfaces, IPAM, etc.)")

    with gr.Row():
        user_box = gr.Textbox(
            label="Your request",
            placeholder="e.g., Does device lond-switch-01 have a primary IPv4? Show its address.",
            lines=3,
        )
    with gr.Row():
        submit = gr.Button("Run")
        clear = gr.Button("Clear")

    chat = gr.Chatbot(label="Conversation")
    tool_json_out = gr.Code(label="Assistant tool JSON", language="json")
    tool_result_out = gr.Code(label="Tool result", language="json")
    final_answer_out = gr.Markdown(label="Final answer")

    state = gr.State(value=[])

    async def on_submit(user_text, st):
        assistant_tool_json, tool_result_text, final_answer, new_state = await handle_user_query(user_text, st or 
[])
        chat_pairs = []
        tmp_state = st.copy() if st else []
        tmp_state.append(("user", user_text))
        tmp_state.append(("assistant", f"Proposed tool call:\n```json\n{assistant_tool_json}\n```"))
        tmp_state.append(("assistant", final_answer))
        for i in range(0, len(tmp_state), 2):
            user_msg = tmp_state[i][1] if i < len(tmp_state) and tmp_state[i][0] == "user" else ""
            asst_msg = tmp_state[i+1][1] if i+1 < len(tmp_state) and tmp_state[i+1][0] == "assistant" else ""
            if user_msg or asst_msg:
                chat_pairs.append([user_msg, asst_msg])
        return chat_pairs, assistant_tool_json, tool_result_text, final_answer, new_state

    submit.click(
        on_submit,
        inputs=[user_box, state],
        outputs=[chat, tool_json_out, tool_result_out, final_answer_out, state],
    )

    def on_clear():
        return [], "", "", "", []
    clear.click(on_clear, outputs=[chat, tool_json_out, tool_result_out, final_answer_out, state])

if __name__ == "__main__":
    asyncio.run(init_utcp())
    demo.launch(server_name="0.0.0.0", server_port=7860)
