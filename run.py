#!/usr/bin/env python
# coding: utf-8
"""
Entry point – spawns a GroupChat with:
Planner → RAG_Agent → Search_Agent → Summarizer_Agent
and streams the conversation.
"""

import json, sys
from autogen import GroupChat, UserProxyAgent, GroupChatManager

from agents.planner import planner
from agents.rag_agent import rag_agent
from agents.search_agent import search_agent
from agents.summarizer_agent import summarizer_agent

# ---------- build group ----------
group = GroupChat(
    agents=[planner, rag_agent, search_agent, summarizer_agent],
    messages=[]
)
manager = GroupChatManager(group)

# ---------- user proxy ----------
user = UserProxyAgent("User")

if __name__ == "__main__":
    # Accept CLI arg: path to image OR ingredient list string
    if len(sys.argv) > 1:
        user_msg = sys.argv[1]
    else:
        # demo: ingredients already detected
        demo_list = ["鸡蛋", "西红柿", "葱"]
        user_msg = f"These are my ingredients: {json.dumps(demo_list, ensure_ascii=False)}"

    user.initiate_chat(manager, message=user_msg)
