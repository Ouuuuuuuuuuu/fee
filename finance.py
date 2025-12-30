import streamlit as st
import pandas as pd
import datetime
from datetime import date
import requests
import json
import base64
from io import StringIO, BytesIO
import os
import fitz  # PyMuPDF
import re
from openai import OpenAI, APITimeoutError
import concurrent.futures
import time

# --- 页面配置 ---
st.set_page_config(page_title="AI 智能账本 Pro", page_icon="💰", layout="wide")

# --- 常量配置 ---
DEFAULT_TARGET_SPEND = 60.0  # 每日体面支出标准
GITHUB_API_URL = "https://api.github.com"
VISION_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct" 
TEXT_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"

# --- 核心工具：OpenAI Client (无缓存) ---
def get_llm_client(api_key):
    return OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

# --- 工具函数：增强版 JSON 提取 ---
def extract_json_from_text(text):
    """增强版JSON提取，支持更多异常格式，返回 (data, error_msg)"""
    if not text: 
        return None, "空响应"
    
    # 保存原始文本用于调试
    original_preview = text[:200].replace('\n', '\\n')
    
    try:
        # 1. 移除所有Markdown代码块标记
        text = text.strip()
        # 尝试匹配 markdown 代码块
        match_code = re.search(r'
