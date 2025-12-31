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
from openai import OpenAI
import concurrent.futures
import time
import plotly.express as px
import plotly.graph_objects as go

# ==================== 页面配置与样式 ====================
st.set_page_config(page_title="AI 智能账本 Pro (PDF视觉版)", page_icon="💰", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: bold; color: #1f2937; }
    div[data-testid="stMetricDelta"] { font-size: 0.9rem; margin-top: 5px; }
    .css-1d391kg { padding-top: 2rem; }
    section[data-testid="stSidebar"] > div { padding-top: 2rem; }
    .dataframe { font-size: 0.9rem; }
    /* 让 Tabs 更紧凑 */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 3rem; }
</style>
""", unsafe_allow_html=True)

# ==================== 常量配置 ====================
GITHUB_API_URL = "https://api.github.com"

# --- 模型设置 ---
VISION_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
TEXT_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"

CHUNK_SIZE = 12000 
BILL_CYCLE_DAY = 10

ALLOWED_CATEGORIES = [
    "餐饮美食", "交通出行", "购物消费", "生活服务", "医疗健康", 
    "工资收入", "理财投资", "转账红包", "其他"
]

# ==================== 核心工具与逻辑 ====================

def get_llm_client(api_key):
    return OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

def get_fiscal_range(current_date, cycle_day=BILL_CYCLE_DAY):
    # ... (保持原有逻辑不变)
    if isinstance(current_date, str):
        current_date = datetime.datetime.strptime(current_date, "%Y-%m-%d").date()
    elif isinstance(current_date, datetime.datetime):
        current_date = current_date.date()

    if current_date.day >= cycle_day:
        start_date = date(current_date.year, current_date.month, cycle_day)
        if current_date.month == 12:
            end_date = date(current_date.year + 1, 1, cycle_day) - datetime.timedelta(days=1)
        else:
            end_date = date(current_date.year, current_date.month + 1, cycle_day) - datetime.timedelta(days=1)
    else:
        if current_date.month == 1:
            start_date = date(current_date.year - 1, 12, cycle_day)
        else:
            start_date = date(current_date.year, current_date.month - 1, cycle_day)
        end_date = date(current_date.year, current_date.month, cycle_day) - datetime.timedelta(days=1)
    return start_date, end_date

# --- 新增：更强大的数据合并与覆盖逻辑 ---
def merge_data_with_overwrite(old_df, new_df):
    """
    策略：将新旧数据合并，基于指纹去重，保留最新的（通过 keep='last' 实现覆盖）。
    指纹规则：日期(标准化)+金额+备注前6位
    """
    if new_df is None or new_df.empty: return old_df, 0
    if old_df.empty: return new_df, len(new_df)
    
    # 标准化日期格式，统一为 YYYY-MM-DD
    def normalize_date(d):
        if pd.isna(d): return ""
        s = str(d)
        # 修复支付宝的 2025/12/30 -> 2025-12-30
        s = s.replace('/', '-')
        s = s.split(' ')[0] # 去掉时间部分
        return s

    # 统一清洗
    old_df_clean = old_df.copy()
    new_df_clean = new_df.copy()
    
    for df in [old_df_clean, new_df_clean]:
        df['日期'] = df['日期'].apply(normalize_date)
        df['备注'] = df['备注'].astype(str)
        df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
    
    # 生成指纹 (为了去重)
    # 针对招商银行等只有日期的情况，依赖备注的前6位来区分同一日同一金额的不同交易
    def get_fp(d): 
        return d['日期'].astype(str) + "_" + d['金额'].astype(str) + "_" + d['备注'].str[:6]

    old_df_clean['_fp'] = get_fp(old_df_clean)
    new_df_clean['_fp'] = get_fp(new_df_clean)
    
    # 合并并去重
    merged_df = pd.concat([old_df_clean, new_df_clean], ignore_index=True)
    
    # 核心逻辑：keep='last' 意味着如果 new_df 里有和 old_df 一样的指纹，new_df 的条目会覆盖 old_df 的
    final_df = merged_df.drop_duplicates(subset=['_fp'], keep='last').drop(columns=['_fp'])
    
    # 排序并规范化类型
    final_df['日期'] = pd.to_datetime(final_df['日期'], errors='coerce').dt.date
    final_df = final_df.sort_values('日期', ascending=False).reset_index(drop=True)

    # 计算新增/更新数量（简单起见，返回新数据条数）
    return final_df, len(new_df)


def get_fund_realtime_valuation(fund_code):
    url = f"http://fundgz.1234567.com.cn/js/{fund_code}.js?rt={int(time.time()*1000)}"
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            content = resp.text
            match = re.search(r'jsonpgz\((.*?)\);', content)
            if match:
                data = json.loads(match.group(1))
                price = data.get('gsz') or data.get('dwjz')
                name = data.get('name')
                time_str = data.get('gztime') or data.get('jzrq')
                return float(price) if price else 0.0, name, time_str
    except Exception:
        pass
    return 0.0, None, None

# ==================== 数据管理类 ====================

class DataManager:
    def __init__(self, github_token=None, repo=None, filename="ledger.csv"):
        self.github_token = github_token
        if repo and repo.startswith("http"):
            self.repo = repo.rstrip("/").split("github.com/")[-1]
        else:
            self.repo = repo
        self.filename = filename
        self.use_github = bool(github_token and self.repo)

    def load_data(self, force_refresh=False):
        if self.use_github:
            if force_refresh: self._fetch_github_content.clear()
            df, sha = self._load_from_github()
        else:
            df, sha = self._load_from_local()

        if "ledger" in self.filename:
            df = self._clean_ledger_types(df)
        elif "funds" in self.filename:
            df = self._clean_fund_types(df)
        return df, sha

    def save_data(self, df, sha=None):
        save_df = df.copy()
        if "ledger" in self.filename and '日期' in save_df.columns:
            save_df['日期'] = save_df['日期'].astype(str)
        if "funds" in self.filename and '基金代码' in save_df.columns:
            save_df['基金代码'] = save_df['基金代码'].astype(str)

        if self.use_github:
            success, new_sha = self._save_to_github(save_df, sha)
            return success, new_sha
        else:
            return self._save_to_local(save_df), None

    @staticmethod
    def _clean_ledger_types(df):
        expected_cols = ["日期", "类型", "金额", "备注", "分类"]
        for col in expected_cols:
            if col not in df.columns: df[col] = ""
        df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0.0)
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce').dt.date
        df['日期'] = df['日期'].fillna(date.today())
        df['类型'] = df['类型'].astype(str).replace('nan', '支出')
        df['分类'] = df['分类'].astype(str).replace('nan', '其他')
        df['备注'] = df['备注'].astype(str).replace('nan', '')
        return df.sort_values('日期', ascending=False).reset_index(drop=True)

    @staticmethod
    def _clean_fund_types(df):
        expected_cols = ["基金代码", "基金名称", "持有份额", "成本金额"]
        for col in expected_cols:
            if col not in df.columns: df[col] = ""
        df['基金代码'] = df['基金代码'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
        df['持有份额'] = pd.to_numeric(df['持有份额'], errors='coerce').fillna(0.0)
        df['成本金额'] = pd.to_numeric(df['成本金额'], errors='coerce').fillna(0.0)
        df['基金名称'] = df['基金名称'].astype(str)
        return df

    def _load_from_local(self):
        if os.path.exists(self.filename):
            try: return pd.read_csv(self.filename, dtype=str), None
            except: pass
        return self._create_empty_df(), None

    def _save_to_local(self, df):
        df.to_csv(self.filename, index=False)
        return True

    @st.cache_data(ttl=300, show_spinner=False)
    def _fetch_github_content(_self):
        # ... (保持原有Github API逻辑)
        headers = {"Authorization": f"token {_self.github_token}", "Accept": "application/vnd.github.v3+json"}
        url = f"{GITHUB_API_URL}/repos/{_self.repo}/contents/{_self.filename}"
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200: return response.json(), None
            elif response.status_code == 404: return None, 404
            else: return None, response.status_code
        except Exception as e: return None, str(e)

    def _load_from_github(self):
        content, error = self._fetch_github_content()
        if content:
            try:
                csv_str = base64.b64decode(content['content']).decode('utf-8')
                df = pd.read_csv(StringIO(csv_str), dtype=str)
                return df, content['sha']
            except: return self._create_empty_df(), content['sha']
        return self._create_empty_df(), None

    def _save_to_github(self, df, sha):
        # ... (保持原有Github API逻辑)
        headers = {"Authorization": f"token {self.github_token}", "Accept": "application/vnd.github.v3+json"}
        url = f"{GITHUB_API_URL}/repos/{self.repo}/contents/{self.filename}"
        csv_str = df.to_csv(index=False)
        content_bytes = base64.b64encode(csv_str.encode('utf-8')).decode('utf-8')
        data = {"message": f"Update {self.filename}", "content": content_bytes}
        if sha: data["sha"] = sha
        try:
            resp = requests.put(url, headers=headers, data=json.dumps(data), timeout=30)
            if resp.status_code in [200, 201]:
                self._fetch_github_content.clear()
                return True, resp.json()['content']['sha']
            elif resp.status_code in [409, 422]:
                self._fetch_github_content.clear()
                latest_content, _ = self._fetch_github_content()
                if latest_content:
                    data["sha"] = latest_content['sha']
                    retry = requests.put(url, headers=headers, data=json.dumps(data), timeout=30)
                    if retry.status_code in [200, 201]:
                        self._fetch_github_content.clear()
                        return True, retry.json()['content']['sha']
                return False, None
        except: return False, None

    def _create_empty_df(self):
        if "ledger" in self.filename:
            return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"])
        elif "funds" in self.filename:
            return pd.DataFrame(columns=["基金代码", "基金名称", "持有份额", "成本金额"])
        return pd.DataFrame()

# ==================== AI 解析器 (高度优化) ====================

class BillParser:
    @staticmethod
    def chunk_text_by_lines(text, max_chars=CHUNK_SIZE):
        if len(text) <= max_chars: return [text]
        lines = text.split('\n')
        chunks, current_chunk, current_len = [], [], 0
        for line in lines:
            line_len = len(line) + 1
            if current_len + line_len > max_chars:
                if current_chunk: chunks.append("\n".join(current_chunk))
                current_chunk, current_len = [line], line_len
            else:
                current_chunk.append(line); current_len += line_len
        if current_chunk: chunks.append("\n".join(current_chunk))
        return chunks

    @staticmethod
    def _pdf_to_images(file_bytes):
        images = []
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    # 适当放大保证清晰度，但也压缩传输
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                    images.append(pix.tobytes("png"))
        except Exception as e:
            st.error(f"PDF转图片错误: {e}")
        return images

    @staticmethod
    def _call_llm_for_text(text_chunk, api_key):
        client = get_llm_client(api_key)
        prompt = f"""
        你是一个财务数据提取专家。
        任务：解析下方的交易记录文本。
        
        **格式要求**：
        请直接返回一个标准 JSON 对象，包含一个名为 "records" 的数组。不要使用 Markdown 代码块。
        字段定义：
        - date: 交易日期，格式 YYYY-MM-DD (处理 2025/12/30 这种格式)
        - type: "支出" 或 "收入"
        - amount: 纯数字金额
        - merchant: 商户名或摘要
        - category: 根据 merchant 从 {ALLOWED_CATEGORIES} 中选择

        **注意**：不要遗漏任何行。如果遇到相同日期和金额的交易，请务必通过 merchant 区分。
        
        文本内容：
        {text_chunk}
        """
        
        try:
            # 使用 JSON Mode 提高效率和准确性
            resp = client.chat.completions.create(
                model=TEXT_MODEL_NAME, 
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, # 强制JSON
                max_tokens=4096, 
                temperature=0.1
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            return parsed.get("records", []), None
        except Exception as e: return None, str(e)

    @staticmethod
    def process_image(filename, image_bytes, api_key, mode="ledger"):
        try:
            b64_img = base64.b64encode(image_bytes).decode('utf-8')
            client = get_llm_client(api_key)

            if mode == "ledger":
                prompt_text = f"""
                分析这张账单图片（可能是银行/支付宝/微信/招商银行流水截图）。
                
                **任务目标**：提取表格中所有交易明细。
                
                **具体要求**：
                1. **日期格式兼容**：如果是支付宝的 "2025/12/30"，转录为 "2025-12-30"。如果是只有日期（如招商银行），请按上下文顺序推断，不需要具体时间。
                2. **金额处理**：如有正负号，"支出" 记为正数，"收入" 记为正数，通过 type 字段区分。
                3. **去重/覆盖**：如果同一天有多笔相同金额的交易，请务必在 merchant 字段中保留唯一的摘要信息（如 "交易1"，"交易2" 或不同的店名），以便后续系统区分。
                4. **分类**：仅从 {ALLOWED_CATEGORIES} 中选。
                
                **输出格式**：
                直接返回标准 JSON 对象，不要 Markdown：
                {{"records": [ {{ "date": "2023-01-01", "type": "支出", "amount": 10.50, "merchant": "肯德基", "category": "餐饮美食" }} ]}}
                """
            else:
                # --- 基金模式 Prompt ---
                prompt_text = """
                提取基金持仓信息。
                **关键**：
                1. 提取 code (基金代码), name (基金名称), share (持有份额), cost (持仓成本)。
                2. **严禁**提取 "市值" 字段。截图上如果有 "持有市值" 或 "参考市值"，请忽略它。
                3. 如果份额显示为 "10000.00"，提取 10000.00。
                
                输出格式：
                {{"records": [ {{ "code": "000001", "name": "华夏成长", "share": 1000, "cost": 1050.00 }} ]}}
                """

            resp = client.chat.completions.create(
                model=VISION_MODEL_NAME,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }],
                response_format={"type": "json_object"}, # 强制JSON
                max_tokens=2048
            )
            
            parsed = json.loads(resp.choices[0].message.content)
            data = parsed.get("records", [])
            
            if not data: return None, "无数据", {}
            
            df = pd.DataFrame(data)

            if mode == "ledger":
                # 字段映射
                cols_map = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
                df = df.rename(columns=cols_map)
                for c in cols_map.values(): 
                    if c not in df.columns: df[c] = ""
                # 强制清洗日期中的斜杠，防止去重失效
                df['日期'] = df['日期'].astype(str).str.replace('/', '-')
                df['日期'] = df['日期'].str.split(' ').str[0]
            else:
                # 基金字段映射
                cols_map = {"code": "基金代码", "name": "基金名称", "share": "持有份额", "cost": "成本金额"}
                df = df.rename(columns=cols_map)
                for c in cols_map.values():
                    if c not in df.columns: df[c] = ""
                df['基金代码'] = df['基金代码'].astype(str).str.replace(r'\D', '', regex=True).str.zfill(6)

            return df, None, {}
        except Exception as e: return None, str(e), {}

    @staticmethod
    def identify_and_parse(filename, file_bytes, api_key):
        try:
            filename_lower = filename.lower()

            # --- 分支 1: PDF (并发处理每一页) ---
            if filename_lower.endswith('.pdf'):
                images = BillParser._pdf_to_images(file_bytes)
                if not images: return None, "PDF转图片失败", {}

                final_pdf_df = pd.DataFrame()
                
                # 使用并发处理每一页，提高PDF处理速度
                with st.status(f"正在处理 PDF (共 {len(images)} 页)...", expanded=False) as status:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        future_to_page = {
                            executor.submit(BillParser.process_image, f"page_{i}", img, api_key, "ledger"): i 
                            for i, img in enumerate(images)
                        }
                        
                        for future in concurrent.futures.as_completed(future_to_page):
                            page_idx = future_to_page[future]
                            try:
                                res, err, _ = future.result()
                                if res is not None and not res.empty:
                                    final_pdf_df = pd.concat([final_pdf_df, res], ignore_index=True)
                                    status.update(label=f"处理页 {page_idx+1} 完成", state="running")
                            except Exception as e:
                                st.toast(f"第 {page_idx+1} 页处理失败: {e}", icon="⚠️")
                    
                    status.update(label="PDF 处理完成", state="complete", expanded=False)

                if final_pdf_df.empty: return None, "PDF未提取到数据", {}
                return final_pdf_df, None, {}

            # --- 分支 2: 图片 (直接视觉) ---
            if filename_lower.endswith(('.png', '.jpg', 'jpeg')):
                return BillParser.process_image(filename, file_bytes, api_key, mode="ledger")

            # --- 分支 3: 文本类 ---
            content_text = ""
            if filename_lower.endswith('.csv'):
                try: content_text = file_bytes.decode('utf-8')
                except: content_text = file_bytes.decode('gbk', errors='ignore')
            elif filename_lower.endswith(('.xls', '.xlsx')):
                xls = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
                content_text = "\n".join([f"{s}\n{d.to_csv(index=False)}" for s, d in xls.items()])
            
            if not content_text.strip(): return None, "空文件", {}
            
            chunks = BillParser.chunk_text_by_lines(content_text, CHUNK_SIZE)
            all_data = []

            with st.status("正在解析文本数据..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {executor.submit(BillParser._call_llm_for_text, chunk, api_key): chunk for chunk in chunks}
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            res, err = future.result()
                            if not err and res:
                                all_data.extend(res)
                        except: continue

            if not all_data: return None, "未提取到数据", {}
            
            df = pd.DataFrame(all_data)
            cols = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
            df = df.rename(columns=cols)
            for c in cols.values(): 
                if c not in df.columns: df[c] = ""
            
            # 文本清洗
            df['日期'] = df['日期'].astype(str).str.replace('/', '-')
            df['日期'] = df['日期'].str.split(' ').str[0]
            df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
            
            return df, None, {}

        except Exception as e: return None, str(e), {}

# ==================== 主程序逻辑 ====================

def main():
    if 'debug_mode' not in st.session_state: st.session_state.debug_mode = False

    # 侧边栏
    with st.sidebar:
        st.title("⚙️ 设置")
        api_key = st.secrets.get("SILICONFLOW_API_KEY") or st.text_input("SiliconFlow API Key", type="password")
        gh_token = st.secrets.get("GITHUB_TOKEN")
        gh_repo = st.secrets.get("GITHUB_REPO")
        
        if gh_token and gh_repo:
            st.success("云端已连接")
            if st.button("☁️ 强制同步云端", use_container_width=True):
                with st.spinner("同步中..."):
                    dm_ledger = DataManager(gh_token, gh_repo, "ledger.csv")
                    st.session_state.ledger_data, st.session_state.ledger_sha = dm_ledger.load_data(force_refresh=True)
                    dm_funds = DataManager(gh_token, gh_repo, "funds.csv")
                    st.session_state.fund_data, st.session_state.fund_sha = dm_funds.load_data(force_refresh=True)
                    st.rerun()

    # 数据管理器初始化
    dm_ledger = DataManager(gh_token, gh_repo, "ledger.csv")
    dm_funds = DataManager(gh_token, gh_repo, "funds.csv")

    # 数据加载
    if 'ledger_data' not in st.session_state:
        df, sha = dm_ledger.load_data()
        st.session_state.ledger_data = df
        st.session_state.ledger_sha = sha
    if 'fund_data' not in st.session_state:
        df, sha = dm_funds.load_data()
        st.session_state.fund_data = df
        st.session_state.fund_sha = sha
    if 'fund_prices' not in st.session_state: st.session_state.fund_prices = {}

    st.title("💰 AI 智能账本 Pro (PDF视觉版)")
    
    # 财务周期
    default_start, default_end = get_fiscal_range(date.today())
    with st.container():
        col_d1 = st.columns([1])[0]
        with col_d1:
            st.caption(f"当前统计周期 (每月{BILL_CYCLE_DAY}号切分)")
            date_range = st.date_input("选择统计时间段", value=(default_start, default_end), format="YYYY-MM-DD", label_visibility="collapsed")

    # 顶部指标计算
    df_ledger = st.session_state.ledger_data.copy()
    df_funds = st.session_state.fund_data.copy()

    cash_net = current_income = current_expense = 0.0
    df_period = pd.DataFrame()

    if not df_ledger.empty:
        df_ledger['金额'] = pd.to_numeric(df_ledger['金额'], errors='coerce').fillna(0)
        cash_net = df_ledger[df_ledger['类型']=='收入']['金额'].sum() - df_ledger[df_ledger['类型']=='支出']['金额'].sum()
        if len(date_range) == 2:
            df_ledger['dt'] = pd.to_datetime(df_ledger['日期'], errors='coerce').dt.date
            mask_period = (df_ledger['dt'] >= date_range[0]) & (df_ledger['dt'] <= date_range[1])
            df_period = df_ledger[mask_period]
            current_income = df_period[df_period['类型']=='收入']['金额'].sum()
            current_expense = df_period[df_period['类型']=='支出']['金额'].sum()

    # 基金市值计算
    fund_total_value = 0.0
    if not df_funds.empty:
        df_funds['持有份额'] = pd.to_numeric(df_funds['持有份额'], errors='coerce').fillna(0)
        for _, row in df_funds.iterrows():
            code = str(row['基金代码'])
            if code in st.session_state.fund_prices:
                fund_total_value += row['持有份额'] * st.session_state.fund_prices[code]['price']

    st.divider()
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("💰 总净资产", f"¥{cash_net + fund_total_value:,.2f}")
    c2.metric("📅 本期支出", f"¥{current_expense:,.2f}", delta_color="inverse")
    c3.metric("📅 本期收入", f"¥{current_income:,.2f}")
    c4.metric("📈 基金市值", f"¥{fund_total_value:,.2f}")

    # Tab 导航
    t_import, t_add, t_history, t_funds, t_stats = st.tabs(["📥 账单导入", "✍️ 手动记账", "📋 历史明细", "📈 基金持仓", "📊 报表"])

    with t_import:
        st.info("💡 **智能识别升级**：支持自动识别支付宝(YYYY/MM/DD)、微信(YYYY-MM-DD)及招商银行格式。**重复上传会自动覆盖旧数据**。")
        files = st.file_uploader("上传账单 (PDF/图片/CSV/Excel)", accept_multiple_files=True)
        
        if files and st.button("🚀 开始识别与合并", type="primary", use_container_width=True):
            if not api_key: st.error("请配置 API Key"); st.stop()

            all_new_df = pd.DataFrame()
            
            # 使用 Status 展示总体进度
            with st.status("正在批量处理文件...", expanded=True) as status:
                # 限制并发文件数，防止过载
                max_files_process = min(len(files), 5) 
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_file = {}
                    for f in files:
                        f.seek(0)
                        file_bytes = f.read()
                        # 提交任务
                        future = executor.submit(BillParser.identify_and_parse, f.name, file_bytes, api_key)
                        future_to_file[future] = f.name
                    
                    # 收集结果
                    for future in concurrent.futures.as_completed(future_to_file):
                        filename = future_to_file[future]
                        try:
                            res, err, _ = future.result()
                            if res is not None and not res.empty:
                                all_new_df = pd.concat([all_new_df, res], ignore_index=True)
                                st.write(f"✅ `{filename}` 提取成功")
                            else:
                                st.write(f"⚠️ `{filename}` 无数据或失败: {err}")
                        except Exception as e:
                            st.write(f"❌ `{filename}` 处理异常: {e}")
                
                status.update(label="所有文件识别完成", state="complete", expanded=False)

            if not all_new_df.empty:
                # 调用新的覆盖式合并逻辑
                st.info("正在进行数据去重与合并...")
                # 这里模拟一点延迟让用户看到处理过程
                time.sleep(0.5)
                
                merged_df, change_count = merge_data_with_overwrite(st.session_state.ledger_data, all_new_df)
                
                ok, sha = dm_ledger.save_data(merged_df, st.session_state.get('ledger_sha'))
                if ok:
                    st.session_state.ledger_data = merged_df
                    st.session_state.ledger_sha = sha
                    st.success(f"数据更新成功！共处理 {len(all_new_df)} 条数据。")
                    st.rerun()
                else: st.error("保存失败")
            else:
                st.warning("未提取到任何有效数据，请检查文件格式或 API Key。")

    with t_add:
        with st.form("manual", clear_on_submit=True):
            st.subheader("快速记账")
            c1, c2, c3 = st.columns(3)
            d = c1.date_input("日期", value=date.today(), label_visibility="collapsed")
            t = c2.selectbox("类型", ["支出", "收入"], label_visibility="collapsed")
            a = c3.number_input("金额", min_value=0.01, step=0.01, label_visibility="collapsed")
            c4, c5 = st.columns([1, 2])
            cat = c4.selectbox("分类", ALLOWED_CATEGORIES, label_visibility="collapsed")
            rem = c5.text_input("备注", placeholder="消费内容...", label_visibility="collapsed")
            
            submitted = st.form_submit_button("保存记录", use_container_width=True)
            if submitted:
                row = pd.DataFrame([{"日期":str(d),"类型":t,"金额":a,"分类":cat,"备注":rem}])
                merged, _ = merge_data_with_overwrite(st.session_state.ledger_data, row)
                ok, sha = dm_ledger.save_data(merged, st.session_state.get('ledger_sha'))
                if ok: 
                    st.session_state.ledger_data = merged
                    st.session_state.ledger_sha = sha
                    st.success("保存成功")
                    st.rerun()

    with t_history:
        if st.session_state.ledger_data.empty: st.info("暂无数据")
        else:
            df_show = st.session_state.ledger_data.copy()
            # 显示时再清洗一下格式好看点
            df_show['日期'] = pd.to_datetime(df_show['日期']).dt.strftime('%Y-%m-%d')
            
            edited_df = st.data_editor(
                df_show, 
                use_container_width=True, 
                num_rows="dynamic",
                column_order=["日期", "类型", "分类", "金额", "备注"],
                key="editor_history",
                column_config={
                    "日期": st.column_config.DateColumn("日期", format="YYYY-MM-DD"),
                    "分类": st.column_config.SelectboxColumn(options=ALLOWED_CATEGORIES),
                    "金额": st.column_config.NumberColumn(format="%.2f"),
                    "类型": st.column_config.SelectboxColumn(options=["支出", "收入"])
                }
            )
            if st.button("💾 保存表格修改", use_container_width=True):
                # 保存前要先转换回标准格式
                save_df = edited_df.copy()
                save_df['日期'] = pd.to_datetime(save_df['日期']).dt.date
                ok, sha = dm_ledger.save_data(save_df, st.session_state.get('ledger_sha'))
                if ok:
                    st.session_state.ledger_data = save_df
                    st.session_state.ledger_sha = sha
                    st.success("修改已保存")
                    time.sleep(0.5); st.rerun()

    with t_funds:
        c_f1, c_f2 = st.columns([1, 3])
        with c_f1:
            # --- 基金导入 (只识别份额) ---
            st.subheader("📸 导入持仓")
            fund_files = st.file_uploader("上传截图", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
            if fund_files and st.button("识别持仓", use_container_width=True):
                if not api_key: st.error("请配置 API Key"); st.stop()
                new_funds = pd.DataFrame()
                with st.status("正在识别..."):
                    for f in fund_files:
                        f.seek(0)
                        res, err, _ = BillParser.process_image(f.name, f.read(), api_key, mode="fund")
                        if res is not None and not res.empty: 
                            new_funds = pd.concat([new_funds, res], ignore_index=True)
                
                if not new_funds.empty:
                    # 基金也使用覆盖逻辑：相同代码的基金，覆盖名称、份额和成本
                    old_funds = st.session_state.fund_data
                    # 合并
                    combined = pd.concat([old_funds, new_funds], ignore_index=True)
                    # 去重：以基金代码为唯一键，保留最新的
                    final_funds = combined.drop_duplicates(subset=['基金代码'], keep='last')
                    
                    ok, sha = dm_funds.save_data(final_funds, st.session_state.get('fund_sha'))
                    if ok:
                        st.session_state.fund_data = final_funds
                        st.success("持仓信息已更新")
                        st.rerun()

        with c_f2:
            # --- 基金列表与行情刷新 ---
            sub_c1, sub_c2 = st.columns([4, 1])
            sub_c1.subheader("📈 持仓详情")
            if sub_c2.button("🔄 刷新", use_container_width=True):
                if st.session_state.fund_data.empty: pass
                else:
                    codes = st.session_state.fund_data['基金代码'].unique()
                    progress = st.progress(0)
                    new_prices = {}
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        future_to_code = {executor.submit(get_fund_realtime_valuation, code): code for code in codes}
                        for i, future in enumerate(concurrent.futures.as_completed(future_to_code)):
                            code, val, name, t_str = future.result()
                            if val > 0: new_prices[code] = {"price": val, "name": name, "time": t_str}
                            progress.progress((i+1)/len(codes))
                    st.session_state.fund_prices.update(new_prices)
                    st.rerun()

            if st.session_state.fund_data.empty: st.info("暂无持仓")
            else:
                display_data = []
                for _, row in st.session_state.fund_data.iterrows():
                    code = str(row['基金代码'])
                    share = float(row['持有份额'])
                    cost = float(row['成本金额'])
                    curr_price = 0
                    
                    if code in st.session_state.fund_prices:
                        curr_price = st.session_state.fund_prices[code]['price']
                        
                    mkt_value = share * curr_price if curr_price > 0 else 0
                    profit = mkt_value - cost if (mkt_value > 0 and cost > 0) else 0
                    name = st.session_state.fund_prices.get(code, {}).get('name', row['基金名称'])

                    display_data.append({
                        "基金代码": code, "基金名称": name,
                        "持有份额": share, "最新净值": curr_price,
                        "持仓市值": mkt_value, "盈亏": profit
                    })
                st.data_editor(pd.DataFrame(display_data), use_container_width=True, column_config={"盈亏": st.column_config.NumberColumn(format="%.2f")}, disabled=["最新净值", "持仓市值", "盈亏"])

    with t_stats:
        if df_period.empty: st.info("本期无数据")
        else:
            df_exp = df_period[df_period['类型'] == '支出']
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                if not df_exp.empty:
                    fig_pie = px.pie(df_exp, values='金额', names='分类', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else: st.info("无支出")
            with col_chart2:
                df_sorted = df_ledger.sort_values('日期')
                df_sorted['net'] = df_sorted.apply(lambda x: x['金额'] if x['类型']=='收入' else -x['金额'], axis=1)
                df_sorted['asset'] = df_sorted['net'].cumsum()
                st.plotly_chart(px.line(df_sorted, x='日期', y='asset'), use_container_width=True)

if __name__ == "__main__":
    main()
