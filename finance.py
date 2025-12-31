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

# --- 页面配置 ---
st.set_page_config(page_title="AI 智能账本 Pro (视觉增强版)", page_icon="💰", layout="wide")

# --- 常量配置 ---
GITHUB_API_URL = "https://api.github.com"
# 推荐使用能力较强的视觉模型，如 Qwen2.5-VL 或 Qwen2-VL-72B
VISION_MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct" 
TEXT_MODEL_NAME = "deepseek-ai/DeepSeek-V3"
CHUNK_SIZE = 12000 
BILL_CYCLE_DAY = 10  # 账单日：每月10号

ALLOWED_CATEGORIES = [
    "餐饮美食", "交通出行", "购物消费", "生活服务", "医疗健康", "工资收入", "理财投资", "转账红包", "其他"
]

# --- 核心工具：OpenAI Client ---
def get_llm_client(api_key):
    # 请确保 base_url 符合你使用的服务商 (如 SiliconFlow, DeepSeek 等)
    return OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

# --- 辅助逻辑 ---
def get_fiscal_range(current_date, cycle_day=BILL_CYCLE_DAY):
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

def repair_truncated_json(json_str):
    json_str = json_str.strip()
    if json_str.endswith("]"): return json_str
    repair_attempts = ["]", "}]", "\"}]", "0}]", "null}]"]
    if json_str.endswith(","): json_str = json_str[:-1]
    for suffix in repair_attempts:
        try:
            temp_str = json_str + suffix
            json.loads(temp_str)
            return temp_str
        except: continue
    return json_str

def extract_json_from_text(text):
    if not text: return None, "空响应"
    try:
        text = text.strip()
        # 尝试提取 Markdown 代码块
        code_block_pattern = r"``" + r"`(?:json)?(.*?)``" + r"`"
        match_code = re.search(code_block_pattern, text, re.DOTALL)
        if match_code: text = match_code.group(1).strip()
        else:
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = text.strip()
        
        text = repair_truncated_json(text)
        # 提取数组部分
        match_array = re.search(r'\[.*\]', text, re.DOTALL)
        if match_array: text_to_parse = match_array.group()
        else: text_to_parse = text
            
        result = json.loads(text_to_parse)
        if isinstance(result, (list, dict)):
            return result if isinstance(result, list) else [result], None
    except: pass
    return None, "JSON提取失败"

# --- 基金相关工具 ---
def get_fund_realtime_valuation(fund_code):
    """通过公开接口获取基金估值"""
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
                return float(price), name, time_str
    except Exception:
        pass
    return 0.0, None, None

# --- 数据管理类 ---
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
    def merge_data(old_df, new_df):
        if new_df is None or new_df.empty: return old_df, 0
        
        def get_fp(d): return d['日期'].astype(str) + d['金额'].astype(str) + d['备注'].str[:5]
        if old_df.empty: return new_df, len(new_df)
        
        old_fp = set(get_fp(old_df))
        new_df['_fp'] = get_fp(new_df)
        to_add = new_df[~new_df['_fp'].isin(old_fp)].drop(columns=['_fp'])
        
        if to_add.empty: return old_df, 0
        merged = pd.concat([old_df, to_add], ignore_index=True)
        merged = DataManager._clean_ledger_types(merged)
        return merged, len(to_add)

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
        df = df.sort_values('日期', ascending=False).reset_index(drop=True)
        return df

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

# --- AI 解析器 (支持 PDF 转图片视觉识别) ---
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
                current_chunk = [line]; current_len = line_len
            else:
                current_chunk.append(line); current_len += line_len
        if current_chunk: chunks.append("\n".join(current_chunk))
        return chunks

    @staticmethod
    def _pdf_to_images(file_bytes):
        """核心：将PDF二进制流转换为高清图片列表"""
        images = []
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    # matrix=fitz.Matrix(2, 2) 放大2倍，提高清晰度
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                    img_bytes = pix.tobytes("png")
                    images.append(img_bytes)
        except Exception as e:
            print(f"PDF转图片失败: {e}")
        return images

    @staticmethod
    def _call_llm_for_text(text_chunk, api_key):
        """纯文本处理通道 (CSV/Excel)"""
        client = get_llm_client(api_key)
        prompt = f"""
        你是一个专业的财务数据提取助手。
        任务：从文本中识别交易记录。
        **强制要求**：
        1. 仅提取包含具体日期、金额的有效交易。
        2. "category" 字段必须根据商户和备注进行**智能推断**，并**严格**从以下列表中选择一项：
           {ALLOWED_CATEGORIES}
        3. 格式必须为纯JSON数组：[{{"date":"YYYY-MM-DD","type":"支出/收入","amount":数字,"merchant":"商户名或备注","category":"上述分类之一"}}]
        
        待处理文本：
        {text_chunk}
        """
        try:
            resp = client.chat.completions.create(
                model=TEXT_MODEL_NAME, messages=[{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.0
            )
            return resp.choices[0].message.content, None
        except Exception as e: return None, str(e)

    @staticmethod
    def process_image(filename, image_bytes, api_key, mode="ledger"):
        """视觉处理通道 (图片 + PDF转换后的图片)"""
        try:
            b64_img = base64.b64encode(image_bytes).decode('utf-8')
            client = get_llm_client(api_key)
            
            if mode == "ledger":
                prompt_text = f"""
                请分析这张账单图片（可能是银行流水截图或PDF页面）。
                
                **任务目标**：提取明细表格中的所有交易。
                
                **关键规则**：
                1. **忽略印章**：请忽略覆盖在文字上的红色印章（如“电子回单专用章”）。
                2. **识别正负数**：
                   - 如果金额列显示为负数（如 -10.40），则 type 为 "支出"，amount 记为正数 10.40。
                   - 如果金额列显示为正数，则 type 为 "收入"。
                3. **字段映射**：
                   - date: 交易日期 (YYYY-MM-DD)
                   - amount: 交易金额 (纯数字)
                   - merchant: 优先取“对手信息”、“交易摘要”或“商户名称”。
                   - category: 根据 merchant 内容，从 {ALLOWED_CATEGORIES} 中智能二选一。
                
                **输出格式**：
                仅返回标准 JSON 数组，无 Markdown 标记：
                [{{ "date": "2023-01-01", "type": "支出", "amount": 10.50, "merchant": "肯德基", "category": "餐饮美食" }}]
                """
            else:
                # 基金模式
                prompt_text = "提取基金持仓信息。识别基金名称(name)、基金代码(code, 6位数字)、持有份额(share)、持仓成本(cost, 可选)。返回JSON数组：[{code, name, share, cost}]"

            resp = client.chat.completions.create(
                model=VISION_MODEL_NAME,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }],
                max_tokens=4096
            )
            data, _ = extract_json_from_text(resp.choices[0].message.content)
            
            if not data: return None, "无数据", {}
            if isinstance(data, dict): data = [data]
            
            df = pd.DataFrame(data)
            
            if mode == "ledger":
                cols = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
                df = df.rename(columns=cols)
                for c in cols.values(): 
                    if c not in df.columns: df[c] = ""
            else:
                cols = {"code": "基金代码", "name": "基金名称", "share": "持有份额", "cost": "成本金额"}
                df = df.rename(columns=cols)
                for c in cols.values():
                    if c not in df.columns: df[c] = ""
                df['基金代码'] = df['基金代码'].astype(str).str.replace(r'\D', '', regex=True)
            
            return df, None, {}
        except Exception as e: return None, str(e), {}

    @staticmethod
    def identify_and_parse(filename, file_bytes, api_key):
        """智能分发入口"""
        try:
            filename_lower = filename.lower()
            
            # --- 分支 1: PDF 文件 (转图片 -> 视觉模型) ---
            if filename_lower.endswith('.pdf'):
                images = BillParser._pdf_to_images(file_bytes)
                if not images: return None, "PDF转图片失败", {}
                
                all_pdf_df = pd.DataFrame()
                # 循环处理每一页 PDF
                for i, img_bytes in enumerate(images):
                    res, err, _ = BillParser.process_image(f"{filename}_p{i}", img_bytes, api_key, mode="ledger")
                    if res is not None and not res.empty:
                        all_pdf_df = pd.concat([all_pdf_df, res], ignore_index=True)
                
                if all_pdf_df.empty: return None, "PDF未提取到数据", {}
                return all_pdf_df, None, {}

            # --- 分支 2: 文本类文件 (CSV/Excel) 走纯文本 ---
            content_text = ""
            if filename_lower.endswith('.csv'):
                try: content_text = file_bytes.decode('utf-8')
                except: content_text = file_bytes.decode('gbk', errors='ignore')
            elif filename_lower.endswith(('.xls', '.xlsx')):
                xls = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
                content_text = "\n".join([f"{s}\n{d.to_csv(index=False)}" for s, d in xls.items()])
            else:
                # 其他格式尝试直接走视觉（如不支持的图片格式漏网之鱼）
                return None, "不支持的文件格式", {}
            
            if not content_text.strip(): return None, "空文件", {}

            chunks = BillParser.chunk_text_by_lines(content_text, CHUNK_SIZE)
            all_data = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(BillParser._call_llm_for_text, chunk, api_key): chunk for chunk in chunks}
                for future in concurrent.futures.as_completed(futures):
                    res, err = future.result()
                    if not err:
                        data, _ = extract_json_from_text(res)
                        if data: all_data.extend(data)
            
            if not all_data: return None, "未提取到数据", {}
            
            df = pd.DataFrame(all_data)
            cols = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
            df = df.rename(columns=cols)
            for c in cols.values(): 
                if c not in df.columns: df[c] = ""
            
            df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
            df['日期'] = df['日期'].astype(str).apply(lambda x: x.split(' ')[0])
            return df, None, {}

        except Exception as e: return None, str(e), {}

# --- 主程序 ---
def main():
    if 'debug_mode' not in st.session_state: st.session_state.debug_mode = False
    
    st.sidebar.title("⚙️ 设置")
    api_key = st.secrets.get("SILICONFLOW_API_KEY") or st.sidebar.text_input("API Key", type="password")
    gh_token = st.secrets.get("GITHUB_TOKEN")
    gh_repo = st.secrets.get("GITHUB_REPO")
    
    dm_ledger = DataManager(gh_token, gh_repo, "ledger.csv")
    dm_funds = DataManager(gh_token, gh_repo, "funds.csv")
    
    if dm_ledger.use_github:
        if st.sidebar.button("☁️ 强制同步云端"):
            with st.spinner("同步中..."):
                df_l, sha_l = dm_ledger.load_data(force_refresh=True)
                st.session_state.ledger_data = df_l
                st.session_state.ledger_sha = sha_l
                df_f, sha_f = dm_funds.load_data(force_refresh=True)
                st.session_state.fund_data = df_f
                st.session_state.fund_sha = sha_f
                st.success("同步完成")
                st.rerun()
    
    if 'ledger_data' not in st.session_state:
        df, sha = dm_ledger.load_data()
        st.session_state.ledger_data = df
        st.session_state.ledger_sha = sha
        
    if 'fund_data' not in st.session_state:
        df, sha = dm_funds.load_data()
        st.session_state.fund_data = df
        st.session_state.fund_sha = sha

    st.title("💰 AI 智能账本 Pro (视觉增强版)")
    
    default_start, default_end = get_fiscal_range(date.today())
    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        st.caption(f"当前统计周期 (每月{BILL_CYCLE_DAY}号切分)")
        date_range = st.date_input("选择统计时间段", value=(default_start, default_end), format="YYYY-MM-DD")

    # --- 计算资产 ---
    df_ledger = st.session_state.ledger_data.copy()
    df_funds = st.session_state.fund_data.copy()
    
    cash_net = 0.0
    current_income = 0.0
    current_expense = 0.0
    
    if not df_ledger.empty:
        df_ledger['金额'] = pd.to_numeric(df_ledger['金额'], errors='coerce').fillna(0)
        cash_net = df_ledger[df_ledger['类型']=='收入']['金额'].sum() - df_ledger[df_ledger['类型']=='支出']['金额'].sum()
        
        if len(date_range) == 2:
            df_ledger['dt'] = pd.to_datetime(df_ledger['日期'], errors='coerce').dt.date
            start_d, end_d = date_range[0], date_range[1]
            mask_period = (df_ledger['dt'] >= start_d) & (df_ledger['dt'] <= end_d)
            df_period = df_ledger[mask_period]
            current_income = df_period[df_period['类型']=='收入']['金额'].sum()
            current_expense = df_period[df_period['类型']=='支出']['金额'].sum()
        else:
            df_period = pd.DataFrame()
    else:
        df_period = pd.DataFrame()

    fund_total_value = 0.0
    if 'fund_prices' not in st.session_state: st.session_state.fund_prices = {}
    
    if not df_funds.empty:
        df_funds['持有份额'] = pd.to_numeric(df_funds['持有份额'], errors='coerce').fillna(0)
        for idx, row in df_funds.iterrows():
            code = str(row['基金代码'])
            share = float(row['持有份额'])
            if code in st.session_state.fund_prices:
                price = st.session_state.fund_prices[code]['price']
                fund_total_value += share * price
    
    total_assets = cash_net + fund_total_value

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 历史总净值", f"¥{total_assets:,.2f}")
    c2.metric("📅 本期支出", f"¥{current_expense:,.2f}")
    c3.metric("📅 本期收入", f"¥{current_income:,.2f}")
    c4.metric("📊 基金市值", f"¥{fund_total_value:,.2f}", delta="点击下方刷新" if fund_total_value==0 else "实时")
    
    st.divider()

    t_import, t_add, t_history, t_funds, t_stats = st.tabs(["📥 账单导入", "✍️ 手动记账", "📋 历史明细", "📈 基金持仓", "📊 报表"])

    with t_import:
        st.info("💡 升级提示：现已支持 PDF 银行账单的视觉识别！自动忽略红章、自动处理负数支出。")
        files = st.file_uploader("上传账单 (PDF/图片/CSV/Excel)", accept_multiple_files=True)
        if files and st.button("🚀 开始识别账单", type="primary"):
            if not api_key: st.error("请配置 API Key"); st.stop()
            
            new_df = pd.DataFrame()
            tasks = []
            
            # 预处理：区分图片/PDF (走视觉) 和 CSV/Excel (走文本)
            # 注意：BillParser.identify_and_parse 内部已经处理了 PDF->图片 的逻辑
            # 我们只需要根据文件后缀传参即可
            
            # 这里为了简化进度条，我们还是把每个文件作为一个任务
            with st.status("正在AI识别...") as status:
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {}
                    for f in files:
                        f.seek(0)
                        file_bytes = f.read()
                        
                        # 如果是图片，直接调 visual 处理 (为了复用逻辑，identify_and_parse 也可以处理，但这里我们显式区分一下更清晰)
                        ext = f.name.split('.')[-1].lower()
                        if ext in ['png', 'jpg', 'jpeg']:
                             futures[executor.submit(BillParser.process_image, f.name, file_bytes, api_key, "ledger")] = f.name
                        else:
                             # PDF, Excel, CSV 都交给 identify_and_parse 智能判断
                             futures[executor.submit(BillParser.identify_and_parse, f.name, file_bytes, api_key)] = f.name
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            res, err, _ = future.result()
                            if res is not None: new_df = pd.concat([new_df, res], ignore_index=True)
                        except Exception as e: st.write(f"Error: {e}")
                status.update(label="完成", state="complete")

            if not new_df.empty:
                merged, added = DataManager.merge_data(st.session_state.ledger_data, new_df)
                if added > 0:
                    ok, sha = dm_ledger.save_data(merged, st.session_state.get('ledger_sha'))
                    if ok:
                        st.session_state.ledger_data = merged
                        st.session_state.ledger_sha = sha
                        st.success(f"成功导入 {added} 条记录")
                    else: st.error("保存失败")
                else: st.warning("未发现新数据 (可能已存在)")

    with t_add:
        with st.form("manual"):
            c1, c2, c3 = st.columns(3)
            d = c1.date_input("日期", date.today())
            t = c2.selectbox("类型", ["支出", "收入"])
            a = c3.number_input("金额", min_value=0.01)
            c4, c5 = st.columns([1,2])
            cat = c4.selectbox("分类", ALLOWED_CATEGORIES)
            rem = c5.text_input("备注")
            if st.form_submit_button("保存", width="stretch"):
                row = pd.DataFrame([{"日期":str(d),"类型":t,"金额":a,"分类":cat,"备注":rem}])
                merged, added = DataManager.merge_data(st.session_state.ledger_data, row)
                ok, sha = dm_ledger.save_data(merged, st.session_state.get('ledger_sha'))
                if ok: 
                    st.session_state.ledger_data = merged
                    st.session_state.ledger_sha = sha
                    st.success("保存成功")

    with t_history:
        if st.session_state.ledger_data.empty: st.info("无数据")
        else:
            df_show = st.session_state.ledger_data.sort_values('日期', ascending=False)
            edited_df = st.data_editor(
                df_show, 
                use_container_width=True, 
                num_rows="dynamic",
                key="editor_history",
                column_config={
                    "日期": st.column_config.DateColumn("日期", format="YYYY-MM-DD"),
                    "分类": st.column_config.SelectboxColumn(options=ALLOWED_CATEGORIES),
                    "金额": st.column_config.NumberColumn(format="%.2f")
                }
            )
            if st.button("💾 保存修改"):
                ok, sha = dm_ledger.save_data(edited_df, st.session_state.get('ledger_sha'))
                if ok:
                    st.session_state.ledger_data = edited_df
                    st.session_state.ledger_sha = sha
                    st.success("已更新")
                    time.sleep(1)
                    st.rerun()

    with t_funds:
        c_f1, c_f2 = st.columns([1, 3])
        with c_f1:
            st.subheader("📸 导入持仓")
            fund_files = st.file_uploader("上传持仓截图", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
            if fund_files and st.button("识别持仓"):
                new_funds = pd.DataFrame()
                with st.status("识别中...") as status:
                    for f in fund_files:
                        f.seek(0)
                        res, err, _ = BillParser.process_image(f.name, f.read(), api_key, mode="fund")
                        if res is not None: new_funds = pd.concat([new_funds, res], ignore_index=True)
                    status.update(label="完成", state="complete")
                
                if not new_funds.empty:
                    current_funds = st.session_state.fund_data
                    merged_funds = pd.concat([current_funds, new_funds], ignore_index=True)
                    merged_funds = DataManager._clean_fund_types(merged_funds)
                    ok, sha = dm_funds.save_data(merged_funds, st.session_state.get('fund_sha'))
                    if ok:
                        st.session_state.fund_data = merged_funds
                        st.session_state.fund_sha = sha
                        st.success("持仓更新")
                        st.rerun()
        
        with c_f2:
            st.subheader("📈 持仓列表")
            col_act1, col_act2 = st.columns([1, 5])
            if col_act1.button("🔄 刷新行情"):
                codes = st.session_state.fund_data['基金代码'].unique()
                progress = st.progress(0)
                for i, code in enumerate(codes):
                    if not code: continue
                    val, name, t_str = get_fund_realtime_valuation(code)
                    if val > 0:
                        st.session_state.fund_prices[code] = {"price": val, "name": name, "time": t_str}
                    progress.progress((i + 1) / len(codes))
                st.rerun()
            
            if st.session_state.fund_data.empty:
                st.info("暂无持仓")
            else:
                display_data = []
                for _, row in st.session_state.fund_data.iterrows():
                    code = str(row['基金代码'])
                    share = float(row['持有份额'])
                    cost = float(row['成本金额'])
                    
                    curr_info = st.session_state.fund_prices.get(code, {})
                    curr_price = curr_info.get('price', 0)
                    curr_name = curr_info.get('name', row['基金名称'])
                    
                    mkt_value = share * curr_price if curr_price > 0 else 0
                    profit = mkt_value - cost if (mkt_value > 0 and cost > 0) else 0
                    
                    display_data.append({
                        "基金代码": code,
                        "基金名称": curr_name,
                        "持有份额": share,
                        "最新净值": curr_price,
                        "持仓市值": mkt_value,
                        "参考盈亏": profit
                    })
                
                df_disp = pd.DataFrame(display_data)
                edited_funds = st.data_editor(
                    df_disp,
                    use_container_width=True,
                    key="editor_funds",
                    column_config={
                        "持有份额": st.column_config.NumberColumn(format="%.2f"),
                        "最新净值": st.column_config.NumberColumn(format="%.4f"),
                        "持仓市值": st.column_config.NumberColumn(format="%.2f"),
                        "参考盈亏": st.column_config.NumberColumn(format="%.2f"),
                    },
                    disabled=["最新净值", "持仓市值", "参考盈亏"]
                )
                if st.button("💾 保存持仓修改"):
                    save_funds = edited_funds[['基金代码', '基金名称', '持有份额']].copy()
                    save_funds['成本金额'] = 0 
                    ok, sha = dm_funds.save_data(save_funds, st.session_state.get('fund_sha'))
                    if ok:
                        st.session_state.fund_data = save_funds
                        st.session_state.fund_sha = sha
                        st.success("已更新")

    with t_stats:
        if df_period.empty:
            st.info("本期无数据")
        else:
            df_exp = df_period[df_period['类型'] == '支出']
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.subheader("支出结构")
                if not df_exp.empty:
                    fig_pie = px.pie(df_exp, values='金额', names='分类', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                st.subheader("资产趋势")
                df_sorted = df_ledger.sort_values('日期')
                df_sorted['net'] = df_sorted.apply(lambda x: x['金额'] if x['类型']=='收入' else -x['金额'], axis=1)
                df_sorted['asset'] = df_sorted['net'].cumsum()
                fig_line = px.line(df_sorted, x='日期', y='asset')
                st.plotly_chart(fig_line, use_container_width=True)

if __name__ == "__main__":
    main()
