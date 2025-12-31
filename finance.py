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
from PIL import Image  # 新增：用于极速图片压缩

# ==================== 页面配置与样式 ====================
st.set_page_config(page_title="AI 账本 Pro (GitHub版)", page_icon="🚀", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 800; color: #2563eb; }
    .stAlert { border: 1px solid #e5e7eb; border-radius: 0.5rem; padding: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { border-radius: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# ==================== 常量配置 ====================
# --- 模型设置 (强制使用指定模型) ---
VISION_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
TEXT_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"

GITHUB_API_URL = "https://api.github.com"
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

# --- 核心提速：极速图片压缩 (修复速度问题的关键) ---
def optimize_image(img_bytes, max_dim=1280, quality=85):
    """将图片压缩至 1280px 宽以内，大幅减少 Token 消耗，提升 API 速度"""
    try:
        img = Image.open(BytesIO(img_bytes))
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        
        if img.width > max_dim or img.height > max_dim:
            ratio = min(max_dim / img.width, max_dim / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()
    except Exception as e:
        return img_bytes

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
                if price: return float(price), name
    except Exception:
        pass
    return 0.0, None

# ==================== 数据管理类 (GitHub 原生逻辑) ====================

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
            # 保存转为字符串
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
        # 修复：强制转为date对象，方便后续计算
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
        if "ledger" in self.filename:
            return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"]), None
        else:
            return pd.DataFrame(columns=["基金代码", "基金名称", "持有份额", "成本金额"]), None

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
            except: 
                if "ledger" in self.filename:
                    return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"]), None
                else:
                    return pd.DataFrame(columns=["基金代码", "基金名称", "持有份额", "成本金额"]), None
        if "ledger" in self.filename:
            return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"]), None
        else:
            return pd.DataFrame(columns=["基金代码", "基金名称", "持有份额", "成本金额"]), None

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

# ==================== AI 解析器 (整合极速逻辑) ====================

class TurboParser:
    @staticmethod
    def _pdf_to_images(file_bytes):
        images = []
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    # 放大一倍保证清晰度
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                    images.append(pix.tobytes("png"))
        except Exception as e:
            st.error(f"PDF转图片错误: {e}")
        return images

    @staticmethod
    def process_image(filename, raw_file_bytes, api_key):
        try:
            # 1. 极速压缩
            optimized_bytes = optimize_image(raw_file_bytes)
            b64_img = base64.b64encode(optimized_bytes).decode('utf-8')
            
            client = get_llm_client(api_key)
            
            prompt_text = f"""
            分析这张账单/流水。
            任务：提取交易明细。
            
            **规则**：
            1. 日期格式转换为 YYYY-MM-DD (兼容 2025/12/30)。
            2. 支出记为 "支出"，收入记为 "收入"。
            3. 自动归入分类，仅从 {ALLOWED_CATEGORIES} 中选。
            4. **去重敏感**：如果同一日有相同金额，请务必通过商户名区分 (如 "星巴克A店", "星巴克B店")。

            **Strict Output JSON**:
            {{"records": [{{"date":"YYYY-MM-DD","type":"支出","amount":10.5,"merchant":"商户","category":"分类"}}]}}
            """

            resp = client.chat.completions.create(
                model=VISION_MODEL_NAME, # Qwen3-VL
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }],
                response_format={"type": "json_object"},
                max_tokens=2048
            )
            
            parsed = json.loads(resp.choices[0].message.content)
            data = parsed.get("records", [])
            
            if not data: return None
            df = pd.DataFrame(data)
            cols_map = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
            return df.rename(columns=cols_map)

        except Exception as e: 
            return None

    @staticmethod
    def identify_and_parse(filename, file_bytes, api_key):
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.pdf'):
            images = TurboParser._pdf_to_images(file_bytes)
            if not images: return None
            
            # 并发处理PDF每一页
            final_df = pd.DataFrame()
            with st.status(f"正在处理 PDF (共 {len(images)} 页)..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(TurboParser.process_image, f"p{i}", img, api_key) for i, img in enumerate(images)]
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        if res is not None:
                            final_df = pd.concat([final_df, res], ignore_index=True)
            return final_df
        
        elif filename_lower.endswith(('.png', '.jpg', 'jpeg')):
            return TurboParser.process_image(filename, file_bytes, api_key)
        
        return None

# ==================== 主程序逻辑 ====================

def main():
    # 初始化 Session State
    if 'ledger_data' not in st.session_state: st.session_state.ledger_data = pd.DataFrame()
    if 'fund_data' not in st.session_state: st.session_state.fund_data = pd.DataFrame()
    if 'fund_prices' not in st.session_state: st.session_state.fund_prices = {}
    if 'api_key' not in st.session_state: st.session_state.api_key = st.secrets.get("SILICONFLOW_API_KEY")

    # 配置
    gh_token = st.secrets.get("GITHUB_TOKEN")
    gh_repo = st.secrets.get("GITHUB_REPO")

    # --- 数据加载 ---
    dm_ledger = DataManager(gh_token, gh_repo, "ledger.csv")
    dm_funds = DataManager(gh_token, gh_repo, "funds.csv")
    
    # 如果是首次运行，从云端加载
    if st.session_state.ledger_data.empty and gh_token:
        df, sha = dm_ledger.load_data()
        st.session_state.ledger_data = df
        st.session_state.ledger_sha = sha
    
    if st.session_state.fund_data.empty and gh_token:
        df, sha = dm_funds.load_data()
        st.session_state.fund_data = df
        st.session_state.fund_sha = sha

    # 侧边栏设置
    with st.sidebar:
        st.title("⚙️ 设置")
        st.session_state.api_key = st.text_input("API Key", value=st.session_state.api_key or "", type="password", label_visibility="collapsed", placeholder="Enter SiliconFlow Key")
        
        if gh_token and gh_repo:
            st.success("☁️ GitHub 已连接")
            if st.button("🔄 强制刷新云端", use_container_width=True):
                with st.spinner("同步中..."):
                    df, sha = dm_ledger.load_data(force_refresh=True)
                    st.session_state.ledger_data = df; st.session_state.ledger_sha = sha
                    df, sha = dm_funds.load_data(force_refresh=True)
                    st.session_state.fund_data = df; st.session_state.fund_sha = sha
                    st.rerun()

    # 财务概览
    default_start, default_end = get_fiscal_range(date.today())
    df_ledger = st.session_state.ledger_data
    if not df_ledger.empty:
        cash_net = df_ledger[df_ledger['类型']=='收入']['金额'].sum() - df_ledger[df_ledger['类型']=='支出']['金额'].sum()
        
        df_ledger['dt'] = pd.to_datetime(df_ledger['日期'], errors='coerce').dt.date
        mask_period = (df_ledger['dt'] >= default_start) & (df_ledger['dt'] <= default_end)
        df_period = df_ledger[mask_period]
        current_income = df_period[df_period['类型']=='收入']['金额'].sum()
        current_expense = df_period[df_period['类型']=='支出']['金额'].sum()
    else:
        cash_net = current_income = current_expense = 0.0

    # 账面资产计算
    fund_val = 0.0
    if not st.session_state.fund_data.empty:
        df_funds = st.session_state.fund_data
        df_funds['持有份额'] = pd.to_numeric(df_funds['持有份额'], errors='coerce').fillna(0)
        # 如果有缓存价格，计算市值
        for code in df_funds['基金代码'].unique():
            if code in st.session_state.fund_prices:
                price = st.session_state.fund_prices[code]
                shares = df_funds[df_funds['基金代码']==code]['持有份额'].sum()
                fund_val += shares * price

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 总净资产", f"¥{cash_net + fund_val:,.2f}")
    c2.metric("📅 本期支出", f"¥{current_expense:,.2f}", delta_color="inverse")
    c3.metric("📅 本期收入", f"¥{current_income:,.2f}")
    c4.metric("📈 基金市值", f"¥{fund_val:,.2f}")

    # Tabs
    t_import, t_add, t_history, t_funds, t_stats, t_copilot = st.tabs(["📥 导入", "✍️ 记账", "📋 明细", "📈 基金", "📊 报表", "🧠 AI Copilot"])

    # --- 导入 ---
    with t_import:
        files = st.file_uploader("上传账单 (PDF/图片)", accept_multiple_files=True, type=['csv', 'png', 'jpg', 'jpeg', 'pdf'])
        if files and st.button("🚀 极速解析", type="primary"):
            if not st.session_state.api_key: st.error("请先输入 API Key"); st.stop()
            
            status = st.status("AI Agent 正在处理...", expanded=True)
            all_new = pd.DataFrame()
            
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(TurboParser.identify_and_parse, f.name, f.read(), st.session_state.api_key): f.name for f in files}
                for future in concurrent.futures.as_completed(futures):
                    fname = futures[future]
                    try:
                        res = future.result()
                        if res is not None and not res.empty:
                            all_new = pd.concat([all_new, res], ignore_index=True)
                            status.write(f"✅ ✨ {fname} 完成 ({len(res)} 条)")
                        else:
                            status.write(f"⚠️ {fname} 未识别到数据")
                    except Exception as e:
                        status.write(f"❌ {fname} 错误")
            
            status.update(label=f"完成! 耗时 {time.time()-start_time:.2f}s", state="complete", expanded=False)
            
            if not all_new.empty:
                # 合并与清洗
                old_df = st.session_state.ledger_data
                # 标准化日期以便合并
                all_new['日期'] = pd.to_datetime(all_new['日期'], errors='coerce').dt.date
                merged_df, cnt = merge_data_with_overwrite(old_df, all_new)
                
                # 保存
                ok, sha = dm_ledger.save_data(merged_df, st.session_state.get('ledger_sha'))
                if ok:
                    st.session_state.ledger_data = merged_df
                    st.session_state.ledger_sha = sha
                    st.success(f"成功合并 {cnt} 条数据！")
                    st.rerun()

    # --- 手动记账 ---
    with t_add:
        with st.form("manual", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            d = c1.date_input("日期", value=date.today(), label_visibility="collapsed")
            t = c2.selectbox("类型", ["支出", "收入"], label_visibility="collapsed")
            a = c3.number_input("金额", min_value=0.01, step=0.01, label_visibility="collapsed")
            c4, c5 = st.columns([1, 2])
            cat = c4.selectbox("分类", ALLOWED_CATEGORIES, label_visibility="collapsed")
            rem = c5.text_input("备注", placeholder="消费内容...", label_visibility="collapsed")
            
            submitted = st.form_submit_button("保存记录", use_container_width=True)
            if submitted:
                row = pd.DataFrame([{"日期":d,"类型":t,"金额":float(a),"分类":cat,"备注":rem}])
                # 确保 row 日期是 date 对象
                row['日期'] = pd.to_datetime(row['日期']).dt.date
                merged, _ = merge_data_with_overwrite(st.session_state.ledger_data, row)
                ok, sha = dm_ledger.save_data(merged, st.session_state.get('ledger_sha'))
                if ok: 
                    st.session_state.ledger_data = merged
                    st.session_state.ledger_sha = sha
                    st.success("保存成功")
                    st.rerun()

    # --- 历史明细 (Bug 修复位置) ---
    with t_history:
        if st.session_state.ledger_data.empty: st.info("暂无数据")
        else:
            df_temp = st.session_state.ledger_data.copy()
            # *** 修复关键步骤 ***
            # 准备给 st.data_editor 的数据：
            # 1. 确保日期列是 datetime.date 对象，不是字符串，否则 column_config.DateColumn 会报错
            # 2. 确保其他类型正确
            df_temp['日期'] = pd.to_datetime(df_temp['日期'], errors='coerce').dt.date
            df_temp = df_temp.sort_values("日期", ascending=False)
            
            # 填充可能存在的 NaN 避免类型歧义
            for col in df_temp.columns:
                if df_temp[col].dtype == 'object': df_temp[col] = df_temp[col].fillna("")

            edited_df = st.data_editor(
                df_temp,
                use_container_width=True,
                num_rows="dynamic",
                column_order=["日期", "类型", "分类", "金额", "备注"],
                key="editor_history",
                column_config={
                    # 明确说明日期格式，因为编辑器内部格式很好
                    "日期": st.column_config.DateColumn("日期", format="YYYY-MM-DD", step=1),
                    "分类": st.column_config.SelectboxColumn(options=ALLOWED_CATEGORIES),
                    "金额": st.column_config.NumberColumn(format="%.2f"),
                    "类型": st.column_config.SelectboxColumn(options=["支出", "收入"])
                }
            )
            if st.button("💾 保存表格修改", use_container_width=True):
                # 编辑后的数据已经是 date 对象了，直接保存
                ok, sha = dm_ledger.save_data(edited_df, st.session_state.get('ledger_sha'))
                if ok:
                    st.session_state.ledger_data = edited_df
                    st.session_state.ledger_sha = sha
                    st.success("修改已保存")
                    time.sleep(0.5); st.rerun()

    # --- 基金 ---
    with t_funds:
        c_f1, c_f2 = st.columns([1, 3])
        with c_f1:
            st.subheader("📸 导入持仓")
            fund_files = st.file_uploader("上传截图", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
            if fund_files and st.button("识别持仓", use_container_width=True):
                if not st.session_state.api_key: st.error("请配置 API Key"); st.stop()
                new_funds = pd.DataFrame()
                with st.status("正在识别..."):
                    for f in fund_files:
                        f.seek(0)
                        # 复用 process_image 逻辑 (略作修改以适配基金)
                        # 这里简化：直接按基金 Prompt 处理
                        try:
                            optimized = optimize_image(f.read())
                            b64 = base64.b64encode(optimized).decode()
                            client = get_llm_client(st.session_state.api_key)
                            prompt = """
                            提取基金持仓。字段: code(代码), name(名称), share(份额), cost(成本)。
                            忽略市值。
                            JSON: {{"records": [{{"code":"000001","name":"华夏","share":1000,"cost":1000}}]}}
                            """
                            resp = client.chat.completions.create(
                                model=VISION_MODEL_NAME,
                                messages=[{"role":"user", "content": [{"type":"text", "text":prompt}, {"type":"image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
                                response_format={"type": "json_object"}
                            )
                            data = json.loads(resp.choices[0].message.content).get("records", [])
                            if data:
                                temp_df = pd.DataFrame(data)
                                temp_df = temp_df.rename(columns={"code":"基金代码", "name":"基金名称", "share":"持有份额", "cost":"成本金额"})
                                new_funds = pd.concat([new_funds, temp_df])
                        except: continue
                
                if not new_funds.empty:
                    old_funds = st.session_state.fund_data
                    combined = pd.concat([old_funds, new_funds], ignore_index=True)
                    final_funds = combined.drop_duplicates(subset=['基金代码'], keep='last')
                    ok, sha = dm_funds.save_data(final_funds, st.session_state.get('fund_sha'))
                    if ok:
                        st.session_state.fund_data = final_funds
                        st.success("持仓更新成功"); st.rerun()

        with c_f2:
            if st.button("🔄 刷新行情", use_container_width=True):
                if st.session_state.fund_data.empty: pass
                else:
                    codes = st.session_state.fund_data['基金代码'].unique()
                    progress = st.progress(0)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        future_to_code = {executor.submit(get_fund_realtime_valuation, code): code for code in codes}
                        for i, future in enumerate(concurrent.futures.as_completed(future_to_code)):
                            code, val, name = future.result()
                            if val > 0: st.session_state.fund_prices[code] = {"price": val, "name": name}
                            progress.progress((i+1)/len(codes))
                    st.rerun()
            
            if st.session_state.fund_data.empty: st.info("暂无持仓")
            else:
                display_data = []
                for _, row in st.session_state.fund_data.iterrows():
                    code = str(row['基金代码'])
                    price = st.session_state.fund_prices.get(code, {}).get('price', 0.0)
                    name = st.session_state.fund_prices.get(code, {}).get('name', row['基金名称'])
                    val = float(row['持有份额']) * price
                    display_data.append({
                        "基金代码": code, "基金名称": name,
                        "持有份额": row['持有份额'], "最新净值": price,
                        "当前市值": val
                    })
                st.data_editor(pd.DataFrame(display_data), use_container_width=True, disabled=["基金名称", "最新净值", "当前市值"])

    # --- 报表 ---
    with t_stats:
        if df_ledger.empty: st.info("暂无数据")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.pie(df_ledger[df_ledger['类型']=='支出'], values='金额', names='分类', hole=0.4), use_container_width=True)
            with col2:
                st.plotly_chart(px.bar(df_ledger, x='日期', y='金额', color='类型'), use_container_width=True)

    # --- AI Copilot (功能突破) ---
    with t_copilot:
        st.markdown("### 💬 向你的财务 AI 提问")
        st.caption("例如：上个月我在哪里花钱最多？统计一下所有的餐饮支出。")
        user_query = st.text_input("你的问题：", key="copilot_query")
        if st.button("🧠 分析", type="secondary"):
            if not st.session_state.api_key: st.error("需要 API Key"); st.stop()
            
            if st.session_state.ledger_data.empty:
                st.warning("没有数据供分析")
            else:
                with st.spinner("AI 正在写代码分析..."):
                    # 创建一段代码环境供 LLM 执行
                    sample = st.session_state.ledger_data.head(5).to_csv(index=False)
                    
                    prompt = f"""
                    你是 Pandas 专家。变量名是 `df`。
                    列名: [{', '.join(st.session_state.ledgerger_data.columns)}].
                    数据样本:
                    {sample}
                    
                    问题：{user_query}
                    
                    请输出 Python 代码。
                    1. 使用 `st.dataframe` 或 `st.metric` 展示结果。
                    2. 忽略无关警告。
                    3. 处理日期时，参考 pandas dt accessor。
                    4. **仅输出代码**。
                    """
                    
                    try:
                        client = get_llm_client(st.session_state.api_key)
                        resp = client.chat.completions.create(
                            model=TEXT_MODEL_NAME,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.1,
                            max_tokens=1024
                        )
                        
                        code_str = resp.choices[0].message.content
                        if "```python" in code_str:
                            code_str = code_str.split("```python")[1].split("```")[0].strip()
                        elif "```" in code_str:
                            code_str = code_str.split("```")[1].split("```")[0].strip()
                        
                        with st.context:
                            exec_globals = {"df": st.session_state.ledger_data, "st": st, "pd": pd}
                            exec(code_str, exec_globals)
                            
                    except Exception as e:
                        st.error(f"AI 分析出错: {e}")

def merge_data_with_overwrite(old_df, new_df):
    if new_df is None or new_df.empty: return old_df, 0
    if old_df.empty: return new_df, len(new_df)
    
    for df in [old_df.copy(), new_df.copy()]:
        df['日期'] = df['日期'].astype(str).str.replace('/', '-')
        df['日期'] = df['日期'].str.split(' ').str[0]
        df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
        df['备注'] = df['备注'].astype(str)
    
    merged_df = pd.concat([old_df, new_df], ignore_index=True)
    def get_fp(d): return d['日期'].astype(str) + "_" + d['金额'].astype(str) + "_" + d['备注'].str[:6]
    merged_df['_fp'] = get_fp(merged_df)
    final_df = merged_df.drop_duplicates(subset=['_fp'], keep='last').drop(columns=['_fp'])
    final_df['日期'] = pd.to_datetime(final_df['日期'], errors='coerce').dt.date
    return final_df.sort_values('日期', ascending=False).reset_index(drop=True), len(new_df)

if __name__ == "__main__":
    main()
