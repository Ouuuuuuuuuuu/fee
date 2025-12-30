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

# --- 页面配置 ---
st.set_page_config(page_title="AI 智能账本 Pro (10号账期版)", page_icon="💰", layout="wide")

# --- 常量配置 ---
GITHUB_API_URL = "https://api.github.com"
VISION_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct" 
TEXT_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
CHUNK_SIZE = 12000 
BILL_CYCLE_DAY = 10  # 账单日：每月10号

# --- 标准分类定义 ---
# 格式： "标准分类": ["关键词1", "关键词2", ...]
CATEGORY_MAPPING = {
    "餐饮美食": ["麦当劳", "肯德基", "饿了么", "美团", "星巴克", "瑞幸", "饭", "面", "吃", "饮", "烧烤", "火锅", "食品", "菜", "酒", "茶", "养生小食坊"],
    "交通出行": ["滴滴", "打车", "地铁", "公交", "交通", "加油", "停车", "铁路", "车", "机票", "一卡通"],
    "购物消费": ["超市", "便利店", "京东", "淘宝", "天猫", "拼多多", "商户消费", "扫二维码付款", "7-11", "全家"],
    "生活服务": ["话费", "电费", "水费", "燃气", "宽带", "理发", "洗", "充值缴费"],
    "娱乐休闲": ["电影", "游戏", "会员", "视频", "KTV", "网吧", "玩", "温泉", "龙悦酒店"],
    "工资收入": ["工资", "薪", "奖金", "补助", "报销", "轧差"],
    "转账红包": ["红包", "转账", "退款"],
    "其他": []  # 兜底
}

# --- 核心工具：OpenAI Client ---
def get_llm_client(api_key):
    return OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

# --- 辅助逻辑：计算账期范围 ---
def get_fiscal_range(current_date, cycle_day=BILL_CYCLE_DAY):
    """
    根据给定的日期和账单日，计算所属的账期范围。
    逻辑：如果今天 >= 10号，则账期是 本月10号 到 下月9号
          如果今天 < 10号，则账期是 上月10号 到 本月9号
    """
    if isinstance(current_date, str):
        current_date = datetime.datetime.strptime(current_date, "%Y-%m-%d").date()
    elif isinstance(current_date, datetime.datetime):
        current_date = current_date.date()

    if current_date.day >= cycle_day:
        start_date = date(current_date.year, current_date.month, cycle_day)
        # 下个月
        if current_date.month == 12:
            end_date = date(current_date.year + 1, 1, cycle_day) - datetime.timedelta(days=1)
        else:
            end_date = date(current_date.year, current_date.month + 1, cycle_day) - datetime.timedelta(days=1)
    else:
        # 上个月
        if current_date.month == 1:
            start_date = date(current_date.year - 1, 12, cycle_day)
        else:
            start_date = date(current_date.year, current_date.month - 1, cycle_day)
        end_date = date(current_date.year, current_date.month, cycle_day) - datetime.timedelta(days=1)
    
    return start_date, end_date

# --- 辅助逻辑：自动分类 ---
def auto_categorize(row):
    """基于备注和原始分类，自动归类到标准分类"""
    # 如果已经是标准分类，直接返回
    if row['分类'] in CATEGORY_MAPPING.keys():
        return row['分类']

    # 组合搜索文本：备注 + 原始分类
    text = f"{str(row['备注'])} {str(row['分类'])}".lower()
    
    # 优先匹配具体关键词
    for category, keywords in CATEGORY_MAPPING.items():
        for kw in keywords:
            if kw.lower() in text:
                return category
    
    # 默认逻辑
    if row['类型'] == '收入':
        return "其他收入"
    
    return "其他" # 无法识别归为其他

# --- 工具函数：JSON 提取与修复 ---
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
        code_block_pattern = r"``" + r"`(?:json)?(.*?)``" + r"`"
        match_code = re.search(code_block_pattern, text, re.DOTALL)
        if match_code: text = match_code.group(1).strip()
        else:
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = text.strip()
        
        text = repair_truncated_json(text)
        match_array = re.search(r'\[.*\]', text, re.DOTALL)
        if match_array: text_to_parse = match_array.group()
        else: text_to_parse = text
            
        result = json.loads(text_to_parse)
        if isinstance(result, (list, dict)):
            return result if isinstance(result, list) else [result], None
    except: pass
    return None, "JSON提取失败"

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
        df = self._clean_df_types(df)
        return df, sha

    def save_data(self, df, sha=None):
        save_df = df.copy()
        if '日期' in save_df.columns:
            save_df['日期'] = save_df['日期'].astype(str)
        if self.use_github:
            success, new_sha = self._save_to_github(save_df, sha)
            return success, new_sha
        else:
            return self._save_to_local(save_df), None

    @staticmethod
    def merge_data(old_df, new_df):
        if new_df is None or new_df.empty: return old_df, 0
        
        # 1. 应用自动分类清洗
        new_df['分类'] = new_df.apply(auto_categorize, axis=1)

        def get_fp(d): return d['日期'].astype(str) + d['金额'].astype(str) + d['备注'].str[:5]
        if old_df.empty: return new_df, len(new_df)
        old_fp = set(get_fp(old_df))
        new_df['_fp'] = get_fp(new_df)
        to_add = new_df[~new_df['_fp'].isin(old_fp)].drop(columns=['_fp'])
        if to_add.empty: return old_df, 0
        merged = pd.concat([old_df, to_add], ignore_index=True)
        merged = DataManager._clean_df_types(merged)
        merged = merged.sort_values('日期', ascending=False).reset_index(drop=True)
        return merged, len(to_add)

    @staticmethod
    def _clean_df_types(df):
        expected_cols = ["日期", "类型", "金额", "备注", "分类"]
        for col in expected_cols:
            if col not in df.columns: df[col] = ""
        df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0.0)
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        df['日期'] = df['日期'].fillna(pd.Timestamp(date.today()))
        df['日期'] = df['日期'].dt.date
        df['类型'] = df['类型'].astype(str).replace('nan', '支出')
        # 如果读取时分类为空或不标准，也可以在这里再洗一次，但一般在merge时做
        df['分类'] = df['分类'].astype(str).replace('nan', '其他')
        df['备注'] = df['备注'].astype(str).replace('nan', '')
        return df

    def _load_from_local(self):
        if os.path.exists(self.filename):
            try: return pd.read_csv(self.filename), None
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
                df = pd.read_csv(StringIO(csv_str))
                return df, content['sha']
            except: return self._create_empty_df(), content['sha']
        return self._create_empty_df(), None

    def _save_to_github(self, df, sha):
        headers = {"Authorization": f"token {self.github_token}", "Accept": "application/vnd.github.v3+json"}
        url = f"{GITHUB_API_URL}/repos/{self.repo}/contents/{self.filename}"
        csv_str = df.to_csv(index=False)
        content_bytes = base64.b64encode(csv_str.encode('utf-8')).decode('utf-8')
        data = {"message": f"Update ledger", "content": content_bytes}
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

    @staticmethod
    def _create_empty_df():
        return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"])

# --- AI 解析器 ---
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
    def _call_llm_for_text(text_chunk, api_key):
        client = get_llm_client(api_key)
        prompt = f"""
        你是一个严谨的财务专家。
        任务：从文本提取交易。
        标准分类：{list(CATEGORY_MAPPING.keys())}。
        要求：
        1. 仅提取含日期、金额的行。
        2. 根据备注或商户名，**必须**将分类映射到上述标准分类之一。
        3. 返回纯JSON数组: [{{"date":"YYYY-MM-DD","type":"支出/收入","amount":数字,"merchant":"备注","category":"标准分类"}}]
        文本：{text_chunk}
        """
        try:
            resp = client.chat.completions.create(
                model=TEXT_MODEL_NAME, messages=[{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.0
            )
            return resp.choices[0].message.content, None
        except Exception as e: return None, str(e)

    @staticmethod
    def identify_and_parse(filename, file_bytes, api_key):
        try:
            content_text = ""
            if filename.endswith('.csv'):
                try: content_text = file_bytes.decode('utf-8')
                except: content_text = file_bytes.decode('gbk', errors='ignore')
            elif filename.endswith(('.xls', '.xlsx')):
                xls = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
                content_text = "\n".join([f"{s}\n{d.to_csv(index=False)}" for s, d in xls.items()])
            elif filename.endswith('.pdf'):
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    content_text = "\n".join([p.get_text() for p in doc])
            
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

    @staticmethod
    def process_image(filename, image_bytes, api_key):
        try:
            b64_img = base64.b64encode(image_bytes).decode('utf-8')
            client = get_llm_client(api_key)
            resp = client.chat.completions.create(
                model=VISION_MODEL_NAME,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": f"提取账单。请归类为：{list(CATEGORY_MAPPING.keys())}。返回JSON数组。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }],
                max_tokens=2048
            )
            data, _ = extract_json_from_text(resp.choices[0].message.content)
            if not data: return None, "无数据", {}
            if isinstance(data, dict): data = [data]
            df = pd.DataFrame(data)
            cols = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
            df = df.rename(columns=cols)
            for c in cols.values(): 
                if c not in df.columns: df[c] = ""
            return df, None, {}
        except Exception as e: return None, str(e), {}

# --- 主程序 ---
def main():
    if 'debug_mode' not in st.session_state: st.session_state.debug_mode = False
    
    st.sidebar.title("⚙️ 设置")
    api_key = st.secrets.get("SILICONFLOW_API_KEY") or st.sidebar.text_input("API Key", type="password")
    gh_token = st.secrets.get("GITHUB_TOKEN")
    gh_repo = st.secrets.get("GITHUB_REPO")
    
    dm = DataManager(gh_token, gh_repo)
    
    if dm.use_github:
        if st.sidebar.button("☁️ 强制同步云端"):
            with st.spinner("同步中..."):
                df, sha = dm.load_data(force_refresh=True)
                st.session_state.ledger_data = df
                st.session_state.github_sha = sha
                st.success("同步完成")
                st.rerun()
    
    if 'ledger_data' not in st.session_state:
        df, sha = dm.load_data()
        st.session_state.ledger_data = df
        st.session_state.github_sha = sha

    # --- 标题与账期选择 ---
    st.title("💰 AI 智能账本 Pro")
    
    # 默认账期：今天所属的账期
    default_start, default_end = get_fiscal_range(date.today())
    
    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        st.caption(f"当前统计周期 (每月{BILL_CYCLE_DAY}号切分)")
        date_range = st.date_input(
            "选择统计时间段",
            value=(default_start, default_end),
            format="YYYY-MM-DD"
        )

    # --- 核心指标计算 ---
    df = st.session_state.ledger_data.copy()
    
    # 指标初始化
    current_income = 0.0
    current_expense = 0.0
    net_asset = 0.0
    
    if not df.empty and len(date_range) == 2:
        df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
        df['dt'] = pd.to_datetime(df['日期'], errors='coerce').dt.date
        
        # 全量净资产 (不受日期筛选影响)
        net_asset = df[df['类型']=='收入']['金额'].sum() - df[df['类型']=='支出']['金额'].sum()
        
        # 筛选当期数据
        start_d, end_d = date_range[0], date_range[1]
        mask_period = (df['dt'] >= start_d) & (df['dt'] <= end_d)
        df_period = df[mask_period]
        
        current_income = df_period[df_period['类型']=='收入']['金额'].sum()
        current_expense = df_period[df_period['类型']=='支出']['金额'].sum()
        
    else:
        df_period = pd.DataFrame()

    # --- 顶部看板 ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 历史总净值", f"¥{net_asset:,.2f}", help="历史所有收入 - 历史所有支出")
    c2.metric("📅 本期支出", f"¥{current_expense:,.2f}", delta=f"-{current_expense/max(1, (date_range[1]-date_range[0]).days):.1f}/天", delta_color="inverse")
    c3.metric("📅 本期收入", f"¥{current_income:,.2f}")
    c4.metric("📊 本期结余", f"¥{current_income - current_expense:,.2f}", delta_color="normal")
    
    st.divider()

    t_import, t_add, t_history, t_stats = st.tabs(["📥 智能导入", "✍️ 手动记账", "📋 历史明细", "📊 可视化报表"])

    with t_import:
        st.info("💡 导入时会自动根据备注关键词（如'麦当劳'->'餐饮美食'）进行标准化归类。")
        files = st.file_uploader("上传文件 (PDF/CSV/Excel/图片)", accept_multiple_files=True)
        if files and st.button("🚀 开始识别", type="primary"):
            if not api_key: st.error("请配置 API Key"); st.stop()
            
            # ... (保持原有的多线程处理逻辑不变，这里简化显示) ...
            # 这里的 identify_and_parse 内部已经调用了 auto_categorize 逻辑（通过 prompt 或者 后处理）
            # 为了保险，我们在 merge_data 时再次应用 auto_categorize
            
            new_df = pd.DataFrame()
            # 模拟处理过程 (复用之前的逻辑)
            tasks_doc, tasks_img = [], []
            for f in files:
                ext = f.name.split('.')[-1].lower()
                f.seek(0); b = f.read()
                if ext in ['png', 'jpg']: tasks_img.append({"name":f.name, "bytes":b})
                else: tasks_doc.append({"name":f.name, "bytes":b})
            
            with st.status("正在AI识别...") as status:
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {}
                    for t in tasks_doc: futures[executor.submit(BillParser.identify_and_parse, t['name'], t['bytes'], api_key)] = t['name']
                    for t in tasks_img: futures[executor.submit(BillParser.process_image, t['name'], t['bytes'], api_key)] = t['name']
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            res, err, _ = future.result()
                            if res is not None:
                                new_df = pd.concat([new_df, res], ignore_index=True)
                        except: pass
                status.update(label="完成", state="complete")

            if not new_df.empty:
                merged, added = DataManager.merge_data(st.session_state.ledger_data, new_df)
                if added > 0:
                    ok, sha = dm.save_data(merged, st.session_state.get('github_sha'))
                    if ok:
                        st.session_state.ledger_data = merged
                        st.session_state.github_sha = sha
                        st.success(f"导入 {added} 条")
                    else: st.error("保存失败")
                else: st.warning("无新数据")

    with t_add:
        with st.form("manual"):
            c1, c2, c3 = st.columns(3)
            d = c1.date_input("日期", date.today())
            t = c2.selectbox("类型", ["支出", "收入"])
            a = c3.number_input("金额", min_value=0.01)
            c4, c5 = st.columns([1,2])
            cat = c4.selectbox("分类", list(CATEGORY_MAPPING.keys()) + ["其他"])
            rem = c5.text_input("备注")
            if st.form_submit_button("保存", width="stretch"):
                row = pd.DataFrame([{"日期":str(d),"类型":t,"金额":a,"分类":cat,"备注":rem}])
                merged, added = DataManager.merge_data(st.session_state.ledger_data, row)
                ok, sha = dm.save_data(merged, st.session_state.get('github_sha'))
                if ok: 
                    st.session_state.ledger_data = merged
                    st.session_state.github_sha = sha
                    st.success("成功")

    with t_history:
        if st.session_state.ledger_data.empty: st.info("无数据")
        else:
            edited = st.data_editor(st.session_state.ledger_data, use_container_width=True, num_rows="dynamic",
                                    column_config={"分类": st.column_config.SelectboxColumn(options=list(CATEGORY_MAPPING.keys()) + ["其他"])})
            if st.button("保存表格"):
                ok, sha = dm.save_data(edited, st.session_state.get('github_sha'))
                if ok:
                    st.session_state.ledger_data = edited
                    st.session_state.github_sha = sha
                    st.success("已更新")

    with t_stats:
        if df_period.empty:
            st.info("本期暂无数据，请调整时间段或导入数据。")
        else:
            df_exp = df_period[df_period['类型'] == '支出']
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("📊 支出结构")
                if not df_exp.empty:
                    fig_pie = px.pie(df_exp, values='金额', names='分类', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.caption("无支出数据")

            with col_chart2:
                st.subheader("📉 每日支出")
                if not df_exp.empty:
                    daily = df_exp.groupby("日期")['金额'].sum().reset_index()
                    fig_bar = px.bar(daily, x='日期', y='金额', color='金额', color_continuous_scale="Blues")
                    fig_bar.update_layout(xaxis_title="", yaxis_title="")
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.caption("无支出数据")

            st.divider()
            st.subheader("📈 资产净值趋势 (全周期)")
            # 净值趋势使用全量数据，因为看净值通常需要看长期的
            if not df.empty:
                df_sorted = df.sort_values('dt')
                df_sorted['net'] = df_sorted.apply(lambda x: x['金额'] if x['类型']=='收入' else -x['金额'], axis=1)
                daily_net = df_sorted.groupby('dt')['net'].sum().reset_index()
                daily_net['asset'] = daily_net['net'].cumsum()
                
                fig_area = px.area(daily_net, x='dt', y='asset', line_shape='spline')
                fig_area.update_layout(xaxis_title="", yaxis_title="净资产", showlegend=False)
                fig_area.update_traces(line_color="#2E86C1", fill_color="rgba(46, 134, 193, 0.2)")
                st.plotly_chart(fig_area, use_container_width=True)

if __name__ == "__main__":
    main()
