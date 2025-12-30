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

# --- 页面配置 ---
st.set_page_config(page_title="AI 智能账本 Pro (净值版)", page_icon="💰", layout="wide")

# --- 常量配置 ---
GITHUB_API_URL = "https://api.github.com"
VISION_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct" 
TEXT_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
CHUNK_SIZE = 12000  # 核心参数：单次喂给 AI 的最大字符数

# --- 核心工具：OpenAI Client ---
def get_llm_client(api_key):
    return OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

# --- 工具函数：增强版 JSON 提取与修复 ---
def repair_truncated_json(json_str):
    """尝试修复因为 Token 耗尽被截断的 JSON 字符串"""
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
    """增强版JSON提取，支持截断修复"""
    if not text: return None, "空响应"
    original_preview = text[:200].replace('\n', '\\n')
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
    except Exception:
        try:
            text_no_comments = re.sub(r'//.*?\n', '\n', text)
            text_no_comments = re.sub(r'/\*.*?\*/', '', text_no_comments, flags=re.DOTALL)
            match_array = re.search(r'\[.*\]', text_no_comments, re.DOTALL)
            if match_array: text_no_comments = match_array.group()
            result = json.loads(text_no_comments)
            return result if isinstance(result, list) else [result], None
        except: pass
    return None, f"JSON提取失败。预览: {original_preview}..."

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
        headers = {
            "Authorization": f"token {_self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
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
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        url = f"{GITHUB_API_URL}/repos/{self.repo}/contents/{self.filename}"
        csv_str = df.to_csv(index=False)
        content_bytes = base64.b64encode(csv_str.encode('utf-8')).decode('utf-8')
        data = {
            "message": f"Update ledger {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "content": content_bytes
        }
        if sha: data["sha"] = sha
        def do_put(payload):
            return requests.put(url, headers=headers, data=json.dumps(payload), timeout=30)
        try:
            resp = do_put(data)
            if resp.status_code in [200, 201]:
                self._fetch_github_content.clear()
                return True, resp.json()['content']['sha']
            elif resp.status_code in [409, 422]:
                self._fetch_github_content.clear()
                latest_content, _ = self._fetch_github_content()
                if latest_content and 'sha' in latest_content:
                    data["sha"] = latest_content['sha']
                    retry_resp = do_put(data)
                    if retry_resp.status_code in [200, 201]:
                        self._fetch_github_content.clear()
                        return True, retry_resp.json()['content']['sha']
            return False, None
        except Exception: return False, None

    @staticmethod
    def _create_empty_df():
        return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"])

# --- AI 解析器 ---
class BillParser:
    @staticmethod
    def chunk_text_by_lines(text, max_chars=CHUNK_SIZE):
        if len(text) <= max_chars: return [text]
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_len = 0
        for line in lines:
            line_len = len(line) + 1
            if current_len + line_len > max_chars:
                if current_chunk: chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_len = line_len
            else:
                current_chunk.append(line)
                current_len += line_len
        if current_chunk: chunks.append("\n".join(current_chunk))
        return chunks

    @staticmethod
    def _call_llm_for_text(text_chunk, api_key, chunk_id=0):
        client = get_llm_client(api_key)
        prompt = f"""
        你是一个严谨的财务数据提取专家。
        任务：从以下文本片段中提取交易记录。这是一个大文件的第 {chunk_id + 1} 部分。
        原则：
        1. 仅提取包含具体日期、金额的交易行。
        2. 如果这部分文本包含表头或无意义数据，请忽略。
        3. 必须返回纯JSON数组，格式：[{{"date":"YYYY-MM-DD","type":"支出/收入","amount":数字,"merchant":"商户/备注","category":"分类"}}]
        
        文本内容：
        {text_chunk}
        """
        try:
            resp = client.chat.completions.create(
                model=TEXT_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.0
            )
            return resp.choices[0].message.content, None
        except Exception as e: return None, str(e)

    @staticmethod
    def identify_and_parse(filename, file_bytes, api_key):
        t_start = time.time()
        debug_log = {"file": filename, "steps": [], "chunks_data": []}
        try:
            t0 = time.time()
            content_text = ""
            file_stream = BytesIO(file_bytes)
            if filename.endswith('.csv'):
                try: content_text = file_bytes.decode('utf-8')
                except:
                    try: content_text = file_bytes.decode('gbk')
                    except: content_text = file_bytes.decode('latin-1', errors='ignore')
            elif filename.endswith(('.xls', '.xlsx')):
                xls = pd.read_excel(file_stream, sheet_name=None)
                parts = []
                for sname, sdf in xls.items(): parts.append(f"Sheet: {sname}\n{sdf.to_csv(index=False)}")
                content_text = "\n".join(parts)
            elif filename.endswith('.pdf'):
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    content_text = "\n".join([p.get_text() for p in doc])
            
            debug_log["steps"].append(f"读取耗时: {time.time()-t0:.4f}s")
            if not content_text.strip(): return None, "内容为空", debug_log

            chunks = BillParser.chunk_text_by_lines(content_text, CHUNK_SIZE)
            all_parsed_data = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_id = {executor.submit(BillParser._call_llm_for_text, chunk, api_key, i): i for i, chunk in enumerate(chunks)}
                for future in concurrent.futures.as_completed(future_to_id):
                    chunk_id = future_to_id[future]
                    raw_json, err = future.result()
                    chunk_log = {"chunk_id": chunk_id, "raw_preview": raw_json[:100]+"..." if raw_json else "None"}
                    if err: chunk_log["error"] = err
                    else:
                        data, parse_err = extract_json_from_text(raw_json)
                        if data:
                            all_parsed_data.extend(data)
                            chunk_log["count"] = len(data)
                        else: chunk_log["parse_error"] = parse_err
                    debug_log["chunks_data"].append(chunk_log)

            if not all_parsed_data: return None, "未提取到数据", debug_log
            df = pd.DataFrame(all_parsed_data)
            cols = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
            df = df.rename(columns=cols)
            for c in cols.values(): 
                if c not in df.columns: df[c] = ""
            df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
            df['日期'] = df['日期'].astype(str).apply(lambda x: x.split(' ')[0])
            df = df.drop_duplicates()
            return df, None, debug_log
        except Exception as e: return None, str(e), debug_log

    @staticmethod
    def process_image(filename, image_bytes, api_key):
        debug_log = {"file": filename, "steps": []}
        try:
            b64_img = base64.b64encode(image_bytes).decode('utf-8')
            client = get_llm_client(api_key)
            resp = client.chat.completions.create(
                model=VISION_MODEL_NAME,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "提取账单明细。返回JSON数组：[{date, type, amount, merchant, category}]"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }],
                max_tokens=2048
            )
            raw_json = resp.choices[0].message.content
            data, parse_err = extract_json_from_text(raw_json)
            if parse_err: return None, f"解析失败: {parse_err}", debug_log
            if isinstance(data, dict): data = [data]
            if not data: return None, "无数据", debug_log
            df = pd.DataFrame(data)
            cols = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
            df = df.rename(columns=cols)
            for c in cols.values():
                if c not in df.columns: df[c] = ""
            return df, None, debug_log
        except Exception as e: return None, str(e), debug_log

# --- 主程序 ---
def main():
    if 'debug_mode' not in st.session_state: st.session_state.debug_mode = False
    
    st.sidebar.title("⚙️ 设置")
    st.session_state.debug_mode = st.sidebar.checkbox("🐞 开启深度调试", value=st.session_state.debug_mode)
    
    api_key = st.secrets.get("SILICONFLOW_API_KEY") or st.sidebar.text_input("API Key", type="password")
    gh_token = st.secrets.get("GITHUB_TOKEN")
    gh_repo = st.secrets.get("GITHUB_REPO")
    
    dm = DataManager(gh_token, gh_repo)
    
    if dm.use_github:
        st.sidebar.success(f"已连接: {dm.repo}")
        if st.sidebar.button("☁️ 强制同步云端"):
            with st.spinner("同步中..."):
                df, sha = dm.load_data(force_refresh=True)
                st.session_state.ledger_data = df
                st.session_state.github_sha = sha
                st.success("同步完成")
                st.rerun()
    else:
        st.sidebar.warning("本地模式")

    # 加载数据
    if 'ledger_data' not in st.session_state:
        df, sha = dm.load_data()
        st.session_state.ledger_data = df
        st.session_state.github_sha = sha

    st.title("💰 AI 智能账本 Pro")
    
    # --- 核心指标计算 ---
    df = st.session_state.ledger_data.copy()
    total_income = 0.0
    total_expense = 0.0
    net_asset = 0.0
    last_7d_expense = 0.0

    if not df.empty:
        # 确保类型安全
        df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
        df['dt'] = pd.to_datetime(df['日期'], errors='coerce')
        
        # 总收支
        total_income = df[df['类型'] == '收入']['金额'].sum()
        total_expense = df[df['类型'] == '支出']['金额'].sum()
        net_asset = total_income - total_expense
        
        # 近7天支出
        seven_days_ago = pd.Timestamp(date.today()) - pd.Timedelta(days=7)
        mask_7d = (df['dt'] >= seven_days_ago) & (df['类型'] == '支出')
        last_7d_expense = df.loc[mask_7d, '金额'].sum()

    # --- 顶部看板 ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 净资产", f"¥{net_asset:,.2f}", help="总收入 - 总支出")
    c2.metric("📉 总支出", f"¥{total_expense:,.2f}")
    c3.metric("📈 总收入", f"¥{total_income:,.2f}")
    c4.metric("🗓️ 近7天支出", f"¥{last_7d_expense:,.2f}")
    
    st.divider()

    t_import, t_add, t_history, t_stats = st.tabs(["📥 智能导入", "✍️ 手动记账", "📋 历史明细", "📊 统计报表"])

    with t_import:
        files = st.file_uploader("支持 PDF/CSV/Excel/图片 (自动分片处理)", accept_multiple_files=True)
        if files and st.button("🚀 批量开始识别", type="primary"):
            if not api_key: st.error("缺少 API Key"); st.stop()
            
            tasks_doc = []; tasks_img = []
            with st.status("预处理文件...") as status:
                for f in files:
                    ext = f.name.split('.')[-1].lower()
                    f.seek(0); bytes_data = f.read()
                    item = {"name": f.name, "bytes": bytes_data}
                    if ext in ['png', 'jpg', 'jpeg']: tasks_img.append(item)
                    else: tasks_doc.append(item)
                status.update(label="准备就绪", state="complete")

            new_df = pd.DataFrame()
            debug_logs = []
            progress = st.progress(0)
            total_tasks = len(tasks_doc) + len(tasks_img)
            completed = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}
                for t in tasks_doc:
                    f = executor.submit(BillParser.identify_and_parse, t['name'], t['bytes'], api_key)
                    futures[f] = t['name']
                for t in tasks_img:
                    f = executor.submit(BillParser.process_image, t['name'], t['bytes'], api_key)
                    futures[f] = t['name']
                
                for future in concurrent.futures.as_completed(futures):
                    fname = futures[future]
                    try:
                        res, err, dbg = future.result()
                        debug_logs.append(dbg)
                        if res is not None and not res.empty:
                            new_df = pd.concat([new_df, res], ignore_index=True)
                            st.toast(f"✅ {fname} 成功")
                        else: st.error(f"❌ {fname}: {err}")
                    except Exception as e: st.error(f"❌ {fname} 异常: {e}")
                    completed += 1
                    progress.progress(completed / total_tasks)

            if st.session_state.debug_mode:
                with st.expander("🔬 深度调试信息", expanded=True): st.json(debug_logs)

            if not new_df.empty:
                merged_df, added = DataManager.merge_data(st.session_state.ledger_data, new_df)
                if added > 0:
                    with st.spinner("保存中..."):
                        ok, new_sha = dm.save_data(merged_df, st.session_state.get('github_sha'))
                        if ok:
                            st.session_state.ledger_data = merged_df
                            st.session_state.github_sha = new_sha
                            st.success(f"🎉 存入 {added} 条记录")
                        else: st.error("保存失败")
                else: st.warning("无新记录")

    with t_add:
        with st.form("manual_add"):
            c1, c2, c3 = st.columns(3)
            d = c1.date_input("日期", date.today())
            t = c2.selectbox("类型", ["支出", "收入"])
            a = c3.number_input("金额", min_value=0.01)
            c4, c5 = st.columns([1, 2])
            cat = c4.selectbox("分类", ["餐饮", "交通", "购物", "居住", "娱乐", "医疗", "工资", "其他"])
            rem = c5.text_input("备注")
            if st.form_submit_button("💾 保存", width="stretch"):
                row = pd.DataFrame([{"日期": str(d), "类型": t, "金额": a, "分类": cat, "备注": rem}])
                merged, added = DataManager.merge_data(st.session_state.ledger_data, row)
                ok, new_sha = dm.save_data(merged, st.session_state.get('github_sha'))
                if ok:
                    st.session_state.ledger_data = merged
                    st.session_state.github_sha = new_sha
                    st.success("保存成功")
                    st.rerun()

    with t_history:
        st.subheader("📋 账单明细")
        if st.session_state.ledger_data.empty: st.info("暂无数据")
        else:
            st.session_state.ledger_data = DataManager._clean_df_types(st.session_state.ledger_data)
            edited_df = st.data_editor(
                st.session_state.ledger_data,
                use_container_width=True,
                num_rows="dynamic",
                key="history_editor",
                column_config={
                    "金额": st.column_config.NumberColumn(format="¥%.2f", required=True),
                    "日期": st.column_config.DateColumn(format="YYYY-MM-DD", required=True),
                    "类型": st.column_config.SelectboxColumn(options=["支出", "收入"], required=True),
                    "分类": st.column_config.SelectboxColumn(options=["餐饮", "交通", "购物", "居住", "娱乐", "医疗", "工资", "其他"])
                }
            )
            if st.button("💾 保存变更"):
                if not edited_df.equals(st.session_state.ledger_data):
                    with st.spinner("同步中..."):
                        ok, new_sha = dm.save_data(edited_df, st.session_state.get('github_sha'))
                        if ok:
                            st.session_state.ledger_data = edited_df
                            st.session_state.github_sha = new_sha
                            st.success("✅ 更新成功")

    with t_stats:
        if st.session_state.ledger_data.empty: st.info("暂无数据")
        else:
            df = st.session_state.ledger_data.copy()
            df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
            df['dt'] = pd.to_datetime(df['日期'], errors='coerce')
            
            # --- 新增：资产净值变化曲线 ---
            st.subheader("📈 资产净值变化趋势")
            df_sorted = df.sort_values('dt')
            # 计算每笔交易的净变动（收入为正，支出为负）
            df_sorted['net_change'] = df_sorted.apply(lambda x: x['金额'] if x['类型'] == '收入' else -x['金额'], axis=1)
            # 按天聚合，防止同一天多笔交易导致曲线锯齿
            daily_net = df_sorted.groupby('dt')['net_change'].sum().reset_index()
            # 计算累计值
            daily_net['cumulative_asset'] = daily_net['net_change'].cumsum()
            
            st.area_chart(daily_net, x='dt', y='cumulative_asset', color="#2E86C1")

            # --- 原有图表 ---
            df_exp = df[df['类型'] == '支出']
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.subheader("📊 支出分类占比")
                if not df_exp.empty:
                    chart_data = df_exp.groupby("分类")['金额'].sum().reset_index()
                    st.bar_chart(chart_data, x="分类", y="金额", color="分类")
            with c_s2:
                st.subheader("📉 每日支出统计")
                if not df_exp.empty:
                    daily_data = df_exp.groupby("日期")['金额'].sum().reset_index()
                    st.line_chart(daily_data, x="日期", y="金额")
            
            st.divider()
            st.subheader("🤖 AI 财务顾问")
            if st.button("生成本月分析"):
                if not api_key: st.error("请配置 API Key")
                else:
                    with st.spinner("AI 分析中..."):
                        summary_csv = df_exp.sort_values('日期', ascending=False).head(100).to_csv(index=False)
                        client = get_llm_client(api_key)
                        try:
                            res = client.chat.completions.create(
                                model=TEXT_MODEL_NAME,
                                messages=[
                                    {"role": "system", "content": "你是一个犀利的理财师。根据支出给出评价和建议。"},
                                    {"role": "user", "content": summary_csv}
                                ],
                                max_tokens=2000
                            )
                            st.markdown(res.choices[0].message.content)
                        except Exception as e: st.error(f"AI 分析失败: {e}")

if __name__ == "__main__":
    main()
