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

# --- 核心工具：OpenAI Client (无缓存，防止线程安全问题) ---
def get_llm_client(api_key):
    return OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

# --- 工具函数：增强版 JSON 提取与修复 ---
def extract_json_from_text(text):
    """增强版JSON提取，支持截断修复、注释清洗"""
    if not text: 
        return None, "空响应"
    
    # 保存原始文本用于调试 (取前200字符预览)
    original_preview = text[:200].replace('\n', '\\n')
    
    try:
        text = text.strip()
        
        # 1. 移除 Markdown 标记
        match_code = re.search(r"``" + r"`(?:json)?(.*?)``" + r"`", text, re.DOTALL)
        if match_code:
            text = match_code.group(1).strip()
        else:
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = text.strip()
        
        # 2. 快速判断空数组
        if text == '[]':
            return [], None

        # 3. 尝试定位数组边界
        # 寻找第一个 [ 和 最后一个 ]
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text_to_parse = text[start_idx:end_idx+1]
        elif start_idx != -1:
            # 只有开头没有结尾，可能是被截断了
            text_to_parse = text[start_idx:]
        else:
            text_to_parse = text

        # 4. 尝试直接解析
        try:
            result = json.loads(text_to_parse)
            if isinstance(result, (list, dict)):
                return result if isinstance(result, list) else [result], None
        except json.JSONDecodeError:
            # 5. 解析失败，尝试修复截断问题
            # 常见情况：结尾少了 ] 或 }
            try:
                # 尝试补全结尾
                fixed_text = text_to_parse.strip()
                if not fixed_text.endswith(']'):
                    if fixed_text.endswith('}'):
                        fixed_text += ']'
                    elif fixed_text.endswith(','):
                        fixed_text = fixed_text[:-1] + '}]' # 假设断在对象间
                    else:
                        # 暴力尝试：找到最后一个 }，截断后补 ]
                        last_brace = fixed_text.rfind('}')
                        if last_brace != -1:
                            fixed_text = fixed_text[:last_brace+1] + ']'
                
                result = json.loads(fixed_text)
                return result if isinstance(result, list) else [result], None
            except:
                pass

            # 6. 尝试移除注释 (//...)
            try:
                text_no_comments = re.sub(r'//.*?\n', '\n', text_to_parse)
                text_no_comments = re.sub(r'/\*.*?\*/', '', text_no_comments, flags=re.DOTALL)
                result = json.loads(text_no_comments)
                return result if isinstance(result, list) else [result], None
            except:
                pass
            
            # 如果还是失败，抛出原始异常以便查看
            return None, f"无法修复的JSON格式。尝试解析片段: {text_to_parse[:100]}..."
            
    except Exception as e:
        return None, f"解析异常: {str(e)}"

# --- 辅助函数：大文本切片 ---
def split_text_into_chunks(text, max_chars=12000):
    """将长文本按行切分为多个片段，避免 LLM 上下文溢出或输出截断"""
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_len = 0
    
    # 保留表头（假设前5行是表头）
    header = "\n".join(lines[:5]) if len(lines) > 5 else ""
    
    for line in lines:
        if current_len + len(line) > max_chars:
            chunk_content = "\n".join(current_chunk)
            # 如果不是第一块，加上表头上下文
            if len(chunks) > 0:
                chunk_content = header + "\n...[接上文]...\n" + chunk_content
            chunks.append(chunk_content)
            current_chunk = []
            current_len = 0
        current_chunk.append(line)
        current_len += len(line) + 1
        
    if current_chunk:
        chunk_content = "\n".join(current_chunk)
        if len(chunks) > 0:
            chunk_content = header + "\n...[接上文]...\n" + chunk_content
        chunks.append(chunk_content)
        
    return chunks

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
            if force_refresh:
                self._fetch_github_content.clear()
            df, sha = self._load_from_github()
        else:
            df, sha = self._load_from_local()
        return self._clean_df_types(df), sha

    def save_data(self, df, sha=None):
        save_df = df.copy()
        if '日期' in save_df.columns:
            save_df['日期'] = save_df['日期'].astype(str)
            
        if self.use_github:
            return self._save_to_github(save_df, sha)
        else:
            return self._save_to_local(save_df), None

    @staticmethod
    def _clean_df_types(df):
        cols = ["日期", "类型", "金额", "备注", "分类"]
        for c in cols:
            if c not in df.columns: df[c] = ""
        
        df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0.0)
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce').dt.date
        df['日期'] = df['日期'].fillna(date.today())
        
        for c in ['类型', '分类', '备注']:
            df[c] = df[c].astype(str).replace('nan', '')
            
        return df

    def _load_from_local(self):
        if os.path.exists(self.filename):
            try:
                return pd.read_csv(self.filename), None
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
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200: return r.json(), None
            elif r.status_code == 404: return None, 404
            else: return None, r.status_code
        except Exception as e: return None, str(e)

    def _load_from_github(self):
        c, e = self._fetch_github_content()
        if c:
            try:
                csv = base64.b64decode(c['content']).decode('utf-8')
                return pd.read_csv(StringIO(csv)), c['sha']
            except: pass
        return self._create_empty_df(), c.get('sha') if c else None

    def _save_to_github(self, df, sha):
        headers = {"Authorization": f"token {self.github_token}", "Accept": "application/vnd.github.v3+json"}
        url = f"{GITHUB_API_URL}/repos/{self.repo}/contents/{self.filename}"
        csv_str = df.to_csv(index=False)
        data = {
            "message": f"Update {datetime.datetime.now()}",
            "content": base64.b64encode(csv_str.encode('utf-8')).decode('utf-8')
        }
        if sha: data["sha"] = sha

        def put(d): return requests.put(url, headers=headers, data=json.dumps(d), timeout=30)

        try:
            r = put(data)
            if r.status_code in [200, 201]:
                self._fetch_github_content.clear()
                return True, r.json()['content']['sha']
            elif r.status_code in [409, 422]: # SHA 冲突修复
                self._fetch_github_content.clear()
                latest, _ = self._fetch_github_content()
                if latest:
                    data["sha"] = latest['sha']
                    r2 = put(data)
                    if r2.status_code in [200, 201]:
                        self._fetch_github_content.clear()
                        return True, r2.json()['content']['sha']
            return False, None
        except: return False, None

    @staticmethod
    def _create_empty_df():
        return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"])

# --- AI 解析器 ---
class BillParser:
    @staticmethod
    def identify_and_parse(filename, file_bytes, api_key):
        """主入口：处理单个文件，支持大文件切片"""
        t_start = time.time()
        debug_log = {"file": filename, "steps": [], "chunks": 0}
        
        try:
            # 1. 提取纯文本
            t0 = time.time()
            text = ""
            file_stream = BytesIO(file_bytes)
            
            if filename.endswith('.csv'):
                try: text = file_bytes.decode('utf-8')
                except: text = file_bytes.decode('gbk', 'ignore')
            elif filename.endswith(('.xls', '.xlsx')):
                xls = pd.read_excel(file_stream, sheet_name=None)
                text = "\n".join([df.to_csv(index=False) for df in xls.values()])
            elif filename.endswith('.pdf'):
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    text = "\n".join([p.get_text() for p in doc])
            
            if not text.strip(): return None, "空文件", debug_log
            debug_log["steps"].append(f"读取耗时: {time.time()-t0:.4f}s")
            
            # 2. 智能切片 (处理超长账单的关键)
            # 如果文本 > 15000 字符，大概率超过 4k output token，需要切片
            chunks = split_text_into_chunks(text, max_chars=15000)
            debug_log["chunks"] = len(chunks)
            
            # 3. 并发处理所有切片
            all_df = pd.DataFrame()
            
            # 使用内部函数处理单个切片
            def process_chunk(chunk_idx, chunk_text):
                t_c = time.time()
                prompt = f"""
                你是一个严谨的财务专家。请从文本中提取交易记录。
                当前是第 {chunk_idx+1} 部分文本。
                原则：宁缺毋假，禁止捏造。只提取有效交易行。
                
                当前年份参考：{datetime.datetime.now().year}
                
                **强制要求**：仅返回纯JSON数组。
                格式：[{{"date":"YYYY-MM-DD","type":"支出/收入","amount":数字,"merchant":"商户/备注","category":"分类"}}]
                
                文本内容：
                {chunk_text}
                """
                
                try:
                    client = get_llm_client(api_key)
                    resp = client.chat.completions.create(
                        model=TEXT_MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=8192, # 尽可能大
                        temperature=0.0
                    )
                    raw = resp.choices[0].message.content
                    data, err = extract_json_from_text(raw)
                    return data, err, raw, time.time()-t_c
                except Exception as e:
                    return None, str(e), "", 0

            # 执行并发
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(process_chunk, i, c): i for i, c in enumerate(chunks)}
                
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    data, err, raw, cost = future.result()
                    
                    # 记录首个切片的调试信息，避免日志爆炸
                    if i == 0:
                        debug_log["ai_response_sample"] = raw[:500] + "..."
                        if err: debug_log["first_chunk_error"] = err
                    
                    if data:
                        all_df = pd.concat([all_df, pd.DataFrame(data)], ignore_index=True)
            
            if all_df.empty:
                return None, "未提取到任何数据 (可能格式不支持或Token超限)", debug_log
                
            # 4. 统一清洗
            cols = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
            all_df = all_df.rename(columns=cols)
            for c in cols.values(): 
                if c not in all_df.columns: all_df[c] = ""
            
            all_df['金额'] = pd.to_numeric(all_df['金额'], errors='coerce').fillna(0)
            
            debug_log["total_time"] = time.time() - t_start
            return all_df, None, debug_log

        except Exception as e:
            return None, str(e), debug_log

    @staticmethod
    def process_image(filename, image_bytes, api_key):
        t_start = time.time()
        debug_log = {"file": filename}
        try:
            b64 = base64.b64encode(image_bytes).decode('utf-8')
            client = get_llm_client(api_key)
            resp = client.chat.completions.create(
                model=VISION_MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "提取账单。返回纯JSON数组：[{date, type, amount, merchant, category}]"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
                max_tokens=4096
            )
            raw = resp.choices[0].message.content
            debug_log["ai_response"] = raw
            data, err = extract_json_from_text(raw)
            
            if err: return None, err, debug_log
            if not data: return None, "无数据", debug_log
            
            df = pd.DataFrame(data if isinstance(data, list) else [data])
            cols = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
            df = df.rename(columns=cols)
            for c in cols.values(): 
                if c not in df.columns: df[c] = ""
            return df, None, debug_log
        except Exception as e:
            return None, str(e), debug_log

    @staticmethod
    def merge_data(old_df, new_df):
        if new_df is None or new_df.empty: return old_df, 0
        def fp(d): return d['日期'].astype(str) + d['金额'].astype(str) + d['备注'].str[:6]
        if old_df.empty: return new_df, len(new_df)
        
        new_df = DataManager._clean_df_types(new_df) # 确保类型一致
        old_fp = set(fp(old_df))
        new_df['_fp'] = fp(new_df)
        to_add = new_df[~new_df['_fp'].isin(old_fp)].drop(columns=['_fp'])
        
        if to_add.empty: return old_df, 0
        merged = pd.concat([old_df, to_add], ignore_index=True)
        merged = merged.sort_values('日期', ascending=False).reset_index(drop=True)
        return merged, len(to_add)

# --- Main ---
def main():
    if 'debug_mode' not in st.session_state: st.session_state.debug_mode = False
    
    st.sidebar.title("⚙️ 设置")
    st.session_state.debug_mode = st.sidebar.checkbox("🐞 调试模式", value=st.session_state.debug_mode)
    api_key = st.secrets.get("SILICONFLOW_API_KEY") or st.sidebar.text_input("API Key", type="password")
    
    dm = DataManager(st.secrets.get("GITHUB_TOKEN"), st.secrets.get("GITHUB_REPO"))
    
    if dm.use_github:
        st.sidebar.success(f"云端: {dm.repo}")
        if st.sidebar.button("☁️ 强制同步"):
            df, sha = dm.load_data(True)
            st.session_state.ledger_data = df
            st.session_state.github_sha = sha
            st.rerun()
    
    payday = st.sidebar.number_input("发薪日", 1, 31, 10)
    assets = st.sidebar.number_input("资产", value=3000.0)

    if 'ledger_data' not in st.session_state:
        df, sha = dm.load_data()
        st.session_state.ledger_data = df
        st.session_state.github_sha = sha

    st.title("💰 AI 智能账本 Pro")
    
    # 概览逻辑
    today = date.today()
    df = st.session_state.ledger_data.copy()
    m_spend = 0.0
    if not df.empty:
        df['dt'] = pd.to_datetime(df['日期'], errors='coerce')
        mask = (df['dt'].dt.month == today.month) & (df['dt'].dt.year == today.year) & (df['类型'] == '支出')
        m_spend = df.loc[mask, '金额'].sum()
        
    next_pay = date(today.year, today.month, payday)
    if today.day >= payday:
        next_pay = (next_pay + pd.DateOffset(months=1)).date()
    days_left = (next_pay - today).days
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("资产", f"¥{assets:,.0f}")
    c2.metric("本月支出", f"¥{m_spend:,.0f}")
    c3.metric("距发薪", f"{days_left}天")
    budget = (assets / max(1, days_left))
    c4.metric("日均可用", f"¥{budget:.0f}", delta=f"{budget-DEFAULT_TARGET_SPEND:.0f}")

    st.divider()
    
    t_imp, t_add, t_his, t_stat = st.tabs(["📥 导入", "✍️ 记账", "📋 明细", "📊 统计"])
    
    with t_imp:
        files = st.file_uploader("传文件 (PDF/图片/Excel)", accept_multiple_files=True)
        if files and st.button("🚀 开始解析", type="primary"):
            if not api_key: st.stop()
            
            tasks_doc, tasks_img = [], []
            for f in files:
                f.seek(0)
                b = f.read()
                ext = f.name.split('.')[-1].lower()
                if ext in ['png','jpg','jpeg']: tasks_img.append((f.name, b))
                else: tasks_doc.append((f.name, b))
            
            new_df = pd.DataFrame()
            dbg_logs = []
            
            prog = st.progress(0)
            tot = len(tasks_doc) + len(tasks_img)
            done = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exc:
                fs = {}
                for n, b in tasks_doc: fs[exc.submit(BillParser.identify_and_parse, n, b, api_key)] = n
                for n, b in tasks_img: fs[exc.submit(BillParser.process_image, n, b, api_key)] = n
                
                for f in concurrent.futures.as_completed(fs):
                    name = fs[f]
                    try:
                        res, err, dbg = f.result()
                        dbg_logs.append(dbg)
                        if res is not None:
                            new_df = pd.concat([new_df, res], ignore_index=True)
                            st.toast(f"✅ {name} 完成")
                        else:
                            st.error(f"❌ {name}: {err}")
                    except Exception as e: st.error(str(e))
                    done += 1
                    prog.progress(done/tot)
            
            if st.session_state.debug_mode:
                st.json(dbg_logs)
                
            if not new_df.empty:
                m_df, cnt = DataManager.merge_data(st.session_state.ledger_data, new_df)
                if cnt > 0:
                    ok, sha = dm.save_data(m_df, st.session_state.get('github_sha'))
                    if ok:
                        st.session_state.ledger_data = m_df
                        st.session_state.github_sha = sha
                        st.balloons()
                        st.success(f"导入 {cnt} 条")
                else: st.warning("无新数据")

    with t_add:
        with st.form("add"):
            c1, c2, c3 = st.columns(3)
            d = c1.date_input("日期")
            t = c2.selectbox("类型", ["支出", "收入"])
            a = c3.number_input("金额", min_value=0.01)
            cat = st.selectbox("分类", ["餐饮", "交通", "购物", "居住", "娱乐", "医疗", "工资", "其他"])
            rem = st.text_input("备注")
            if st.form_submit_button("保存", use_container_width=True):
                r = pd.DataFrame([{"日期": str(d), "类型": t, "金额": a, "分类": cat, "备注": rem}])
                m_df, cnt = DataManager.merge_data(st.session_state.ledger_data, r)
                ok, sha = dm.save_data(m_df, st.session_state.get('github_sha'))
                if ok:
                    st.session_state.ledger_data = m_df
                    st.session_state.github_sha = sha
                    st.success("已保存")
                    st.rerun()

    with t_his:
        st.session_state.ledger_data = DataManager._clean_df_types(st.session_state.ledger_data)
        edf = st.data_editor(st.session_state.ledger_data, use_container_width=True, num_rows="dynamic",
                             column_config={"金额": st.column_config.NumberColumn(format="¥%.2f"),
                                            "日期": st.column_config.DateColumn(format="YYYY-MM-DD"),
                                            "类型": st.column_config.SelectboxColumn(options=["支出", "收入"])})
        if st.button("同步修改"):
            if not edf.equals(st.session_state.ledger_data):
                ok, sha = dm.save_data(edf, st.session_state.get('github_sha'))
                if ok:
                    st.session_state.ledger_data = edf
                    st.session_state.github_sha = sha
                    st.success("已同步")

    with t_stat:
        if not df.empty:
            df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
            exp = df[df['类型']=='支出']
            c1, c2 = st.columns(2)
            with c1:
                if not exp.empty: st.bar_chart(exp.groupby("分类")['金额'].sum())
            with c2:
                if not exp.empty: st.line_chart(exp.groupby("日期")['金额'].sum())
            
            if st.button("生成AI月报") and api_key:
                with st.spinner("AI 分析中..."):
                    csv = exp.sort_values('日期', ascending=False).head(100).to_csv(index=False)
                    try:
                        client = get_llm_client(api_key)
                        r = client.chat.completions.create(model=TEXT_MODEL_NAME, messages=[
                            {"role":"system","content":"简辣点评消费习惯，给出省钱建议。"},
                            {"role":"user","content":csv}], max_tokens=1000)
                        st.markdown(r.choices[0].message.content)
                    except Exception as e: st.error(str(e))

if __name__ == "__main__":
    main()
