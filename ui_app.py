import streamlit as st
import sys
import os
import glob # ファイルリスト取得用
import random # ランダム選択用
import streamlit.components.v1 as components # HTML埋め込み用
import json # クリップボードコピー用

# st.set_page_config() はStreamlitコマンドの最初に配置する必要があります。
st.set_page_config(page_title="カスタマーサポートAIデモ")

# app.pyからデータをロードする関数とエージェント作成関数をインポートします
# プロジェクトのディレクトリ構造に合わせてimportパスを調整してください。
try:
    from app import load_faq_data_from_py, create_agent_app
except ImportError as e:
    st.error(f"エラー: app.pyから必要な関数をインポートできませんでした。ファイルパスと関数名を確認してください。詳細: {e}")
    st.stop() # インポートに失敗した場合は処理を停止

# Langchainのメッセージタイプをインポート
# ★★★ この行が正しく実行される必要があります ★★★
from langchain_core.messages import HumanMessage, AIMessage

# クリップボードにコピーするJavaScriptコードを埋め込む関数
def copy_to_clipboard_js(text_to_copy):
    """クリップボードにテキストをコピーするJavaScriptを実行するHTMLを埋め込みます。"""
    js_code = f"""
    <script>
    navigator.clipboard.writeText({json.dumps(text_to_copy)}).then(function() {{
        console.log('クリップボードにコピーしました');
        // 必要であれば、ここでStreamlitにイベントを通知することも可能ですが、今回は単にコピーするだけです。
        // Streamlit.setComponentValue({{copied: true, text: "{text_to_copy}"}});
    }}, function(err) {{
        console.error('クリップボードへのコピーに失敗しました: ', err);
        alert('クリップボードへのコピーに失敗しました。手動でコピーしてください。');
    }});
    </script>
    """
    components.html(js_code, height=0, width=0) # UIには表示しない

# ドキュメント選択のUIを追加
doc_dir = "doc"
# Streamlitスクリプトの場所からの相対パスでdocディレクトリ内の.pyファイルをリストアップ
script_dir = os.path.dirname(__file__)
doc_abs_dir = os.path.join(script_dir, doc_dir)

available_docs = glob.glob(os.path.join(doc_abs_dir, "*.py"))
available_doc_names = [os.path.basename(doc) for doc in available_docs]

if not available_doc_names:
    st.warning(f"'{doc_dir}' ディレクトリに利用可能なドキュメントファイルが見つかりません（.pyファイル）。FAQデータがロードできません。")
    st.stop() # ファイルがない場合はアプリを停止

# Streamlitのセッションステートで選択されたドキュメント名を管理
if "selected_doc_name" not in st.session_state:
    st.session_state.selected_doc_name = available_doc_names[0] # デフォルトで最初に見つかったファイルを選択

# ドキュメント選択用のセレクトボックス
selected_doc_name = st.selectbox(
    "利用するドキュメントを選択してください:",
    available_doc_names,
    index=available_doc_names.index(st.session_state.selected_doc_name),
    key="doc_selector" # セッションを跨いで状態を維持するためのキー
)

# 選択が変更されたかチェック
if selected_doc_name != st.session_state.selected_doc_name:
    st.session_state.selected_doc_name = selected_doc_name
    # ドキュメントが変更されたらチャット履歴とコンパイル済みアプリをクリア
    st.session_state.messages = []
    if 'langgraph_app' in st.session_state:
         del st.session_state.langgraph_app # 古いアプリインスタンスを削除
    # ドキュメント変更時にチャット入力欄の初期値をクリア
    st.session_state.chat_input_key_counter = 0 # 新しい chat_input のキーカウンターをリセット
    st.rerun() # 変更を適用し、チャットとアプリをクリアして再実行


# 選択されたドキュメントのデータをロード
selected_doc_path = os.path.join(doc_abs_dir, st.session_state.selected_doc_name)
loaded_data_dict = load_faq_data_from_py(selected_doc_path)

# ドキュメントが正常にロードされたか確認し、エージェントアプリを作成
langgraph_app = None
agent_identity = "AIアシスタント" # デフォルトのアイデンティティ
qa_data = [] # FAQデータリストを初期化

if loaded_data_dict:
    qa_data = loaded_data_dict.get('data', [])
    # データアイテムに 'カテゴリー' キーが存在する場合のみ抽出
    categories = list(set(item.get('カテゴリー') for item in qa_data if item.get('カテゴリー')))
    # メタデータからアイデンティティを取得、なければデフォルトを使用
    agent_identity = loaded_data_dict.get('metadata', {}).get('description', 'AIアシスタント')

    # --- プロンプトのロード (任意：選択可能にする場合は別途UI追加) ---
    # 現時点では、docディレクトリ内のpromptsサブディレクトリにある default.txt をデフォルトプロンプトとする
    prompt_dir = os.path.join(doc_abs_dir, "prompts") # docディレクトリ内のpromptsディレクトリを想定
    default_prompt_path = os.path.join(prompt_dir, "default.txt") # 例：デフォルトプロンプトファイル

    system_prompt = f"あなたは{agent_identity}です。" # エージェントのアイデンティティを反映したデフォルトプロンプト
    try:
        if os.path.exists(default_prompt_path):
            with open(default_prompt_path, 'r', encoding='utf-8') as f:
                # ファイルの内容を読み込み、デフォルトプロンプトとして使用
                system_prompt = f.read().strip().replace("{agent_identity}", agent_identity) # ファイル内で {agent_identity} を置き換える
                # print(f"Loaded system prompt from {default_prompt_path}") # デバッグ出力
        # else:
            # print(f"Default system prompt file not found at {default_prompt_path}. Using generated default prompt.") # デバッグ出力
    except Exception as e:
        print(f"エラー: システムプロンプトファイルの読み込み中にエラーが発生しました: {e}。生成されたデフォルトプロンプトを使用します。")
        # エラー発生時も生成したデフォルトプロンプトを使用
        system_prompt = f"あなたは{agent_identity}です。"
    # ----------------------------------------------------------

    # StreamlitのセッションステートにLangGraphアプリがまだない場合のみ作成・コンパイル
    # これにより、新しいチャットメッセージが送信される度に再コンパイルされるのを防ぐ
    if 'langgraph_app' not in st.session_state:
         try:
            st.session_state.langgraph_app = create_agent_app(qa_data, categories, agent_identity, system_prompt)
            # st.success(f"ドキュメント '{st.session_state.selected_doc_name}' に基づいてAIエージェントを準備しました。")
         except Exception as e:
            st.error(f"AIエージェントの作成中にエラーが発生しました: {e}")
            st.session_state.langgraph_app = None # エージェント作成失敗

    # セッションステートからコンパイル済みのアプリインスタンスを取得
    langgraph_app = st.session_state.get('langgraph_app')

else:
     st.error(f"選択されたドキュメント '{st.session_state.selected_doc_name}' の読み込みに失敗しました。ファイル形式（'_JSON'で終わる辞書変数を含む.pyファイルで、'data'/'metadata'キーがあること）を確認してください。")
     st.session_state.langgraph_app = None # データロード失敗時はアプリインスタンスをNoneに
     langgraph_app = None # ローカル変数もNoneに


# サイドバーにドキュメント情報を表示
with st.sidebar:
    st.subheader("📚 ドキュメント情報")
    st.markdown(f"**アイデンティティ**: {agent_identity}")

    # カテゴリー一覧を表示
    if categories:
        st.markdown("**利用可能なカテゴリー**:")
        for category in sorted(categories):
            st.markdown(f"- {category}")

    # システムプロンプトの表示（折りたたみ可能）
    with st.expander("システムプロンプト設定", expanded=False):
        st.markdown("```")
        st.markdown(system_prompt)
        st.markdown("```")

    st.markdown("---")  # 区切り線


# チャット履歴を初期化
if "messages" not in st.session_state:
    # 選択されたドキュメント/アイデンティティに基づいた初期挨拶
    initial_greeting = f"こんにちは！{agent_identity}です。どのようなご用件でしょうか？"
    # ★★★ ここで AIMessage が使われています ★★★
    st.session_state.messages = [AIMessage(content=initial_greeting)]

# セッションステートにチャット入力のキーカウンターを初期化
# これを使って st.chat_input をリセットする
if "chat_input_key_counter" not in st.session_state:
    st.session_state.chat_input_key_counter = 0

# --- 質問テンプレートの表示（カテゴリー非表示、各カテゴリーからランダム1件を縦並びで） ---
if qa_data: # データが正常にロードされた場合のみ表示
    st.subheader("💡 よくある質問")

    # カテゴリーごとに質問をグループ化
    questions_by_category = {}
    for item in qa_data: # qa_data はすでにリスト形式
        category = item.get("カテゴリー", "その他") # カテゴリーがない場合は「その他」に分類
        question = item.get("質問")
        if question: # 質問内容が空でない場合のみ追加
             if category not in questions_by_category:
                 questions_by_category[category] = []
             questions_by_category[category].append(question)

    # 各カテゴリーからランダムに1件選択し、ボタンを表示
    if questions_by_category:
        for category, questions in questions_by_category.items():
            if questions:
                random_question = random.choice(questions)
                button_key = f"copy_button_{category}_{random_question[:10]}" # ユニークなキーを生成

                # JavaScriptに渡すために質問テキスト内の引用符をエスケープ
                # f-stringのエラーを避けるため、エスケープ処理は外で行う
                escaped_question = random_question.replace("'", "\\'").replace('"', '\\"')

                # JavaScriptを使用してクリップボードにコピーするボタン
                # 幅をフルにするためにdivで囲む
                copy_button_html = f"""
                <div style="width: 100%; margin-bottom: 5px;">
                    <button onclick="navigator.clipboard.writeText('{escaped_question}').then(function() {{
                        /* クリップボードへのコピー成功 */
                        console.log('Async: Copying to clipboard was successful!');
                    }}, function(err) {{
                        /* クリップボードへのコピー失敗 */
                        console.error('Async: Could not copy text: ', err);
                    }});" style="width: 100%; text-align: left; padding: 10px;">{random_question}</button>
                </div>
                """
                st.components.v1.html(copy_button_html, height=45) # heightを少し調整
    else:
        st.info("表示できる質問テンプレートがありません。")
# ----------------------------------------------------------------------


# チャット履歴を表示
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
         with st.chat_message("assistant"):
            st.markdown(message.content)
    # ToolMessageはUI上には表示しない想定

# ユーザー入力を受け付け
# エージェントアプリが正常に作成された場合のみ入力を有効化
# key を変更することで入力欄をリセットできる (value 引数の代わり)
prompt = st.chat_input(
    "質問を入力してください",
    key=f"chat_input_{st.session_state.chat_input_key_counter}", # キーをカウンターで変更
    disabled=langgraph_app is None
)

# ユーザー入力（手動またはボタン）があった場合のみ処理を実行
# ボタンクリック時は prompt は None なので、手動入力があった場合のみここで処理
if prompt:
    user_input = prompt
    # chat_input_key_counter をインクリメントして入力欄をリセット
    st.session_state.chat_input_key_counter += 1

    # ユーザーメッセージをチャットコンテナに表示
    with st.chat_message("user"):
        st.markdown(user_input)

    # ユーザーメッセージを履歴に追加
    st.session_state.messages.append(HumanMessage(content=user_input))

    # LangGraphアプリへの入力形式を準備
    inputs = {"messages": [HumanMessage(content=user_input)]}

    try:
        # セッションステートから取得したコンパイル済みのアプリインスタンスを使用
        final_state = langgraph_app.invoke(inputs)

        # 最終的なAIからのメッセージを状態から抽出
        # final_state['messages'] の最後の要素が最終応答と想定
        ai_message = final_state.get('messages', [])[-1] if final_state.get('messages') else AIMessage(content="申し訳ございません、回答を生成できませんでした。")

        # アシスタントの応答をチャットコンテナに表示
        with st.chat_message("assistant"):
            st.markdown(ai_message.content)

        # アシスタントの応答を履歴に追加
        st.session_state.messages.append(ai_message)

    except Exception as e:
        st.error(f"リクエスト処理中にエラーが発生しました: {e}")
        # エラーメッセージをチャット履歴に追加することも考慮
        st.session_state.messages.append(AIMessage(content="申し訳ございません、処理中にエラーが発生しました。時間をおいて再度お試しください。"))

    # 応答生成後、Streamlitを再実行してUIを更新し、chat_input をリセットする
    st.rerun()
