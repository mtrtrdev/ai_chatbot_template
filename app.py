import os
import sys
import importlib.util
import importlib.machinery
import json
import pandas as pd # Keep pandas import for potential future use or context
from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv(verbose=True)

print("[DEBUG] 必要なモジュールをインポートしました")

# 環境変数にAPIキーを設定 (リポジトリに直接記述しない)
# ローカルでの開発時には、ターミナルで環境変数を設定するか、.env ファイル + python-dotenv を使用してください。
# 例: export GOOGLE_API_KEY='YOUR_API_KEY_HERE' (Linux/macOS) または $env:GOOGLE_API_KEY='YOUR_API_KEY_HERE' (PowerShell)
# Streamlit Cloudなどのデプロイ環境では、各サービスのシークレット管理機能を利用してください。
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    print("[ERROR] 環境変数 GOOGLE_API_KEY が設定されていません。有効なAPIキーを設定してください。")
    # 環境変数が設定されていない場合、アプリケーションが正常に動作しない可能性が高いです。
    # ここでsys.exitなどでプログラムを終了させることも検討できますが、
    # Streamlitの実行モデルに合わせ、ここではエラーメッセージ出力に留めます。
    # LLMインスタンスの初期化時にエラーが発生する可能性が高いです。
else:
    # LangChainが参照するためにos.environに設定
    os.environ["GOOGLE_API_KEY"] = google_api_key
    print("[DEBUG] 環境変数GOOGLE_API_KEYを読み込みました")

# PythonファイルからFAQデータ辞書を読み込む関数
def load_faq_data_from_py(file_path: str) -> dict | None:
    """PythonファイルからFAQデータ辞書をロードします。
    ファイルは辞書形式の変数（'_JSON'で終わる名前を想定）を含んでいる必要があります。
    """
    print(f"[DEBUG] load_faq_data_from_pyが呼び出されました: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"[DEBUG] エラー: ドキュメントファイルが見つかりません: {file_path}")
        return None

    try:
        # ファイルのディレクトリを取得し、sys.pathに一時的に追加してインポート可能にする
        file_dir = os.path.dirname(file_path)
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"[DEBUG] モジュール名: {module_name}")

        # sys.pathに一時的に追加
        sys.path.insert(0, file_dir)
        print(f"[DEBUG] sys.pathに追加: {file_dir}")

        # モジュールをインポート
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            print(f"[DEBUG] エラー: Specを作成できませんでした: {file_path}")
            sys.path.pop(0) # パスを削除
            return None
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        print(f"[DEBUG] モジュールをインポートしました: {module_name}")

        # '_JSON'で終わる名前の辞書形式の変数を探す
        data_variable = None
        for var_name in dir(module):
            if var_name.endswith('_JSON') and isinstance(getattr(module, var_name), dict):
                data_variable = getattr(module, var_name)
                print(f"[DEBUG] データ変数を見つけました: {var_name}")
                break

        # 一時的に追加したパスをsys.pathから削除
        sys.path.pop(0)
        print("[DEBUG] sys.pathから一時パスを削除しました")

        if data_variable and 'data' in data_variable and 'metadata' in data_variable:
            print(f"[DEBUG] ドキュメント '{file_path}' を正常に読み込みました")
            return data_variable
        else:
            print(f"[DEBUG] エラー: ドキュメントファイル '{file_path}' から必要なデータが見つかりません")
            return None

    except Exception as e:
        print(f"[DEBUG] エラー: ドキュメント '{file_path}' の読み込み中に予期せぬエラーが発生しました: {e}")
        return None

# エージェントの状態を定義
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    predicted_category: str

print("[DEBUG] AgentStateクラスを定義しました")

# LLMインスタンス
# GOOGLE_API_KEY環境変数が設定されていない場合、ここでエラーが発生する可能性があります。
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
relevance_scorer_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
print("[DEBUG] LLMインスタンスを初期化しました")

# LangGraphエージェントアプリを作成・コンパイルする関数
def create_agent_app(qa_data: List[dict], categories: List[str], agent_identity: str, system_prompt: str) -> StateGraph:
    """指定されたQAデータ、カテゴリー、アイデンティティ、システムプロンプトでLangGraphエージェントアプリを作成・コンパイルします。"""
    print("[DEBUG] create_agent_appが呼び出されました")
    print(f"[DEBUG] パラメータ: qa_data長={len(qa_data)}, categories={categories}, agent_identity={agent_identity}")

    if not qa_data or not categories:
        print("[DEBUG] 警告: QAデータまたはカテゴリーが空です")
        # データがない場合の代替ツール定義
        @tool
        def search_qa_by_category(query: str, category: str) -> str:
             """データが利用できないため検索できません。"""
             print("[DEBUG] 空のデータセットに対する検索が試みられました")
             return "申し訳ございません、現在参照できるFAQデータがありません。"
        tools = [search_qa_by_category]
    else:
        # search_qa_by_category ツールをこの関数内で定義
        @tool
        def search_qa_by_category(query: str, category: str) -> str:
            """指定されたカテゴリー内で、ユーザーの質問に関連する回答を検索します。"""
            print(f"[DEBUG] search_qa_by_categoryが呼び出されました: query='{query}', category='{category}'")
            # 検索対象データ長とカテゴリーをログ出力
            print(f"[DEBUG] 検索対象データ長: {len(qa_data) if qa_data else 0}")
            print(f"[DEBUG] 検索対象カテゴリー: {category}")

            best_match_answer = "申し訳ございません、お探しの情報は見つかりませんでした。"
            max_relevance_score = -1

            # カテゴリーでデータをフィルタリング
            filtered_qa_data = [item for item in qa_data if item.get('カテゴリー') == category]
            # フィルタリング後のデータ数をログ出力
            print(f"[DEBUG] フィルタリング後のQAデータ数（カテゴリー'{category}'）: {len(filtered_qa_data)}")

            if not filtered_qa_data:
                print(f"[DEBUG] カテゴリー '{category}' にデータがありません")
                return f"申し訳ございません、指定されたカテゴリー「{category}」には関連情報がありませんでした。"

            # 各QAペアをLLMに評価させるための形式に変換
            qa_for_llm_evaluation = []
            for idx, item in enumerate(filtered_qa_data):
                question = item.get('質問', '')
                if question:
                    qa_for_llm_evaluation.append(f"QA_PAIR_{idx+1}: 質問: {question}")

            if not qa_for_llm_evaluation:
                print("[DEBUG] 有効な質問データがありません")
                return f"指定されたカテゴリー「{category}」には有効な質問データがありませんでした。"

            qa_block = "\n".join(qa_for_llm_evaluation)
            # 評価用QAブロックの先頭部分をログ出力
            print(f"[DEBUG] 評価用QAブロック（最初の500文字）:\n{qa_block[:500]}{'...' if len(qa_block) > 500 else ''}")

            # LLMに一括で関連度を評価させるプロンプト
            evaluation_prompt = f"""
ユーザーの質問: "{query}"

以下の社内ドキュメントの質問リストについて、それぞれの質問がユーザーの質問と意味的にどの程度関連しているかを評価してください。
関連度を0から100の整数で評価し、最も関連性の高いQAペアのインデックス（QA_PAIR_NのN）と、その関連度スコアを特定してください。

評価結果は、JSON形式で出力してください。
JSONの形式は以下の通りです。
{{
    "evaluations": [
        {{ "index": N, "score": S }},
        ...
    ],
    "most_relevant_index": N_most_relevant,
    "max_score": S_max
}}
- "index": QA_PAIR_NのNです。
- "score": 関連度スコア（0-100の整数）です。
- "most_relevant_index": 関連度が最も高いQAペアのNです。
- "max_score": 最も高い関連度スコアです。

---
社内ドキュメントの質問リスト（カテゴリー: {category}）:
{qa_block}
---
評価結果:
"""
            # 評価用プロンプトは長い場合があるのでログ出力はコメントアウト
            # print(f"[DEBUG] 評価用プロンプト（最初の500文字）:\n{evaluation_prompt[:500]}{'...' if len(evaluation_prompt) > 500 else ''}")

            try:
                # LLMによる評価を実行
                evaluation_response = relevance_scorer_llm.invoke(evaluation_prompt).content.strip()
                # LLMからの生の応答をログ出力
                print(f"[DEBUG] LLMからの生の評価応答:\n{evaluation_response}")

                # LLMの応答をJSONとしてパース
                # 応答に不要なマークダウンが含まれている場合を考慮してクリーンアップを試みる
                response_text_cleaned = evaluation_response.replace("```json", "").replace("```", "").strip()
                print(f"[DEBUG] クリーンアップ後の評価応答:\n{response_text_cleaned}")

                # クリーンアップしたテキストに対して json.loads を試みる
                parsed_result = json.loads(response_text_cleaned)

                # パース結果からインデックスとスコアを取得
                most_relevant_index = parsed_result.get("most_relevant_index")
                max_relevance_score = parsed_result.get("max_score", -1)

                # パース結果をログ出力
                print(f"[DEBUG] LLM評価結果のパース後: most_relevant_index={most_relevant_index}, max_score={max_relevance_score}")

                RELEVANCE_THRESHOLD = 70 # 関連度閾値

                # 最も関連性の高いQAペアのインデックスが有効かつ閾値以上のスコアの場合
                if most_relevant_index is not None and 1 <= most_relevant_index <= len(filtered_qa_data):
                    if max_relevance_score >= RELEVANCE_THRESHOLD:
                         # '回答例' キーが存在することを確認して回答を取得
                         best_match_answer = filtered_qa_data[most_relevant_index - 1].get('回答例', '回答が見つかりませんでした。')
                         print(f"[DEBUG] 閾値({RELEVANCE_THRESHOLD})以上の関連度({max_relevance_score})で回答候補を見つけました。")
                    else:
                         best_match_answer = "申し訳ございません、お探しの情報は見つかりませんでした。別の言葉でお試しいただくか、より詳細な情報をお知らせください。"
                         print(f"[DEBUG] 最大関連度スコア({max_relevance_score})が閾値({RELEVANCE_THRESHOLD})未満です。")
                else:
                    best_match_answer = "申し訳ございません、LLMが適切なQAペアを特定できませんでした。別の言葉でお試しください。"
                    print(f"[DEBUG] LLMから適切な most_relevant_index ({most_relevant_index}) が得られませんでした（範囲外またはNone）。")

            except json.JSONDecodeError as e:
                print(f"[DEBUG] エラー: LLMの応答がJSONとしてパースできませんでした: {e}")
                # パースに失敗した応答をログ出力
                print(f"[DEBUG] パースに失敗した応答:\n{evaluation_response}")
                best_match_answer = "申し訳ございません、関連情報の評価中にエラーが発生しました。別の言葉でお試しください。"
            except Exception as e:
                print(f"[DEBUG] エラー: 評価中に予期せぬエラーが発生しました: {e}")
                import traceback
                # 予期せぬエラーの場合、完全なトレースバックをログ出力
                traceback.print_exc()
                best_match_answer = "申し訳ございません、情報の検索中に問題が発生しました。再度お試しください。"

            # search_qa_by_category 関数の最終結果をログ出力
            print(f"[DEBUG] search_qa_by_category 最終結果: {best_match_answer} (最大関連度スコア: {max_relevance_score if max_relevance_score != -1 else 'N/A'})")
            return best_match_answer

        tools = [search_qa_by_category] # search_qa_by_category のリスト

    # ツールをLLMインスタンスにバインド
    llm_with_tools = llm.bind_tools(tools)


    # ノードの定義
    def classify_category(state: AgentState) -> AgentState:
        """ユーザーの質問がどのカテゴリーに属するかを分類します。"""
        print("[DEBUG] classify_category ノードが実行されました。")
        print(f"[DEBUG] 入力状態: {state}")
        last_message = state["messages"][-1]
        print(f"[DEBUG] 最後のメッセージ: {last_message}")

        # create_agent_app 関数に渡された categories を使用
        category_list_str = ", ".join(f"'{cat}'" for cat in categories)
        print(f"[DEBUG] 利用可能なカテゴリー: {category_list_str}")

        classifier_prompt = f"""
{system_prompt}

ユーザーからの以下の質問が、以下のカテゴリーのどれに最も当てはまるかを判断してください。

利用可能なカテゴリー: {category_list_str}

もし質問が上記のカテゴリーのどれにも当てはまらない、または非常に一般的な質問や個人的な質問の場合は、「その他」と回答してください。
（「その他」というカテゴリーも上記のリストに含まれています）

回答は、上記のカテゴリー名（例: 'ファンクラブ・会員サービス', 'チケット・イベント' など）のいずれか、または「その他」という単語のみを返してください。
余計な文字は含めないでください。

質問: {last_message.content}
分類:
"""
        print(f"[DEBUG] 分類用プロンプト:\n{classifier_prompt}")

        classification_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        classification_response = classification_llm.invoke(classifier_prompt).content.strip()
        print(f"[DEBUG] LLMからの生の応答: {classification_response}")

        # 渡された categories リストに対してチェック
        if classification_response not in categories:
            print(f"[DEBUG] 分類結果 '{classification_response}' が不正なカテゴリーです。")
            classification_response = "その他" # フォールバック
            print(f"[DEBUG] 「その他」にフォールバックしました。")

        print(f"[DEBUG] 最終的な分類結果: {classification_response}")
        return {"predicted_category": classification_response}


    def call_search_tool(state: AgentState) -> AgentState:
        """LLMが `search_qa_by_category` ツールを選択し、ToolCallメッセージを生成します。"""
        print("[DEBUG] call_search_tool ノードが実行されました。")
        last_message = state["messages"][-1]
        predicted_category = state["predicted_category"]

        try:
            # この関数内でバインドされた llm_with_tools を使用
            # LLMを呼び出し、ToolCallを含むAIMessageを生成しようとする
            ai_message_with_tool_call = llm_with_tools.invoke([
                HumanMessage(content=f"ユーザーの質問「{last_message.content}」について、予測されたカテゴリーが「{predicted_category}」です。この情報を使って、社内ドキュメント検索ツール `search_qa_by_category` を使用して情報を検索してください。検索クエリはユーザーの質問内容そのままを渡してください。")
            ])

            if ai_message_with_tool_call.tool_calls:
                print(f"[DEBUG] モデルがツール呼び出しを提案しました: {ai_message_with_tool_call.tool_calls}")
                # ツール呼び出しを含むメッセージを次のノードに渡す
                return {"messages": [ai_message_with_tool_call]}
            else:
                print("[DEBUG] モデルはツール呼び出しを提案しませんでした。最終応答生成に直接進みます。")
                # ツール呼び出しが含まれないAIMessageを次のノードに渡す
                return {"messages": [ai_message_with_tool_call]}

        except Exception as e:
            print(f"[DEBUG] エラー: call_search_toolノードでLLM呼び出し中に予期せぬエラーが発生しました: {e}")
            import traceback
            # エラーの完全なトレースバックをログ出力
            traceback.print_exc()
            # エラーメッセージをUIに表示するため、エラー内容を含むAIMessageを返す
            error_message_content = f"申し訳ございません、ツール呼び出しの準備中にエラーが発生しました。\nエラー詳細: {e}"
            return {"messages": [AIMessage(content=error_message_content)]}


    def generate_final_response(state: AgentState) -> AgentState:
        """最終的なテキスト応答を生成します。"""
        print("[DEBUG] generate_final_response ノードが実行されました。")
        last_message = state["messages"][-1]

        # 履歴から元の HumanMessage の内容を探す
        original_query = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                original_query = msg.content
                break
        if not original_query and state["messages"]:
             # 履歴にHumanMessageがない場合のフォールバック (通常は発生しない想定)
             original_query = state["messages"][0].content


        final_response_content = ""

        if isinstance(last_message, ToolMessage):
            tool_result = last_message.content

            response_prompt = f"""
あなたは{agent_identity}です。
ユーザーの質問「{original_query}」に対する社内ドキュメントの検索結果は以下の通りです。
---
検索結果: {tool_result}
---
この検索結果と{agent_identity}として、ユーザーに分かりやすく、丁寧かつ親しみやすい言葉で回答を生成してください。
もし検索結果が「申し訳ございません、お探しの情報が見つかりませんでした。」または「指定されたカテゴリーには関連情報がありませんでした。」という内容であった場合、ユーザーの質問を理解できなかったことを丁寧に伝え、他に何かお手伝いできることがないか尋ねるようにしてください。
"""
            final_response_content = llm.invoke(response_prompt).content
        else:
             # ツール呼び出しが行われなかった場合（例: カテゴリー分類が「その他」になった場合など）
             general_prompt = f"""
あなたは{agent_identity}です。
ユーザーからの以下の質問「{original_query}」に、丁寧かつ親しみやすい言葉で回答してください。
もし回答できない内容であれば、その旨を伝え、他に何かお手伝いできることがないか尋ねてください。
"""
             final_response_content = llm.invoke(general_prompt).content


        print(f"[DEBUG] 最終応答: {final_response_content}")
        return {"messages": [AIMessage(content=final_response_content)]}


    # LangGraph の構築
    graph = StateGraph(AgentState)

    # ノードの追加
    graph.add_node("classify_category", classify_category)
    graph.add_node("call_search_tool", call_search_tool)
    graph.add_node("tool_executor", ToolNode(tools)) # このスコープで定義された tools を使用
    graph.add_node("generate_final_response", generate_final_response)


    # グラフの開始点
    graph.set_entry_point("classify_category")

    graph.add_edge("classify_category", "call_search_tool")
    graph.add_edge("call_search_tool", "tool_executor")
    graph.add_edge("tool_executor", "generate_final_response")
    graph.add_edge("generate_final_response", END)


    # グラフをコンパイル
    compiled_app = graph.compile()

    print("LangGraph エージェントがコンパイルされました。")
    
    print(f"読み込んだFAQデータ:")
    print(f"- 総レコード数: {len(qa_data)}")
    print(f"- カテゴリー数: {len(categories)}")
    print(f"- カテゴリー一覧: {', '.join(categories)}")
    print(f"- エージェントアイデンティティ: {agent_identity}")

    return compiled_app

# 注: UIから呼び出すため、ここでは app インスタンスを直接作成せず、関数として提供します。
# 実行例はコメントアウトまたは削除してください。

# ### 実行例
# ... (コメントアウトまたは削除)