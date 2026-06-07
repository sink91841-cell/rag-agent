"""RAG Agent 演示脚本 - 智能文档知识库问答.

用法:
    python demo.py ingest <路径>       # 摄入文档到知识库
    python demo.py ask "<问题>"        # 提问
    python demo.py stats               # 查看知识库统计
    python demo.py reset               # 清空知识库
    python demo.py demo                # 完整演示流程
"""

import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.agent import RagAgent

# Weaviate API key (from Dify docker deployment)
WEAVIATE_API_KEY = "WVF5YThaHlkYwhGUSmCRgsX3tD5ngdN8pkih"

console = Console(force_terminal=True)


def cmd_ingest(agent: RagAgent, args: list[str]) -> None:
    """摄入文档命令."""
    path = args[0] if args else "."
    target = Path(path)

    console.print(f"[bold cyan][INGEST] 正在加载文档: {path}[/]")
    start = time.time()

    if target.is_file():
        count = agent.ingest_file(str(target))
    elif target.is_dir():
        count = agent.ingest_directory(str(target))
    else:
        console.print(f"[red]路径不存在: {path}[/]")
        return

    elapsed = time.time() - start
    if count > 0:
        console.print(f"[green][OK] 成功摄入 {count} 个文本块 (耗时 {elapsed:.1f}s)[/]")
    else:
        console.print("[yellow][WARN] 未摄入任何内容，请检查文件格式是否支持[/]")
        console.print("[dim]支持格式: PDF, Markdown, TXT, DOCX, 代码文件[/]")


def cmd_ask(agent: RagAgent, args: list[str], top_k: int = 5) -> None:
    """问答命令."""
    if not args:
        console.print("[red]请提供问题[/]")
        return

    query = " ".join(args)
    # 临时修改 top_k
    agent.retriever.top_k = top_k

    console.print(f"[bold cyan][SEARCH] 检索中: {query}[/]")
    start = time.time()

    results = agent.retrieve(query)
    context = agent.retriever.build_context(results)
    elapsed = time.time() - start

    # 展示检索结果
    table = Table(title=f"检索结果 ({len(results)} 条, {elapsed:.2f}s)", show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("来源", style="cyan", max_width=40)
    table.add_column("片段", style="dim", width=5)
    table.add_column("相似度", style="green", width=8)
    table.add_column("内容预览", max_width=60)

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            Path(r["source"]).name,
            str(r["chunk_index"]),
            f"{r['score']:.4f}",
            r["content"][:120].replace("\n", " "),
        )

    console.print(table)
    console.print()

    # 展示上下文（供 LLM 使用或人工阅读）
    console.print(Panel(context[:3000], title="[上下文] 检索结果 (可直接喂给 LLM)", border_style="blue"))


def cmd_stats(agent: RagAgent) -> None:
    """查看知识库统计."""
    try:
        s = agent.stats()
        table = Table(title="知识库统计")
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="green")
        table.add_row("已索引块数", str(s.chunk_count))
        console.print(table)
    except Exception as e:
        console.print(f"[red]获取统计失败: {e}[/]")


def cmd_reset(agent: RagAgent) -> None:
    """清空知识库."""
    agent.reset()
    console.print("[yellow][CLEAR] 知识库已清空[/]")


def cmd_demo(agent: RagAgent) -> None:
    """完整演示: 摄入示例文档 → 多次问答."""
    console.print(Panel.fit("[bold]RAG Agent 完整演示[/]", border_style="green"))

    # 1. 创建示例文档
    demo_dir = Path(__file__).parent / "data" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    demo_file = demo_dir / "rag_intro.md"
    demo_content = """# RAG 技术详解

## 什么是 RAG

RAG (Retrieval-Augmented Generation) 即检索增强生成，是一种将信息检索与文本生成相结合的技术架构。

### 核心思想

传统 LLM 受限于训练数据的截止时间和未见的私有知识。RAG 通过以下流程解决这些问题:

1. **离线索引阶段**: 将文档分割成块 (chunk)，使用 embedding 模型将每个块转为向量，存入向量数据库。
2. **在线查询阶段**: 用户提问 → 同样转为向量 → 在向量库中检索最相似的 K 个块 → 将检索结果作为上下文注入 prompt → LLM 基于这些上下文生成答案。

### 关键组件

- **Embedding 模型**: 将文本映射到高维向量空间，语义相近的文本向量距离近。常用模型包括 text-embedding-3-small, all-MiniLM-L6-v2 等。
- **向量数据库**: 存储和检索高维向量。代表产品有 Weaviate, Milvus, Pinecone, Chroma, Qdrant 等。
- **分块策略**: 太大则检索精度下降，太小则丢失上下文。常见策略为 512 token + 64 token 重叠。
- **检索策略**: 纯向量检索、混合检索（向量 + BM25 关键词）、重排序（cross-encoder rerank）等。

## RAG 的应用场景

1. **企业知识库问答**: 将公司内部文档索引后，员工可用自然语言查询规章制度、产品文档等。
2. **客户支持**: 基于产品手册自动回答用户问题，减少人工客服负担。
3. **学术研究**: 快速检索大量论文，辅助文献综述写作。
4. **法律合规**: 检索法规条文和判例，辅助律师工作。

## 进阶技术

### Agentic RAG

将 RAG 与 AI Agent 结合，Agent 可以自主决定:
- 是否需要检索（而不是每次都检）
- 检索哪些数据源
- 是否需要多轮检索
- 检索结果是否需要验证

### 多模态 RAG

不仅检索文本，还能检索图片、表格、图表等多模态内容，适用于更复杂的知识管理场景。

### Graph RAG

结合知识图谱和向量检索，既能做模糊语义匹配，又能利用实体关系做精确推理。
"""
    demo_file.write_text(demo_content, encoding="utf-8")
    console.print("[dim]已创建示例文档[/]")

    # 2. 摄入
    console.print()
    cmd_ingest(agent, [str(demo_file)])

    # 3. 多次问答
    questions = [
        "RAG 的核心原理是什么?",
        "RAG 有哪些关键组件?",
        "什么是 Agentic RAG?",
    ]
    for q in questions:
        console.print()
        console.rule(f"[bold yellow]{q}[/]")
        cmd_ask(agent, [q], top_k=3)


def main() -> None:
    if len(sys.argv) < 2:
        console.print("[bold]RAG Agent - 智能文档知识库问答系统[/]")
        console.print()
        console.print("用法:")
        console.print("  python demo.py ingest <文件/目录>    摄入文档")
        console.print("  python demo.py ask <问题>             检索问答")
        console.print("  python demo.py stats                  查看统计")
        console.print("  python demo.py reset                  清空知识库")
        console.print("  python demo.py demo                   完整演示")
        console.print()
        console.print("示例:")
        console.print("  python demo.py ingest ./docs/")
        console.print("  python demo.py ask '什么是 RAG?'")
        return

    command = sys.argv[1].lower()
    agent = RagAgent(weaviate_api_key=WEAVIATE_API_KEY)

    try:
        if command == "ingest":
            cmd_ingest(agent, sys.argv[2:])
        elif command == "ask":
            # 支持 --top-k N 参数
            args = sys.argv[2:]
            top_k = 5
            if "--top-k" in args:
                idx = args.index("--top-k")
                top_k = int(args[idx + 1])
                args = args[:idx] + args[idx + 2 :]
            cmd_ask(agent, args, top_k=top_k)
        elif command == "stats":
            cmd_stats(agent)
        elif command == "reset":
            cmd_reset(agent)
        elif command == "demo":
            cmd_demo(agent)
        else:
            console.print(f"[red]未知命令: {command}[/]")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
