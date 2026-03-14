"""
hdrag_gradio - Gradio UI for HdRAG.

Wires together InferenceEngine (generation) and HdRAG (memory retrieval).
The "Use Memory" toggle is the dependency boundary: when off, zero calls
to the HDC memory engine.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hdrag_model import (
    Config,
    HuggingFaceTokenizer,
    LlamaServer,
    InferenceEngine,
    ConversationLogger,
    resolve_gguf,
)
from hdrag import HdRAG


def create_gradio_app(
    inference: InferenceEngine,
    hdrag: HdRAG,
    chat_log: ConversationLogger,
    config: Config,
    server: LlamaServer,
    config_path: str = "hdrag_config.yaml",
):
    model_max = inference.context_length
    slider_max = max(min(model_max // 2, 131072), config.max_context_tokens)

    with gr.Blocks(title="HdRAG") as app:
        gr.Markdown(
            f"# \U0001f9e0 HdRAG\n**Model:** {Path(config.gguf_model).stem}"
            f" | **Memories:** {hdrag.count:,}"
        )

        with gr.Tab("\U0001f4ac Chat"):
            chatbot = gr.Chatbot(height=450, label="Conversation")
            with gr.Row():
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Ask something...",
                    scale=4,
                    lines=2,
                )
                budget = gr.Slider(
                    500,
                    slider_max,
                    value=config.max_context_tokens,
                    step=100,
                    label="Token Budget",
                    scale=1,
                )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("New Conversation")
                use_mem = gr.Checkbox(value=True, label="Use Memory")
                top_only = gr.Checkbox(value=False, label="Top Document")
                resp_hist_cb = gr.Checkbox(value=False, label="Response History")

            with gr.Accordion("\U0001f50d Debug: Request Viewer", open=False):
                debug_out = gr.Code(
                    label="Messages",
                    language="markdown",
                    lines=20,
                    value="Send a message to see the request...",
                )

            def respond(message, history, bgt, mem, top1, resp_hist):
                if not message.strip():
                    yield "", history, ""
                    return
                extract = lambda c: (
                    " ".join(b.get("text", "") for b in c if b.get("type") == "text")
                    if isinstance(c, list)
                    else str(c)
                )

                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": "..."})
                yield "", history, f"\U0001f50d Query: {message}"

                # ── Memory retrieval (only if requested) ──
                memory = ""
                if mem and hdrag.count > 0:
                    if top1:
                        results = hdrag.search(message, bgt, track=True)
                        if results:
                            memory = results[0]["memory"]["text"]
                    else:
                        memory = hdrag.get_context(message, bgt, track=True)
                ctx = inference.context_length
                gen_budget = config.max_new_tokens
                sys_prompt = inference.build_system_prompt(memory)
                sys_tokens = inference.count_tokens(sys_prompt)
                msg_tokens = inference.count_tokens(message)
                framing = 4  # ~4 tokens per message for chat template

                # ── Response history (working memory from past LLM outputs) ──
                resp_history_text = ""
                resp_hist_tokens = 0
                if resp_hist:
                    past_turns = history[:-2]
                    resp_budget = max(0, gen_budget - msg_tokens)
                    resp_parts = []
                    resp_total = 0
                    # Newest first — prioritize recent context
                    for t in reversed(past_turns):
                        if t["role"] != "assistant":
                            continue
                        content = extract(t["content"])
                        if not content or content == "...":
                            continue
                        n = inference.count_tokens(content) + 1  # +1 for separator
                        if resp_total + n > resp_budget:
                            break
                        resp_parts.append(content)
                        resp_total += n
                    if resp_parts:
                        resp_parts.reverse()  # restore chronological order
                        resp_history_text = "\n\n".join(resp_parts)
                        resp_hist_tokens = resp_total
                        # Inject into system prompt and reduce gen headroom
                        sys_prompt += (
                            f"\n\n<response_history>\n"
                            f"{resp_history_text}\n"
                            f"</response_history>"
                        )
                        sys_tokens = inference.count_tokens(sys_prompt)
                        gen_budget = max(256, gen_budget - resp_hist_tokens)

                fixed = sys_tokens + msg_tokens + gen_budget + framing * 3
                hist_budget = ctx - fixed

                past = history[:-2]
                hist_turns = []
                for t in past:
                    content = extract(t["content"])
                    n = inference.count_tokens(content) + framing
                    hist_turns.append((t["role"], content, n))

                hist_total = sum(n for _, _, n in hist_turns)
                while hist_total > hist_budget and hist_turns:
                    _, _, dropped = hist_turns.pop(0)
                    hist_total -= dropped

                msgs = [{"role": "system", "content": sys_prompt}]
                msgs += [
                    {"role": role, "content": content}
                    for role, content, _ in hist_turns
                ]
                msgs.append({"role": "user", "content": message})

                # Build debug info before streaming starts
                debug_json = (
                    json.dumps(msgs, indent=2).replace("\\n", "\n").replace("\\t", "\t")
                )
                n_hist = len(hist_turns)
                n_dropped = len(past) - n_hist
                budget_info = (
                    f"ctx={ctx} sys={sys_tokens} hist={hist_total} "
                    f"({n_hist} turns, {n_dropped} dropped) "
                    f"msg={msg_tokens} gen={gen_budget}"
                )
                if resp_hist_tokens > 0:
                    budget_info += f" resp_hist={resp_hist_tokens}"
                debug_text = (
                    f"\U0001f50d Query: {message}\n"
                    f"\U0001f4ca Budget: {budget_info}"
                    f"\n\n\U0001f916 Messages:\n{debug_json}"
                )

                # Stream tokens into the chatbot
                response = ""
                for chunk in inference.generate_stream(msgs):
                    response += chunk
                    history[-1] = {"role": "assistant", "content": response}
                    yield "", history, debug_text

                chat_log.log(message, response)
                server.clear_kv_cache()
                hdrag.compact(server.process)

            def clear():
                chat_log.new_conversation()
                return [], "Send a message to see the request..."

            chat_io = [
                msg,
                chatbot,
                budget,
                use_mem,
                top_only,
                resp_hist_cb,
            ]
            chat_out = [msg, chatbot, debug_out]
            msg.submit(respond, chat_io, chat_out)
            send_btn.click(respond, chat_io, chat_out)
            clear_btn.click(clear, outputs=[chatbot, debug_out])

        with gr.Tab("\u2699\ufe0f Settings"):
            with gr.Row():
                c_dims = gr.Number(label="HDC Dimensions", value=config.hdc_dimensions)
                c_seed = gr.Number(label="HDC Seed", value=config.hdc_seed)
                c_ngram = gr.Number(label="HDC N-gram", value=config.hdc_ngram)
                c_ctx = gr.Slider(
                    500,
                    slider_max,
                    value=config.max_context_tokens,
                    step=100,
                    label="Default Token Budget",
                )
            with gr.Row():
                c_temp = gr.Slider(0, 2, value=config.temperature, label="Temperature")
                c_topp = gr.Slider(0, 1, value=config.top_p, label="Top P")
                c_maxt = gr.Number(label="Max New Tokens", value=config.max_new_tokens)

            with gr.Accordion("Datasets", open=True):
                sc = hdrag.db.source_counts()
                if sc:
                    labels = [f"{s} ({n:,})" for s, n in sc.items()]
                    ds_cb = gr.CheckboxGroup(
                        choices=labels,
                        value=labels,
                        label="Enabled Datasets",
                    )
                    ds_st = gr.Textbox(
                        label="Status",
                        interactive=False,
                        value=f"{len(sc)} datasets enabled",
                    )

                    def upd_src(sel):
                        hdrag.enabled_sources = {s.rsplit(" (", 1)[0] for s in sel}
                        return f"\u2713 {len(hdrag.enabled_sources)} datasets enabled"

                    ds_cb.change(upd_src, [ds_cb], [ds_st])
                else:
                    gr.Markdown("*No indexed datasets. Click 'Index' to build.*")

            with gr.Row():
                save_btn = gr.Button("\U0001f4be Save Config", variant="secondary")
                reindex_btn = gr.Button("\U0001f504 Index", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

            cfg_inputs = [
                c_dims,
                c_seed,
                c_ngram,
                c_ctx,
                c_temp,
                c_topp,
                c_maxt,
            ]

            def save_cfg(dims, seed, ngram, ctx, temp, topp, maxt):
                c = config
                c.hdc_dimensions = int(dims)
                c.hdc_seed = int(seed)
                c.hdc_ngram = int(ngram)
                c.max_context_tokens = int(ctx)
                c.temperature = float(temp)
                c.top_p = float(topp)
                c.max_new_tokens = int(maxt)
                c.save(config_path)
                return "\u2713 Saved"

            save_btn.click(save_cfg, cfg_inputs, status)
            reindex_btn.click(
                lambda *_: f"\u2713 Indexed {hdrag.build_index()} memories",
                cfg_inputs,
                status,
            )

        with gr.Tab("\U0001f4ca Stats"):
            st_out = gr.JSON(label="System Statistics")
            with gr.Row():
                sp_plot = gr.Plot(label="HDV Sparsity")
                sim_plot = gr.Plot(label="Corpus Similarity Distribution")
            dim_plot = gr.Plot(label="Dimension Activation")
            ref_btn = gr.Button("Refresh Stats", variant="primary")

            def compute_stats():
                stats = hdrag.stats()

                sp = hdrag.db.compute_sparsity()
                vals = [sp["positive"], sp["zero"], sp["negative"]]
                sp_fig = go.Figure(
                    go.Bar(
                        x=["+1", "0", "-1"],
                        y=vals,
                        marker_color=[
                            "#4CAF50",
                            "#9E9E9E",
                            "#f44336",
                        ],
                        text=[f"{v:.1%}" for v in vals],
                        textposition="auto",
                    )
                )
                sp_fig.update_layout(
                    title="Ternary Distribution",
                    yaxis_title="Fraction",
                    yaxis_tickformat=".0%",
                    height=300,
                    margin=dict(t=40, b=40),
                )

                sims = hdrag.db.sample_similarities()
                sim_fig = go.Figure()
                if sims:
                    sim_fig.add_trace(
                        go.Histogram(
                            x=sims,
                            nbinsx=50,
                            marker_color="#2196F3",
                            opacity=0.75,
                        )
                    )
                    sim_fig.add_vline(
                        x=np.median(sims),
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"median: {np.median(sims):.0f}",
                    )
                    sim_fig.update_layout(
                        title="Pairwise Similarity",
                        xaxis_title="HDC Score",
                        yaxis_title="Count",
                        height=300,
                        margin=dict(t=40, b=40),
                    )
                else:
                    sim_fig.add_annotation(text="Not enough documents", showarrow=False)

                pf, nf = hdrag.db.dimension_activation()
                dim_fig = go.Figure()
                if len(pf) > 0:
                    step = max(1, len(pf) // 500)
                    x = np.arange(0, len(pf), step)
                    dim_fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=("Positive", "Negative"),
                        vertical_spacing=0.1,
                    )
                    dim_fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=pf[::step],
                            mode="lines",
                            line=dict(color="#4CAF50", width=1),
                        ),
                        row=1,
                        col=1,
                    )
                    dim_fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=nf[::step],
                            mode="lines",
                            line=dict(color="#f44336", width=1),
                        ),
                        row=2,
                        col=1,
                    )
                    dim_fig.update_layout(
                        height=350,
                        margin=dict(t=40, b=40),
                        showlegend=False,
                    )
                    dim_fig.update_xaxes(title_text="Dimension", row=2, col=1)
                    dim_fig.update_yaxes(title_text="Freq", tickformat=".0%")
                else:
                    dim_fig.add_annotation(text="No index loaded", showarrow=False)
                return stats, sp_fig, sim_fig, dim_fig

            ref_btn.click(
                compute_stats,
                outputs=[st_out, sp_plot, sim_plot, dim_plot],
            )
    return app


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="HdRAG")
    parser.add_argument("--config", default="hdrag_config.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    for name in ("httpx", "httpcore", "gradio"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logger = logging.getLogger("hdrag")

    config_path = args.config
    config = Config.load(config_path)
    gguf_path = resolve_gguf(config.gguf_model, config.model_dir)
    if gguf_path:
        logger.info(f"GGUF resolved: {gguf_path}")
    else:
        logger.warning(f"GGUF '{config.gguf_model}' not found in {config.model_dir}")

    tokenizer = HuggingFaceTokenizer.from_gguf(gguf_path, logger=logger)
    server = LlamaServer(config, logger, gguf_path)
    server.start()
    inference = InferenceEngine(config, logger, server, gguf_path, tokenizer=tokenizer)
    hdrag = HdRAG(config, tokenizer=tokenizer, logger=logger)
    chat_log = ConversationLogger(config.chat_history_dir)

    logger.info(f"Server context: {server.context_length:,} tokens")

    create_gradio_app(inference, hdrag, chat_log, config, server, config_path).launch(
        server_port=config.gradio_port
    )


if __name__ == "__main__":
    main()
