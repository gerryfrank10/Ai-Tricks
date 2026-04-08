/**
 * Ask Docs — AI chat widget for AI Tricks documentation
 *
 * Powered by Claude API (direct browser access).
 * Users supply their own Anthropic API key, stored in sessionStorage.
 *
 * Features:
 *  - Floating action button on every page
 *  - Current page content auto-included as context
 *  - Streaming responses via SSE
 *  - Markdown rendering for code blocks
 *  - API key prompt with privacy notice
 */

(function () {
  "use strict";

  // ── Config ──────────────────────────────────────────────────────────────────
  const API_URL = "https://api.anthropic.com/v1/messages";
  const MODEL = "claude-haiku-4-5-20251001";
  const STORAGE_KEY = "aitricks_claude_api_key";
  const MAX_CONTEXT_CHARS = 6000; // Limit page content to keep costs low

  // ── State ───────────────────────────────────────────────────────────────────
  let isOpen = false;
  let isStreaming = false;
  let conversationHistory = [];

  // ── DOM helpers ─────────────────────────────────────────────────────────────
  function getPageContext() {
    const content = document.querySelector(".md-content__inner");
    if (!content) return "";
    return (content.innerText || content.textContent || "")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, MAX_CONTEXT_CHARS);
  }

  function getPageTitle() {
    const h1 = document.querySelector(".md-content h1");
    return h1 ? h1.textContent.trim() : document.title;
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function renderMarkdown(text) {
    // Lightweight markdown rendering for the chat bubble
    return escapeHtml(text)
      // Code blocks ```lang\n...\n```
      .replace(
        /```(\w*)\n([\s\S]*?)```/g,
        (_, lang, code) =>
          `<pre style="background:rgba(0,0,0,0.15);padding:0.6em;border-radius:6px;overflow-x:auto;margin:0.5em 0;font-size:0.8em"><code>${code.trim()}</code></pre>`
      )
      // Inline code `...`
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      // Bold **...**
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      // Italic *...*
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      // Newlines
      .replace(/\n/g, "<br>");
  }

  // ── Build widget HTML ────────────────────────────────────────────────────────
  function buildWidget() {
    // FAB button
    const fab = document.createElement("button");
    fab.id = "ask-docs-fab";
    fab.title = "Ask AI Docs";
    fab.setAttribute("aria-label", "Ask AI about this page");
    fab.innerHTML = "🤖";

    // Panel
    const panel = document.createElement("div");
    panel.id = "ask-docs-panel";
    panel.className = "hidden";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "Ask AI Docs");
    panel.innerHTML = `
      <div class="ask-docs-header">
        <div>
          <h4>Ask AI Docs</h4>
          <span>Powered by Claude</span>
        </div>
        <button class="ask-docs-close" aria-label="Close" id="ask-docs-close">✕</button>
      </div>
      <div class="ask-docs-messages" id="ask-docs-messages">
        <div class="ask-docs-msg ask-docs-msg--system">
          Ask anything about <strong id="ask-docs-page-title"></strong>.<br>
          Responses are grounded in this page's content.
        </div>
      </div>
      <div class="ask-docs-input-area">
        <input
          type="text"
          id="ask-docs-input"
          placeholder="Ask a question…"
          autocomplete="off"
          maxlength="500"
        />
        <button id="ask-docs-send" type="button">Send</button>
      </div>
      <div class="ask-docs-apikey-notice" id="ask-docs-key-notice" style="display:none">
        🔑 <a href="#" id="ask-docs-set-key">Set API key</a> ·
        <a href="https://console.anthropic.com/" target="_blank" rel="noopener">Get a key</a>
      </div>
    `;

    document.body.appendChild(fab);
    document.body.appendChild(panel);

    return { fab, panel };
  }

  // ── API key management ───────────────────────────────────────────────────────
  function getApiKey() {
    return sessionStorage.getItem(STORAGE_KEY) || "";
  }

  function promptApiKey() {
    const existing = getApiKey();
    const key = window.prompt(
      "Enter your Anthropic API key to use Ask AI Docs.\n\n" +
      "Your key is stored in sessionStorage (this tab only) and sent directly to api.anthropic.com — never to any third-party server.\n\n" +
      "Get a free key at: console.anthropic.com",
      existing
    );
    if (key && key.startsWith("sk-ant-")) {
      sessionStorage.setItem(STORAGE_KEY, key.trim());
      return key.trim();
    } else if (key !== null) {
      window.alert("Invalid API key. It should start with 'sk-ant-'.");
    }
    return null;
  }

  // ── Message rendering ────────────────────────────────────────────────────────
  function addMessage(role, content, streaming = false) {
    const msgs = document.getElementById("ask-docs-messages");
    const div = document.createElement("div");
    div.className = `ask-docs-msg ask-docs-msg--${role === "user" ? "user" : "ai"}`;
    if (streaming) div.id = "ask-docs-streaming-msg";
    div.innerHTML = role === "user" ? escapeHtml(content) : renderMarkdown(content);
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return div;
  }

  function showTypingIndicator() {
    const msgs = document.getElementById("ask-docs-messages");
    const div = document.createElement("div");
    div.className = "ask-docs-typing";
    div.id = "ask-docs-typing";
    div.innerHTML = "<span></span><span></span><span></span>";
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function removeTypingIndicator() {
    const el = document.getElementById("ask-docs-typing");
    if (el) el.remove();
  }

  // ── Streaming API call ───────────────────────────────────────────────────────
  async function askClaude(userMessage) {
    const apiKey = getApiKey();
    if (!apiKey) {
      const key = promptApiKey();
      if (!key) return;
    }

    const pageContext = getPageContext();
    const systemPrompt =
      `You are an expert AI assistant embedded in the AI Tricks documentation site.\n` +
      `Answer questions helpfully and concisely based on the page content provided.\n` +
      `If the answer isn't in the page context, say so and give a brief general answer.\n` +
      `Use markdown formatting: **bold**, \`code\`, and code blocks for examples.\n\n` +
      `Current page: "${getPageTitle()}"\n\n` +
      `Page content:\n${pageContext}`;

    conversationHistory.push({ role: "user", content: userMessage });

    isStreaming = true;
    document.getElementById("ask-docs-send").disabled = true;
    document.getElementById("ask-docs-input").disabled = true;

    showTypingIndicator();

    try {
      const resp = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": getApiKey(),
          "anthropic-version": "2023-06-01",
          "anthropic-dangerous-direct-browser-access": "true",
        },
        body: JSON.stringify({
          model: MODEL,
          max_tokens: 1024,
          system: systemPrompt,
          messages: conversationHistory,
          stream: true,
        }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        const msg = err?.error?.message || `HTTP ${resp.status}`;
        if (resp.status === 401) {
          sessionStorage.removeItem(STORAGE_KEY);
          throw new Error("Invalid API key. Please set a valid key.");
        }
        throw new Error(msg);
      }

      removeTypingIndicator();

      // Stream the response
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let fullText = "";
      let msgDiv = addMessage("assistant", "▌", true);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") break;
          try {
            const json = JSON.parse(data);
            if (json.type === "content_block_delta" && json.delta?.text) {
              fullText += json.delta.text;
              msgDiv.innerHTML = renderMarkdown(fullText) + "<span style='opacity:0.5'>▌</span>";
              document.getElementById("ask-docs-messages").scrollTop =
                document.getElementById("ask-docs-messages").scrollHeight;
            }
          } catch (_) {}
        }
      }

      // Finalise
      msgDiv.id = "";
      msgDiv.innerHTML = renderMarkdown(fullText);
      conversationHistory.push({ role: "assistant", content: fullText });

    } catch (err) {
      removeTypingIndicator();
      addMessage(
        "assistant",
        `⚠️ Error: ${err.message}\n\nIf this is an API key issue, click **Set API key** below.`
      );
      document.getElementById("ask-docs-key-notice").style.display = "block";
    } finally {
      isStreaming = false;
      const sendBtn = document.getElementById("ask-docs-send");
      const input = document.getElementById("ask-docs-input");
      if (sendBtn) sendBtn.disabled = false;
      if (input) { input.disabled = false; input.focus(); }
    }
  }

  // ── Event handlers ───────────────────────────────────────────────────────────
  function sendMessage() {
    if (isStreaming) return;
    const input = document.getElementById("ask-docs-input");
    const text = (input.value || "").trim();
    if (!text) return;

    input.value = "";
    addMessage("user", text);
    askClaude(text);
  }

  function togglePanel() {
    isOpen = !isOpen;
    const panel = document.getElementById("ask-docs-panel");
    const fab = document.getElementById("ask-docs-fab");
    if (isOpen) {
      panel.classList.remove("hidden");
      fab.innerHTML = "✕";
      // Set page title in greeting
      const titleEl = document.getElementById("ask-docs-page-title");
      if (titleEl) titleEl.textContent = getPageTitle();
      document.getElementById("ask-docs-input").focus();
      // Show key notice if no key set
      if (!getApiKey()) {
        document.getElementById("ask-docs-key-notice").style.display = "block";
      }
      // Reset conversation on new page
      if (conversationHistory.length === 0) {
        // Already clear
      }
    } else {
      panel.classList.add("hidden");
      fab.innerHTML = "🤖";
    }
  }

  // ── Init ─────────────────────────────────────────────────────────────────────
  function init() {
    const { fab } = buildWidget();

    // FAB click
    fab.addEventListener("click", togglePanel);

    // Close button
    document.getElementById("ask-docs-close").addEventListener("click", togglePanel);

    // Send button
    document.getElementById("ask-docs-send").addEventListener("click", sendMessage);

    // Enter key
    document.getElementById("ask-docs-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    // Set API key link
    document.getElementById("ask-docs-set-key").addEventListener("click", (e) => {
      e.preventDefault();
      promptApiKey();
    });

    // Close on Escape
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && isOpen) togglePanel();
    });

    // Reset conversation when page changes (MkDocs instant nav)
    document.addEventListener("DOMContentLoaded", () => {
      conversationHistory = [];
    });

    // Expose public API
    window.askDocs = {
      open: () => { if (!isOpen) togglePanel(); },
      close: () => { if (isOpen) togglePanel(); },
    };
  }

  // Run after DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
