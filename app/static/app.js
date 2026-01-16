const LS_KEY = "rag_chats_v3";

function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

function loadState() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return { chats: [], activeId: null };
    return JSON.parse(raw);
  } catch {
    return { chats: [], activeId: null };
  }
}

function saveState(state) {
  localStorage.setItem(LS_KEY, JSON.stringify(state));
}

function makeChat() {
  return {
    id: uid(),
    title: "New chat",
    createdAt: Date.now(),
    messages: [],
  };
}

function titleFromText(t) {
  const s = (t || "").trim().replace(/\s+/g, " ");
  if (!s) return "New chat";
  return s.length > 30 ? s.slice(0, 30) + "…" : s;
}

const els = {
  chatList: document.getElementById("chatList"),
  newChatBtn: document.getElementById("newChatBtn"),
  searchInput: document.getElementById("searchInput"),
  messages: document.getElementById("messages"),
  composerSection: document.getElementById("composerSection"),
  input: document.getElementById("input"),
  sendBtn: document.getElementById("sendBtn"),
  stopBtn: document.getElementById("stopBtn"),
  chatTitle: document.getElementById("chatTitle"),
  debugToggle: document.getElementById("debugToggle"),
  colbertToggle: document.getElementById("colbertToggle"),
  fusionSelect: document.getElementById("fusionSelect"),
  ragToggle: document.getElementById("ragToggle"),
  moreBtn: document.getElementById("moreBtn"),
  moreMenu: document.getElementById("moreMenu"),
  agentToggle: document.getElementById("agentToggle"),
  webToggle: document.getElementById("webToggle"),
};

let state = loadState();
let aborter = null;

// Initialize feature flags with defaults (agent on by default)
state.features = state.features || { rag: true, web: false, agent: true, fusion: "rrf" };
if (state.features.fusion !== "lr" && state.features.fusion !== "rrf") state.features.fusion = "rrf";
saveState(state);

function getActiveChat() {
  if (!state.activeId) return null;
  return state.chats.find(c => c.id === state.activeId) || null;
}

function ensureActiveChat(createIfMissing = true) {
  let chat = getActiveChat();
  if (!chat) {
    if (createIfMissing) {
      chat = makeChat();
      state.chats.unshift(chat);
      state.activeId = chat.id;
      saveState(state);
    } else {
      return null;
    }
  }
  return chat;
}

function renderChatList() {
  const q = (els.searchInput.value || "").toLowerCase();
  els.chatList.innerHTML = "";

  const chats = state.chats
    .filter(c => (c.title || "").toLowerCase().includes(q))
    .sort((a, b) => b.createdAt - a.createdAt);

  for (const chat of chats) {
    const active = chat.id === state.activeId;

    const item = document.createElement("div");
    item.className = "group flex items-center gap-2 px-3 py-2 rounded-xl cursor-pointer";
    item.style.background = active ? "rgba(17,24,39,0.06)" : "transparent";

    item.onmouseenter = () => { if (!active) item.style.background = "rgba(17,24,39,0.04)"; };
    item.onmouseleave = () => { item.style.background = active ? "rgba(17,24,39,0.06)" : "transparent"; };

    const title = document.createElement("div");
    title.className = "flex-1 text-sm truncate";
    title.style.color = "#1f2937";
    title.textContent = chat.title || "New chat";

    const del = document.createElement("button");
    del.className = "opacity-0 group-hover:opacity-100 text-xs px-2 py-1 rounded-lg border";
    del.style.borderColor = "rgba(17,24,39,0.12)";
    del.style.color = "rgba(17,24,39,0.7)";
    del.textContent = "Delete";
    del.onclick = (e) => {
      e.stopPropagation();
      deleteChat(chat.id);
    };

    item.onclick = () => {
      state.activeId = chat.id;
      saveState(state);
      renderAll();
    };

    item.appendChild(title);
    item.appendChild(del);
    els.chatList.appendChild(item);
  }
}

function deleteChat(id) {
  state.chats = state.chats.filter(c => c.id !== id);
  if (state.activeId === id) {
    state.activeId = state.chats.length ? state.chats[0].id : null;
  }
  saveState(state);
  renderAll();
}

function markdownToSafeHtml(mdText) {
  const raw = marked.parse(mdText || "", { breaks: true, gfm: true });
  return DOMPurify.sanitize(raw);
}

function renderMessages() {
  const chat = ensureActiveChat(false);
  els.messages.innerHTML = "";

  if (!chat) {
    if (els.composerSection) els.composerSection.style.display = "none";
    els.chatTitle.textContent = "Welcome";
    const wrap = document.createElement("div");
    wrap.className = "h-full flex items-center justify-center";

    const card = document.createElement("div");
    card.className = "max-w-xl w-full text-center p-10 rounded-2xl border";
    card.style.background = "var(--panel)";
    card.style.borderColor = "var(--border)";
    card.style.boxShadow = "var(--shadow)";

    const title = document.createElement("div");
    title.className = "text-2xl font-semibold mb-3";
    title.textContent = "Welcome to HKU Agent";

    const subtitle = document.createElement("div");
    subtitle.className = "text-sm mb-6";
    subtitle.style.color = "var(--muted)";
    subtitle.textContent = "Start a new chat to ask about HKU CS courses, RAG, or web-enabled queries.";

    const btn = document.createElement("button");
    btn.className = "rounded-xl px-4 py-3 text-sm font-semibold";
    btn.style.background = "var(--accent)";
    btn.style.color = "#fff";
    btn.textContent = "New chat";
    btn.onclick = () => {
      els.newChatBtn.click();
    };

    card.appendChild(title);
    card.appendChild(subtitle);
    card.appendChild(btn);
    wrap.appendChild(card);
    els.messages.appendChild(wrap);
    return;
  }

  if (els.composerSection) els.composerSection.style.display = "block";
  els.chatTitle.textContent = chat.title || "New chat";

  const container = document.createElement("div");
  container.className = "max-w-3xl mx-auto space-y-6";

  for (const m of chat.messages) {
    container.appendChild(renderMessage(m));
  }

  els.messages.appendChild(container);
  els.messages.scrollTop = els.messages.scrollHeight;
}

function renderMessage(m) {
  const row = document.createElement("div");
  row.className = "w-full flex";
  row.style.justifyContent = m.role === "user" ? "flex-end" : "flex-start";

  const bubble = document.createElement("div");
  bubble.className = "rounded-2xl border";
  bubble.style.boxShadow = "0 8px 28px rgba(17, 24, 39, 0.06)";

  if (m.role === "user") {
    bubble.style.maxWidth = "70%";
    bubble.style.background = "#111827";
    bubble.style.borderColor = "rgba(17,24,39,0.25)";
    bubble.style.color = "#ffffff";
    bubble.style.padding = "12px 14px";
  } else {
    bubble.style.maxWidth = "100%";
    bubble.style.background = "#ffffff";
    bubble.style.borderColor = "rgba(17,24,39,0.10)";
    bubble.style.color = "#1f2937";
    bubble.style.padding = "16px 18px";
  }

  const head = document.createElement("div");
  head.className = "text-xs mb-2";
  head.style.color = m.role === "user" ? "rgba(255,255,255,0.7)" : "rgba(31,41,55,0.55)";
  head.textContent = m.role === "user" ? "You" : "Assistant";

  const content = document.createElement("div");
  if (m.role === "assistant") {
    content.className = "md";
    content.innerHTML = markdownToSafeHtml(m.content || "");
  } else {
    content.className = "text-sm whitespace-pre-wrap leading-6";
    content.textContent = m.content || "";
  }

  bubble.appendChild(head);
  bubble.appendChild(content);

  if (m.meta) {
    const details = document.createElement("details");
    details.className = "mt-4 text-xs";
    details.style.color = "rgba(31,41,55,0.6)";

    const sum = document.createElement("summary");
    sum.className = "cursor-pointer select-none";
    sum.textContent = "Details";

    const pre = document.createElement("pre");
    pre.className = "mt-2 p-3 rounded-xl border overflow-auto";
    pre.style.background = "#fbfbfb";
    pre.style.borderColor = "rgba(17,24,39,0.10)";
    pre.textContent = JSON.stringify(m.meta, null, 2);

    details.appendChild(sum);
    details.appendChild(pre);
    bubble.appendChild(details);
  }

  row.appendChild(bubble);
  return row;
}

function renderAll() {
  renderChatList();
  renderMessages();
  renderFeatureToggles();
}

function pushMessage(role, content, meta = null) {
  const chat = ensureActiveChat();
  chat.messages.push({ role, content, meta });

  if (role === "user" && chat.messages.length === 1) {
    chat.title = titleFromText(content);
  }
  chat.createdAt = Date.now();
  saveState(state);
  renderAll();
}

function updateLastAssistant(deltaText) {
  const chat = ensureActiveChat();
  const last = chat.messages[chat.messages.length - 1];
  if (!last || last.role !== "assistant") return;

  last.content = (last.content || "") + (deltaText || "");
  saveState(state);

  // Update UI incrementally: re-render markdown for the last assistant bubble
  const container = els.messages.querySelector(".max-w-3xl");
  if (!container) return;
  const lastRow = container.lastElementChild;
  if (!lastRow) return;

  const mdNode = lastRow.querySelector(".md");
  if (!mdNode) return;

  mdNode.innerHTML = markdownToSafeHtml(last.content);
  els.messages.scrollTop = els.messages.scrollHeight;
}

function setLastAssistantMeta(meta) {
  const chat = ensureActiveChat();
  const last = chat.messages[chat.messages.length - 1];
  if (!last || last.role !== "assistant") return;
  last.meta = meta;
  saveState(state);
  renderAll();
}

function appendTrace(event, payload) {
  const chat = ensureActiveChat();
  const last = chat.messages[chat.messages.length - 1];
  if (!last || last.role !== "assistant") return;
  if (!last.meta) last.meta = {};
  if (!last.meta.trace) last.meta.trace = [];
  last.meta.trace.push({ event, payload });
  saveState(state);
  renderAll();
}

async function send() {
  const text = els.input.value.trim();
  const use_colbert = !!state.features.rag && !!els.colbertToggle.checked;
  const rag_enabled = !!state.features.rag;
  const fusion_mode = (state.features.fusion === "lr" ? "lr" : "rrf");
  if (!text) return;
  if (aborter) return;

  pushMessage("user", text);
  pushMessage("assistant", "");

  els.input.value = "";
  autosize();

  const debug = !!els.debugToggle.checked;
  const enable_agent = !!state.features.agent;
  const enable_web = !!state.features.web;
  aborter = new AbortController();
  els.stopBtn.disabled = false;

   const chat = ensureActiveChat();
   const session_id = chat.id;
   const max_steps = 6;

  try {
    const res = await fetch("/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        debug,
        use_colbert,
        rag_enabled,
        fusion_mode,
        session_id,
        max_steps,
        agent_enabled: enable_agent,
        web_enabled: enable_web,
      }),
      signal: aborter.signal,
    });

    if (!res.ok || !res.body) {
      updateLastAssistant(`\n[HTTP ${res.status}]`);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buf = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buf += decoder.decode(value, { stream: true });

      const parts = buf.split("\n\n");
      buf = parts.pop() || "";

      for (const block of parts) {
        const lines = block.split("\n").filter(Boolean);
        let event = "message";
        let data = "";

        for (const ln of lines) {
          if (ln.startsWith("event:")) event = ln.slice(6).trim();
          if (ln.startsWith("data:")) data += ln.slice(5).trim();
        }

        if (!data) continue;

        let obj = null;
        try { obj = JSON.parse(data); } catch { obj = { raw: data }; }

        if (event === "meta") {
          setLastAssistantMeta(obj);
        } else if (event === "plan_update" || event === "tool_start" || event === "tool_end") {
          appendTrace(event, obj);
        } else if (event === "delta") {
          updateLastAssistant(obj.text || "");
        } else if (event === "error") {
          updateLastAssistant(`\n[Error] ${obj.message || "unknown"}`);
        } else if (event === "done") {
          // Non-stream cases deliver the full answer only in the done payload.
          const chat = ensureActiveChat();
          const last = chat.messages[chat.messages.length - 1];
          const hasContent = !!(last && last.role === "assistant" && last.content);
          if (obj.answer && !hasContent) updateLastAssistant(obj.answer);
          appendTrace("done", obj);
        }
      }
    }
  } catch (e) {
    updateLastAssistant(`\n[Request aborted or failed]`);
  } finally {
    aborter = null;
    els.stopBtn.disabled = true;
  }
}

function autosize() {
  const el = els.input;
  el.style.height = "auto";
  const minH = 52;
  const maxH = 220;
  const h = Math.min(Math.max(el.scrollHeight, minH), maxH);
  el.style.height = h + "px";
  el.style.overflowY = el.scrollHeight > maxH ? "auto" : "hidden";
}

els.newChatBtn.onclick = () => {
  const chat = makeChat();
  // Add a fresh chat to the top; keep history
  state.chats.unshift(chat);
  state.activeId = chat.id;
  saveState(state);
  renderAll();
};

function renderFeatureToggles() {
  // RAG button visual
  if (els.ragToggle) {
    const active = !!state.features.rag;
    els.ragToggle.classList.toggle("on", active);
    const stateLabel = els.ragToggle.querySelector(".state");
    if (stateLabel) stateLabel.textContent = active ? "On" : "Off";
    els.ragToggle.setAttribute("aria-pressed", active ? "true" : "false");
  }

  // Checkboxes in dropdown
  if (els.agentToggle) els.agentToggle.checked = !!state.features.agent;
  if (els.webToggle) els.webToggle.checked = !!state.features.web;
  if (els.colbertToggle) {
    els.colbertToggle.disabled = !state.features.rag;
  }
  if (els.fusionSelect) {
    els.fusionSelect.value = (state.features.fusion === "lr" ? "lr" : "rrf");
    els.fusionSelect.disabled = !state.features.rag;
  }
}

if (els.ragToggle) {
  els.ragToggle.onclick = () => {
    state.features.rag = !state.features.rag;
    saveState(state);
    renderFeatureToggles();
  };
}

if (els.moreBtn && els.moreMenu) {
  els.moreBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    els.moreMenu.classList.toggle("hidden");
  });

  // Keep menu clickable
  els.moreMenu.addEventListener("click", (e) => {
    e.stopPropagation();
  });

  document.addEventListener("click", () => {
    els.moreMenu.classList.add("hidden");
  });
}

if (els.agentToggle) {
  els.agentToggle.onchange = () => {
    state.features.agent = !!els.agentToggle.checked;
    saveState(state);
  };
}

if (els.webToggle) {
  els.webToggle.onchange = () => {
    state.features.web = !!els.webToggle.checked;
    saveState(state);
  };
}

if (els.fusionSelect) {
  els.fusionSelect.onchange = () => {
    const v = (els.fusionSelect.value || "rrf").toLowerCase();
    state.features.fusion = (v === "lr" ? "lr" : "rrf");
    saveState(state);
  };
}

els.sendBtn.onclick = send;

els.stopBtn.onclick = () => {
  if (aborter) aborter.abort();
};

els.searchInput.oninput = renderChatList;

els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

els.input.addEventListener("input", autosize);

function bootstrap() {
  renderAll();
  autosize();
}

bootstrap();
